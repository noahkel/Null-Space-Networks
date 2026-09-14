"""Radon operator for limited-angle CT.

One backend: ``MatrixRadonAdapter`` builds the explicit sparse system matrices A
(all angles) and A_la (the measured angles only) with the ASTRA Toolbox, and
keeps a truncated SVD of A_la. That makes the pseudoinverse, the range projector
and the null-space projector exact tensor algebra rather than the output of an
iterative solver -- and, because they are ordinary tensor products, PyTorch
differentiates them, which is what lets an attack gradient travel from the image
back into the sinogram.

Images are (B, C, res, res) and sinograms are (B, C, n_angles, det_count): the
limited-angle operators return a *full-shape* sinogram with the unmeasured rows
zeroed, so every sinogram in a run has the same shape.
"""
import hashlib
import math
import os
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.sparse
import torch


# ---------------------------------------------------------------------------
# Sparse-matrix backend
#
# Stores two system matrices -- A (all angles) and A_la (the measured angles
# only) -- plus a truncated SVD of A_la, which is what makes the pseudoinverse
# and the null-space projector exact rather than iterative.
# ---------------------------------------------------------------------------

class MatrixRadonAdapter:
    """
    Radon adapter backed by precomputed sparse system matrices A and A_la,
    with truncated SVD factors of A_la stored for pseudoinverse and null-space
    operations.

    Operator algebra (image x in R^{n}, n = resolution^2; sinogram in R^{m}).
    A is the full-angle system matrix, A_la its restriction to the measured
    (limited-angle) rows. With the truncated SVD  A_la = U_k Σ_k V_k^T  (keeping
    singular values s_i ≥ svd_threshold · s_max):

      forward        y      = A x            (·dx)          -> forward()
      forward_la     y_la   = A_la x                        -> forward_la()
      A_la^+         = V_k Σ_k^{-1} U_k^T     (pseudoinverse) -> backward_la()
      P_ran          = U_k U_k^T, projector onto range(A_la) -> proj_ran()
      P_N            = I - A_la^+ A_la = I - V_k V_k^T       -> proj_null_image()
                     image-domain projector onto null(A_la);  A_la P_N = 0
      decompose      e_ran = A_la^+ A_la e,  e_nul = (I - A_la^+ A_la) e,
                     with ||e||^2 = ||e_ran||^2 + ||e_nul||^2  -> decompose_error()
      ||A||_2        largest singular value of A (power iteration) -> norm_A / norm_A2

    Note that P_ran is *not* the 0/1 mask that keeps the measured rows: it is the
    projector onto range(A_la), a subspace of the measured rows, because not
    every sinogram supported on the measured angles is the measurement of some
    image.

    Parameters
    ----------
    resolution : int
        Square image side length (pixels).
    angles : np.ndarray
        All projection angles in radians.
    det_count : int
        Number of detector elements.
    phi : (float, float)
        Limited-angle window [lo, hi) in radians.  Required.
    svd_threshold : float
        Relative singular-value cutoff: retain singular values >= threshold * s_max.
        Must be > 0 to build SVD factors and enable backward_la / null-space methods.
    dataset : str or None
        Optional label (stored, not used internally).
    dx : float
        Pixel-spacing scale factor applied to forward output.
    estimate_norm : bool
        Run power iteration to estimate norm_A and norm_A2 from the sparse A.
    norm_iters : int
        Maximum power-iteration steps.
    device : torch.device or None
        Target device for tensors.
    dtype : torch.dtype
        Floating-point dtype.
    dense : bool
        Store A and A_la as dense tensors and apply them with cuBLAS matmuls
        instead of sparse CSR kernels. Radon matrices are only ~1% sparse-
        friendly on GPU, so dense float32 is typically much faster and avoids
        the beta sparse-CSR autograd kernels. Costs O(m*n) memory per matrix
        (~2 GB for 128x128 / 180 angles in float32).
    cache_dir : str or Path or None
        Directory for caching matrices and SVD factors.
    """

    def __init__(
        self,
        resolution: int,
        angles: np.ndarray,
        det_count: int,
        phi: Tuple[float, float],
        svd_threshold: float = 0.0,
        dataset: Union[str, None] = None,
        dx: float = 1.0,
        estimate_norm: bool = True,
        norm_iters: int = 20,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        dense: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        if phi is None:
            raise ValueError("phi=(lo, hi) is required for MatrixRadonAdapter.")

        self.resolution = int(resolution)
        self.det_count = int(det_count)
        self.angles = np.asarray(angles, dtype=np.float64)
        self.phi = phi
        self.svd_threshold = float(svd_threshold)
        self.dx = float(dx)
        self.dataset = (dataset or "").lower()
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.dense = bool(dense)
        self.norm_A: Optional[float] = None
        self.norm_A2: Optional[float] = None

        min_det = math.ceil(math.sqrt(2) * self.resolution)
        if self.det_count < min_det:
            warnings.warn(
                f"det_count={self.det_count} may clip image corners "
                f"(recommended minimum: {min_det}).",
                UserWarning, stacklevel=2,
            )

        # Limited-angle angle subset
        _la_mask = (self.angles >= phi[0]) & (self.angles < phi[1])
        self.angles_la: np.ndarray = self.angles[_la_mask]
        self._la_row_mask: np.ndarray = np.repeat(_la_mask, self.det_count)

        cache_path = Path(cache_dir) / self._cache_key() if cache_dir is not None else None
        print(f"Cache path: {cache_path}")
        if cache_path is not None and self._try_load_cache(cache_path):
            print(f"Loaded matrix cache from {cache_path}")
        else:
            try:
                import astra as _astra
            except ImportError:
                raise ImportError(
                    "astra-toolbox is required. Install with:\n"
                    "  conda install -c astra-toolbox astra-toolbox"
                )
            self._build_matrices(_astra)
            if cache_path is not None:
                print(f"Saving matrix cache to {cache_path}")
                self._save_cache(cache_path)

        if estimate_norm:
            self._estimate_operator_norm(iters=norm_iters)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_la(self) -> int:
        """Number of limited-angle projections."""
        return int(self._la_row_mask.sum()) // self.det_count

    # ------------------------------------------------------------------
    # Matrix / SVD construction
    # ------------------------------------------------------------------

    def _build_matrices(self, astra) -> None:
        """Build sparse A, sparse A_la, and (if svd_threshold > 0) the SVD of A_la.

        A itself is never decomposed: nothing in the pipeline inverts the
        full-angle operator, so its SVD would be the most expensive computation
        in the run for no consumer.
        """
        vol_geom = astra.create_vol_geom(self.resolution, self.resolution)
        proj_geom = astra.create_proj_geom('parallel', 1.0, self.det_count, self.angles)

        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        try:
            matrix_id = astra.projector.matrix(proj_id)
            try:
                csr: scipy.sparse.csr_matrix = astra.matrix.get(matrix_id)
            finally:
                astra.matrix.delete(matrix_id)
        finally:
            astra.projector.delete(proj_id)

        # Cast by the adapter's dtype. Casting by layout (float32 whenever dense)
        # would hand a dense float64 adapter float32-rounded entries to decompose.
        csr = csr.astype(np.float32 if self.dtype == torch.float32 else np.float64)

        # Full system matrix
        self._A = self._csr_to_torch(csr)
        print(f"Built sparse A, shape {tuple(csr.shape)}")

        # Limited-angle submatrix. ASTRA's matrix is angle-major (row index =
        # angle * det_count + detector), which is what _la_row_mask assumes.
        csr_la = csr[self._la_row_mask, :]
        self._A_la = self._csr_to_torch(csr_la)
        print(f"Built sparse A_la, shape {tuple(csr_la.shape)}")

        if self.svd_threshold > 0:
            print("Computing SVD of A_la ...")
            self._U_k_la, self._s_k_la, self._Vt_k_la = self._truncated_svd(csr_la)

    def _truncated_svd(
        self, csr: scipy.sparse.csr_matrix
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the truncated SVD of a sparse matrix.

        Returns U_k (m,k), s_k (k,), Vt_k (k,n) as torch tensors on self.device,
        retaining singular values >= svd_threshold * s_max.

        """
        np_dtype = np.float32 if self.dtype == torch.float32 else np.float64
        m, n = csr.shape

        def _t(arr) -> torch.Tensor:
            if isinstance(arr, torch.Tensor):
                return arr.to(device=self.device, dtype=self.dtype)
            return torch.from_numpy(np.asarray(arr)).to(
                device=self.device, dtype=self.dtype
            )

        def _cut_and_return(U_np, s_np, Vt_np, source: str):
            s_np = np.asarray(s_np, dtype=np.float64)
            cutoff = self.svd_threshold * s_np[0]
            keep = s_np >= cutoff
            print(f"  {m}×{n}: {keep.sum()}/{len(s_np)} singular values retained "
                  f"(s_max={s_np[0]:.3e}, cutoff={cutoff:.3e})  [{source}]")
            U_k  = _t(U_np[:, keep])
            s_k  = _t(s_np[keep])
            Vt_k = _t(Vt_np[keep, :])
            for name, t in (("U", U_k), ("s", s_k), ("Vt", Vt_k)):
                if not torch.isfinite(t).all():
                    raise RuntimeError(
                        f"SVD factor {name} from [{source}] contains NaN/Inf. "
                        "Try a larger svd_threshold."
                    )
            return U_k, s_k, Vt_k

        # ------------------------------------------------------------------
        # GPU path: full thin SVD on the densified matrix
        # ------------------------------------------------------------------

        if self.device.type == "cuda":
            itemsize = np.dtype(np_dtype).itemsize
            print(f"  densifying {m}×{n} on GPU ({m * n * itemsize / 1e9:.1f} GB, {np_dtype.__name__})")
            dense = None
            try:
                dense = torch.from_numpy(csr.toarray().astype(np_dtype)).to(self.device)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    U_t, s_t, Vh_t = torch.linalg.svd(dense, full_matrices=False)
                result = _cut_and_return(
                    U_t.cpu().numpy(), s_t.cpu().numpy(), Vh_t.cpu().numpy(), "GPU"
                )
                del dense, U_t, s_t, Vh_t
                torch.cuda.empty_cache()
                return result
            except Exception as exc:
                if dense is not None:
                    del dense
                torch.cuda.empty_cache()
                print(f"  GPU SVD failed ({exc}); falling back to CPU ...")

        # ------------------------------------------------------------------
        # CPU path: dense LAPACK
        # ------------------------------------------------------------------
        itemsize = np.dtype(np_dtype).itemsize
        print(f"  densifying {m}×{n} on CPU ({m * n * itemsize / 1e9:.1f} GB, {np_dtype.__name__}) ...")
        dense = csr.toarray().astype(np_dtype)
        U, s_cpu, Vt = scipy.linalg.svd(dense, full_matrices=False)
        del dense
        return _cut_and_return(U, s_cpu, Vt, "CPU LAPACK")

    # ------------------------------------------------------------------
    # Cache key / save / load
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        h = hashlib.sha256()
        h.update(str(self.resolution).encode())
        h.update(str(self.det_count).encode())
        h.update(repr(self.dx).encode())
        h.update(repr(self.phi).encode())
        h.update(repr(self.svd_threshold).encode())
        h.update(self.angles.tobytes())
        # The decomposition now runs in self.dtype, so a float32 and a float64
        # adapter no longer produce the same factors and must not share a cache.
        h.update(str(self.dtype).encode())
        return h.hexdigest()[:16]

    # Written last into a finished entry. Its presence is what makes an entry a
    # cache hit, so a directory without it is never read as a complete cache.
    _CACHE_MARKER = "COMPLETE"

    def _save_cache(self, path: Path) -> None:
        """Publish the cache entry atomically.

        Every file is written into a private temporary directory beside the
        final one, which is then renamed into place in one step. Pipeline stages
        run concurrently as a Slurm array over noise levels with an identical
        geometry, so several tasks can miss the same key at once: without the
        rename a reader could load an entry another task was still writing, and
        a job killed mid-write would leave a corrupt entry behind for good. If
        another task publishes first, the rename fails and this copy is
        discarded -- both copies were computed from the same inputs.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            self._write_cache_files(tmp)
            (tmp / self._CACHE_MARKER).write_text("ok\n", encoding="utf-8")
            try:
                os.rename(tmp, path)
            except OSError:
                # Lost the race (or a stale incomplete entry is in the way); a
                # complete entry from another task is equally good.
                if not (path / self._CACHE_MARKER).exists():
                    raise
                print(f"Cache {path.name} was published concurrently; keeping that copy.")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    def _write_cache_files(self, path: Path) -> None:
        # Saved in the adapter's own dtype; the cache key includes it.
        for name, mat in [("A", self._A), ("A_la", self._A_la)]:
            t = mat.cpu()
            if t.layout == torch.strided:
                csr = scipy.sparse.csr_matrix(t.numpy())
            else:
                csr = scipy.sparse.csr_matrix(
                    (t.values().numpy(), t.col_indices().numpy(), t.crow_indices().numpy()),
                    shape=t.shape,
                )
            scipy.sparse.save_npz(str(path / f"{name}.npz"), csr)

        for name, tensor in [
            ("U_k_la", getattr(self, "_U_k_la", None)),
            ("s_k_la", getattr(self, "_s_k_la", None)),
            ("Vt_k_la",getattr(self, "_Vt_k_la",None)),
        ]:
            if tensor is not None:
                np.save(str(path / f"{name}.npy"), tensor.cpu().numpy())

    def _try_load_cache(self, path: Path) -> bool:
        """Load a cache entry if it is complete and valid; report whether it was.

        An entry carrying the completion marker is trusted to be whole. An entry
        without it predates the marker, or was left by a writer that crashed
        before the atomic publish existed: it is loaded and validated, adopted
        by writing the marker if it passes, and removed if it does not. No live
        writer can be behind a markerless entry, because writers now publish by
        renaming a finished directory into place.
        """
        if not path.is_dir():
            return False
        complete = (path / self._CACHE_MARKER).exists()
        try:
            self._load_cache(path)
        except Exception as exc:
            if complete:
                raise
            print(f"Discarding incomplete cache {path.name} ({exc}); rebuilding.")
            shutil.rmtree(path, ignore_errors=True)
            return False
        if not complete:
            (path / self._CACHE_MARKER).write_text("ok\n", encoding="utf-8")
            print(f"Adopted pre-existing cache {path.name} after validation.")
        return True

    def _load_cache(self, path: Path) -> None:
        self._A    = self._csr_to_torch(scipy.sparse.load_npz(str(path / "A.npz")))
        self._A_la = self._csr_to_torch(scipy.sparse.load_npz(str(path / "A_la.npz")))

        n = self.resolution ** 2
        m_la = int(self._la_row_mask.sum())
        if tuple(self._A_la.shape) != (m_la, n):
            raise ValueError(f"cached A_la has shape {tuple(self._A_la.shape)}, "
                             f"expected {(m_la, n)}")

        if self.svd_threshold > 0:
            for name in ("U_k_la", "s_k_la", "Vt_k_la"):
                p = path / f"{name}.npy"
                if not p.exists():
                    raise FileNotFoundError(f"cache is missing {name}.npy")
                setattr(self, f"_{name}",
                        torch.from_numpy(np.load(str(p))).to(device=self.device, dtype=self.dtype))
            k = self._s_k_la.numel()
            if (tuple(self._U_k_la.shape) != (m_la, k)
                    or tuple(self._Vt_k_la.shape) != (k, n)):
                raise ValueError(
                    f"cached factors have inconsistent shapes U{tuple(self._U_k_la.shape)} "
                    f"s({k},) Vt{tuple(self._Vt_k_la.shape)}")
            # Validated once here rather than on every application: a check in
            # backward_la would force a GPU sync inside the PGD inner loop.
            for name, t in (("U", self._U_k_la), ("s", self._s_k_la), ("Vt", self._Vt_k_la)):
                if not torch.isfinite(t).all():
                    raise ValueError(f"cached SVD factor {name} contains NaN/Inf")

    def _csr_to_torch(self, mat: scipy.sparse.csr_matrix) -> torch.Tensor:
        if self.dense:
            return torch.from_numpy(mat.toarray()).to(dtype=self.dtype, device=self.device)
        crow = torch.from_numpy(mat.indptr.astype(np.int64))
        col  = torch.from_numpy(mat.indices.astype(np.int64))
        val  = torch.from_numpy(mat.data.astype(np.float64))  # sparse_csr_tensor casts
        t = torch.sparse_csr_tensor(crow, col, val, size=mat.shape, dtype=self.dtype)
        return t.to(self.device)

    @staticmethod
    def _matmul(mat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply A (or A^T) to a dense matrix, for either operator layout."""
        if mat.layout == torch.strided:
            return mat @ x
        return torch.sparse.mm(mat, x)

    # ------------------------------------------------------------------
    # Forward operators
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full-angle forward projection: y = A x.

        Parameters
        ----------
        x : (B, C, res, res)

        Returns
        -------
        y : (B, C, n_angles, det_count)
        """
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, H * W).to(dtype=self.dtype, device=self.device)
        y_flat = self._matmul(self._A, x_flat.t()).t()
        return (y_flat.reshape(B, C, len(self.angles), self.det_count)
                .to(device=x.device, dtype=x.dtype) * self.dx)

    def forward_la(self, x: torch.Tensor) -> torch.Tensor:
        """
        Limited-angle forward projection: y_la = A_la x.

        Returns a full-shape sinogram (same shape as forward()) with non-LA
        rows zeroed out.

        Parameters
        ----------
        x : (B, C, res, res)

        Returns
        -------
        y_la : (B, C, n_angles, det_count)  — non-LA rows are zero
        """
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, H * W).to(dtype=self.dtype, device=self.device)
        y_compact = self._matmul(self._A_la, x_flat.t()).t()   # (B*C, n_la*det)
        y_compact = y_compact.reshape(B * C, self.n_la, self.det_count)

        return (self._compact_to_full(y_compact, B, C)
            .to(device=x.device, dtype=x.dtype) * self.dx)

    # ------------------------------------------------------------------
    # Sinogram-space projection  —  SVD-based
    # ------------------------------------------------------------------

    def _la_mask(self) -> np.ndarray:
        """Boolean numpy mask of shape (n_angles,) selecting LA angles."""
        return (self.angles >= self.phi[0]) & (self.angles < self.phi[1])

    def _compact_to_full(self, y_compact: torch.Tensor, B: int, C: int) -> torch.Tensor:
        """
        Embed a compact LA sinogram (B*C, n_la, det_count) into a full-shape
        tensor (B, C, n_angles, det_count) with non-LA rows zeroed.
        """
        la_mask = torch.from_numpy(self._la_mask()).to(device=self.device)
        y_full = torch.zeros(
            B * C, len(self.angles), self.det_count,
            device=self.device, dtype=self.dtype,
        )
        y_full[:, la_mask, :] = y_compact
        return y_full.reshape(B, C, len(self.angles), self.det_count)

    def proj_ran(self, y: torch.Tensor) -> torch.Tensor:
        """
        Project sinogram onto range(A_la) using the SVD left-singular vectors:
            y_r = U_kl (U_kl^T y_la)
        where y_la are the LA rows of y.  Result is full-shape with non-LA rows
        set to zero.

        This is strictly smaller than "keep the measured rows": range(A_la) is a
        k-dimensional subspace of the n_la*det_count measured coordinates.
        """
        self._require_svd("_U_k_la", "proj_ran")
        la_mask = self._la_mask()
        y_compact = y[..., la_mask, :]                      # (B,C,n_la,det_count)
        orig_device, orig_dtype = y.device, y.dtype
        B, C, n_la, nd = y_compact.shape
        y_flat = y_compact.reshape(B * C, n_la * nd).to(dtype=self.dtype, device=self.device)
        # U_k_la : (n_la*det, k)
        coeffs   = y_flat @ self._U_k_la                   # (B*C, k)
        y_proj   = (coeffs @ self._U_k_la.t()              # (B*C, n_la*det)
                    ).reshape(B * C, n_la, nd)
        return self._compact_to_full(y_proj, B, C).to(device=orig_device, dtype=orig_dtype)

    # ------------------------------------------------------------------
    # Pseudoinverse (backward) operator  —  A_la^+
    # ------------------------------------------------------------------

    def backward_la(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the limited-angle pseudoinverse: x = A_la^+ y_la.

        Uses the truncated SVD: A_la^+ = Vt_kl.T diag(1/s_kl) U_kl.T

        Expects a full-shape sinogram (B, C, n_angles, det_count) with non-LA rows
        zeroed out (as produced by forward_la / proj_ran).  The LA rows are
        extracted internally before applying the pseudoinverse.

        Parameters
        ----------
        y : (B, C, n_angles, det_count)  — non-LA rows should be zero

        Returns
        -------
        x : (B, C, res, res)
        """
        self._require_svd("_U_k_la", "backward_la (limited-angle pseudoinverse)")
        la_mask = self._la_mask()
        y_compact = y[..., la_mask, :]                       # (B,C,n_la,det_count)
        orig_device, orig_dtype = y_compact.device, y_compact.dtype
        B, C, n_la, nd = y_compact.shape
        y_flat = (y_compact / self.dx).reshape(B * C, n_la * nd).to(dtype=self.dtype, device=self.device)
        x_flat = self._apply_pseudoinverse(y_flat, self._U_k_la, self._s_k_la, self._Vt_k_la)
        # The factors are checked for NaN/Inf once, when built or loaded; this
        # function sits in the PGD inner loop and must not force a GPU sync.
        return x_flat.reshape(B, C, self.resolution, self.resolution).to(device=orig_device, dtype=orig_dtype)

    @staticmethod
    def _apply_pseudoinverse(
        y_flat: torch.Tensor,
        U_k: torch.Tensor,
        s_k: torch.Tensor,
        Vt_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply A^+ = Vt_k.T diag(1/s_k) U_k.T to a batch of flat measurement vectors.

        Parameters
        ----------
        y_flat : (batch, m)
        U_k    : (m, k)
        s_k    : (k,)
        Vt_k   : (k, n)

        Returns
        -------
        x_flat : (batch, n)
        """
        z = (y_flat @ U_k) / s_k   # (batch, k)
        return z @ Vt_k             # (batch, n)

    # ------------------------------------------------------------------
    # Null-space projection
    # ------------------------------------------------------------------

    def proj_null_image(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project image v onto null(A_la): v_n = v - V_kl V_kl^T v.

        Equivalently: v - Vt_kl.T @ (Vt_kl @ v_flat.T)

        Parameters
        ----------
        v : (B, C, res, res)

        Returns
        -------
        v_n : (B, C, res, res)  — component of v in null(A_la)
        """
        self._require_svd("_Vt_k_la", "proj_null_image")
        orig_device, orig_dtype = v.device, v.dtype
        B, C, H, W = v.shape
        Vt_k = self._Vt_k_la
        v_flat = v.reshape(B * C, H * W).to(dtype=self.dtype, device=self.device)
        coeffs = v_flat @ Vt_k.t()                 # (B*C, k)
        v_range = coeffs @ Vt_k                    # (B*C, n)  — range component
        result = (v_flat - v_range).reshape(B, C, H, W)
        return result.to(device=orig_device, dtype=orig_dtype)

    def decompose_error(self, e: torch.Tensor):
        """
        Exact SVD-based error decomposition: e = e_ran + e_null where
          e_ran  = Vt_kl.T (Vt_kl e)   (range component)
          e_null = e - e_ran            (null-space component)

        Returns
        -------
        (e_ran, e_null) : Tuple[torch.Tensor, torch.Tensor]
        """
        self._require_svd("_Vt_k_la", "decompose_error")
        orig_device, orig_dtype = e.device, e.dtype
        B, C, H, W = e.shape
        e_flat = e.reshape(B * C, H * W).to(dtype=self.dtype, device=self.device)
        coeffs = e_flat @ self._Vt_k_la.t()  # (B*C, k)
        e_ran_flat = coeffs @ self._Vt_k_la  # (B*C, n)
        e_ran = e_ran_flat.reshape(B, C, H, W).to(device=orig_device, dtype=orig_dtype)
        return e_ran, e - e_ran

    # ------------------------------------------------------------------
    # Operator norm estimation  (sparse power iteration on A, not A^+)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _estimate_operator_norm(self, iters: int = 20, tol: float = 1e-6, seed: int = 0) -> None:
        """Estimate ||A|| and ||A||² via power iteration using the sparse matrix."""
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        x = torch.randn((self.resolution ** 2, 1), device=self.device, dtype=self.dtype, generator=g)
        x /= x.norm() + 1e-12

        lam, last_lam = None, None
        for _ in range(iters):
            y = self._matmul(self._A, x)
            x_new = self._matmul(self._A.t(), y)
            lam = (x_new * x).sum().abs().item() / (x * x).sum().clamp_min(1e-12).item()
            x = x_new / (x_new.norm() + 1e-12)
            if last_lam is not None and abs(lam - last_lam) / max(lam, 1e-12) < tol:
                break
            last_lam = lam

        self.norm_A2 = float(lam if lam is not None else 0.0)
        self.norm_A = float(math.sqrt(self.norm_A2))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_svd(self, attr: str, method: str) -> None:
        if not hasattr(self, attr):
            raise RuntimeError(
                f"{method} requires SVD factors. "
                "Pass svd_threshold > 0 at construction."
            )
