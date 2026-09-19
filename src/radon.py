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
        Floating-point dtype the matrices and SVD factors are stored and applied
        in. The decomposition itself always runs in float64 (see _truncated_svd).
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
        if cache_path is not None and cache_path.exists():
            print(f"Loading matrix cache from {cache_path}")
            self._load_cache(cache_path)
            if hasattr(self, "_U_k_la"):
                self._check_factors(f"cache {cache_path}",
                                    hint=f" Delete {cache_path} to have it rebuilt.")
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
            # Before the factors can reach the cache: a bad decomposition is
            # never saved.
            self._check_factors("the decomposition")

    def _truncated_svd(
        self, csr: scipy.sparse.csr_matrix
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the truncated SVD of a sparse matrix.

        Returns U_k (m,k), s_k (k,), Vt_k (k,n) as torch tensors on self.device,
        retaining singular values >= svd_threshold * s_max.

        The decomposition runs in float64 whatever self.dtype, and only the
        factors are stored in self.dtype. In float32, torch.linalg.svd on the GPU
        (default driver) returned factors of this operator that were orthonormal
        and consistent with A_la only to ~5e-3, far above float32 rounding; the
        null-space projector built from them leaked into the measurements, and
        the Nullspace Network trained on it learned to use the leak. Rounding
        float64 factors to float32 costs ~1e-7. _check_factors verifies them.
        """
        store_np = np.float32 if self.dtype == torch.float32 else np.float64
        m, n = csr.shape

        def _t(arr) -> torch.Tensor:
            if isinstance(arr, torch.Tensor):
                return arr.to(device=self.device, dtype=self.dtype)
            return torch.from_numpy(np.asarray(arr)).to(
                device=self.device, dtype=self.dtype
            )

        def _cut_and_return(U_np, s_np, Vt_np, source: str):
            # Cut on the singular values as they will be stored, so that k here
            # is the k that src.truncation.k_for_tau reads off a stored s_k.
            s_np = np.asarray(s_np).astype(store_np).astype(np.float64)
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
            print(f"  densifying {m}×{n} on GPU ({m * n * 8 / 1e9:.1f} GB, float64)")
            dense = None
            try:
                dense = torch.from_numpy(csr.toarray().astype(np.float64)).to(self.device)
                U_t, s_t, Vh_t = torch.linalg.svd(dense, full_matrices=False)
                result = _cut_and_return(
                    U_t.cpu().numpy(), s_t.cpu().numpy(), Vh_t.cpu().numpy(), "GPU, float64"
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
        print(f"  densifying {m}×{n} on CPU ({m * n * 8 / 1e9:.1f} GB, float64) ...")
        dense = csr.toarray().astype(np.float64)
        U, s_cpu, Vt = scipy.linalg.svd(dense, full_matrices=False)
        del dense
        return _cut_and_return(U, s_cpu, Vt, "CPU LAPACK, float64")

    # ------------------------------------------------------------------
    # Factor check
    # ------------------------------------------------------------------

    # Largest defect _check_factors accepts. Float64 factors stored in float32
    # come in near 1e-7; the float32 GPU decomposition it guards against was
    # at 5e-3 to 1e-2.
    FACTOR_TOL = 1e-4

    def _check_factors(self, source: str, hint: str = "") -> None:
        """Refuse factors that are not a truncated SVD of A_la.

        The Nullspace Network's guarantees are these identities: only if V_k has
        orthonormal rows is P_N = I - V_k^T V_k a projector, and only if also
        A_la V_k^T = U_k S_k do P_ran A_la P_N = 0 (data consistency) and
        V_k P_N = 0 (the range floor) hold. Factors that miss them by a percent
        still reconstruct plausibly, so nothing downstream would notice.

        Measures, with float64 accumulation, ||V_k V_k^T - I||_2 and
        ||U_k^T U_k - I||_2 by power iteration and the backward residual
        ||A_la V_k^T z - U_k S_k z|| / ||S_k z|| on random z; prints all three
        and raises if any exceeds FACTOR_TOL.
        """
        U, Vt = self._U_k_la, self._Vt_k_la
        g = torch.Generator(device=self.device).manual_seed(0)
        z0 = torch.randn(int(self._s_k_la.numel()), 4, generator=g,
                         device=self.device, dtype=torch.float64)
        z0 = z0 / torch.linalg.norm(z0, dim=0)

        def gram_defect(gram) -> float:
            # every iterate is a lower bound on the norm; keep the largest
            z, est = z0, 0.0
            for _ in range(6):
                z = gram(z) - z
                est = max(est, float(torch.linalg.norm(z, dim=0).max()))
                z = z / torch.linalg.norm(z, dim=0).clamp_min(1e-300)
            return est

        v_def = gram_defect(lambda z: self._mm64(Vt, self._mm64_t(Vt, z)))
        u_def = gram_defect(lambda z: self._mm64_t(U, self._mm64(U, z)))
        sz = self._s_k_la.to(torch.float64)[:, None] * z0
        res = self._mm64(self._A_la, self._mm64_t(Vt, z0)) - self._mm64(U, sz)
        f_def = float((torch.linalg.norm(res, dim=0) / torch.linalg.norm(sz, dim=0)).max())

        print(f"  factor check [{source}]: ||V V^T - I|| = {v_def:.1e}, "
              f"||U^T U - I|| = {u_def:.1e}, ||A V^T z - U S z|| / ||S z|| = {f_def:.1e}")
        worst = max(v_def, u_def, f_def)
        if not worst <= self.FACTOR_TOL:          # written so that NaN fails too
            raise RuntimeError(
                f"SVD factors from {source} are not a decomposition of A_la to "
                f"working precision: largest defect {worst:.1e}, tolerance "
                f"{self.FACTOR_TOL:.0e}. The null-space projector built from them "
                f"would leak into the measurements.{hint}")

    @staticmethod
    def _mm64(mat: torch.Tensor, x: torch.Tensor, block: int = 2048) -> torch.Tensor:
        """mat @ x in float64, converting a dense mat one row block at a time."""
        if mat.layout != torch.strided:
            return torch.sparse.mm(mat.to(torch.float64), x)
        return torch.cat([mat[i:i + block].to(torch.float64) @ x
                          for i in range(0, mat.shape[0], block)])

    @staticmethod
    def _mm64_t(mat: torch.Tensor, x: torch.Tensor, block: int = 2048) -> torch.Tensor:
        """mat^T @ x in float64, converting mat one row block at a time."""
        out = torch.zeros(mat.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
        for i in range(0, mat.shape[0], block):
            out += mat[i:i + block].to(torch.float64).t() @ x[i:i + block]
        return out

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
        # The factors are stored in self.dtype, so a float32 and a float64
        # adapter must not share a cache.
        h.update(str(self.dtype).encode())
        # They are decomposed in float64 whatever the dtype. Entries from when a
        # float32 adapter decomposed in float32 lack this tag and are never read
        # again: their factors were off by ~5e-3 (see _truncated_svd).
        h.update(b"svd:float64")
        return h.hexdigest()[:16]

    def _save_cache(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

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

    def _load_cache(self, path: Path) -> None:
        self._A    = self._csr_to_torch(scipy.sparse.load_npz(str(path / "A.npz")))
        self._A_la = self._csr_to_torch(scipy.sparse.load_npz(str(path / "A_la.npz")))

        for name in ("U_k_la", "s_k_la", "Vt_k_la"):
            p = path / f"{name}.npy"
            if p.exists():
                setattr(self, f"_{name}",
                        torch.from_numpy(np.load(str(p))).to(device=self.device, dtype=self.dtype))

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
        # The factors are checked for NaN/Inf once, when built; this function
        # sits in the PGD inner loop and must not force a GPU sync.
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
