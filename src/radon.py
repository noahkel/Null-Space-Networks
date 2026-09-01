"""Radon operators for limited-angle CT.

One module, two backends behind the same interface (``_RadonBase``):

  * ``AstraRadonAdapter``  — matrix-free forward/backprojection through the ASTRA
    Toolbox, wrapped in autograd Functions so gradients flow. Null-space
    projection and the error decomposition fall back to conjugate gradients.
  * ``MatrixRadonAdapter`` — explicit sparse system matrices A and A_la plus
    their truncated SVDs, so the pseudoinverse, the range projector and the
    null-space projector are exact tensor algebra (and differentiable). This is
    the backend every result uses (``matrix_mode=1``).

Shapes are the same for both: images are (B, C, res, res) and sinograms are
(B, C, n_angles, det_count) — the limited-angle operators return a *full-shape*
sinogram with the unmeasured rows zeroed, so the two backends are drop-in
interchangeable.
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
import torch.nn.functional as F


def construct_fourier_filter_torch(size: int, filter_name: str, device, dtype=torch.float32) -> torch.Tensor:
    """
    Build the Fourier-domain filter as a 1D torch tensor of shape (size,).
    """
    if size % 2 != 0:
        raise ValueError(f"size must be even, got {size}")

    filter_name = filter_name.lower()

    # Create spatial-domain impulse response f, then FFT -> frequency filter
    n = torch.cat(
        (
            torch.arange(1, size // 2 + 1, 2, device=device, dtype=torch.int64),
            torch.arange(size // 2 - 1, 0, -2, device=device, dtype=torch.int64),
        ),
        dim=0,
    )

    f = torch.zeros(size, device=device, dtype=dtype)
    f[0] = 0.25
    f[1::2] = -1.0 / (math.pi * n.to(dtype)) ** 2

    fourier_filter = 2.0 * torch.real(torch.fft.fft(f))

    if filter_name in ("ramp", "ram-lak"):
        pass

    elif filter_name == "shepp-logan":
        # omega = pi * freq, skip DC
        omega = math.pi * torch.fft.fftfreq(size, device=device, dtype=dtype)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega

    elif filter_name == "cosine":
        freq = torch.linspace(0, math.pi, size, device=device, dtype=dtype, requires_grad=False)
        cosine_filter = torch.fft.fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter

    elif filter_name == "hamming":
        fourier_filter *= torch.fft.fftshift(torch.hamming_window(size, device=device, dtype=dtype))

    elif filter_name == "hann":
        fourier_filter *= torch.fft.fftshift(torch.hann_window(size, device=device, dtype=dtype))

    else:
        raise ValueError(
            f"Unknown filter type '{filter_name}'. "
            "Available: 'ramp'/'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'."
        )

    return fourier_filter  # (size,)


def filter_sinogram(
    Y: torch.Tensor,
    filter_name: str = "ramp",
    fourier_filter_cache: Optional[dict] = None,
) -> torch.Tensor:
    """
    Apply FBP-style 1D frequency filtering to sinograms in shape (B, C, H, W),
    filtering along W (detectors). Assumes H = angles.

    Args:
        Y: (B, C, H, W) tensor
        filter_name: filter type
        fourier_filter_cache: optional dict to cache filters by padded_size/device/dtype/name

    Returns:
        Filtered tensor with same shape as Y.
    """
    if Y.ndim != 4:
        raise ValueError(f"Expected input of shape (B, C, H, W), got {tuple(Y.shape)}")

    device = Y.device
    real_dtype = torch.float32 if Y.dtype in (torch.float16, torch.bfloat16) else Y.dtype
    B, C, n_angles, size = Y.shape

    # padded_size = max(64, next_pow2(2*size))
    padded_size = max(64, 1 << math.ceil(math.log2(2 * size)))
    pad = padded_size - size

    # Pad on the last dimension only
    Yf = F.pad(Y.to(real_dtype), (0, pad))  # (B, C, H, padded_size)

    # FFT along detector axis
    sino_fft = torch.fft.fft(Yf, dim=-1)  # complex

    # Build / cache filter
    cache_key = None
    if fourier_filter_cache is not None:
        cache_key = (padded_size, filter_name, device.type, str(device), str(real_dtype))
        f = fourier_filter_cache.get(cache_key)
    else:
        f = None

    if f is None:
        f = construct_fourier_filter_torch(padded_size, filter_name, device=device, dtype=real_dtype)
        # make complex for multiplication with FFT
        f = f.to(torch.complex64 if real_dtype == torch.float32 else torch.complex128)
        if fourier_filter_cache is not None:
            fourier_filter_cache[cache_key] = f

    # Broadcast multiply: (B,C,H,W) * (W,)
    filtered_fft = sino_fft * f.view(1, 1, 1, -1)

    # iFFT back, crop, scale
    filtered = torch.fft.ifft(filtered_fft, dim=-1).real  # (B,C,H,padded_size)
    filtered = filtered[..., :size]  # (B,C,H,W)
    filtered = filtered * (math.pi / (2.0 * n_angles))

    return filtered.to(dtype=Y.dtype)


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _RadonBase:
    """
    Base class for Radon adapter implementations.

    Provides limited-angle masking, FBP, power-iteration norm estimation and
    the CG null-space projection / error decomposition that a backend without
    SVD factors falls back to.

    Subclasses must implement `forward` and `backward` and set
    the following attributes in their ``__init__``:

        resolution, det_count, angles (np.ndarray), dx, device, dtype, phi,
        norm_A, norm_A2,
        _ran_mask (torch.Tensor, shape 1×1×n_angles×det_count),
        _nsn_mask (torch.Tensor, same shape).
    """

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _build_ran_mask_np(self) -> np.ndarray:
        """
        Build a limited-angle mask selecting angles in [lo, hi).

        Returns
        -------
        mask : np.ndarray, shape (1, 1, n_angles, det_count)
            1.0 for angles in-range, 0.0 otherwise.
        """
        lo, hi = self.phi
        ang_mask = ((self.angles >= lo) & (self.angles < hi)).astype(np.float32)
        mask2d = np.repeat(ang_mask.reshape(-1, 1), self.det_count, axis=1)
        return mask2d[None, None, :, :].astype(np.float32)

    def _build_null_mask_np(self) -> np.ndarray:
        """Complement mask: 1 where angles are NOT in [lo, hi), 0 otherwise."""
        return 1.0 - self._build_ran_mask_np()

    # ------------------------------------------------------------------
    # Abstract interface (implemented by subclasses)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Limited-angle and FBP methods
    # ------------------------------------------------------------------

    def proj_ran(self, y: torch.Tensor) -> torch.Tensor:
        """Project onto the 'range' (selected) angle set: y * ran_mask."""
        return y * self._ran_mask.to(device=y.device, dtype=y.dtype)

    def forward_la(self, x: torch.Tensor) -> torch.Tensor:
        """Limited-angle forward projection: y = P_phi (A x)."""
        return self.proj_ran(self.forward(x))

    def backward_la(self, y: torch.Tensor) -> torch.Tensor:
        """Limited-angle backprojection: x = A^T (P_phi y)."""
        return self.backward(self.proj_ran(y))

    def fbp(self, y: torch.Tensor, filter_name: str = "ram-lak") -> torch.Tensor:
        """Full-angle filtered backprojection: x = A^T( F(y) )."""
        return self.backward(filter_sinogram(y, filter_name=filter_name))

    def fbp_la(self, y: torch.Tensor, filter_name: str = "ram-lak") -> torch.Tensor:
        """Limited-angle filtered backprojection: x = A^T( F(P_phi y) )."""
        return self.backward(filter_sinogram(self.proj_ran(y), filter_name=filter_name))

    def proj_nsn(self, y: torch.Tensor) -> torch.Tensor:
        """Project onto the 'null' (complement) angle set: y * nsn_mask."""
        return y * self._nsn_mask.to(device=y.device, dtype=y.dtype)

    
    def proj_null_image(self, v: torch.Tensor, iters: int = 50, tol: float = 1e-6) -> torch.Tensor:
        """
        Project image v onto null(A_la): returns v - A_la^+ A_la v.

        Uses conjugate gradients to solve A_la^T A_la x = A_la^T v, then
        returns v - x. Differentiable; subclasses may override with a faster
        exact implementation (e.g. via truncated SVD factors).
        """

        def AtA(x: torch.Tensor) -> torch.Tensor:
            return self.backward_la(self.forward_la(x))

        b = AtA(v)
        x = torch.zeros_like(v)
        r = b.clone()
        p = r.clone()
        rr = (r * r).sum()

        for _ in range(iters):
            if rr.item() < tol ** 2:
                break
            Ap = AtA(p)
            alpha = rr / (p * Ap).sum().clamp_min(1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = (r * r).sum()
            p = r + (rr_new / rr.clamp_min(1e-12)) * p
            rr = rr_new

        return v - x

    def decompose_error(self, e: torch.Tensor, iters: int = 50, tol: float = 1e-6):
        """
        Decompose error e = e_ran + e_null where:
          e_ran  = A_la^+ A_la e   (range component, visible to measurements)
          e_null = e - e_ran       (null-space component, invisible)

        Uses CG to approximate A_la^+ A_la; subclasses with SVD factors
        (e.g. MatrixRadonAdapter) override this with an exact decomposition.

        Returns
        -------
        (e_ran, e_null) : Tuple[torch.Tensor, torch.Tensor]
        """
        e_null = self.proj_null_image(e, iters=iters, tol=tol)
        return e - e_null, e_null

    # ------------------------------------------------------------------
    # Operator norm estimation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _estimate_operator_norm(
        self,
        iters: int = 20,
        tol: float = 1e-6,
        seed: int = 0,
    ) -> None:
        """
        Estimate ||A|| and ||A||^2 via power iteration on A^T A.

        Sets self.norm_A2 and self.norm_A.
        """
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        x = torch.randn(
            (1, 1, self.resolution, self.resolution),
            device=self.device,
            dtype=self.dtype,
            generator=g,
        )
        x /= x.norm() + 1e-12

        last_lambda = None
        lam = None

        for _ in range(iters):
            y = self.forward(x)
            x_new = self.backward(y)

            lam = (x_new * x).sum().abs().item() / (x * x).sum().clamp_min(1e-12).item()
            x = x_new / (x_new.norm() + 1e-12)

            if last_lambda is not None:
                if abs(lam - last_lambda) / max(lam, 1e-12) < tol:
                    break
            last_lambda = lam

        self.norm_A2 = float(lam if lam is not None else 0.0)
        self.norm_A = float(math.sqrt(self.norm_A2))


# ---------------------------------------------------------------------------
# Differentiable ASTRA autograd wrappers
# ---------------------------------------------------------------------------

def _astra_batch(x: torch.Tensor, single_fn, out_shape: tuple) -> torch.Tensor:
    x_np = x.detach().cpu().float().numpy()
    out = np.empty(out_shape, dtype=np.float32)
    B, C = x.shape[:2]
    for b in range(B):
        for c in range(C):
            out[b, c] = single_fn(x_np[b, c])
    return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
    
class _AstraFP(torch.autograd.Function):
    """
    Differentiable forward projection via ASTRA.

    forward : x (B,C,H,W) -> FP(x) (B,C,n_angles,det_count)
    backward: grad_out     -> BP(grad_out)   [adjoint of FP]
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, adapter: "AstraRadonAdapter") -> torch.Tensor:
        ctx.adapter = adapter
        B, C = x.shape[:2]
        return _astra_batch(x, adapter._fp_single, (B, C, len(adapter.angles), adapter.det_count))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        adapter = ctx.adapter
        B, C = grad_output.shape[:2]
        grad = _astra_batch(grad_output, adapter._bp_single, (B, C, adapter.resolution, adapter.resolution))
        return grad, None


class _AstraBP(torch.autograd.Function):
    """
    Differentiable backprojection via ASTRA.

    forward : y (B,C,n_angles,det_count) -> BP(y) (B,C,H,W)
    backward: grad_out                   -> FP(grad_out)   [adjoint of BP]
    """

    @staticmethod
    def forward(ctx, y: torch.Tensor, adapter: "AstraRadonAdapter") -> torch.Tensor:
        ctx.adapter = adapter
        B, C = y.shape[:2]
        return _astra_batch(y, adapter._bp_single, (B, C, adapter.resolution, adapter.resolution))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        adapter = ctx.adapter
        B, C = grad_output.shape[:2]
        grad = _astra_batch(grad_output, adapter._fp_single, (B, C, len(adapter.angles), adapter.det_count))
        return grad, None


# ---------------------------------------------------------------------------
# ASTRA Toolbox backend
# ---------------------------------------------------------------------------

class AstraRadonAdapter(_RadonBase):
    """
    Radon adapter backed by the ASTRA Toolbox.

    The ASTRA Toolbox must be installed separately::

        conda install -c astra-toolbox astra-toolbox             # CPU
        conda install -c astra-toolbox/label/cuda astra-toolbox  # GPU

    Parameters
    ----------
    resolution : int
        Square image side length (pixels).
    angles : np.ndarray
        Projection angles in radians.
    det_count : int
        Number of detector elements.
    clip_to_circle : bool
        Not supported; a warning is issued if True.
    dataset : str or None
        Optional label (stored, not used internally).
    dx : float
        Pixel-spacing scale factor. forward multiplies by dx, backward divides by dx.
    estimate_norm : bool
        Run power iteration to estimate ``norm_A`` and ``norm_A2``.
    norm_iters : int
        Maximum power-iteration steps.
    device : torch.device or None
        Target device for output tensors and norm estimation.
    dtype : torch.dtype
        Floating-point dtype.
    phi : (float, float) or None
        Limited-angle window ``[lo, hi)`` in radians.

    Notes
    -----
    ASTRA operates on NumPy arrays internally.  Input tensors are moved to
    CPU for projection (as float32) and the result is moved back to the
    original device.  For GPU workflows the CUDA algorithms run on ASTRA's
    own GPU memory; the tensor-to-numpy copy is the only overhead.
    """

    def __init__(
        self,
        resolution: int,
        angles: np.ndarray,
        det_count: int,
        clip_to_circle: bool = False,
        dataset: Union[str, None] = None,
        dx: float = 1.0,
        estimate_norm: bool = True,
        norm_iters: int = 20,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        phi: Optional[Tuple[float, float]] = None,
    ):
        try:
            import astra as _astra
        except ImportError:
            raise ImportError(
                "astra-toolbox is required. Install with:\n"
                "  conda install -c astra-toolbox astra-toolbox\n"
                "  # or for CUDA support:\n"
                "  conda install -c astra-toolbox/label/cuda astra-toolbox"
            )
        self._astra = _astra

        if clip_to_circle:
            import warnings
            warnings.warn(
                "clip_to_circle=True is not supported by AstraRadonAdapter; ignoring.",
                UserWarning,
                stacklevel=2,
            )

        self.resolution = int(resolution)
        self.det_count = int(det_count)
        self.angles = np.asarray(angles, dtype=np.float64)
        self.dx = float(dx)
        self.dataset = (dataset or "").lower()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.phi = phi
        self.norm_A: Optional[float] = None
        self.norm_A2: Optional[float] = None

        # ASTRA geometry descriptors
        self._vol_geom = _astra.create_vol_geom(self.resolution, self.resolution)
        self._proj_geom = _astra.create_proj_geom(
            'parallel', 1.0, self.det_count, self.angles
        )

        # Use GPU algorithms only if the device is CUDA and ASTRA has CUDA
        self._use_gpu = (
            str(self.device).startswith('cuda') and _astra.use_cuda()
        )

        # CPU projector (not needed for GPU path)
        if not self._use_gpu:
            self._proj_id: Optional[int] = _astra.create_projector(
                'strip', self._proj_geom, self._vol_geom
            )
        else:
            self._proj_id = None

        # Build masks
        self._ran_mask_np = self._build_ran_mask_np()
        self._nsn_mask_np = self._build_null_mask_np()
        self._ran_mask = torch.from_numpy(self._ran_mask_np).to(device=self.device, dtype=self.dtype)
        self._nsn_mask = torch.from_numpy(self._nsn_mask_np).to(device=self.device, dtype=self.dtype)

        if estimate_norm:
            self._estimate_operator_norm(iters=norm_iters)

    # ------------------------------------------------------------------
    # Single-image ASTRA calls
    # ------------------------------------------------------------------

    def _fp_single(self, x_np: np.ndarray) -> np.ndarray:
        """
        Forward project one image.

        Parameters
        ----------
        x_np : np.ndarray, shape (resolution, resolution), float32

        Returns
        -------
        np.ndarray, shape (n_angles, det_count), float32
        """
        astra = self._astra
        vol_id = astra.data2d.create('-vol', self._vol_geom, data=x_np)
        sino_id = astra.data2d.create('-sino', self._proj_geom)
        try:
            if self._use_gpu:
                cfg = astra.astra_dict('FP_CUDA')
            else:
                cfg = astra.astra_dict('FP')
                cfg['ProjectorId'] = self._proj_id
            cfg['VolumeDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            alg_id = astra.algorithm.create(cfg)
            try:
                astra.algorithm.run(alg_id)
                return astra.data2d.get(sino_id).copy()
            finally:
                astra.algorithm.delete(alg_id)
        finally:
            astra.data2d.delete([vol_id, sino_id])

    def _bp_single(self, sino_np: np.ndarray) -> np.ndarray:
        """
        Backproject one sinogram.

        Parameters
        ----------
        sino_np : np.ndarray, shape (n_angles, det_count), float32

        Returns
        -------
        np.ndarray, shape (resolution, resolution), float32
        """
        astra = self._astra
        vol_id = astra.data2d.create('-vol', self._vol_geom)
        sino_id = astra.data2d.create('-sino', self._proj_geom, data=sino_np)
        try:
            if self._use_gpu:
                cfg = astra.astra_dict('BP_CUDA')
            else:
                cfg = astra.astra_dict('BP')
                cfg['ProjectorId'] = self._proj_id
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            alg_id = astra.algorithm.create(cfg)
            try:
                astra.algorithm.run(alg_id)
                return astra.data2d.get(vol_id).copy()
            finally:
                astra.algorithm.delete(alg_id)
        finally:
            astra.data2d.delete([vol_id, sino_id])

    # ------------------------------------------------------------------
    # Batched forward / backward (public interface)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Radon forward projection A x, scaled by dx.

        Differentiable: gradients flow back through ASTRA via BP (the adjoint).

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, resolution, resolution)

        Returns
        -------
        torch.Tensor, shape (B, C, n_angles, det_count)
        """
        return _AstraFP.apply(x, self) * self.dx

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply adjoint / backprojection A^T y, scaled by 1/dx.

        Differentiable: gradients flow back through ASTRA via FP (the adjoint of BP).

        Parameters
        ----------
        y : torch.Tensor, shape (B, C, n_angles, det_count)

        Returns
        -------
        torch.Tensor, shape (B, C, resolution, resolution)
        """
        return _AstraBP.apply(y / self.dx, self)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Release the ASTRA CPU projector if one was created."""
        proj_id = getattr(self, '_proj_id', None)
        if proj_id is not None:
            try:
                self._astra.projector.delete(proj_id)
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Sparse-matrix backend
#
# Stores two system matrices -- A (all angles) and A_la (the measured angles
# only) -- plus a truncated SVD of each, which is what makes the pseudoinverse
# and the null-space projector exact rather than iterative.
# ---------------------------------------------------------------------------

class MatrixRadonAdapter(_RadonBase):
    """
    Radon adapter backed by precomputed sparse system matrices A and A_la,
    with truncated SVD factors stored for pseudoinverse and null-space operations.

    Operator algebra (image x in R^{n}, n = resolution^2; sinogram in R^{m}).
    A is the full-angle system matrix, A_la its restriction to the measured
    (limited-angle) rows. With the truncated SVD  A_la = U_k Σ_k V_k^T  (keeping
    singular values s_i ≥ svd_threshold · s_max):

      forward        y      = A x            (·dx)          -> forward()
      forward_la     y_la   = A_la x                        -> forward_la()
      A_la^+         = V_k Σ_k^{-1} U_k^T     (pseudoinverse) -> backward_la()
      P_ran          keep measured rows                     -> proj_ran()
      P_nsn          = I - P_ran (unmeasured rows)          -> proj_nsn()
      P_N            = I - A_la^+ A_la = I - V_k V_k^T       -> proj_null_image()
                     image-domain projector onto null(A_la);  A_la P_N = 0
      decompose      e_ran = A_la^+ A_la e,  e_nul = (I - A_la^+ A_la) e,
                     with ||e||^2 = ||e_ran||^2 + ||e_nul||^2  -> decompose_error()
      ||A||_2        largest singular value of A (power iteration) -> norm_A / norm_A2

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
        Must be > 0 to build SVD factors and enable backward / null-space methods.
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
        dtype: torch.dtype = torch.float64,
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

        # Masks used by _RadonBase helpers (proj_ran / proj_nsn)
        self._ran_mask_np = self._build_ran_mask_np()
        self._nsn_mask_np = self._build_null_mask_np()
        self._ran_mask = torch.from_numpy(self._ran_mask_np).to(device=self.device, dtype=self.dtype)
        self._nsn_mask = torch.from_numpy(self._nsn_mask_np).to(device=self.device, dtype=self.dtype)

        cache_path = Path(cache_dir) / self._cache_key() if cache_dir is not None else None
        print(f"Cache path: {cache_path}")
        if cache_path is not None and cache_path.exists():
            print(f"Loading matrix cache from {cache_path}")
            self._load_cache(cache_path)
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
        """Build sparse A, sparse A_la, and (if svd_threshold > 0) their SVD factors."""
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

        csr = csr.astype(np.float32 if self.dense else np.float64)

        # Full system matrix
        self._A = self._csr_to_torch(csr)
        print(f"Built sparse A, shape {tuple(csr.shape)}")

        # Limited-angle submatrix
        csr_la = csr[self._la_row_mask, :]
        self._A_la = self._csr_to_torch(csr_la)
        print(f"Built sparse A_la, shape {tuple(csr_la.shape)}")

        if self.svd_threshold > 0:
            print("Computing SVD of A ...")
            self._U_k, self._s_k, self._Vt_k = self._truncated_svd(csr)

            print("Computing SVD of A_la ...")
            self._U_k_la, self._s_k_la, self._Vt_k_la = self._truncated_svd(csr_la)

    def _truncated_svd(
        self, csr: scipy.sparse.csr_matrix
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the truncated SVD of a sparse matrix.

        Returns U_k (m,k), s_k (k,), Vt_k (k,n) as torch tensors on self.device,
        retaining singular values >= svd_threshold * s_max.

        GPU strategy: densify as float32, run torch.linalg.svd under the MAGMA
        backend (full thin SVD in one pass — avoids cuSOLVER gesvdj which overflows
        its int32 workspace counter for large sketch matrices).  Falls back to CPU
        LAPACK / ARPACK if MAGMA is unavailable or the matrix is too large for VRAM.

        CPU strategy: dense scipy.linalg.svd for matrices that fit in RAM,
        otherwise scipy.sparse.linalg.svds (ARPACK) directly on the sparse matrix.
        """
        m, n = csr.shape

        def _t(arr) -> torch.Tensor:
            if isinstance(arr, torch.Tensor):
                return arr.to(device=self.device, dtype=self.dtype)
            return torch.from_numpy(np.asarray(arr, dtype=np.float64)).to(
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
                        "Try a larger svd_threshold or increase the dense-SVD memory limit."
                    )
            return U_k, s_k, Vt_k

        # ------------------------------------------------------------------
        # GPU path: full thin SVD via MAGMA (single pass, no cuSOLVER limits)
        # ------------------------------------------------------------------

        if self.device.type == "cuda":
            mem_gb = m * n * 8 / 1e9  # float64
            print(f"  densifying {m}×{n} on GPU ({mem_gb:.1f} GB fp64)")
            dense_f64 = None
            try:
                dense_f64 = torch.from_numpy(csr.toarray().astype(np.float64)).to(self.device)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    #torch.backends.cuda.preferred_linalg_library("magma")
                    U_t, s_t, Vh_t = torch.linalg.svd(dense_f64, full_matrices=False)
                    #torch.backends.cuda.preferred_linalg_library("default")
                result = _cut_and_return(
                    U_t.cpu().numpy(), s_t.cpu().numpy(), Vh_t.cpu().numpy(), "GPU MAGMA"
                )
                del dense_f64, U_t, s_t, Vh_t
                torch.cuda.empty_cache()
                return result
            except Exception as exc:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    torch.backends.cuda.preferred_linalg_library("default")
                if dense_f64 is not None:
                    del dense_f64
                torch.cuda.empty_cache()
                print(f"  MAGMA SVD failed ({exc}); falling back to CPU ...")

        # ------------------------------------------------------------------
        # CPU path: dense LAPACK for small matrices, ARPACK for large ones, LAPACK is prefered
        # ------------------------------------------------------------------
        mem_gb_f64 = m * n * 8 / 1e9
        if mem_gb_f64 <= 8.0:
            print(f"  densifying {m}×{n} on CPU ({mem_gb_f64:.1f} GB fp64) ...")
            dense = csr.toarray().astype(np.float64)
            U, s_cpu, Vt = scipy.linalg.svd(dense, full_matrices=False)
            del dense
            return _cut_and_return(U, s_cpu, Vt, "CPU LAPACK")

        from scipy.sparse.linalg import svds as sp_svds

        sparse_max_q = min(m, n) - 1
        sparse_q = min(256, sparse_max_q)
        while True:
            k = min(sparse_q, sparse_max_q)

            if k > sparse_max_q // 2:
                warnings.warn(
                    f"ARPACK SVD requested k={k} which is >{sparse_max_q//2} "
                    f"(>half of sparse_max_q={sparse_max_q}). Results may be "
                    "numerically unstable. Consider increasing svd_threshold or "
                    "raising the dense-SVD memory limit.",
                    UserWarning, stacklevel=3,
                )

            print(f"  sparse CPU SVD (ARPACK), k={k} ...")
            # Try PROPACK first (faster for large k), fall back to ARPACK.
            try:
                U_cpu, s_arr, Vt_cpu = sp_svds(csr, k=k, which="LM", solver="propack")
            except Exception:
                U_cpu, s_arr, Vt_cpu = sp_svds(csr, k=k, which="LM")

            # svds returns singular values in ascending order — reverse them.
            idx = np.argsort(s_arr)[::-1]
            s_arr = s_arr[idx]
            U_cpu = U_cpu[:, idx]
            Vt_cpu = Vt_cpu[idx, :]

            if s_arr[-1] < self.svd_threshold * s_arr[0] or k >= sparse_max_q:
                break
            next_q = min(k * 2, sparse_max_q)
            print(f"  all {k} values above threshold, retrying with k={next_q} ...")
            sparse_q = next_q

        return _cut_and_return(U_cpu, s_arr, Vt_cpu, "sparse CPU")
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
        # Canonical dtype string: cached artifacts are stored as float64 npz/npy
        # and cast to self.dtype on load, so float32 and float64 adapters share
        # one cache. Hashing the literal keeps existing float64 caches valid.
        h.update(str(torch.float64).encode())
        return h.hexdigest()[:16]

    def _save_cache(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Artifacts are always saved as float64 (the cache is shared across
        # adapter dtypes; _load_cache casts to self.dtype).
        for name, mat in [("A", self._A), ("A_la", self._A_la)]:
            t = mat.cpu().to(torch.float64)
            if t.layout == torch.strided:
                csr = scipy.sparse.csr_matrix(t.numpy())
            else:
                csr = scipy.sparse.csr_matrix(
                    (t.values().numpy(), t.col_indices().numpy(), t.crow_indices().numpy()),
                    shape=t.shape,
                )
            scipy.sparse.save_npz(str(path / f"{name}.npz"), csr)

        for name, tensor in [
            ("U_k",    getattr(self, "_U_k",    None)),
            ("s_k",    getattr(self, "_s_k",    None)),
            ("Vt_k",   getattr(self, "_Vt_k",   None)),
            ("U_k_la", getattr(self, "_U_k_la", None)),
            ("s_k_la", getattr(self, "_s_k_la", None)),
            ("Vt_k_la",getattr(self, "_Vt_k_la",None)),
        ]:
            if tensor is not None:
                np.save(str(path / f"{name}.npy"),
                        tensor.cpu().to(torch.float64).numpy())

    def _load_cache(self, path: Path) -> None:
        self._A    = self._csr_to_torch(scipy.sparse.load_npz(str(path / "A.npz")).astype(np.float64))
        self._A_la = self._csr_to_torch(scipy.sparse.load_npz(str(path / "A_la.npz")).astype(np.float64))

        for name in ("U_k", "s_k", "Vt_k", "U_k_la", "s_k_la", "Vt_k_la"):
            p = path / f"{name}.npy"
            if p.exists():
                setattr(self, f"_{name}",
                        torch.from_numpy(np.load(str(p))).to(device=self.device, dtype=self.dtype))

    def _csr_to_torch(self, mat: scipy.sparse.csr_matrix) -> torch.Tensor:
        if self.dense:
            return torch.from_numpy(mat.toarray()).to(dtype=self.dtype, device=self.device)
        crow = torch.from_numpy(mat.indptr.astype(np.int64))
        col  = torch.from_numpy(mat.indices.astype(np.int64))
        val  = torch.from_numpy(mat.data.astype(np.float64))
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
        rows zeroed out, so that both adapters share the same sinogram convention.

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
    # Sinogram-space projections  —  SVD-based
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

    def proj_nsn(self, y: torch.Tensor) -> torch.Tensor:
        """
        Project sinogram onto the null-sinogram space (complement of range(A_la)):
            y_n = y - proj_ran(y)
        """
        return y - self.proj_ran(y)

    # ------------------------------------------------------------------
    # FBP — filter + sparse adjoint A^T, matching AstraRadonAdapter behaviour
    # ------------------------------------------------------------------

    def _backproject(self, y: torch.Tensor) -> torch.Tensor:
        """Apply sparse adjoint A^T to a full-shape sinogram (not the pseudoinverse)."""
        orig_device, orig_dtype = y.device, y.dtype
        B, C, n_a, nd = y.shape
        y_flat = (y / self.dx).reshape(B * C, n_a * nd).to(dtype=self.dtype, device=self.device)
        x_flat = self._matmul(self._A.t(), y_flat.t()).t()
        return x_flat.reshape(B, C, self.resolution, self.resolution).to(device=orig_device, dtype=orig_dtype)

    def _backproject_la(self, y: torch.Tensor) -> torch.Tensor:
        """Apply sparse adjoint A_la^T to a full-shape sinogram (extracts LA rows first)."""
        orig_device, orig_dtype = y.device, y.dtype
        la_mask = self._la_mask()
        y_compact = y[..., la_mask, :]                        # (B,C,n_la,det_count)
        B, C, n_la, nd = y_compact.shape
        y_flat = (y_compact / self.dx).reshape(B * C, n_la * nd).to(dtype=self.dtype, device=self.device)
        x_flat = self._matmul(self._A_la.t(), y_flat.t()).t()
        return x_flat.reshape(B, C, self.resolution, self.resolution).to(device=orig_device, dtype=orig_dtype)

    def fbp(self, y: torch.Tensor, filter_name: str = "ram-lak") -> torch.Tensor:
        """
        Full-angle filtered backprojection: x = A^T( filter(y) ).

        Matches AstraRadonAdapter behaviour so that fbp(forward(v)) ≈ v and
        the NSN null-space approximation works correctly.
        Use backward() for the exact pseudoinverse A^+ instead.

        Parameters
        ----------
        y : (B, C, n_angles, det_count)

        Returns
        -------
        x : (B, C, res, res)
        """
        return self._backproject(filter_sinogram(y, filter_name=filter_name))

    def fbp_la(self, y: torch.Tensor, filter_name: str = "ram-lak") -> torch.Tensor:
        """
        Limited-angle filtered backprojection: x = A_la^T( filter(proj_ran(y)) ).

        Matches AstraRadonAdapter behaviour so that the NSN null-space approximation
        is consistent across adapters.
        Use backward_la() for the exact limited-angle pseudoinverse A_la^+ instead.

        Parameters
        ----------
        y : (B, C, n_angles, det_count)  — non-LA rows are ignored

        Returns
        -------
        x : (B, C, res, res)
        """
        return self._backproject_la(filter_sinogram(self.proj_ran(y), filter_name=filter_name))

    # ------------------------------------------------------------------
    # Pseudoinverse (backward) operators  —  A^+ and A_la^+
    # ------------------------------------------------------------------

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the full-angle pseudoinverse: x = A^+ y.

        Uses the truncated SVD: A^+ = Vt_k.T diag(1/s_k) U_k.T

        Parameters
        ----------
        y : (B, C, n_angles, det_count)

        Returns
        -------
        x : (B, C, res, res)
        """
        self._require_svd("_U_k", "backward (full pseudoinverse)")
        orig_device, orig_dtype = y.device, y.dtype
        B, C, n_a, nd = y.shape
        y_flat = (y / self.dx).reshape(B * C, n_a * nd).to(dtype=self.dtype, device=self.device)
        x_flat = self._apply_pseudoinverse(y_flat, self._U_k, self._s_k, self._Vt_k)
        result = x_flat.reshape(B, C, self.resolution, self.resolution).to(device=orig_device, dtype=orig_dtype)
        if not torch.isfinite(result).all():
            warnings.warn(
                "backward() produced non-finite values. The SVD factors may be corrupted "
                "(common when ARPACK is used with k close to matrix rank). "
                "Try a larger svd_threshold.",
                RuntimeWarning, stacklevel=2,
            )
        return result

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
        la_mask = (self.angles >= self.phi[0]) & (self.angles < self.phi[1])
        y_compact = y[..., la_mask, :]                       # (B,C,n_la,det_count)
        orig_device, orig_dtype = y_compact.device, y_compact.dtype
        B, C, n_la, nd = y_compact.shape
        y_flat = (y_compact / self.dx).reshape(B * C, n_la * nd).to(dtype=self.dtype, device=self.device)
        x_flat = self._apply_pseudoinverse(y_flat, self._U_k_la, self._s_k_la, self._Vt_k_la)
        result = x_flat.reshape(B, C, self.resolution, self.resolution).to(device=orig_device, dtype=orig_dtype)
        if not torch.isfinite(result).all():
            warnings.warn(
                "backward_la() produced non-finite values. The SVD factors may be corrupted. "
                "Try a larger svd_threshold.",
                RuntimeWarning, stacklevel=2,
            )
        return result

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
    # Null-space projections
    # ------------------------------------------------------------------

    def proj_null_la(self, v: torch.Tensor) -> torch.Tensor:
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
        self._require_svd("_Vt_k_la", "proj_null_la")
        return self._proj_null(v, self._Vt_k_la)

    def proj_null_image(self, v: torch.Tensor) -> torch.Tensor:
        """Alias for proj_null_la (overrides _RadonBase CG fallback)."""
        return self.proj_null_la(v)

    def proj_null(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project image v onto null(A): v_n = v - V_k V_k^T v.

        Parameters
        ----------
        v : (B, C, res, res)

        Returns
        -------
        v_n : (B, C, res, res)  — component of v in null(A)
        """
        self._require_svd("_Vt_k", "proj_null")
        return self._proj_null(v, self._Vt_k)

    def _proj_null(self, v: torch.Tensor, Vt_k: torch.Tensor) -> torch.Tensor:
        """
        Shared null-space projection: v - Vt_k.T @ (Vt_k @ v_flat.T).

        Parameters
        ----------
        v    : (B, C, res, res)
        Vt_k : (k, n)

        Returns
        -------
        (B, C, res, res)
        """
        orig_device, orig_dtype = v.device, v.dtype
        B, C, H, W = v.shape
        v_flat = v.reshape(B * C, H * W).to(dtype=self.dtype, device=self.device)
        coeffs = v_flat @ Vt_k.t()                 # (B*C, k)
        v_range = coeffs @ Vt_k                    # (B*C, n)  — range component
        result = (v_flat - v_range).reshape(B, C, H, W)
        return result.to(device=orig_device, dtype=orig_dtype)

    def decompose_error(self, e: torch.Tensor, iters: int = 50, tol: float = 1e-6):
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
                "Pass svd_threshold > 0 at construction (and ensure phi is set for _la variants)."
            )
