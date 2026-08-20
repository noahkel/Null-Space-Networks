import torch.nn as nn
from src.radon import _RadonBase
import torch

# ---------------------------------------------------------------------------
# Notation (image x in R^{HxW}, sinogram y).  A_la is the *limited-angle*
# forward operator (only the measured projection angles); A is the full-angle
# operator.  The building blocks used by the models below:
#
#   N            the learned UNet correction                      N(x)
#   A x          forward projection (radon.forward)
#   P_ran        projector onto the measured rows (radon.proj_ran)
#   P_nsn        = I - P_ran, projector onto the unmeasured rows (radon.proj_nsn)
#   FBP_la       filtered back-projection from measured angles (radon.fbp_la)
#   FBP          filtered back-projection, full (radon.fbp)
#   A_la^+       truncated-SVD pseudoinverse of A_la, A_la^+ = V_k Σ_k^{-1} U_k^T
#   P_N          = I - A_la^+ A_la = I - V_k V_k^T,  image-domain projector onto
#                null(A_la)  (radon.proj_null_image);  A_la P_N = 0
#   Π_β(v)       Euclidean projection onto the L2 ball of radius β,
#                Π_β(v) = v · min(1, β / ||v||)   (_proj_l2_ball below)
#
# Every model returns  f(x) = x + (correction);  they differ only in how the
# correction is constrained to stay consistent with the measurements A_la x.
# ---------------------------------------------------------------------------


def _proj_l2_ball(v: torch.Tensor, radius: float) -> torch.Tensor:
    """Project a batch of tensors onto an L2 ball of the given radius.

    Π_r(v) = v · min(1, r / ||v||_2)   (per sample; identity inside the ball,
    radial shrink to the sphere outside it).
    """
    B = v.shape[0]
    n = torch.linalg.norm(v.view(B, -1), dim=1).clamp_min(1e-12)
    scale = torch.minimum(torch.ones_like(n), (radius / n)).view(B, 1, 1, 1)
    return v * scale


class RESNET(nn.Module):
    """
    Residual wrapper: output = x + N(x).

    The network learns a residual correction which is added to the input.
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x, y_delta=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        y_delta : torch.Tensor, optional
            Unused (kept for interface compatibility).

        Returns
        -------
        torch.Tensor
            Residual-enhanced output x + UNet(x).
        """
        # f(x) = x + N(x)   — unconstrained residual: the correction may live
        # anywhere in image space, so A_la f(x) need not equal A_la x.
        res = self.unet(x)
        return x + res


class NSN(nn.Module):
    """
    Null-Space Network (NSN).

    Learns corrections that live in the null space of the Radon operator
    by projecting UNet outputs onto unmeasured angles and backprojecting.

    """

    def __init__(self, unet: nn.Module, radon: _RadonBase):
        super().__init__()
        self.unet = unet
        self.radon = radon

    def forward(self, x, y_delta=None):
        """
        Forward pass applying null-space correction.

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        y_delta : torch.Tensor, optional
            Unused (kept for interface compatibility).

        Returns
        -------
        torch.Tensor
            Input image plus null-space correction.
        """
        # f(x) = x + P_N N(x),   P_N = I - A_la^+ A_la  (projector onto null(A_la)).
        # Since A_la P_N = 0, the correction is invisible to the measurements:
        # A_la f(x) = A_la x, so the reconstruction is data-consistent by design.
        res = self.unet(x)
        x_nsn = self.radon.proj_null_image(res) #x_nsn = self.radon.fbp(self.radon.proj_nsn(self.radon.forward(res))) #self.radon.proj_null_image(res)
        return x + x_nsn