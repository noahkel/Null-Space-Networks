import torch.nn as nn

from src.radon import MatrixRadonAdapter

# ---------------------------------------------------------------------------
# Notation (image x in R^{HxW}, sinogram y).  A_la is the *limited-angle*
# forward operator (only the measured projection angles).  The building blocks
# used by the models below:
#
#   N            the learned UNet correction                      N(x)
#   A_la^+       truncated-SVD pseudoinverse of A_la, A_la^+ = V_k Σ_k^{-1} U_k^T
#   P_N          = I - A_la^+ A_la = I - V_k V_k^T,  image-domain projector onto
#                null(A_la)  (radon.proj_null_image);  A_la P_N = 0
#
# Both models return  f(x) = x + (correction);  they differ only in whether the
# correction is constrained to stay invisible to the measurements A_la x.
# ---------------------------------------------------------------------------


class RESNET(nn.Module):
    """
    Residual wrapper: output = x + N(x).

    The network learns a residual correction which is added to the input.
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image (the initial reconstruction).

        Returns
        -------
        torch.Tensor
            Residual-enhanced output x + UNet(x).
        """
        # f(x) = x + N(x)   — unconstrained residual: the correction may live
        # anywhere in image space, so A_la f(x) need not equal A_la x.
        return x + self.unet(x)


class NSN(nn.Module):
    """
    Null-Space Network (NSN).

    Identical to RESNET except that the UNet correction is projected onto the
    null space of the limited-angle Radon operator before it is added.
    """

    def __init__(self, unet: nn.Module, radon: MatrixRadonAdapter):
        super().__init__()
        self.unet = unet
        self.radon = radon

    def forward(self, x):
        """
        Forward pass applying null-space correction.

        Parameters
        ----------
        x : torch.Tensor
            Input image (the initial reconstruction).

        Returns
        -------
        torch.Tensor
            Input image plus null-space correction.
        """
        # f(x) = x + P_N N(x),   P_N = I - A_la^+ A_la  (projector onto null(A_la)).
        # Since A_la P_N = 0, the correction is invisible to the measurements:
        # A_la f(x) = A_la x, so the reconstruction is data-consistent by design.
        return x + self.radon.proj_null_image(self.unet(x))
