import torch.nn as nn

from src.radon import MatrixRadonAdapter

# ---------------------------------------------------------------------------
# Notation (image x in R^{HxW}, sinogram y).  A_la is the *limited-angle*
# forward operator (only the measured projection angles).  The building blocks
# used by the models below:
#
#   N            the learned UNet correction                      N(x)
#   A_la^+       truncated-SVD pseudoinverse of A_la, A_la^+ = V_k Σ_k^{-1} U_k^T
#   P_k          = U_k U_k^T, projector onto the retained sinogram directions
#                (radon.proj_ran)
#   P_N          = I - A_la^+ A_la = I - V_k V_k^T,  image-domain projector onto
#                the numerical null space N_tau (radon.proj_null_image);
#                P_k A_la P_N = 0, while A_la P_N has norm sigma_{k+1} != 0
#
# Both models return  f(x) = x + (correction);  they differ only in whether the
# correction is constrained to stay invisible to the retained measurements
# P_k A_la x.
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
    numerical null space of the truncated limited-angle Radon operator before
    it is added.
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
        # f(x) = x + P_N N(x),   P_N = I - A_la^+ A_la  (projector onto N_tau).
        # Since P_k A_la P_N = 0, the correction is invisible to the retained
        # measurements: P_k A_la f(x) = P_k A_la x, so the reconstruction is
        # data-consistent on them by design. A_la P_N itself is not zero: the
        # discarded readings do see the correction, weakly (norm sigma_{k+1}).
        return x + self.radon.proj_null_image(self.unet(x))
