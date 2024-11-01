from torch import nn

from samba_mixer.math.initializer import segm_init_weights  # noqa: F401


class SambaInputProjection(nn.Module):
    """Parent base class for all Samba input projection modules.

    Reflects shared functionality, defines the interface and is used for static type hints.
    """

    def __init__(self, d_model: int) -> None:
        """Initialize SambaInputProjection.

        Args:
            d_model (int): Hidden dimensions of the model.
        """
        super().__init__()
        self.d_model = d_model

    def _init_modules(self) -> None:
        """Re-initializes all specified modules the SambaInputProjection."""
        # NOTE Sascha: for now this is deactivated since I noticed on the test meetrics that performance get worse.
        pass
        # self.apply(segm_init_weights)
