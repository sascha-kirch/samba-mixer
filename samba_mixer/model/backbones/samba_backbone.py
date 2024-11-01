from abc import ABC
from typing import Dict
from typing import Optional

from mamba_ssm.models.mixer_seq_simple import _init_weights  # noqa: F401
from torch import nn


class SambaBackbone(nn.Module, ABC):
    """Abstract base class for all Samba backbone modules.

    Reflects shared functionality, defines the interface and is used for static type hints.
    """

    def _init_modules(self, n_layer: int, initializer_cfg: Optional[Dict] = None) -> None:
        """Re-initializes all specified modules the SambaBackbone.

        Args:
            n_layer (int): Number of stacked SambaBlocks in the SambaBackbone
            initializer_cfg (Optional[Dict], optional): Configuration for the initializer from a hydra config file.
                Defaults to None.
        """
        # NOTE Sascha: for now this is deactivated since I noticed on the test metrics that performance get worse.
        pass
        # self.apply(
        #     partial(
        #         _init_weights,
        #         n_layer=n_layer,
        #         **(initializer_cfg if initializer_cfg is not None else {}),
        #     )
        # )
