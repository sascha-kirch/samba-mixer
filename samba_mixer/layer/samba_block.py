from functools import partial
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from mamba_ssm.modules.mamba_simple import Block
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
from timm.models.layers import DropPath
from torch import nn

from samba_mixer.layer.bi_samba import BiSamba


class SambaBlock(Block):
    """Custom Mamba Block adding DropPath to original implementation."""

    def __init__(
        self,
        layer_idx: int,
        d_model: int,
        mixer_cls: Mamba,
        norm_cls: Union[RMSNorm, nn.LayerNorm],
        drop_path: float = 0.0,
    ) -> None:
        """Initialize SambaBlock.

        Args:
            layer_idx (int): Index of the layer to be created.
            d_model (int): Hidden dimensions of the model.
            mixer_cls (Mamba): Partial of a Mamba object.
            norm_cls (Union[RMSNorm, nn.LayerNorm]): Partial of a normalization object.
            drop_path (float, optional): Probability for the DropPath layer. Defaults to 0.0.
        """
        super().__init__(
            dim=d_model,
            mixer_cls=mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=False,
            residual_in_fp32=False,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute SambaBlock.

        Args:
            hidden_states (torch.Tensor): Output of previous layer of shape `[batch, num_tokens, d_model]`
            residual (Optional[torch.Tensor], optional): Residual connection from previous layer of shape
                `[batch, num_tokens, d_model]`. Defaults to None.
            inference_params (Optional[Any], optional): Inference parameters required by Mamba class. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(hidden_states, residual)` both of shape `[batch, num_tokens, d_model]`
        """
        residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


def create_samba_block(
    layer_idx: int,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    rms_norm: bool,
    drop_path: float = 0.0,
    bimamba: bool = False,
) -> SambaBlock:
    """Creates a Samba Block using either standard Mamba SSM or bi-directional SSM.

    Args:
        layer_idx (int): Index of the layer to be created.
        d_model (int): Hidden dimensions of the model.
        d_state (int): Dimensions of the state of the model.
        d_conv (int): Kernel size of the convolution
        expand (int): Expansion factor.
        rms_norm (bool): Wether to use RMSNorm or LayerNorm
        drop_path (float, optional): Probability for the DropPath layer. Defaults to 0.0.
        bimamba (bool, optional): If true uses `BiSamba` as SSM, otherwise `Mamba`. Defaults to False.

    Returns:
        SambaBlock: Samba Block Module.
    """
    mixer_cls = partial(
        (BiSamba if bimamba else Mamba),
        layer_idx=layer_idx,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    norm_cls = partial(RMSNorm if rms_norm else nn.LayerNorm, eps=1e-5)
    return SambaBlock(layer_idx, d_model, mixer_cls, norm_cls, drop_path)
