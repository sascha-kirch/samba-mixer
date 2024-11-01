import torch
from mamba_ssm.ops.triton.layernorm import RMSNorm
from timm.models.layers import DropPath
from torch import nn

from samba_mixer.layer.samba_block import create_samba_block
from samba_mixer.model.backbones.samba_backbone import SambaBackbone


class VanillaMambaBackbone(SambaBackbone):
    """Basic Mamba Backbone with some addaptations.

    1. Adding DropPath with stochastic depth decay to the residuals like in VisionMamba/VideoMamba.
    2. Option to either use a standard Mamba SSM or a Bidirectional Mamba SSM like introduced in VisionMamba.
        I.e. the one defined as v2 in their code.
    3. Option to do bidirectional inference also inspired by VideoMamba.
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_state: int,
        d_conv: int,
        expand: int,
        rms_norm: bool,
        drop_path: float = 0.1,
        bimamba: bool = False,
        bidirectional_execution: bool = False,
    ) -> None:
        """Initialize VanillaMambaBackbone.

        Args:
            d_model (int): Hidden dimensions of the model.
            n_layer (int): Number of consecutivly stacked SambaBlocks.
            d_state (int): Dimensions of the state of the model.
            d_conv (int): Kernel size of the convolution.
            expand (int): Expansion factor.
            rms_norm (bool): Wether to use RMSNorm or LayerNorm.
            drop_path (float, optional): Probability for the DropPath layer. Defaults to 0.1.
            bimamba (bool, optional): If true, use BiMamba block from VisionMamba, otherwise use standard Mamba block.
                Defaults to False.
            bidirectional_execution (bool, optional): If true, infers two MambaBlocks at a time. One as usualy and one
                with the reversed sequence of tokens and then combines their results. Defaults to False.
        """
        super().__init__()

        self.bidirectional_execution = bidirectional_execution

        drop_path_block = [x.item() for x in torch.linspace(0, drop_path, n_layer)]  # stochastic depth decay rule
        drop_path_block = [0.0, *drop_path_block]
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_samba_block(
                    layer_idx=i,
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    rms_norm=rms_norm,
                    drop_path=drop_path_block[i],
                    bimamba=bimamba,
                )
                for i in range(n_layer)
            ]
        )
        self.norm_f = (RMSNorm if rms_norm else nn.LayerNorm)(d_model, eps=1e-5)
        self._init_modules(n_layer=n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the VanillaMambaBackbone.

        If `bidirectional_execution` is True, for each forward pass two SambaBlocks are infered at once. The first block
        is infered as usual an the second block get's the reversed sequence of tokens as input. After that both outputs
        and residuals are added. This part is inspiered by VisionMamba.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_length, d_model)`

        Returns:
            torch.Tensor: Output of shape `(batch, seq_length, d_model)`, same as input.
        """
        residual: torch.Tensor = None
        hidden_states = x

        if self.bidirectional_execution:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                # forward
                hidden_states_f, residual_f = self.layers[i * 2](hidden_states, residual)

                # backward
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual is None else residual.flip([1])
                )

                # combine both
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])
        else:
            for layer in self.layers:
                hidden_states, residual = layer(hidden_states, residual)

        residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
