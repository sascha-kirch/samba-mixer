import torch
from mamba_ssm.ops.triton.layernorm import RMSNorm
from timm.models.layers import DropPath
from torch import nn

from samba_mixer.layer.samba_mixer import SambaMixerBlock
from samba_mixer.model.backbones.samba_backbone import SambaBackbone


class SambaMixerBackbone(SambaBackbone):
    """Backbone inspired by the MambaMixer.

    Paper: https://arxiv.org/pdf/2403.19888v1

    Note that MambaMixer code is not yet released as of May 2024. This backbone is implemented by re-using code of the
    other backbones and the architectual definitions/drawings from MambaMixer

    Further note that the MambaMixer paper got updated during the development of this module. We followed v1.
    """

    def __init__(
        self,
        d_model: int,
        num_tokens: int,
        n_layer: int,
        d_state: int,
        d_conv: int,
        expand: int,
        rms_norm: bool,
        drop_path: float = 0.1,
    ) -> None:
        """Initialize SambaMixerBackbone.

        Args:
            d_model (int): Hidden dimensions of the model.
            num_tokens (int): Number of Tokens input into the Backbone (incl. optional CLS tokens).
            n_layer (int): Number of consecutivly stacked SambaBlocks.
            d_state (int): Dimensions of the state of the model.
            d_conv (int): Kernel size of the convolution.
            expand (int): Expansion factor.
            rms_norm (bool): Wether to use RMSNorm or LayerNorm.
            drop_path (float, optional): Probability for the DropPath layer. Defaults to 0.1.
        """
        super().__init__()

        drop_path_block = [x.item() for x in torch.linspace(0, drop_path, n_layer)]  # stochastic depth decay rule
        drop_path_block = [0.0, *drop_path_block]
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.layers = nn.ModuleList(
            [
                SambaMixerBlock(
                    d_model=d_model,
                    num_tokens=num_tokens,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    rms_norm=rms_norm,
                    layer_idx=i,
                    drop_path=drop_path_block[i],
                    token_mixer_bimamba=False,
                    channel_mixer_bimamba=True,
                )
                for i in range(n_layer)
            ]
        )
        self.norm_f = (RMSNorm if rms_norm else nn.LayerNorm)(d_model, eps=1e-5)
        self._init_modules(n_layer=n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the SambaMixerBackbone.

        Note that each output of each each individual TokenMixer and ChannelMixer is stored and used in the next layer,
        where they get combined in the WeightedAverage block. This is done by concatenating the outputs allong axis 1.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_length, d_model)`

        Returns:
            torch.Tensor: Output of shape `(batch, seq_length, d_model)`, same as input.
        """
        residual: torch.Tensor = None
        x = torch.unsqueeze(x, dim=1)
        hidden_states_token_mixer = x.clone()
        hidden_states_channel_mixer = x.clone()

        for layer in self.layers:
            hidden_states_token_mixer_layer_l, hidden_states_channel_mixer_layer_l, residual = layer(
                hidden_states_token_mixer, hidden_states_channel_mixer, residual
            )

            # combine results with previous results.
            hidden_states_token_mixer = torch.concat(
                (hidden_states_token_mixer, hidden_states_token_mixer_layer_l[:, None, ...]), dim=1
            )
            hidden_states_channel_mixer = torch.concat(
                (hidden_states_channel_mixer, hidden_states_channel_mixer_layer_l[:, None, ...]), dim=1
            )

        residual = (
            (self.drop_path(hidden_states_channel_mixer[:, -1, ...]) + residual)
            if residual is not None
            else hidden_states_channel_mixer[:, -1, ...]
        )
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
