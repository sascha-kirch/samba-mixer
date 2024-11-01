from typing import List
from typing import Optional

import torch
from torch import nn

from samba_mixer.model.backbones.samba_backbone import SambaBackbone
from samba_mixer.model.backbones.samba_mixer import SambaMixerBackbone


class HierachicalSambaMixerStage(nn.Module):
    """Single Stage of a `HierachicalSambaMixerBackbone`."""

    def __init__(
        self,
        d_model: int,
        num_tokens: int,
        n_layer: int,
        d_state: int,
        d_conv: int,
        expand: int,
        rms_norm: bool,
        downsample: bool,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()

        self.mixer_stage = SambaMixerBackbone(
            d_model=d_model,
            num_tokens=num_tokens,
            n_layer=n_layer,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            rms_norm=rms_norm,
            drop_path=drop_path,
        )
        self.downsample_stage: Optional[nn.Module] = None

        if downsample:
            self.downsample_stage = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model * 2,
                stride=2,
                kernel_size=2,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the HierachicalSambaStage.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_length, d_model)`

        Returns:
            torch.Tensor: Output of shape `(batch, seq_length, d_model)`, same as input.
        """
        x = self.mixer_stage(x)
        if self.downsample_stage is not None:
            x = torch.einsum("BLC->BCL", x)
            x = self.downsample_stage(x)
            x = torch.einsum("BCL->BLC", x)

        return x


class HierachicalSambaMixerBackbone(SambaBackbone):
    """Multi-Stage backbone that doubles d_model and halves seq_length after every stage."""

    def __init__(
        self,
        stage_layers: List[int],
        d_model: int,
        num_tokens: int,
        d_state: int,
        d_conv: int,
        expand: int,
        rms_norm: bool,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_stages = len(stage_layers)
        self.stage_layers = stage_layers

        # TODO Sascha: improve stability of this code by adding some checks (e.g. d_model and num_tokens).
        # TODO Sascha: only works with none token. Need to add some sort of check in general to ensure compatibility (e.g. which cls token etc.)

        self.stages = nn.ModuleList(
            [
                HierachicalSambaMixerStage(
                    d_model=d_model // (2 ** ((self.num_stages - 1) - stage)),
                    num_tokens=num_tokens // (2**stage),
                    n_layer=self.stage_layers[stage],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    rms_norm=rms_norm,
                    downsample=stage != (self.num_stages - 1),  # note the last stage
                    drop_path=drop_path,
                )
                for stage in range(self.num_stages)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the HierachicalSambaMixerBackbone.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_length, d_model/2**(num_stages-1))`

        Returns:
            torch.Tensor: Output of shape `(batch, seq_length/2**(num_stages-1), d_model)`, same as input.
        """
        for stage in self.stages:
            x = stage(x)

        return x
