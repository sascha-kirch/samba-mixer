from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from mamba_ssm.modules.mamba_simple import Mamba
from torch import nn

from samba_mixer.ops.selective_scan_interface import mamba_inner_fn_no_out_proj


# NOTE Sascha: inspired by VideoMamba's patch of Mamba: https://github.com/OpenGVLab/VideoMamba/blob/f3427e42cb8453a523aec3a6f86d57b5bc1de5c3/mamba/mamba_ssm/modules/mamba_simple.py#L34
class BiSamba(Mamba):
    """Adds a backward path to the standard Mamba module to make it bidirectional."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        layer_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize BiSamba module.

        Args:
            d_model (int): Hidden dimensions of the model.
            d_state (int): Dimensions of the state of the model.
            d_conv (int): Kernel size of the convolution
            expand (int): Expansion factor.
            layer_idx (Optional[int], optional): Index of the layer to be created. Defaults to None.
            **kwargs (Any): Additional keyword args fed to the Mamba base class
        """
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=True,
            layer_idx=layer_idx,
            **kwargs,
        )

        A_backward = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_backward_log = torch.log(A_backward)  # Keep A_b_log in fp32

        self.A_backward_log = nn.Parameter(A_backward_log)
        self.A_backward_log._no_weight_decay = True

        self.conv1d_backward = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj_backward = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj_backward = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.D_backward = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D_backward._no_weight_decay = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params: Optional[Any] = None,
    ) -> torch.Tensor:
        """Execute BiSamba module.

        Args:
            hidden_states (torch.Tensor): Output of previous layer of shape `[batch, num_tokens, d_model]`
            inference_params (Optional[Any], optional): Inference parameters required by Mamba class. Defaults to None.

        Returns:
            torch.Tensor: Tensor of shape `[batch, num_tokens, d_model]`
        """
        batch, seqlen, _ = hidden_states.shape

        conv_state, ssm_state = None, None

        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # Forward computation
        A_forward = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        out_forward = mamba_inner_fn_no_out_proj(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A_forward,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        # Backward computation
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        A_backward = -torch.exp(self.A_backward_log.float())
        out_backward = mamba_inner_fn_no_out_proj(
            xz.flip([-1]),
            self.conv1d_backward.weight,
            self.conv1d_backward.bias,
            self.x_proj_backward.weight,
            self.dt_proj_backward.weight,
            A_backward,
            None,
            None,
            self.D_backward.float(),
            delta_bias=self.dt_proj_backward.bias.float(),
            delta_softplus=True,
        )

        # combine forward and backward outputs
        return F.linear(
            rearrange(out_forward + out_backward.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias
        )
