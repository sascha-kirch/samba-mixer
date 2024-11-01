from typing import Any
from typing import Optional
from typing import Tuple

import torch
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
from timm.models.layers import DropPath
from torch import nn

from samba_mixer.layer.bi_samba import BiSamba
from samba_mixer.layer.weighted_average import WeightedAverageFromPreviousMixers


class SambaMixerBlock(nn.Module):
    """Custom implementation of the MambaMixer (No code released).

    Block consists of two sub-modules: A TokenMixer and a ChannelMixer.

    Paper: https://arxiv.org/pdf/2403.19888v1
    """

    def __init__(
        self,
        d_model: int,
        num_tokens: int,
        d_state: int,
        d_conv: int,
        expand: int,
        rms_norm: bool,
        layer_idx: int,
        drop_path: float = 0.0,
        token_mixer_bimamba: bool = False,
        channel_mixer_bimamba: bool = False,
    ) -> None:
        """Initialize SambaMixerBlock.

        Args:
            d_model (int): Hidden dimensions of the model.
            num_tokens (int): Number of Tokens input into the Backbone (incl. optional CLS tokens).
            d_state (int): Dimensions of the state of the model.
            d_conv (int): Kernel size of the convolution
            expand (int): Expansion factor.
            rms_norm (bool): Wether to use RMSNorm or LayerNorm
            layer_idx (int): Index of the current SambaMixerBlock.
            drop_path (float, optional): Probability for the DropPath layer. Defaults to 0.0.
            token_mixer_bimamba (bool, optional): If true, TokenMixer is of type BiSamba. If false, standard Mamba.
                Defaults to False.
            channel_mixer_bimamba (bool, optional): If true, ChannelMixer is of type BiSamba. If false, standard Mamba.
                Defaults to False.
        """
        super().__init__()

        # NOTE Sascha: caution with the layer index!!! It must be unique for each mamba block because it is used to obtain values from a memory_dict.
        token_mixer_index = layer_idx * 2
        channel_mixer_index = layer_idx * 2 + 1

        self.token_mixer = TokenMixer(
            d_model=d_model,
            num_tokens=num_tokens,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            rms_norm=rms_norm,
            layer_idx=layer_idx,
            mixer_idx=token_mixer_index,
            drop_path=drop_path,
            bimamba=token_mixer_bimamba,
        )

        self.channel_mixer = ChannelMixer(
            d_model=d_model,
            num_tokens=num_tokens,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            rms_norm=rms_norm,
            layer_idx=layer_idx,
            mixer_idx=channel_mixer_index,
            drop_path=drop_path,
            bimamba=channel_mixer_bimamba,
        )

    def forward(
        self,
        hidden_states_token_mixer: torch.Tensor,
        hidden_states_channel_mixer: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward the SambaMixer.

        Args:
            hidden_states_token_mixer (torch.Tensor): Contains outputs of all previous TokenMixer layers.
                Shape `(batch, num_outputs, num_tokens, d_model)`
            hidden_states_channel_mixer (torch.Tensor): Contains outputs of all previous ChannelMixer layers.
                Shape `(batch, num_outputs, num_tokens, d_model)`
            residual (Optional[torch.Tensor], optional): Residual connection from previous layer of shape
                `[batch, num_tokens, d_model]`. Defaults to None.

        Returns:
            hidden_states_token_mixer_layer_l (torch.Tensor): Hiden states from TokenMixer.
            hidden_states_channel_mixer_layer_l (torch.Tensor): Hiden states from ChannelMixer.
            residual (torch.Tensor): Residual Connection
        """
        hidden_states_token_mixer_layer_l, residual = self.token_mixer(
            hidden_states_token_mixer, hidden_states_channel_mixer, residual
        )
        hidden_states_channel_mixer_layer_l, residual = self.channel_mixer(
            hidden_states_token_mixer_layer_l, hidden_states_token_mixer, hidden_states_channel_mixer, residual
        )

        return hidden_states_token_mixer_layer_l, hidden_states_channel_mixer_layer_l, residual


class TokenMixer(nn.Module):
    """Sub-module of SambaMixer to mix along the token axis."""

    def __init__(
        self,
        d_model: int,
        num_tokens: int,
        d_state: int,
        d_conv: int,
        expand: int,
        rms_norm: bool,
        layer_idx: int,
        mixer_idx: int,
        drop_path: float = 0.0,
        bimamba: bool = False,
    ) -> None:
        """Initialize TokenMixer.

        Args:
            d_model (int): Hidden dimensions of the model.
            num_tokens (int): Number of Tokens input into the Backbone (incl. optional CLS tokens).
            d_state (int): Dimensions of the state of the model.
            d_conv (int): Kernel size of the convolution
            expand (int): Expansion factor.
            rms_norm (bool): Wether to use RMSNorm or LayerNorm
            layer_idx (int): Index of the current SambaMixerBlock.
            mixer_idx (int): Index of the individual mixer blocks.
            drop_path (float, optional): Probability for the DropPath layer. Defaults to 0.0.
            bimamba (bool, optional): If true, mixer is of type BiSamba. If false, standard Mamba. Defaults to False.
        """
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm = (RMSNorm if rms_norm else nn.LayerNorm)(d_model, eps=1e-5)

        # +1 because embedding is layer 0 is also added
        prev_token_mixer = prev_channel_mixer = layer_idx + 1

        self.weighted_average_input = (
            WeightedAverageFromPreviousMixers(
                num_tokens=num_tokens,
                num_token_mixer_inputs=prev_token_mixer,
                num_channel_mixer_inputs=prev_channel_mixer,
            )
            if mixer_idx > 0
            else None
        )

        self.mixer = (BiSamba if bimamba else Mamba)(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=mixer_idx,
        )

    def forward(
        self,
        hidden_states_token_mixer: torch.Tensor,
        hidden_states_channel_mixer: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executed the TokenMixer module.

        1. Add residual connection with hidenstates from previous layer
        2. Pre-Normalization
        3. WeightedAvaraging of previous inputs to obtain hiden state for input for current TokenMixer
        4. Execute Mamba mixer.

        Args:
            hidden_states_token_mixer (torch.Tensor): Contains outputs of all previous TokenMixer layers.
                Shape `(batch, num_outputs, num_tokens, d_model)`
            hidden_states_channel_mixer (torch.Tensor): Contains outputs of all previous ChannelMixer layers.
                Shape `(batch, num_outputs, num_tokens, d_model)`
            residual (Optional[torch.Tensor], optional): Residual connection from previous layer of shape
                `[batch, num_tokens, d_model]`. Defaults to None.
            inference_params (Optional[Any], optional): Inference parameters required by Mamba class. Defaults to None.

        Returns:
            hidden_states (torch.Tensor): Hiden states from TokenMixer.
            residual (torch.Tensor): Residual Connection
        """
        # extract output of last MixerLayer which is either the channel mixer or the input embedding.
        # Then combine it it with the residual
        output_last_layer = hidden_states_channel_mixer[:, -1, ...]
        residual = (residual + self.drop_path(output_last_layer)) if residual is not None else output_last_layer
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

        if self.weighted_average_input is not None:
            # recombine the processed hiden_states into the hidden_states_of_channel_mixer outputs
            hidden_states_channel_mixer = torch.concat(
                (hidden_states_channel_mixer[:, :-1, ...], hidden_states[:, None, ...]), dim=1
            )
            hidden_states = self.weighted_average_input(hidden_states_token_mixer, hidden_states_channel_mixer)

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        return hidden_states, residual


class ChannelMixer(nn.Module):
    """Sub-module of SambaMixer to mix along the channel axis.

    In contrast to the TokenMixer, this module transposes the hidden state before and after calling the Mamba block.
    """

    def __init__(
        self,
        d_model: int,
        num_tokens: int,
        d_state: int,
        d_conv: int,
        expand: int,
        rms_norm: bool,
        layer_idx: int,
        mixer_idx: int,
        drop_path: float = 0.0,
        bimamba: bool = False,
    ) -> None:
        """Initialize ChannelMixer.

        Args:
            d_model (int): Hidden dimensions of the model.
            num_tokens (int): Number of Tokens input into the Backbone (incl. optional CLS tokens).
            d_state (int): Dimensions of the state of the model.
            d_conv (int): Kernel size of the convolution
            expand (int): Expansion factor.
            rms_norm (bool): Wether to use RMSNorm or LayerNorm
            layer_idx (int): Index of the current SambaMixerBlock.
            mixer_idx (int): Index of the individual mixer blocks.
            drop_path (float, optional): Probability for the DropPath layer. Defaults to 0.0.
            bimamba (bool, optional): If true, mixer is of type BiSamba. If false, standard Mamba. Defaults to False.
        """
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm = (RMSNorm if rms_norm else nn.LayerNorm)(d_model, eps=1e-5)

        # +1 because embedding is layer 0 is also added.
        prev_token_mixer = prev_channel_mixer = layer_idx + 1

        self.weighted_average_input = WeightedAverageFromPreviousMixers(
            num_tokens=num_tokens,
            num_token_mixer_inputs=prev_token_mixer + 1,  # +1 because the token mixer of the same layer is considered.
            num_channel_mixer_inputs=prev_channel_mixer,
        )

        self.mixer = (BiSamba if bimamba else Mamba)(
            d_model=num_tokens,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=mixer_idx,
        )

    def forward(
        self,
        output_token_mixer: torch.Tensor,
        hidden_states_token_mixer: torch.Tensor,
        hidden_states_channel_mixer: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executed the ChannelMixer module.

        1. Add residual connection with hidenstates from previous layer
        2. Pre-Normalization
        3. Add output from TokenMixer of same layer to previous TokenMixer outputs.
        4. WeightedAvaraging of previous inputs to obtain hiden state for input for current ChannelMixer
        5. Transpose hiden states
        6. Execute Mamba mixer.
        7. Transpose hidden states.

        Args:
            output_token_mixer (torch.Tensor): Hiden States from TokenMixer of same layer.
            hidden_states_token_mixer (torch.Tensor): Contains outputs of all previous TokenMixer layers.
                Shape `(batch, num_outputs, num_tokens, d_model)`
            hidden_states_channel_mixer (torch.Tensor): Contains outputs of all previous ChannelMixer layers.
                Shape `(batch, num_outputs, num_tokens, d_model)`
            residual (Optional[torch.Tensor], optional): Residual connection from previous layer of shape
                `[batch, num_tokens, d_model]`. Defaults to None.
            inference_params (Optional[Any], optional): Inference parameters required by Mamba class. Defaults to None.

        Returns:
            hidden_states (torch.Tensor): Hiden states from ChannelMixer.
            residual (torch.Tensor): Residual Connection
        """
        # extract output of last MixerLayer which is either the channel mixer or the input embedding.
        # Then combine it it with the residual
        residual = (residual + self.drop_path(output_token_mixer)) if residual is not None else output_token_mixer
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

        # recomine the processed hiden_states into the hidden_states_token_mixer outputs
        hidden_states_token_mixer = torch.concat((hidden_states_token_mixer, hidden_states[:, None, ...]), dim=1)
        hidden_states = self.weighted_average_input(hidden_states_token_mixer, hidden_states_channel_mixer)

        hidden_states = torch.einsum("bsd->bds", hidden_states)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states = torch.einsum("bds->bsd", hidden_states)

        return hidden_states, residual
