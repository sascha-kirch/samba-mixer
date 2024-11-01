import torch
from torch import nn


class WeightedAverageFromPreviousMixers(nn.Module):
    """Layer that combines all outputs from previous TokenMixer and a ChannelMixer blocks from the SambaMixer Backbone.

    Reimplemented from SambaMixer: https://arxiv.org/pdf/2403.19888v1, since code not published.
    """

    def __init__(
        self,
        num_tokens: int,
        num_token_mixer_inputs: int,
        num_channel_mixer_inputs: int,
    ) -> None:
        """Initialize the WeightedAverageFromPreviousMixers module.

        Args:
            num_tokens (int): Number of tokens feed into the backbone.
            num_token_mixer_inputs (int): Number of outputs from previous TokenMixer Blocks that are averaged.
            num_channel_mixer_inputs (int): Number of outputs from previous ChannelMixer Blocks that are averaged.
        """
        super().__init__()
        self.num_previous_token_mixer = num_token_mixer_inputs
        self.num_previous_channel_mixer = num_channel_mixer_inputs

        self.alphas = nn.Parameter(torch.ones(num_token_mixer_inputs, num_tokens))
        self.betas = nn.Parameter(torch.ones(num_channel_mixer_inputs, num_tokens))

    def forward(
        self,
        token_mixer_outputs: torch.Tensor,
        channel_mixer_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the WeightedAverageFromPreviousMixers module.

        Args:
            token_mixer_outputs (torch.Tensor): TokenMixer otputs. Shape `[batch, num_outputs, num_tokens, d_model]`
            channel_mixer_outputs (torch.Tensor): ChannelMixer otputs. Shape `[batch, num_outputs, num_tokens, d_model]`

        Returns:
            torch.Tensor: Weighted average of all inputs. Shape: `[batch, num_tokens, d_model]`
        """
        alpha_term = self._multiply(token_mixer_outputs, self.alphas) / self.num_previous_token_mixer
        beta_term = self._multiply(channel_mixer_outputs, self.betas) / self.num_previous_channel_mixer

        return (alpha_term + beta_term) / 2

    def _multiply(
        self,
        mixer_otuputs: torch.Tensor,
        param: nn.Parameter,
    ) -> torch.Tensor:
        """Multiplies all of a mixer's outputs with a learnable parameter.

        Multiplication is done element-wise along the axes `num_outputs` and `num_tokens` and is broadcasted over
        `batch` axis and `d_model` axis. The result is then reduce-summed over the `num_output` axis.

        Args:
            mixer_otuputs (torch.Tensor): TokenMixer|ChannelMixer otputs: `[batch, num_outputs, num_tokens, d_model]`
            param (nn.Parameter): _description_

        Returns:
            torch.Tensor: _description_
        """
        # "bstd" is the shape of the mixer_otuputs with b=batch, l=previous_layers, t=num_tokens, d=d_mode
        # "st" is the shape of the param.
        # "btd" is the shape of the resulting output
        # This comand means: do element-wise multiplication allong axis lt, broadcast b and d and finally reduce sum over l
        return torch.einsum("bltd, lt -> btd", mixer_otuputs, param)
