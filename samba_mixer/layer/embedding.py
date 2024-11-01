import math

import torch
from torch import nn


class SinusoidalEmbeddingLayer(nn.Module):
    """Sinusodial positional encoding for integer timestamps.

    The positional encoding (in contrast to the standard way), is not determined by the index of the sample,
    but the time itself serves as index for the sinusodial embedding. That implies it cannot be pre comuted, but
    must be computed for every datainput.
    """

    def __init__(self, d_model: int) -> None:
        """Initialize SinusoidalEmbeddingLayer.

        Args:
            d_model (int): Hidden dimensions of the model.
        """
        super(SinusoidalEmbeddingLayer, self).__init__()

        if d_model % 2 != 0:
            ValueError(f"d_model should be divisable by 2, but got d_model={d_model}")

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sinusoidal embedding layer.

        Args:
        x (torch.Tensor): Input tensor with shape (batch_size, seq_len) representing integer timestamps.

        Returns:
        torch.Tensor: Embedded tensor with shape (batch_size, seq_len, d_model).
        """
        return self._sinusoidal_embedding(x)

    def _sinusoidal_embedding(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Generates sinusoidal embeddings for the given timestamps.

        Args:
            timestamps (torch.Tensor):  Tensor of shape (batch_size, seq_len) containing the timestamps.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, seq_len, d_model) containing the sinusoidal embeddings.
        """
        batch_size, seq_len = timestamps.size()
        embedding = torch.zeros(
            batch_size, seq_len, self.d_model, device=timestamps.device
        )  # (batch_size, seq_len, d_model)

        position = timestamps.unsqueeze(2)  # (batch_size, seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=timestamps.device) * -(math.log(10_000.0) / self.d_model)
        )  # (batch_size, seq_len, d_model/2)

        embedding[:, :, 0::2] = torch.sin(position * div_term)
        embedding[:, :, 1::2] = torch.cos(position * div_term)

        return embedding
