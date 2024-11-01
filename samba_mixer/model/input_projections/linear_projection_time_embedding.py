from typing import Dict

import torch

from samba_mixer.layer.embedding import SinusoidalEmbeddingLayer
from samba_mixer.model.input_projections.linear_projection import LinearProjection
from samba_mixer.model.input_projections.samba_input_projection import SambaInputProjection


class LinearProjectionWithLocalTimeEmbedding(SambaInputProjection):
    """Projects time signals into d_model-dimensional space and adds an positional encoding for local timestamp.

    The positional encoding (in contrast to the standard way), is not determined by the index of the sample,
    but the timestamp given in seconds serves as index for the sinusodial embedding. That implies it cannot be
    pre-computed, but must be computed for every new forward call.

    The idea behind using the time and not the index is that the timesignal is not sampled at a constant sample rate.
    """

    def __init__(self, d_model: int) -> None:
        """Initialize LinearProjectionWithLocalTimeEmbedding.

        Args:
            d_model (int): Hidden dimensions of the model.
        """
        super(LinearProjectionWithLocalTimeEmbedding, self).__init__(d_model=d_model)

        if self.d_model % 2 != 0:
            ValueError(f"d_model should be divisable by 2, but got d_model={self.d_model}")

        self.input_projection = LinearProjection(self.d_model)
        self.timestamp_embedding = SinusoidalEmbeddingLayer(self.d_model)
        self._init_modules()
        # patch_size= 4
        # self.patchifyer = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=(patch_size),stride=patch_size)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of `LinearProjectionWithLocalTimeEmbedding`.

        Args:
        x (Dict[str,torch.Tensor]): Dict witch values beeing batched input tensors. Sequence data is accessed with key
            - `x["sequence"]` (torch.Tensor): Sequence data of shape (batch_size, seq_len, feature_dim). feature_dim
            should be 4 (voltage, current, temperature, timestamp). Last feature needs to be the timestamp!
            - Optionally (depending on the dataset), there might be other global metadata for a each sequence, like
                - `x["cycle_id"]` (torch.Tensor): Integer id of the current cycle.
                - `x["battery_id"]` (torch.Tensor): Integer id of the battery the current cycle belongs to.
                - `x["time_diff_hours"]` (torch.Tensor): Time diff to previous cycle as integer in hours.

        Returns:
        torch.Tensor: Embedded tensor with shape (batch_size, seq_len, d_model).
        """
        projected_input = self.input_projection(x)

        timestamp_sec = x["sequence"][..., -1]
        timestamp_embeddings = self.timestamp_embedding(timestamp_sec)
        output = projected_input + timestamp_embeddings

        return output
