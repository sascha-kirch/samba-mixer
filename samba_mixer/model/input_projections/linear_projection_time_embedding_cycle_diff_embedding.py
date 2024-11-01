from typing import Dict

import torch

from samba_mixer.layer.embedding import SinusoidalEmbeddingLayer
from samba_mixer.model.input_projections.linear_projection_time_embedding import LinearProjectionWithLocalTimeEmbedding
from samba_mixer.model.input_projections.samba_input_projection import SambaInputProjection


class LinearProjectionWithLocalTimeAndGlobalDiffEmbedding(SambaInputProjection):
    """Project signals into d_model-dimensional space and adds pos. encodings for local timestamp and global time diff.

    Adds two seperate pos. encoding onto the the projected time signal.
    1. Encodes the local time stamp within a battery cycle using a SinusoidalEmbeddingLayer
    2. Encodes the global time diff given in hours between two cycles using a SinusoidalEmbeddingLayer

    The idea behind adding the difference in time between two cycles is that battery recuperate their capacity after
    and the model should learn this from the global time difference.
    """

    def __init__(self, d_model: int) -> None:
        """LinearProjectionWithLocalTimeAndGlobalDiffEmbedding.

        Args:
            d_model (int): Hidden dimensions of the model.
        """
        super(LinearProjectionWithLocalTimeAndGlobalDiffEmbedding, self).__init__(d_model=d_model)

        self.input_projection_with_local_time_embedding = LinearProjectionWithLocalTimeEmbedding(self.d_model)
        self.global_time_diff_embedding = SinusoidalEmbeddingLayer(self.d_model)
        self._init_modules()

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of `LinearProjectionWithLocalTimeAndGlobalDiffEmbedding`.

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
        projected_input = self.input_projection_with_local_time_embedding(x)

        # add extra axis because SinusoidalEmbeddingLayer needs tensor of  shape [batch, seq_length].
        # Unsqueeze results in [batch, 1], which is then broadcasted.
        time_diff_hours = x["time_diff_hours"].unsqueeze(-1)
        global_time_diff_embeddings = self.global_time_diff_embedding(time_diff_hours)

        return projected_input + global_time_diff_embeddings
