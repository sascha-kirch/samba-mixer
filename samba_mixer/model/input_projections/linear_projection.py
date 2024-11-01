from typing import Dict

import torch
from torch import nn

from samba_mixer.model.input_projections.samba_input_projection import SambaInputProjection


class LinearProjection(SambaInputProjection):
    """Projects time signals into d_model-dimensional space.

    The projection is done for each sample individually and without possitional embeddings.
    """

    def __init__(self, d_model: int) -> None:
        """Initialize LinearProjection.

        Args:
            d_model (int): Hidden dimensions of the model.
        """
        super(LinearProjection, self).__init__(d_model=d_model)

        self.input_projection = nn.Linear(in_features=3, out_features=self.d_model)
        self._init_modules()

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of LinearProjection.

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
        sequence_data = x["sequence"]
        time_signals, _ = sequence_data[..., :-1], sequence_data[..., -1]

        return self.input_projection(time_signals)
