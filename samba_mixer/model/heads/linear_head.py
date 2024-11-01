from typing import List
from typing import Optional

import torch
from torch import nn

from samba_mixer.model.heads.samba_head import SambaHead


class LinearHead(SambaHead):
    """Head using linear projection."""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        final_sigmoid: bool,
    ) -> None:
        """Initialize the LinearHead.

        Args:
            d_model (int): Hidden dimensions of the model.
            n_layer (int): Number of consecutavly stacked linear layers.
            final_sigmoid (bool): Weather or not to apply a signmoid function befor outputting the prediction.
        """
        super().__init__(criterion=nn.MSELoss())

        layers: List[nn.Module] = []

        for _ in range(n_layer):
            layers.extend([nn.Linear(d_model, d_model), nn.SiLU()])

        layers.append(nn.Linear(d_model, 1))

        if final_sigmoid:
            layers.append(nn.Sigmoid())

        self.head = nn.Sequential(*layers)

        self._init_modules()

    def forward(self, x: torch.Tensor, position_cls_token: Optional[int] = None) -> torch.Tensor:
        """Forward the LinearHead.

        If `position_cls_token` is None, all tokens are reduced into a single token using the mean() along the
        `num_tokens` axis (i.e. axis 1).

        If `position_cls_token` is an integer, regression is applied on that single token.

        Args:
            x (torch.Tensor): Output of the backbone of shape (batch, num_tokens, d_model)
            position_cls_token (Optional[int], optional): position of the cls_token. Defaults to None.

        Returns:
            torch.Tensor: Predicted output using regression of shape [batch, 1].
        """
        x = self.token_selection(x, position_cls_token)
        if len(x.shape) == 3:
            x = torch.mean(x, dim=1)  # mean to reduce multiple tokens into a single one!
        return self.head(x)
