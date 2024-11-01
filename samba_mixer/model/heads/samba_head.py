from typing import Callable
from typing import Optional

import torch
from torch import nn

from samba_mixer.math.initializer import segm_init_weights  # noqa: F401


class SambaHead(nn.Module):
    """Parent base class for all Samba head modules.

    Reflects shared functionality, defines the interface and is used for static type hints.
    """

    def __init__(self, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        """Initialize SambaHead.

        Args:
            criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Any loss function that takes two tensors
                (usually a prediction and a label) and returns a loss.
        """
        super().__init__()
        self.criterion = criterion

    def _init_modules(self) -> None:
        """Re-initializes all specified modules the SambaHead."""
        # NOTE Sascha: for now this is deactivated since I noticed on the test meetrics that performance get worse.
        pass
        # self.apply(segm_init_weights)

    def token_selection(
        self,
        x: torch.Tensor,
        position_cls_token: Optional[int] = None,
    ) -> torch.Tensor:
        """If `position_cls_token` is not None, selects only the cls_token, otherwise passes all tokens to the head.

        Args:
            x (torch.Tensor): Output of the backbone of shape (batch, num_tokens, d_model)
            position_cls_token (Optional[int], optional): Position of the cls_token. Defaults to None.

        Returns:
            torch.Tensor: cls_token if `position_cls_token` not None. All tokens otherwise.
        """
        if position_cls_token is not None:
            return x[:, position_cls_token, :]
        return x
