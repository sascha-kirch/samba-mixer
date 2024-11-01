from timm.models.layers import lecun_normal_
from timm.models.layers import trunc_normal_
from torch import nn


def segm_init_weights(module: nn.Module) -> None:
    """Initialization for Heads and input projection.

    Inspired by VisionMamba: https://github.com/hustvl/Vim/blob/6143d07b3dd31f904c63840a19e22d95d1124493/vim/models_mamba.py

    Args:
        module (nn.Module): Module to be initialized
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
