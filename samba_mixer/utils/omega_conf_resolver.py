from typing import Literal

from omegaconf import OmegaConf

from samba_mixer.utils.enums import ClsTokenType


def register_new_resolver() -> None:
    """Registers custom resolvers to OmegaConf so functions can be used within the hydra config files."""
    OmegaConf.register_new_resolver("get_num_classes", lambda x: len(x) - 1)
    OmegaConf.register_new_resolver("get_num_tokens", _get_num_tokens)
    OmegaConf.register_new_resolver("get_dataloader_reload_period", _get_dataloader_reload_period)


def _get_num_tokens(
    num_samples_target: int,
    cls_token_type: str,
) -> int:
    """Calculates the resulting number of tokens from target length of resampled sequence and the cls token type.

    Args:
        num_samples_target (int): Number of samples the time series data shall be resampled to.
        cls_token_type (str): type of the cls that shall be inserted.

    Raises:
        ValueError: if unsupported cls_token_type provided.

    Returns:
        int: Number of tokens.
    """
    cls_token_type_enum = ClsTokenType(cls_token_type)
    if cls_token_type_enum in [ClsTokenType.HEAD, ClsTokenType.MIDDLE, ClsTokenType.TAIL]:
        return num_samples_target + 1
    if cls_token_type_enum == ClsTokenType.NONE:
        return num_samples_target
    raise ValueError(f"Provided unsupported cls_token_type: {cls_token_type_enum}.")


def _get_dataloader_reload_period(
    resample_type_train: Literal["linear", "random", "anchors"],
    oversample: Literal["none", "x3", "x2", "max"],
) -> int:
    """Determines wheather the dataloaders need to be recreated every epoch or just once in the beginning.

    Args:
        resample_type_train (Literal["linear", "random", "anchors"]): type of resampling to use.
        oversample (Literal["none", "x3", "x2", "max"]): type of oversampling to use.

    Returns:
        int: 1 if dataloader is recreated every epoch, 0 if created only once
    """
    if resample_type_train in ["random", "anchors"]:
        return 1

    if oversample in ["x3", "x2", "max"]:
        return 1

    return 0
