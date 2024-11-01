from typing import Optional

import pandas as pd


def calc_eol_threshold(
    fade_in_percent: pd.Series,
    c0: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculates the End of Life threshold given a fade in % and optional rated capacity c0.

    Args:
        fade_in_percent (pd.Series): Fade in percent that defines the EOL limit.
        c0 (Optional[pd.Series], optional): Rated capacity inf Ahr. Defaults to None.

    Returns:
        pd.Series: If `c0` is None, threshold as SOH in %, if `c0` is not None, threshold in Ahr.
    """
    if c0 is None:  # SOH in %
        return 100 - fade_in_percent
    return c0 * (100 - fade_in_percent) / 100  # capacity in Ahr


def calc_eol_indicator(
    eol_threshold: int,
    capacity_k: pd.Series,
    cycle_id: pd.Series,
) -> Optional[int]:
    """Calculates the EOL Indicator which is the cycle where the capacity drops bellow the threshold for the first time.

    Note: eol_threshold and capacity_k should be in the same unit. So, either both in SOH in % or in Ahr.

    Args:
        eol_threshold (int): End of Life threshold
        capacity_k (pd.Series): Capacity of all cycles k for of a single battery.
        cycle_id (pd.Series): cycle id for each capacity value in capacity_k.

    Returns:
        Optional[int]: The cycle where the capacity falls bellow the threshold (if it does), otherwise NaN.
    """
    # shift sequence by one
    capacity_shifted = capacity_k.shift(-1)

    # search for crossings, where one value is above and the shifted sequence bellow the threshold
    crossing = ((capacity_k >= eol_threshold) & (capacity_shifted < eol_threshold)) | (capacity_k == eol_threshold)

    # of all cycle IDs that are at a crossing, take the largest (last occurance) one because we consider recuperation
    # effects. +1 because we where using the next capacity at the current step, so we need to use the next step.
    return cycle_id[crossing].max() + 1
