import math
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from samba_mixer.utils.enums import OversampleType


def drop_selected_cycles(per_battery_data: pd.DataFrame, cycle_ids_to_drop: List[int]) -> pd.DataFrame:
    """Drops selected cycles from the battery data.

    Args:
        per_battery_data (pd.DataFrame): Data of a single battery
        cycle_ids_to_drop (List[int]): List of cycle ids to drop.

    Returns:
        pd.DataFrame: Battery data with the selected cycles dropped.
    """
    return per_battery_data[~per_battery_data["cycle_id"].isin(cycle_ids_to_drop)]


def drop_soh_outlier(per_battery_data: pd.DataFrame, *, threshold: Union[float, int]) -> pd.DataFrame:
    """Drops cycles where the soh value of a given battery exceeds the threshold and hence is seen as outlier.

    Args:
        per_battery_data (pd.DataFrame): Data of a single battery
        threshold (Union[float,int]): threshold in [%]

    Returns:
        pd.DataFrame: battery data with the outliers dropped
    """
    fdiff = (per_battery_data["soh"] - per_battery_data["soh"].shift(1)).abs()
    bdiff = (per_battery_data["soh"] - per_battery_data["soh"].shift(-1)).abs()
    mask_fdiff = fdiff > threshold
    mask_bdiff = bdiff > threshold
    mask_borders = bdiff.isna() | fdiff.isna()
    return per_battery_data[~(mask_fdiff & mask_bdiff | mask_borders & mask_fdiff | mask_borders & mask_bdiff)]


def filter_discharge_disconnected_load(discharge_time_signals: pd.DataFrame) -> pd.DataFrame:
    """Detects the point where the load is removed in a discharge cycle and removes samples after that point.

    Args:
        discharge_time_signals (pd.DataFrame): Time signals of a single cycle.

    Returns:
        pd.DataFrame: Filtered tieme signals
    """
    index_cutoff = discharge_time_signals["voltage_measured"].idxmin()
    return discharge_time_signals[discharge_time_signals.index < index_cutoff]


def filter_k_first_cycles(
    df: pd.DataFrame,
    test_battery_cycle_start: Dict[int, int],
) -> pd.DataFrame:
    """Filters cycles according to the provided `test_battery_cycle_start`.

    Args:
        df (pd.DataFrame): BAttery Data
        test_battery_cycle_start (Dict[int,int]): Dict with keys beeing battery_ids and values start cycles.

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    for battery_id_to_filter, start_cycle in test_battery_cycle_start.items():
        df = df.query(f" \
                (battery_id == {battery_id_to_filter} and cycle_id >= {start_cycle})\
                or (battery_id != {battery_id_to_filter})")
        df.loc[df["battery_id"] == battery_id_to_filter, "cycle_id"] -= start_cycle
    return df


def filter_charger_disconnected(
    charge_time_signals: pd.DataFrame,
    *,
    threshold: Union[float, int],
) -> pd.DataFrame:
    """Detects the point where the charger is deactivated in a charge cycle and removes samples after that point.

    Args:
        charge_time_signals (pd.DataFrame): Time signals of a single cycle.
        threshold (Union[float,int]): threshold of edge detector level in volt.

    Returns:
        pd.DataFrame: Filtered time signals
    """
    voltage_charge = charge_time_signals["voltage_charge"]
    voltage_charge_shifted = charge_time_signals["voltage_charge"].shift(-1)

    # search for crossings, where one value is above and the shifted sequence bellow the threshold
    crossing = ((voltage_charge >= threshold) & (voltage_charge_shifted < threshold)) | (voltage_charge == threshold)

    # of all cycle IDs that are at a crossing, take the smallest (first occurance) one. +1 because we where using the next capacity at the current step, so we need to use the next step.
    index_cutoff = crossing.index[crossing].min() + 1
    if math.isnan(index_cutoff):
        return charge_time_signals
    return charge_time_signals[charge_time_signals.index < index_cutoff]


# TODO Sascha: df must contain train and test samples, otherwise it does not work...
def oversample(
    df: pd.DataFrame,
    oversample_type: OversampleType,
    bin_width: int,
) -> pd.DataFrame:
    """Selects the appropiated oversample function based on the `oversample_type` and applies it to the data in `df`.

    Oversampling adds new rows to the dataframe, where each row corresponds to a cycle of the battery's charging or
    discharge cycle.

    Args:
        df (pd.DataFrame): Dataframe containing individual battery cycles as rows.
        oversample_type (OversampleType): type of oversampling.
        bin_width (int): width of the bins in percent points that shall be used to divide the data into.

    Returns:
        pd.DataFrame: Potentially oversampled dataframe.
    """
    if oversample_type == OversampleType.X2:
        return _oversample_x_test(df, bin_width, x=2)
    if oversample_type == OversampleType.X3:
        return _oversample_x_test(df, bin_width, x=3)
    if oversample_type == OversampleType.MAX:
        return _oversample_max_train(df, bin_width)

    return df


def _oversample_x_test(
    df: pd.DataFrame,
    bin_width: int,
    x: int = 2,
) -> pd.DataFrame:
    """Oversamples the training dataset.

    Divides the SOH predictions into bins of width `bin_width`. Further, compares the number of samples in each bin and
    then oversamples the training data, until it has roughly `x`-times more samples as the test data in the same bin.
    If a bin has test data but no train data, no data is oversampled.

    Args:
        df (pd.DataFrame): Battery data with individual cycles as rows. Must have a collumn '"split"', that contains weather a cycle belongs to the `test` or the `train` set.
        bin_width (int): Width of the bin in percentage points.
        x (int, optional): Factor how much train samples than test samples there should be in a bin. Defaults to 2.

    Returns:
        pd.DataFrame: Oversampled dataset.
    """
    df["bin"] = pd.cut(df["soh"], bins=np.arange(0, 110, bin_width), include_lowest=True).astype(str)

    rows_to_add: List[pd.DataFrame] = []

    for bin, data in df.groupby(by="bin"):
        train_data = data[data["split"] == "train"]
        test_data = data[data["split"] == "test"]

        samples_in_bin_train = (train_data["bin"] == bin).sum()
        samples_in_bin_test = (test_data["bin"] == bin).sum()

        if samples_in_bin_test < 1:
            continue

        if samples_in_bin_train < x * samples_in_bin_test:
            samples_to_draw = x * samples_in_bin_test - samples_in_bin_train
            if samples_in_bin_train > 0:
                rows_to_add.append(train_data.sample(n=samples_to_draw, replace=True, random_state=47))

    resampled_rows = pd.concat(rows_to_add)
    return pd.concat([df, resampled_rows], axis=0)


def _oversample_max_train(
    df: pd.DataFrame,
    bin_width: int,
) -> pd.DataFrame:
    """Oversamples the training dataset.

    Divides the SOH predictions into bins of width `bin_width`. Further, compares the number of samples in each bin and
    then oversamples the training data, until it has roughly the same number of samples as the bin of the training set
    with the maximum number of samples.
    If a bin has test data but no train data, no data is oversampled.


    Args:
        df (pd.DataFrame): Battery data with individual cycles as rows. Must have a collumn `"split"`, that contains weather a cycle belongs to the `test` or the `train` set.
        bin_width (int): Width of the bin in percentage points.

    Returns:
        pd.DataFrame: Oversampled dataset.
    """
    df["bin"] = pd.cut(df["soh"], bins=np.arange(0, 110, bin_width), include_lowest=True).astype(str)
    max_train_count = df.loc[df["split"] == "train", "bin"].value_counts().max()

    rows_to_add: List[pd.DataFrame] = []

    for bin, data in df.groupby(by="bin"):
        train_data = data[data["split"] == "train"]
        test_data = data[data["split"] == "test"]

        samples_in_bin_train = (train_data["bin"] == bin).sum()
        samples_in_bin_test = (test_data["bin"] == bin).sum()

        if samples_in_bin_test < 1:
            continue

        if samples_in_bin_train < max_train_count:
            samples_to_draw = max_train_count - samples_in_bin_train
            if samples_in_bin_train > 0:
                rows_to_add.append(train_data.sample(n=samples_to_draw, replace=True, random_state=47))

    resampled_rows = pd.concat(rows_to_add)
    return pd.concat([df, resampled_rows], axis=0)


def pad_timesignals_discharge(
    per_battery_data: pd.DataFrame,
) -> pd.DataFrame:
    """Pads time signals with 0V and 0A, the last temperature and adds new sample times.

    Args:
        per_battery_data (pd.DataFrame): Data of a single battery

    Returns:
        pd.DataFrame: padded timesignals.
    """
    # for each of the cycles, get the max time value.
    max_time_per_cycle = per_battery_data["data"].apply(lambda cycle_data: cycle_data["time"].max())
    # get the maximum time of the longest cycle (likely the first one)
    max_time = max_time_per_cycle.max()
    new_max_time = max_time + max_time // 20

    for index, row_cycle in tqdm(per_battery_data.iterrows(), desc="Pad discharge time signals."):
        last_sample_time = row_cycle["data"]["time"].iloc[-1]

        # Take the difference in sample time for the padded samples.
        last_sample_time_diff = last_sample_time - row_cycle["data"]["time"].iloc[-2]

        time_missing = new_max_time - last_sample_time
        cycles_to_add = int(time_missing // last_sample_time_diff + 1)

        new_rows = [
            pd.DataFrame(
                {
                    "current_measured": [0.0],
                    "voltage_measured": [0.0],
                    "current_load": [0.0],
                    "voltage_load": [0.0],
                    "temperature_measured": [0.0],
                    "time": [last_sample_time + last_sample_time_diff * cycle],
                }
            )
            for cycle in range(1, cycles_to_add + 1)
        ]
        df_new_rows = pd.concat(new_rows)

        # update original dataframe.
        per_battery_data.at[index, "data"] = pd.concat([row_cycle["data"], df_new_rows], ignore_index=True)

    # update length column
    per_battery_data["num_samples_padded"] = per_battery_data["data"].apply(lambda x: len(x))

    return per_battery_data
