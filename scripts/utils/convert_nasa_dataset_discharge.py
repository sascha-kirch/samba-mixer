import argparse
import datetime
import os
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml
from scipy import integrate
from tqdm import tqdm

from samba_mixer.dataset.preprocessing import drop_selected_cycles
from samba_mixer.dataset.preprocessing import drop_soh_outlier
from samba_mixer.dataset.preprocessing import filter_discharge_disconnected_load
from samba_mixer.dataset.preprocessing import pad_timesignals_discharge


# fmt: off
parser = argparse.ArgumentParser(description="Convert Discharge Cycles of Nasa Battery Dataset")
parser.add_argument("-f","--filter", action="store_true", help="Applies filters as determined by dataset analysis.")
parser.add_argument("-p","--pad", action="store_true", help="Pads the time series signals.")
args = parser.parse_args()
# fmt: on

DATASET_BASE_PATH: Path = Path("/home/dev_user/samba-mixer/datasets/nasa_batteries_orig")
path_str = "/home/dev_user/samba-mixer/datasets/nasa_batteries_preprocessed_discharge"
if args.filter:
    path_str += "_filtered"
if args.pad:
    path_str += "_padded"
OUTPUT_PATH: Path = Path(path_str)


def get_discharge_cycle_df(cycles: np.ndarray, battery_id: str) -> pd.DataFrame:
    cycle_data = []
    cycles_charge = cycles[cycles["type"] == "charge"]
    cycles_discharge = cycles[cycles["type"] == "discharge"]

    for cycle_charge_id, cycle_charge_data in enumerate(cycles_charge):
        time_cycle_start = datetime.datetime(*cycle_charge_data["time"][0].astype(int))
        ambient_temperature = cycle_charge_data["ambient_temperature"][0][0]
        current_measured = (cycle_charge_data["data"]["Current_measured"][0][0][0],)
        voltage_measured = (cycle_charge_data["data"]["Voltage_measured"][0][0][0],)
        current_charge = (cycle_charge_data["data"]["Current_charge"][0][0][0],)
        voltage_charge = (cycle_charge_data["data"]["Voltage_charge"][0][0][0],)
        temperature_measured = (cycle_charge_data["data"]["Temperature_measured"][0][0][0],)
        time = (cycle_charge_data["data"]["Time"][0][0][0],)
        data = pd.DataFrame(
            np.concatenate(
                [
                    current_measured,
                    voltage_measured,
                    current_charge,
                    voltage_charge,
                    temperature_measured,
                    time,
                ]
            ).T,
            columns=[
                "current_measured",
                "voltage_measured",
                "current_charge",
                "voltage_charge",
                "temperature_measured",
                "time",
            ],
        )
        num_samples = len(data)
        cycle_data.append(
            [
                "charge",
                int(battery_id[1::]),
                cycle_charge_id,
                time_cycle_start,
                ambient_temperature,
                data,
                num_samples,
            ]
        )

    for cycle_discharge_id, cycle_discharge_data in enumerate(cycles_discharge):
        time_cycle_start = datetime.datetime(*cycle_discharge_data["time"][0].astype(int))
        ambient_temperature = cycle_discharge_data["ambient_temperature"][0][0]
        current_measured = (cycle_discharge_data["data"]["Current_measured"][0][0][0],)
        voltage_measured = (cycle_discharge_data["data"]["Voltage_measured"][0][0][0],)
        current_load = (cycle_discharge_data["data"]["Current_load"][0][0][0],)
        voltage_load = (cycle_discharge_data["data"]["Voltage_load"][0][0][0],)
        temperature_measured = (cycle_discharge_data["data"]["Temperature_measured"][0][0][0],)
        time = (cycle_discharge_data["data"]["Time"][0][0][0],)
        capacity_k_Ahr = cycle_discharge_data["data"]["Capacity"][0][0][0][0]
        data = pd.DataFrame(
            np.concatenate(
                [
                    current_measured,
                    voltage_measured,
                    current_load,
                    voltage_load,
                    temperature_measured,
                    time,
                ]
            ).T,
            columns=[
                "current_measured",
                "voltage_measured",
                "current_load",
                "voltage_load",
                "temperature_measured",
                "time",
            ],
        )
        num_samples = len(data)
        cycle_data.append(
            [
                "discharge",
                int(battery_id[1::]),
                cycle_discharge_id,
                time_cycle_start,
                ambient_temperature,
                data,
                num_samples,
                capacity_k_Ahr,
            ]
        )

    df = pd.DataFrame(
        cycle_data,
        columns=[
            "cycle_type",
            "battery_id",
            "cycle_id",
            "time_cycle_start",
            "ambient_temperature",
            "data",
            "num_samples",
            "capacity_k",
        ],
    )

    # Sort according to starting time of the cycle
    df.sort_values(by="time_cycle_start", inplace=True)

    # remove first cycle if it is a discharge cycle, because then we try to discharge an empty battery
    if args.filter and df.iloc[0]["cycle_type"] == "discharge":
        df.drop(df.head(1).index, inplace=True)

    return df[df["cycle_type"] == "discharge"]


def append_instantanious_capacity(data: np.ndarray, capacity_k: float) -> np.ndarray:
    data["capacity_t"] = capacity_k + integrate.cumulative_trapezoid(
        data["current_measured"], x=data["time"], initial=0.0
    ) / (3_600)
    return data


def process_battery(battery_id: str, battery_meta_data: Dict[str, str]) -> pd.DataFrame:
    """Process each individual battery data.

    Writes all discharge cycle data of a given battery into individual npy-files and returns a list with metadata of
    each cycle.

    Args:
        battery_id (str): The name of the battery (e.g. B0005)
        battery_meta_data (Dict[str, str]): Dictionarry containing the metadata of each battery as taken from the
            README.txt of the original NASA dataset

    Returns:
        pd.DataFrame: Information about each individual cycle of the given batery.
    """
    mat_file = DATASET_BASE_PATH / battery_meta_data["sub_dir"] / f"{battery_id}.mat"
    mat_db = sio.loadmat(mat_file)[battery_id]

    # ndarray of individual cycles of either CHARGE,DISCHARGE or IMPEDANCE. Shape (<NUM_CYCLES>,)
    cycles = mat_db["cycle"][0, 0][0, :]

    discharge_df = get_discharge_cycle_df(cycles, battery_id)

    # ADD METADATA
    discharge_df["capacity_0"] = battery_meta_data["c0"]
    discharge_df["fade"] = battery_meta_data["fade_in_percent"]
    discharge_df["cutoff_voltage"] = battery_meta_data["discharge"]["cutoff_voltage"]
    discharge_df["discharge_type"] = battery_meta_data["discharge"]["discharge_type"]
    discharge_df["discharge_amplitude"] = battery_meta_data["discharge"]["discharge_amplitude"]
    discharge_df["discharge_frequency"] = battery_meta_data["discharge"]["discharge_frequency"]
    discharge_df["discharge_dutycycle"] = battery_meta_data["discharge"]["discharge_dutycycle"]

    # CALC SOH
    discharge_df["soh"] = discharge_df["capacity_k"] / discharge_df["capacity_0"] * 100

    # Apply filters on Cycle Level
    if args.filter:
        discharge_df = drop_selected_cycles(discharge_df, battery_meta_data["discharge"]["drop_cycles"])
        discharge_df = drop_soh_outlier(discharge_df, threshold=10)

    # Apply filter on timeseries data.
    if args.filter:
        discharge_df["data"] = discharge_df.apply(
            lambda per_cycle_data: filter_discharge_disconnected_load(per_cycle_data["data"]),
            axis=1,
        )

    if args.pad:
        discharge_df = pad_timesignals_discharge(discharge_df)

    # CALC INSTANTANIOUS CAPACITY AT TIME t FOR EACH CYCLE
    discharge_df["data"] = discharge_df[["data", "capacity_k"]].apply(
        lambda df: append_instantanious_capacity(df["data"], df["capacity_k"]), axis=1
    )
    # ADD FILENAME
    discharge_df["data_file"] = discharge_df["cycle_id"].apply(
        lambda cycle_id: f"{battery_id}/discharge/discharge_{battery_id}_{cycle_id}.npy"
    )

    # SAVE NPY
    discharge_df[["data", "data_file"]].apply(
        lambda x: np.save(OUTPUT_PATH / x["data_file"], x["data"].to_numpy()), axis=1
    )

    # DROP DATA COLUMN
    discharge_df.drop(columns=["data", "cycle_type"], inplace=True)

    return discharge_df


def _calc_global_time_diff_in_hours(new_dataset: pd.DataFrame) -> pd.DataFrame:
    # Get the time difference between cycles in hours.
    new_dataset["time_diff_hours"] = new_dataset["time_cycle_start"].diff().dt.total_seconds() // 3_600
    grp_battary_id = new_dataset[["battery_id", "time_diff_hours"]].groupby(by="battery_id")

    # replace first value of a battery's cycle time diff to NaT
    new_dataset.loc[grp_battary_id.head(1).index, "time_diff_hours"] = 0

    # convert new column to int. Has to be done once all NaN values are filles, otherwise raises error.
    new_dataset["time_diff_hours"] = new_dataset["time_diff_hours"].astype("int")

    return new_dataset


def post_process(new_dataset: pd.DataFrame) -> pd.DataFrame:
    new_dataset = _calc_global_time_diff_in_hours(new_dataset)
    return new_dataset


if __name__ == "__main__":
    global_data: List[pd.DataFrame] = []

    # Note that the meta data dict is only to iterate through the data and is not the final format of the file for the dataset
    with open("/home/dev_user/samba-mixer/scripts/utils/nasa_battery_metadata.yml", "r") as yaml_file:
        batteries = yaml.safe_load(yaml_file)

    for battery_id, battery_meta_data in tqdm(batteries.items(), desc="Process batteries.", colour="red"):
        if args.filter and not battery_meta_data["usable"]:
            continue
        os.makedirs(OUTPUT_PATH / battery_id / "discharge", exist_ok=True)
        battery_data = process_battery(battery_id, battery_meta_data)
        global_data.append(battery_data)

    new_dataset = pd.concat(global_data, ignore_index=True)

    new_dataset = post_process(new_dataset)

    new_dataset.to_csv(OUTPUT_PATH / "data.csv", index=False)
    print(new_dataset)
