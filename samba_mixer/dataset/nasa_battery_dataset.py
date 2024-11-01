import logging
from collections import defaultdict
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.figure
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from omegaconf import DictConfig
from scipy import interpolate
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from samba_mixer.dataset.preprocessing import filter_k_first_cycles
from samba_mixer.dataset.preprocessing import oversample
from samba_mixer.dataset.samba_data_module import SambaDataModule
from samba_mixer.math.battery_formulations import calc_eol_indicator
from samba_mixer.math.battery_formulations import calc_eol_threshold
from samba_mixer.math.converter import ck_to_soh
from samba_mixer.math.converter import soh_to_ck
from samba_mixer.math.metrics import mape
from samba_mixer.math.metrics import root_mean_squared_error
from samba_mixer.utils.enums import DataFormat
from samba_mixer.utils.enums import OversampleType
from samba_mixer.utils.enums import TimeSignalResamplingType
from samba_mixer.utils.typing import Number
from samba_mixer.utils.visualization import get_figure_battery_prediction
from samba_mixer.utils.visualization import get_figure_predicted_over_true


log = logging.getLogger(__name__)


class NasaBatteryDataset(Dataset):
    """Dataset for the NasaBattery data.

    Number 5:
    https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
    """

    def __init__(
        self,
        sequences: List[Tuple[pd.DataFrame, float, int, int, int]],
        augmentation_functions: Optional[List[Callable[[pd.DataFrame], pd.DataFrame]]] = None,
    ) -> None:
        """Initialize the NasaBatteryDataset.

        Args:
            sequences (List[Tuple[pd.DataFrame, float, int, int, int]]): Pre-processed dataset.
            augmentation_functions (Optional[List[Callable[[pd.DataFrame], pd.DataFrame]]]): List of functions applied during
                data loading on each individual data sample.
        """
        self.sequences = sequences
        self.augmentation_functions: List[Callable[[pd.DataFrame], pd.DataFrame]] = []

        if augmentation_functions is not None:
            self.augmentation_functions = augmentation_functions

    def __len__(self) -> int:
        """Calculates the lenght of the dataset which are the number of battery cycles.

        Returns:
            int: Number of battery cycles within the Dataset.
        """
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Gets a single item from the dataset.

        Args:
            index (int): Index to querry the dataset.

        Returns:
            Dict[str, torch.Tensor]: Data of single cycle of a single battery.
        """
        sequence, label, cycle_id, battery_id, time_diff_hours = self.sequences[index]

        # CAUTION: Tensor() creates a new Tensor object and tensor() converts a number into a tensor!!!
        return {
            "sequence": torch.Tensor(sequence.to_numpy()),
            "label": torch.tensor(label),
            "cycle_id": torch.tensor(cycle_id),
            "battery_id": torch.tensor(battery_id),
            "time_diff_hours": torch.tensor(time_diff_hours),
        }


class NasaBatteryDataModule(SambaDataModule):
    """Implementation of the Abstract SambaDataModule for the Nasa Battery Dataset.

    Serves as base class for the more specialized and task-dependent DataModules:
    - For regression use `NasaBatteryRegressionDataModule`
    - For classification use `NasaBatteryClassificationDataModule`
    """

    def __init__(
        self,
        dataset_root_path: Path,
        train_batteries: List[int],
        test_batteries: List[int],
        num_samples_target: int,
        columns_all_timesignals: List[str],
        columns_target_timesignals: List[str],
        range_target_timesignals: Dict[str, Dict[str, Number]],
        batch_size: int,
        resample_type_train_signal: TimeSignalResamplingType,
        oversample_type: OversampleType,
        label_format: DataFormat,
        label_scalar: Number,
        test_battery_cycle_start: Optional[Dict[int, int]] = None,
    ) -> None:
        """Initialize NasaBatteryDataModule.

        Args:
            dataset_root_path (Path): Rootpath of the previously converted dataset containing the data.csv file.
            train_batteries (List[int]): List of integer battery ids that shall be used for the training set.
            test_batteries (List[int]): List of integer battery ids that shall be used for the testing set.
            num_samples_target (int): Number of samples the timeseries data shall be resampled to during preprocessing.
            columns_all_timesignals (List[str]): Collumn names of all columns in the timeseries data.
            columns_target_timesignals (List[str]): Collumn names that shall be used in the final dataset.
            range_target_timesignals (Dict[str, Dict[str, Number]]): Dict containing the expected range of signal for
                each column. Format: {<COLUMN_NAME>: {"min":<MIN_VALUE>, "max":<MAX_VALUE>}}
            batch_size (int): Batchsize for training and validation
            resample_type_train_signal (TimeSignalResamplingType): If not TimeSignalResamplingType.LINEAR, applies
                specified resampling as resample augmentation per cycle inside the data loader.
            oversample_type (OversampleType): Type of oversampling to use.
            label_format (DataFormat): Format the label should have. [SOH | CK]
            label_scalar (Number): scalar used to scale the label coulmn for training and prediction.
            test_battery_cycle_start(Optional[Dict[int,int]]): Dictionary containing the battery id as key and the cycle
                number as value, where the test dataset shall start. If not provided, the test dataset will start at the
                first cycle of each battery. Defaults to None.

        """
        super().__init__()

        self.dataset_root_path = dataset_root_path
        self.train_batteries = train_batteries
        self.test_batteries = test_batteries
        self.num_samples_target = num_samples_target
        self.columns_all_timesignals = columns_all_timesignals
        self.columns_target_timesignals = columns_target_timesignals
        self.range_target_timesignals = range_target_timesignals
        self.batch_size = batch_size
        self.label_format = label_format
        self.label_scalar = label_scalar
        self.resample_type_train_signal = resample_type_train_signal
        self.oversample_type = oversample_type
        self.test_battery_cycle_start = test_battery_cycle_start if test_battery_cycle_start is not None else {}

        self.label_column = label_format.value

        self.battery_data = self.get_battery_data()

        # TODO Sascha: Rename to more meaningfull
        self.register: Dict[str, torch.Tensor] = {}

        self.conversion_look_up: Dict[Tuple[DataFormat, DataFormat], Callable[[torch.Tensor, float], torch.Tensor]] = {
            (DataFormat.CK, DataFormat.SOH): ck_to_soh,
            (DataFormat.SOH, DataFormat.CK): soh_to_ck,
        }
        self.register_df: pd.DataFrame = None

        self.test_dataset_initialized = False

        # Will be initialized later!
        self.train_batteries_data: Optional[pd.DataFrame] = None
        self.test_batteries_data: Optional[pd.DataFrame] = None
        self.train_dataset: Optional[NasaBatteryDataset] = None
        self.test_dataset: Optional[NasaBatteryDataset] = None

    def get_battery_data(self) -> pd.DataFrame:
        """Reads the data.csv of a battery dataset and does some global preprocessing.

        Returns:
            pd.DataFrame: Preprocessed battery dataset.
        """
        battery_data = pd.read_csv(self.dataset_root_path / "data.csv")
        return self._preprocess_dataset_on_global_level(battery_data)

    def prepare_train_dataset(self, battery_data: pd.DataFrame) -> NasaBatteryDataset:
        """Prepares the train-split of the dataset.

        Args:
            battery_data (pd.DataFrame): Dataframe where each row represents a cycle (e.g. discharge)

        Returns:
            NasaBatteryDataset: Training dataset.
        """
        data_oversampled = oversample(battery_data, self.oversample_type, bin_width=2)
        self.train_batteries_data = data_oversampled[data_oversampled["split"] == "train"]
        train_sequences = self._get_sequences_with_label(self.train_batteries_data, split="train")
        return NasaBatteryDataset(train_sequences)

    def prepare_val_dataset(self, battery_data: pd.DataFrame) -> NasaBatteryDataset:
        """Prepares the validation-split of the dataset.

        Same as the test dataset.

        Args:
            battery_data (pd.DataFrame): Dataframe where each row represents a cycle (e.g. discharge)

        Returns:
            NasaBatteryDataset: Validation dataset.
        """
        return self.prepare_test_dataset(battery_data)

    def prepare_test_dataset(self, battery_data: pd.DataFrame) -> NasaBatteryDataset:
        """Prepares the test-split of the dataset.

        Args:
            battery_data (pd.DataFrame): Dataframe where each row represents a cycle (e.g. discharge)

        Returns:
            NasaBatteryDataset: Testing dataset.
        """
        self.test_batteries_data = battery_data[battery_data["split"] == "test"]
        self.test_batteries_data = filter_k_first_cycles(self.test_batteries_data, self.test_battery_cycle_start)

        test_sequences = self._get_sequences_with_label(self.test_batteries_data, split="test")
        return NasaBatteryDataset(test_sequences)

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        self.train_dataset = self.prepare_train_dataset(self.battery_data)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=cpu_count())

    # HACK Sascha: for now we only dvide into train and test, so we use the test set for validation as well.
    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        # Test data is equal for every iteration. Traindata might vary from iteration to iteration
        if not self.test_dataset_initialized:
            self.test_dataset = self.prepare_test_dataset(self.battery_data)
            self.test_dataset_initialized = True
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())

    def _get_sequences_with_label(
        self,
        battery_data: pd.DataFrame,
        split: Literal["train", "test", "val"],
    ) -> List[Tuple[pd.DataFrame, float, int, int, int]]:
        """Reads npy-file for each cycle provided in the `batery_data`, resamples to provided length and associates a label.

        Args:
            battery_data (pd.DataFrame): Dataframe where each row represents a cycle (e.g. discharge)
            split (Literal["train","test","val"]): which type of dataset split is provided, so
                augmentations/preprocessing is only applied to the desired type of dataset.

        Returns:
            List[Tuple[pd.DataFrame, float, int, int]]: List[(resampled_sequence, label, cycle_id, battery_id)]
        """
        sequences_and_labels: List[Tuple[pd.DataFrame, float, int, int, int]] = []

        for _, battery_cycle in tqdm(battery_data.iterrows(), desc=f"Prepare sequence data for {split}-split"):
            cycle_timesignals = self._load_cycle_timesignals(file_name=battery_cycle.data_file)
            cycle_timesignals = self._preprocess_dataset_on_cycle_level(cycle_timesignals, split)

            sequences_and_labels.append(
                (
                    cycle_timesignals,
                    battery_cycle[self.label_column],
                    battery_cycle["cycle_id"],
                    battery_cycle["battery_id"],
                    battery_cycle["time_diff_hours"],
                )
            )
        return sequences_and_labels

    def _preprocess_dataset_on_global_level(
        self,
        battery_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Preprcesses the Nasa Baatery dataset on a global level, i.e. the metadata for each cycle.

        Args:
            battery_data (pd.DataFrame): Battery data with individual cycles as rows.

        Returns:
            pd.DataFrame:  Preprocessed Battery data.
        """
        battery_data = self._add_split_info(battery_data)
        battery_data = self._drop_cycles_with_abnormal_measurements(battery_data)
        self._rescale_label(battery_data)
        return battery_data

    def _add_split_info(
        self,
        battery_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Adds a collumn containing the split the cycle belongs to.

        Args:
            battery_data (pd.DataFrame): Battery data with individual cycles as rows.

        Returns:
            pd.DataFrame: Battery data with an extra column `"split"`.
        """
        mask_train = battery_data["battery_id"].isin(self.train_batteries)
        mask_test = battery_data["battery_id"].isin(self.test_batteries)
        battery_data = battery_data[mask_train | mask_test]
        battery_data.loc[mask_train, "split"] = "train"
        battery_data.loc[mask_test, "split"] = "test"
        return battery_data

    def _preprocess_dataset_on_cycle_level(
        self,
        timesignals: pd.DataFrame,
        split: Literal["train", "test", "val"],
    ) -> pd.DataFrame:
        """Preprocesses the timeseries data of a single battery cycle.

        Args:
            timesignals (pd.DataFrame): Timesignals of a single battery cycle.
            split (Literal["train","test","val"]): which type of dataset split is provided, so
                augmentations/preprocessing is only applied to the desired type of dataset.

        Returns:
            pd.DataFrame: Preprocessed timesignals.
        """
        resample_type = self.resample_type_train_signal if split == "train" else TimeSignalResamplingType.LINEAR

        timesignals = self._resample_timesignals(timesignals, resample_type=resample_type)
        timesignals = self._quantize_sample_time(timesignals)
        timesignals = self._rescale_timesignals(timesignals)
        return timesignals

    def _load_cycle_timesignals(self, file_name: str) -> pd.DataFrame:
        """Load the per-cycle timesignals from npy-files and returns columns given in `self.columns_target_timesignals`.

        Args:
            file_name (str): Filename of the data npy-file for a the desired discharge cycle.

        Returns:
            pd.DataFrame: Timesignals containing columns `self.columns_target_timesignals`
        """
        timesignals_np = np.load(self.dataset_root_path / file_name)
        timesignals = pd.DataFrame(timesignals_np, columns=self.columns_all_timesignals)
        return timesignals[self.columns_target_timesignals]

    # TODO Sascha: might be better in pre-processing file
    def _resample_timesignals(
        self,
        cycle_data: pd.DataFrame,
        resample_type: TimeSignalResamplingType,
    ) -> pd.DataFrame:
        """Consumes the measurement data of a single cycle and resamples and interpolates to `num_samples_target`.

        Args:
            cycle_data (pd.DataFrame): Measurement data of a single cycle
            resample_type (TimeSignalResamplingType): Type of resampling to apply.

        Returns:
            pd.DataFrame: Resampled data.
        """
        # Create interpolation functions
        interp_funcs = {
            col: interpolate.interp1d(cycle_data["sample_time"], cycle_data[col])
            for col in cycle_data.columns
            if col != "sample_time"
        }

        # Generate new "sample_time" array
        sample_time_min = cycle_data["sample_time"].min()
        sample_time_max = cycle_data["sample_time"].max()

        if resample_type == TimeSignalResamplingType.LINEAR:
            # Linear spacing
            new_time = np.linspace(sample_time_min, sample_time_max, self.num_samples_target)

        elif resample_type == TimeSignalResamplingType.RANDOM:
            # Random uniform samples but min and max are ensured to be contained
            random_samples = np.random.uniform(
                low=sample_time_min, high=sample_time_max, size=[self.num_samples_target - 2]
            )
            new_time = np.concatenate([[sample_time_min], random_samples, [sample_time_max]])

        elif resample_type == TimeSignalResamplingType.ANCHORS:
            # Linearly spaced anchors to ensure it is distributed and then add noise
            intervall = sample_time_max / self.num_samples_target
            noise = np.random.uniform(low=-(intervall / 2), high=(intervall / 2), size=[self.num_samples_target])
            anchors = np.linspace(sample_time_min, sample_time_max, self.num_samples_target)
            new_time = np.clip(anchors + noise, a_min=sample_time_min, a_max=sample_time_max)

        elif resample_type == TimeSignalResamplingType.ANCHORS_NEW:
            num_samples_center = self.num_samples_target - 2
            intervall = sample_time_max / num_samples_center
            noise = np.random.uniform(low=-(intervall / 2), high=(intervall / 2), size=[num_samples_center])
            anchors = np.linspace(sample_time_min + intervall, sample_time_max - intervall, num_samples_center)
            new_time = np.concatenate([[sample_time_min], anchors + noise, [sample_time_max]])
        else:
            raise NotImplementedError()

        # Generate new samples
        df_resampled = pd.DataFrame({col: func(new_time) for col, func in interp_funcs.items()})
        df_resampled["sample_time"] = new_time
        return df_resampled

    def _rescale_timesignals(
        self,
        sequence: pd.DataFrame,
    ) -> pd.DataFrame:
        """Rescales the data based on the range of the target signals provided in `range_target_timesignals`.

        Args:
            sequence (pd.DataFrame): Data to be rescaled.

        Returns:
            pd.DataFrame: Rescaled data.
        """
        for feature_coulumn in sequence.columns:
            if feature_coulumn not in self.range_target_timesignals:
                continue
            min_scale = self.range_target_timesignals[feature_coulumn]["min"]
            max_scale = self.range_target_timesignals[feature_coulumn]["max"]
            sequence[feature_coulumn] = (sequence[feature_coulumn] - min_scale) / (max_scale - min_scale)

        return sequence

    # TODO Sascha: what is abnormal? Only if capacity==0.0 or would 0.05 also be abnormal?
    def _drop_cycles_with_abnormal_measurements(self, battery_data: pd.DataFrame) -> pd.DataFrame:
        """Checks all battery cycles and drops those if the capacity of that cycle is 0.0.

        The Nasa Battery dataset has some abnormal measurements cycles where the capacity is very low in certain cycles.

        Args:
            battery_data (pd.DataFrame): Battery data with individual cycles as rows.

        Returns:
            pd.DataFrame: Battery data with abnormal cycles dropped.
        """
        return battery_data.drop(battery_data[battery_data["capacity_k"] == 0.0].index)

    def _quantize_sample_time(self, sequence: pd.DataFrame) -> pd.DataFrame:
        """Quantizes the timestamp into a full hour integer representation.

        Args:
            sequence (pd.DataFrame): sequence data of a single cycle.

        Returns:
            pd.DataFrame: sequence data of a single cycle with quntized `time` column.
        """
        sequence["sample_time"] = sequence["sample_time"].astype(int)
        return sequence

    def combine_and_register_batched_test_outputs(self, batched_outputs: List[Dict[str, torch.Tensor]]) -> None:
        """Combines all outputs from the batched test_step() into a single data structure.

        Args:
            batched_outputs (List[Dict[str, torch.Tensor]]): List containing the output batches of the test_step() method.

        """
        batched_tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for batch in batched_outputs:
            for key, data_tensor in batch.items():
                batched_tensors[key].append(data_tensor)

        for key, tensor_list in batched_tensors.items():
            self.register[key] = torch.concat(tensor_list, axis=0)  # axis=0 is batch axis

        self._convert_register_to_dataframe()

    def clear_register(self) -> None:
        self.register.clear()
        self.register_df = self.register_df.drop(self.register_df.index)

    def get_dummy_data_batch(self, to_device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
        """Generates dummy batched data to infere a model.

        Args:
            to_device (Optional[Union[str, torch.device]]): device where to place the tensors.

        Returns:
            Dict[str,torch.Tensor]: Dummy data.
        """
        return {
            "sequence": torch.rand((self.batch_size, self.num_samples_target, len(self.columns_target_timesignals))).to(
                device=to_device
            ),
            "label": torch.rand((self.batch_size)).to(device=to_device),
            "cycle_id": torch.rand((self.batch_size)).to(device=to_device),
            "battery_id": torch.rand((self.batch_size)).to(device=to_device),
            "time_diff_hours": torch.rand((self.batch_size)).to(device=to_device),
        }

    def _convert_register_to_dataframe(self) -> None:
        data_types = {"label": "float32", "cycle_id": "int32", "battery_id": "int32", "output": "float32"}
        data_numpy = torch.vstack(tuple(self.register.values())).cpu().numpy().T
        self.register_df = pd.DataFrame(data_numpy, columns=self.register.keys()).astype(data_types)

        # Joins the test dataset with the register containing predictions of cycles not dropped during pre-processing.
        self.register_df = pd.merge(
            self.register_df,
            self.test_batteries_data,
            how="left",
            left_on=["battery_id", "cycle_id"],
            right_on=["battery_id", "cycle_id"],
        )

        self.register_df["output_soh"] = self._convert_to_target_format("output", DataFormat.SOH)
        self.register_df["label_soh"] = self._convert_to_target_format("label", DataFormat.SOH)
        self.register_df["output_ck"] = self._convert_to_target_format("output", DataFormat.CK)
        self.register_df["label_ck"] = self._convert_to_target_format("label", DataFormat.CK)

        self.register_df[["output_soh", "label_soh", "output_ck", "label_ck"]] = self.register_df[
            ["output_soh", "label_soh", "output_ck", "label_ck"]
        ].apply(self._unscale)

    def _convert_to_target_format(
        self,
        key: Literal["output", "label"],
        target_format: DataFormat,
    ) -> pd.Series:
        if target_format == self.label_format:
            return self.register_df[key]

        conversion = (self.label_format, target_format)

        if conversion not in self.conversion_look_up:
            raise KeyError(f"Conversion {conversion} not available. Only got: {self.conversion_look_up.keys()}")

        conversion_func = self.conversion_look_up[(self.label_format, target_format)]

        return conversion_func(self.register_df[key], self.register_df["capacity_0"])

    @property
    def metrics_function_lookup(self) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        return {
            "MAE": sklearn.metrics.mean_absolute_error,
            "RMSE": root_mean_squared_error,
            "MAPE": mape,
        }

    def _get_metrics(self, outputs: pd.Series, labels: pd.Series, specifier: str) -> Dict[str, float]:
        return {
            f"{metric_name}/{specifier}": metric_func(outputs, labels)
            for metric_name, metric_func in self.metrics_function_lookup.items()
        }

    def _calc_global_metrics_for_intervals(self) -> Dict[str, float]:
        intervalls = [[110, 90], [90, 80], [80, 70], [70, 60], [60, 50], [50, 0]]

        metrics: Dict[str, float] = {}

        for upper, lower in intervalls:
            mask = (self.register_df["label_soh"] < upper) & (self.register_df["label_soh"] >= lower)

            if mask.any():
                metrics.update(
                    self._get_metrics(
                        outputs=self.register_df.loc[mask, "output_soh"],
                        labels=self.register_df.loc[mask, "label_soh"],
                        specifier=f"ALL_SOH_{upper}_{lower}",
                    )
                )
        return metrics

    def calc_metrics(self) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {}
        metrics.update(self._get_metrics(self.register_df["output_soh"], self.register_df["label_soh"], "ALL_SOH"))
        # metrics.update(self._calc_global_metrics_for_intervals())
        for battery_id, battery_data in self.register_df.groupby("battery_id"):
            metrics.update(
                self._get_metrics(battery_data["output_soh"], battery_data["label_soh"], f"B{battery_id}_SOH")
            )
            metrics.update({f"NUM_CYCLES/B{battery_id}_SOH": len(battery_data)})
        return metrics

    def prepare_prediction_plots(self) -> Dict[str, matplotlib.figure.Figure]:
        plots: Dict[str, matplotlib.figure.Figure] = {}
        plots.update(self._prepare_ck_soh_line_plot(DataFormat.SOH))
        plots.update(self._prepare_prediction_over_true_scatter_plot(DataFormat.SOH))
        return plots

    def _prepare_prediction_over_true_scatter_plot(
        self, data_format: DataFormat
    ) -> Dict[str, matplotlib.figure.Figure]:
        plots: Dict[str, matplotlib.figure.Figure] = {}

        output_column = {
            DataFormat.CK: "output_ck",
            DataFormat.SOH: "output_soh",
        }
        label_column = {
            DataFormat.CK: "label_ck",
            DataFormat.SOH: "label_soh",
        }
        figure = get_figure_predicted_over_true(
            battery_data=self.register_df,
            column_label=label_column[data_format],
            column_prediction=output_column[data_format],
        )
        plots.update({f"prediction_over_true_{data_format.value}": figure})
        return plots

    def _prepare_ck_soh_line_plot(self, data_format: DataFormat) -> Dict[str, matplotlib.figure.Figure]:
        """Generates a plot per battery to compare predictions against labels and plots those to the experiment logger.

        Args:
            data_format (DataFormat): Format of the output and label data to be used.

        Returns:
            Dict[str, matplotlib.figure.Figure]: key: name of the plot, value: Figure object.
        """
        plots: Dict[str, matplotlib.figure.Figure] = {}
        plot_meta_data = {
            DataFormat.CK: {"y_lim_pred": (0, 2.3), "y_lim_error": (-0.5, 0.5), "y_label": "C_k [Ahr]"},
            DataFormat.SOH: {"y_lim_pred": (40, 110), "y_lim_error": (-15, 15), "y_label": "SOH [%]"},
        }
        output_column = {
            DataFormat.CK: "output_ck",
            DataFormat.SOH: "output_soh",
        }
        label_column = {
            DataFormat.CK: "label_ck",
            DataFormat.SOH: "label_soh",
        }

        for battery_id, battery_data in self.register_df.groupby("battery_id"):
            outputs = battery_data[output_column[data_format]]
            labels = battery_data[label_column[data_format]]
            capacity_0 = battery_data["capacity_0"]
            fade = battery_data["fade"]
            cycle_id = battery_data["cycle_id"]
            error = outputs - labels

            # TODO sascha: make an independent metric out of this.
            eol_threshold = calc_eol_threshold(fade, capacity_0 if data_format == DataFormat.CK else None)
            eol_indicator_outputs = calc_eol_indicator(eol_threshold, outputs, cycle_id)
            eol_indicator_labels = calc_eol_indicator(eol_threshold, labels, cycle_id)

            figure = get_figure_battery_prediction(
                battery_id,
                cycle_id,
                outputs,
                labels,
                error,
                eol_threshold=eol_threshold,
                eol_indicator_outputs=eol_indicator_outputs,
                eol_indicator_labels=eol_indicator_labels,
                **plot_meta_data[data_format],
            )
            plots.update({f"prediction_{data_format.value}/battery_{battery_id}": figure})
        return plots

    def _rescale_label(self, battery_data: pd.DataFrame) -> pd.DataFrame:
        battery_data[self.label_column] = self.label_scalar * battery_data[self.label_column]
        return battery_data

    def _unscale(self, scaled_series: pd.Series) -> pd.Series:
        """Undoes the data scaling that is performed for training the model so it is in its original range.

        Since the label is scaled with the `label_scalar` member variable, the output of the model also needs to be
        unscaled since it is trained to match the scaled label.

        Args:
            scaled_series (pd.Series): Tensor that has been scaled for training. (Usually label or model output)

        Returns:
            pd.Series: Unscaled series
        """
        return scaled_series / self.label_scalar


class NasaBatteryDataModuleFactory:
    """Static factory class to instantiate a certain NasaBatteryDataModule."""

    @staticmethod
    def get_nasa_battery_data_module(dataset_config: DictConfig) -> NasaBatteryDataModule:
        """Instantiates and returns a certain instance of a NasaBatteryDataModule subclass.

        Args:
            dataset_config (DictConfig): Configuration for the dataset from a hydra config file.

        Returns:
            NasaBatteryDataModule: Instance of the requested NasaBatteryDataModule class with the provided model config.
        """
        return NasaBatteryDataModule(
            dataset_root_path=Path(dataset_config.cycle_type.dataset_root_path),
            train_batteries=dataset_config.train_batteries,
            test_batteries=dataset_config.test_batteries,
            num_samples_target=dataset_config.num_samples_target,
            columns_all_timesignals=dataset_config.cycle_type.columns_all_timesignals,
            columns_target_timesignals=dataset_config.cycle_type.columns_target_timesignals,
            range_target_timesignals=dataset_config.range_target_timesignals,
            batch_size=dataset_config.batch_size,
            resample_type_train_signal=TimeSignalResamplingType(dataset_config.resample_type_train),
            oversample_type=OversampleType(dataset_config.oversample),
            label_format=DataFormat.CK,
            label_scalar=0.5,
            test_battery_cycle_start=dataset_config.test_battery_cycle_start,
        )
