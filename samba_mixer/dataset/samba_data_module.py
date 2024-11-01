from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import matplotlib.figure
import pytorch_lightning as pl
import torch


class SambaDataModule(pl.LightningDataModule, ABC):
    """Abstract base class for all Samba data modules.

    Reflects shared functionality, defines the interface and is used for static type hints.
    """

    """
    # TODO Sascha: not so clean that the register is expected but not part of the abstract class...
    A solution might be to add an register property to the ABC that is called
    """

    @abstractmethod
    def combine_and_register_batched_test_outputs(self, batched_outputs: List[Dict[str, torch.Tensor]]) -> None:
        """Takes results from all testing_step()/validation_step() calls and combines them in a single data structure.

        Results are saved into member variable self.register.

        Args:
            batched_outputs (List[Dict[str, torch.Tensor]]): A list of testing_steps results where each element of the
                list corresponds to a batch
        """

    @abstractmethod
    def clear_register(self) -> None:
        """Clears the content from the register used to accumulate data at xxx_step_end"""

    @property
    @abstractmethod
    def metrics_function_lookup(self) -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """Property returning a dict of metric functions.

        Returns:
            Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]: key: metric name, value metric function.
        """

    # TODO Sascha: return type shall be either a torch tensor or a float.
    @abstractmethod
    def calc_metrics(self) -> Dict[str, torch.Tensor]:
        """Calculate metrics for testing data and return results in a dict.

        Can be used to log multiple metrics at once to e.g. tensorboard.

        Returns:
            Dict[str, torch.Tensor]: key: name of the loss, value: loss.
        """

    @abstractmethod
    def prepare_prediction_plots(self) -> Dict[str, matplotlib.figure.Figure]:
        """Generates plots and returns their figures in a dict.

        Can be used to log multiple figures at once to e.g. tensorboard.

        Returns:
            Dict[str, matplotlib.figure.Figure]: key: name of the figure, value: figure object.
        """

    @abstractmethod
    def get_dummy_data_batch(self, to_device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
        """Generates a single batch of random data following the structure of the dataset.

        Might be used as a testing input to a model to ensure shapes are as expected.

        Args:
            to_device (Optional[Union[str, torch.device]], optional): Device where to place tensor. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: key: name of the data item, value: data in form of a tensor.
        """
