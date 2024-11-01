from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from timm.models.layers import trunc_normal_
from torch import nn
from torch import optim

from samba_mixer.dataset.samba_data_module import SambaDataModule
from samba_mixer.model.backbones.backbone_factory import SambaBackboneFactory
from samba_mixer.model.backbones.samba_backbone import SambaBackbone
from samba_mixer.model.heads.head_factory import SambaHeadFactory
from samba_mixer.model.heads.samba_head import SambaHead
from samba_mixer.model.input_projections.input_projection_factory import SambaInputProjectionFactory
from samba_mixer.model.input_projections.samba_input_projection import SambaInputProjection
from samba_mixer.utils.enums import ClsTokenType


class SambaModel(nn.Module):
    """SambaModel combines an input projection, a backbone and a head into a single model and inserts a cls token."""

    def __init__(
        self,
        input_projection: SambaInputProjection,
        backbone: SambaBackbone,
        head: SambaHead,
        cls_token_type: ClsTokenType,
    ) -> None:
        """Initialize SambaModel.

        Args:
            input_projection (SambaInputProjection): Input projection module
            backbone (SambaBackbone): backbone module
            head (SambaHead): head module
            cls_token_type (ClsTokenType): type for the CLS token [NONE | HEAD | TAIL | MIDDLE]
        """
        super().__init__()

        self.input_projection = input_projection
        self.backbone = backbone
        self.head = head
        self.cls_token_type = cls_token_type
        self.pos_token_for_pred: Optional[int] = None
        self.cls_token: Optional[nn.Parameter] = None

        if self.cls_token_type != ClsTokenType.NONE:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.input_projection.d_model))
            trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Infers the Samba model.

        Args:
        x (Dict[str,torch.Tensor]): Dict witch values beeing batched input tensors. Sequence data is accessed with key
            - `x["sequence"]` (torch.Tensor): Sequence data of shape (batch_size, seq_len, feature_dim). feature_dim
            should be 4 (voltage, current, temperature, timestamp). Last feature needs to be the timestamp!
            - Optionally (depending on the dataset), there might be other global metadata for a each sequence, like
                - `x["cycle_id"]` (torch.Tensor): Integer id of the current cycle.
                - `x["battery_id"]` (torch.Tensor): Integer id of the battery the current cycle belongs to.
                - `x["time_diff_hours"]` (torch.Tensor): Time diff to previous cycle as integer in hours.

        Returns:
            torch.Tensor: Output of the samba model
        """
        projection = self.input_projection(x)
        input_tokens = self._insert_cls_token(projection)
        backbone_output = self.backbone(input_tokens)
        return self.head(backbone_output, self.pos_token_for_pred)

    def _insert_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Inserts a a learnable CLS token to the already projected output of the input projection.

        Note that we don't add a positional embedding, since in our case we do not use an index based positional
        embedding but a timebased one. Hence, we would need to interpolate a time which is 1. not very meaning full and
        two not possible if the token is inserted at the end. We hypothesize that it is learned anyway.

        If a token is inserted, it can either be inserted
        1. at the head (before the first token)
        2. at the tail (after the last token)
        3. in the middle

        Args:
            x (torch.Tensor): Projected tensor of shape [batch, num_tokens, d_model]

        Raises:
            NotImplementedError: if requested token type is not a valid optinon.

        Returns:
            torch.Tensor: Tensor with potentially added token of shape [batch, num_tokens + 1, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # TODO Sascha: add another type of CLS token, that does not reuturn a token but takes the last output.
        if self.cls_token_type == ClsTokenType.NONE:
            self.pos_token_for_pred = None
            return x

        cls_token = self.cls_token.expand(batch_size, -1, -1)

        if self.cls_token_type == ClsTokenType.HEAD:
            self.pos_token_for_pred = 0
            return torch.cat((cls_token, x), dim=1)

        if self.cls_token_type == ClsTokenType.MIDDLE:
            self.pos_token_for_pred = seq_len // 2
            return torch.cat((x[:, : self.pos_token_for_pred, :], cls_token, x[:, self.pos_token_for_pred :, :]), dim=1)

        if self.cls_token_type == ClsTokenType.TAIL:
            self.pos_token_for_pred = -1
            return torch.cat((x, cls_token), dim=1)

        raise NotImplementedError(f"Requested token_type: {self.cls_token_type} not supported.")

    @property
    def criterion(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Property to retrieve criterion defined in the head of the model.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: Criterion for loss calculation.
        """
        return self.head.criterion


class SambaPredictor(pl.LightningModule):
    """LightningModule of the SambaModel used for training, validation, testing and inference."""

    def __init__(
        self,
        model_config: DictConfig,
        trainer_config: DictConfig,
        data_module: SambaDataModule,
    ) -> None:
        super().__init__()
        self.trainer_config = trainer_config
        self.model = SambaModel(
            input_projection=SambaInputProjectionFactory.get_input_projection(model_config),
            backbone=SambaBackboneFactory.get_backbone(model_config),
            head=SambaHeadFactory.get_head(model_config),
            cls_token_type=ClsTokenType(model_config.cls_token_type),
        )
        self.criterion = self.model.criterion
        self.data_module = data_module

        # NOTE Sascha: save_hyperparameters crashes if a non serializable object is passed (e.g. data_module)
        self.save_hyperparameters("model_config", "trainer_config")  # , "data_module")

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Calls the SmbaPredictor.

        Args:
        x (Dict[str,torch.Tensor]): Dict witch values beeing batched input tensors. Sequence data is accessed with key
            - `x["sequence"]` (torch.Tensor): Sequence data of shape (batch_size, seq_len, feature_dim). feature_dim
            should be 4 (voltage, current, temperature, timestamp). Last feature needs to be the timestamp!
            - Optionally (depending on the dataset), there might be other global metadata for a each sequence, like
                - `x["cycle_id"]` (torch.Tensor): Integer id of the current cycle.
                - `x["battery_id"]` (torch.Tensor): Integer id of the battery the current cycle belongs to.
                - `x["time_diff_hours"]` (torch.Tensor): Time diff to previous cycle as integer in hours.

        labels (Optional[torch.Tensor], optional): Ground truth labels. Shape depends on the task. Defaults to None.

        Returns:
            Tuple[Optional[torch.Tensor], torch.Tensor]:
                loss (Optional[torch.Tensor]): Loss between predictions and GT labels.
                output (torch.Tensor): Output of the model.
        """
        output = torch.squeeze(self.model(x))
        loss = None
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Optional[torch.Tensor]]:  # noqa: D102
        labels = batch.pop("label")
        loss, _ = self(batch, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Optional[torch.Tensor]]:  # noqa: D102
        labels = batch.pop("label")
        loss, output = self(batch, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        return {
            "label": labels,
            "cycle_id": batch["cycle_id"],
            "battery_id": batch["battery_id"],
            "output": output,
        }

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:  # noqa: D102
        labels = batch.pop("label")
        _, output = self(batch)
        return {
            "label": labels,
            "cycle_id": batch["cycle_id"],
            "battery_id": batch["battery_id"],
            "output": output,
        }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:  # noqa: D102
        self.data_module.combine_and_register_batched_test_outputs(outputs)
        self._calc_and_log_test_metrics()
        self._plot_predictions()
        self.data_module.clear_register()

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:  # noqa: D102
        self.data_module.combine_and_register_batched_test_outputs(outputs)
        self._calc_and_log_test_metrics()
        self._plot_predictions()
        self.data_module.clear_register()

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[Dict[str, Any]]]:  # noqa: D102
        # using getattr to be able to use the config file to instantiate optimizer and scheduler objects using strings.
        optimizer = getattr(torch.optim, self.trainer_config["optimizer"]["name"])(
            params=self.parameters(),
            **self.trainer_config["optimizer"]["parameter"],
        )

        lr_scheduler = {
            "scheduler": getattr(torch.optim.lr_scheduler, self.trainer_config["scheduler"]["name"])(
                optimizer=optimizer,
                **self.trainer_config["scheduler"]["parameter"],
            ),
            "name": "learning_rate",
        }
        return [optimizer], [lr_scheduler]

    def _calc_and_log_test_metrics(self) -> None:
        """Calculates metrics from the prediction and the labels and logs them to the experiment logger."""
        metrics = self.data_module.calc_metrics()
        self.log_dict(metrics, logger=True, prog_bar=False)

    def _plot_predictions(self) -> None:
        """Plots the predictions as defined in the task-specific data module."""
        plots = self.data_module.prepare_prediction_plots()
        for key, figure in plots.items():
            self.logger.experiment.add_figure(key, figure, global_step=self.current_epoch)
