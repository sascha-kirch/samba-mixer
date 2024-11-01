import logging
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from samba_mixer.dataset.nasa_battery_dataset import NasaBatteryDataModuleFactory
from samba_mixer.model.samba import SambaPredictor
from samba_mixer.utils.omega_conf_resolver import register_new_resolver


warnings.filterwarnings("ignore")

register_new_resolver()

pl.seed_everything(42)
# configure logging at the root level of Lightning
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    log.debug(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Output directory: {output_dir}")

    data_module = NasaBatteryDataModuleFactory.get_nasa_battery_data_module(dataset_config=cfg.dataset)
    model = SambaPredictor.load_from_checkpoint(checkpoint_path=cfg.checkpoint, data_module=data_module)

    logger = TensorBoardLogger(save_dir=output_dir / "tensorboard", name=None, version=None)
    if cfg.log_graph:
        logger.experiment.add_graph(
            model.model.to(device="cuda"),
            input_to_model=data_module.get_dummy_data_batch(to_device="cuda"),
        )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            ModelSummary(max_depth=2),
        ],
        accelerator="gpu",
        devices=cfg.gpu_devices,
    )

    metrics = trainer.test(model, dataloaders=data_module.test_dataloader())

    logger.log_hyperparams(cfg, metrics=metrics[0])


if __name__ == "__main__":
    main()
