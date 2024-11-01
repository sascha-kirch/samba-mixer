from omegaconf import DictConfig

from samba_mixer.model.heads.linear_head import LinearHead
from samba_mixer.model.heads.samba_head import SambaHead


class SambaHeadFactory:
    """Static factory class to instantiate a certain SambaHead."""

    @staticmethod
    def get_head(model_config: DictConfig) -> SambaHead:
        """Instantiates and returns a certain instance of a SambaHead subclass.

        Args:
            model_config (DictConfig): Configuration for the model from a hydra config file.

        Returns:
            SambaHead: An instance of the requested SambaHead subclass with the provided model config.
        """
        return LinearHead(
            d_model=model_config.d_model,
            n_layer=model_config.head.n_layer,
            final_sigmoid=True,
        )
