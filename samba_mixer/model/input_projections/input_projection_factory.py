from omegaconf import DictConfig

from samba_mixer.model.input_projections.linear_projection import LinearProjection
from samba_mixer.model.input_projections.linear_projection_time_embedding import LinearProjectionWithLocalTimeEmbedding
from samba_mixer.model.input_projections.linear_projection_time_embedding_cycle_diff_embedding import (
    LinearProjectionWithLocalTimeAndGlobalDiffEmbedding,
)
from samba_mixer.model.input_projections.samba_input_projection import SambaInputProjection


class SambaInputProjectionFactory:
    """Static factory class to instantiate a certain SambaInputProjection."""

    @staticmethod
    def get_input_projection(model_config: DictConfig) -> SambaInputProjection:
        """Instantiates and returns a certain instance of a SambaInputProjection subclass.

        Args:
            model_config (DictConfig): Configuration for the model from a hydra config file.

        Raises:
            ValueError: if a requested projection_name is not a valid choice.

        Returns:
            SambaInputProjection: Instance of the requested SambaInputProjection class with the provided model config.
        """
        projection_name = model_config.input_projection.name
        d_model = SambaInputProjectionFactory.get_d_model(model_config)
        if projection_name == "linear_projection":
            return LinearProjection(d_model)
        if projection_name == "local_time_embedding":
            return LinearProjectionWithLocalTimeEmbedding(d_model)
        if projection_name == "local_time_global_diff_embedding":
            return LinearProjectionWithLocalTimeAndGlobalDiffEmbedding(d_model)
        raise ValueError(f"Provided projection_name '{projection_name}' is not a valid option.")

    # HACK Sascha: super ugly implementation...
    @staticmethod
    def get_d_model(model_config: DictConfig) -> int:
        if model_config.backbone.name == "hierachical_samba_mixer":
            return model_config.d_model // (2 ** (len(model_config.backbone.stage_layers) - 1))

        return model_config.d_model
