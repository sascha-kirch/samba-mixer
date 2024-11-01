from omegaconf import DictConfig

from samba_mixer.model.backbones.bidirectional_mamba import BidirectionalMambaBackbone
from samba_mixer.model.backbones.hierachical_samba_mixer import HierachicalSambaMixerBackbone
from samba_mixer.model.backbones.samba_backbone import SambaBackbone
from samba_mixer.model.backbones.samba_mixer import SambaMixerBackbone
from samba_mixer.model.backbones.vanilla_mamba import VanillaMambaBackbone


class SambaBackboneFactory:
    """Static factory class to instantiate a certain SambaBackbone."""

    @staticmethod
    def get_backbone(model_config: DictConfig) -> SambaBackbone:
        """Instantiates and returns a certain instance of a SambaBackbone subclass.

        Args:
            model_config (DictConfig): Configuration for the model from a hydra config file.

        Raises:
            ValueError: if a requested head_name is not a valid choice.

        Returns:
            SambaBackbone: An instance of the requested SambaBackbone subclass with the provided model config.
        """
        backbone_name = model_config.backbone.name
        if backbone_name == "vanilla_mamba":
            return VanillaMambaBackbone(
                d_model=model_config.d_model,
                n_layer=model_config.backbone.n_layer,
                d_state=model_config.backbone.d_state,
                d_conv=model_config.backbone.d_conv,
                expand=model_config.backbone.expand,
                rms_norm=model_config.backbone.rms_norm,
                drop_path=model_config.backbone.drop_path,
            )
        if backbone_name == "bidirectional_mamba":
            return BidirectionalMambaBackbone(
                d_model=model_config.d_model,
                n_layer=model_config.backbone.n_layer,
                d_state=model_config.backbone.d_state,
                d_conv=model_config.backbone.d_conv,
                expand=model_config.backbone.expand,
                rms_norm=model_config.backbone.rms_norm,
                drop_path=model_config.backbone.drop_path,
            )
        if backbone_name == "samba_mixer":
            return SambaMixerBackbone(
                d_model=model_config.d_model,
                num_tokens=model_config.backbone.num_tokens,
                n_layer=model_config.backbone.n_layer,
                d_state=model_config.backbone.d_state,
                d_conv=model_config.backbone.d_conv,
                expand=model_config.backbone.expand,
                rms_norm=model_config.backbone.rms_norm,
                drop_path=model_config.backbone.drop_path,
            )
        if backbone_name == "hierachical_samba_mixer":
            return HierachicalSambaMixerBackbone(
                d_model=model_config.d_model,
                num_tokens=model_config.backbone.num_tokens,
                stage_layers=model_config.backbone.stage_layers,
                d_state=model_config.backbone.d_state,
                d_conv=model_config.backbone.d_conv,
                expand=model_config.backbone.expand,
                rms_norm=model_config.backbone.rms_norm,
                drop_path=model_config.backbone.drop_path,
            )

        raise ValueError(f"Provided backbone_name '{backbone_name}' is not a valid option.")
