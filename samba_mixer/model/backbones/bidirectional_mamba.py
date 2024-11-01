from samba_mixer.model.backbones.vanilla_mamba import VanillaMambaBackbone


class BidirectionalMambaBackbone(VanillaMambaBackbone):
    """VanillaMambaBackbone with bidirectional execution of MambaBlocks and using BiSamba blocks instead of Mamba."""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_state: int,
        d_conv: int,
        expand: int,
        rms_norm: bool,
        drop_path: float = 0.1,
    ) -> None:
        """Initialize BidirectionalMambaBackbone.

        Args:
            d_model (int): Hidden dimensions of the model.
            n_layer (int): Number of consecutivly stacked SambaBlocks.
            d_state (int): Dimensions of the state of the model.
            d_conv (int): Kernel size of the convolution.
            expand (int): Expansion factor.
            rms_norm (bool): Wether to use RMSNorm or LayerNorm.
            drop_path (float, optional): Probability for the DropPath layer. Defaults to 0.1.
        """
        super().__init__(
            d_model, n_layer, d_state, d_conv, expand, rms_norm, drop_path, bimamba=True, bidirectional_execution=True
        )
