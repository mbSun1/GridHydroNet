"""Combine a selectable grid parameter encoder with daily HBV92 equations."""

from torch import nn

from .grid_bigru_parameterizer import GridBiGRUParameterizer
from .grid_hbv_daily import GridHBV92Daily
from .grid_patch_transformer_parameterizer import GridPatchTransformerParameterizer
from .grid_tcn_parameterizer import GridTCNParameterizer


class ParameterizedGridHBV92(nn.Module):
    """Generate static grid parameters with the selected temporal encoder."""

    def __init__(
        self,
        terrain_channels: int,
        parameter_hidden: int,
        parameter_layers: int,
        parameter_cell_batch_size: int = 256,
        parameter_encoder: str = "bigru",
    ):
        super().__init__()
        if parameter_encoder == "bigru":
            self.parameterization = GridBiGRUParameterizer(
                terrain_channels=terrain_channels,
                hidden=parameter_hidden,
                layers=parameter_layers,
                cell_batch_size=parameter_cell_batch_size,
            )
        elif parameter_encoder == "tcn":
            self.parameterization = GridTCNParameterizer(
                terrain_channels=terrain_channels,
                cell_batch_size=parameter_cell_batch_size,
                parameter_head_width=parameter_hidden,
            )
        elif parameter_encoder == "patch_transformer":
            self.parameterization = GridPatchTransformerParameterizer(
                terrain_channels=terrain_channels,
                cell_batch_size=parameter_cell_batch_size,
                parameter_head_width=parameter_hidden,
            )
        else:
            raise ValueError(f"Unsupported parameter_encoder={parameter_encoder!r}")
        self.parameter_encoder = parameter_encoder
        self.hbv = GridHBV92Daily()

    def generate_parameters(self, terrain, climate_sequence, basin_mask):
        """Generate spatial parameters from the fixed training context."""
        return self.parameterization(terrain, climate_sequence, basin_mask)

    def initial_state(self, context):
        """Create HBV92 states matching the parameter-grid shape."""
        return self.hbv.initial_state(context)

    def forward_step(
        self,
        precipitation,
        temperature,
        pet,
        state,
        context,
    ):
        """Advance the HBV92 equations by one day."""
        return self.hbv.forward_step(
            precipitation,
            temperature,
            pet,
            state,
            context,
        )
