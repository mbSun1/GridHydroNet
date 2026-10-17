"""Bind one prepared basin manifest to the current daily-model interfaces.

Data preparation is the only supported input path. The training and testing
workflow consumes exactly the manifest written by ``disthybridcore.prepare_data``
and never guesses filenames, temporal resolution, CRS, or basin layout.
"""

import json
from pathlib import Path

import xarray as xr


def bind_prepared_basin_catalog(params, manifest: dict) -> None:
    """Bind a validated daily preparation manifest to model parameters."""
    required = {
        "dem",
        "topology",
        "static_attributes",
        "discharge",
        "forcing",
        "grid_gauges",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(
            "preparation_manifest.json lacks required fields: "
            + ", ".join(missing)
        )
    forcing = manifest["forcing"]
    if not isinstance(forcing, dict) or not {"prec", "t", "pet"}.issubset(forcing):
        raise ValueError(
            "preparation_manifest.json forcing must contain prec, t, and pet"
        )

    from disthybridcore.prepare_data.basin_validation import validate_simulation_period

    params.path_dem = manifest["dem"]
    # The prepared DEM stores the basin coverage fraction used as the model
    # grid's authoritative catchment mask; no second mask source is accepted.
    params.path_catchments = manifest["dem"]
    params.catchment_variable = "basin_fraction"
    params.path_bafg = manifest["discharge"]
    params.path_discharge = manifest["discharge"]
    params.path_precipitation = forcing["prec"]
    params.path_temperature = forcing["t"]
    params.path_pet = forcing["pet"]
    params.path_prepared_topology = manifest["topology"]
    params.path_terrain_attributes = manifest["static_attributes"]
    params.path_grid_gauges = manifest["grid_gauges"]
    params.path_river_lines = manifest.get("river_lines")
    params.path_basin_boundary = manifest.get("boundary")
    params.path_ssrd = None
    params.dem_variable = "dem"
    params.discharge_variable = "discharge"
    params.precipitation_variable = "prec"
    params.temperature_variable = "t"
    params.pet_variable = "pet"
    params.normalization_dir = str(Path(params.path_pet).parent / "normalization")
    params.topology_dir = str(Path(params.path_prepared_topology).parent)
    params.prepared_daily = True

    with xr.open_dataset(params.path_dem, engine="h5netcdf") as data:
        params.grid_height = int(data.sizes["y"])
        params.grid_width = int(data.sizes["x"])
        params.grid_size = params.grid_width
        params.native_cell_size_km = float(data.attrs["grid_resolution_m"]) / 1000.0
        params.grid_crs_wkt = data.attrs["crs_wkt"]

    if params.mode != "prepare":
        validate_simulation_period(manifest, params)


def prepare_data_catalog(project_root: Path, params) -> None:
    """Load the current basin manifest or fail with an actionable message."""
    root = Path(project_root) / "data_repository" / f"basin_{params.basin}"
    manifest_path = root / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Basin {params.basin!r} is not prepared: {manifest_path}. "
            "Run --mode prepare first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 4:
        raise ValueError(
            f"Unsupported preparation manifest schema: {manifest.get('schema')!r}; "
            "prepare the basin again with the current data pipeline."
        )
    bind_prepared_basin_catalog(params, manifest)
