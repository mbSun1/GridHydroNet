# Data repository

TopoHydroNet expects basin boundaries and river-network vectors, Copernicus DEM, MSWEP precipitation, ERA5 2 m air temperature, GLEAM potential evapotranspiration, and USGS daily discharge observations.

The raw and processed datasets used in the Pascagoula experiment are not stored in Git because they are large and remain subject to their original providers' licences. Recreate the local `data_repository/` hierarchy with the preparation utilities in `disthybridcore/prepare_data/`, then point the experiment configuration to that directory.

Generated model outputs and checkpoints belong under `outputs/` and are ignored by Git.
