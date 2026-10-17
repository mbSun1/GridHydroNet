"""Resolve prepared basin-mean forcing and unique outlet observations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .observations import outlet_metadata


@dataclass(frozen=True)
class LumpedDataCatalog:
    basin: str
    basin_root: Path
    forcing: Path
    discharge: Path
    basin_area_km2: float
    outlet_id: str
    outlet_name: str
    official_area_km2: float
    forcing_start: str
    forcing_end_inclusive: str
    input_signature: str


def load_lumped_catalog(project_root: str | Path, basin: str) -> LumpedDataCatalog:
    project_root = Path(project_root).resolve()
    basin = str(basin).strip().lower()
    basin_root = project_root / "data_repository" / f"basin_{basin}"
    path = basin_root / "forcing" / "basin_mean" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Basin-mean forcing manifest is missing: {path}. Run --mode prepare first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("basin") != basin:
        raise ValueError(f"Invalid basin-mean forcing manifest: {path}")
    forcing = Path(manifest["forcing"])
    discharge = Path(manifest["discharge_source"])
    if not forcing.is_file() or not discharge.is_file():
        raise FileNotFoundError("Basin-mean forcing or prepared discharge file is missing")
    observed = outlet_metadata(discharge)
    if observed["outlet_id"] != str(manifest["outlet_id"]):
        raise ValueError("Basin-mean manifest and discharge outlet IDs disagree")
    return LumpedDataCatalog(
        basin=basin,
        basin_root=basin_root,
        forcing=forcing,
        discharge=discharge,
        basin_area_km2=float(manifest["basin_area_km2"]),
        outlet_id=observed["outlet_id"],
        outlet_name=observed["outlet_name"],
        official_area_km2=float(observed["official_area_km2"]),
        forcing_start=str(manifest["start"]),
        forcing_end_inclusive=str(manifest["end_inclusive"]),
        input_signature=str(manifest["input_signature"]),
    )
