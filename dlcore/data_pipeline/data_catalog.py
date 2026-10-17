"""Resolve basin-mean forcing and the unique outlet observation file."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from dlcore.data_pipeline.observations import outlet_metadata


@dataclass(frozen=True)
class DLDataCatalog:
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


def resolve_manifest_path(path_value, fallback_directory: Path, pattern: str) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path.resolve()
    matches = sorted(fallback_directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Manifest path is unavailable and fallback is not unique: {path}"
        )
    return matches[0].resolve()


def load_dl_catalog(project_root: str | Path, basin: str) -> DLDataCatalog:
    project_root = Path(project_root).resolve()
    basin = str(basin).strip().lower()
    basin_root = project_root / "data_repository" / f"basin_{basin}"
    manifest_path = basin_root / "forcing" / "basin_mean" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Basin-mean forcing manifest is missing: {manifest_path}. "
            "Run the lumped forcing preparation first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("basin") != basin:
        raise ValueError(f"Invalid basin-mean forcing manifest: {manifest_path}")
    expected_units = {"prec": "mm/day", "t": "degree_Celsius", "pet": "mm/day"}
    if manifest.get("variables") != expected_units:
        raise ValueError("Basin-mean manifest variables or units do not match P/T/PET contract")
    forcing = resolve_manifest_path(
        manifest["forcing"], basin_root / "forcing" / "basin_mean", "*.nc"
    )
    discharge = resolve_manifest_path(
        manifest["discharge_source"], basin_root / "discharge" / "processed", "*.nc"
    )
    observed = outlet_metadata(discharge)
    if observed["outlet_id"] != str(manifest["outlet_id"]):
        raise ValueError("Basin-mean manifest and discharge outlet IDs disagree")
    return DLDataCatalog(
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
