"""建立流域工作空间、识别输入、检查缺文件，不执行下载。"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import shapefile


@dataclass(frozen=True)
class BasinWorkspace:
    root: Path
    region_shp: Path
    dem_tiles: Path
    dem_processed: Path
    hydro_grid: Path
    forcing: Path
    forcing_grid: Path
    forcing_basin_mean: Path
    normalization: Path
    discharge_raw: Path
    discharge_processed: Path
    terrain: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "preparation_manifest.json"


def normalize_station_id(value) -> str:
    """Return stable string IDs for integer-like SHP attribute values."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if (
        isinstance(value, (float, np.floating))
        and np.isfinite(value)
        and float(value).is_integer()
    ):
        return str(int(value))
    return str(value)


def create_basin_workspace(repository: Path, basin_name: str) -> BasinWorkspace:
    """验证流域标识并在数据仓库内创建独立工作目录。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", basin_name):
        raise ValueError("basin_name 只能包含英文字母、数字、下划线和短横线")
    root = repository.resolve() / f"basin_{basin_name.lower()}"
    workspace = BasinWorkspace(
        root, root / "region_shp", root / "dem" / "raw", root / "dem" / "processed",
        root / "hydro_grid", root / "forcing", root / "forcing" / "grid",
        root / "forcing" / "basin_mean", root / "forcing" / "grid" / "normalization",
        root / "discharge" / "raw", root / "discharge" / "processed", root / "terrain",
    )
    for directory in vars(workspace).values():
        directory.mkdir(parents=True, exist_ok=True)
    return workspace


def write_json(path: Path, payload: dict) -> None:
    """先写临时文件，再替换；不留下貌似完整的中断报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def input_signature(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def read_vector(path: Path, target_crs="EPSG:3035") -> list[dict]:
    """读取 SHP 几何和属性，并根据 .prj 真正重投影；不猜测坐标单位。"""
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        if not path.with_suffix(suffix).is_file():
            raise FileNotFoundError(f"SHP 配套文件缺失：{path.with_suffix(suffix)}")
    source = CRS.from_wkt(path.with_suffix(".prj").read_text(encoding="utf-8"))
    transformer = Transformer.from_crs(source, target_crs, always_xy=True)
    encoding_path = path.with_suffix(".cpg")
    encoding = encoding_path.read_text().strip() if encoding_path.exists() else "utf-8"
    if encoding == "65001":
        encoding = "utf-8"

    def project_coordinates(coordinates):
        if isinstance(coordinates[0], (float, int)):
            return list(transformer.transform(coordinates[0], coordinates[1]))
        return [project_coordinates(part) for part in coordinates]

    features = []
    with shapefile.Reader(str(path), encoding=encoding, encodingErrors="replace") as reader:
        for item in reader.iterShapeRecords():
            geometry = dict(item.shape.__geo_interface__)
            geometry["coordinates"] = project_coordinates(geometry["coordinates"])
            features.append({"geometry": geometry, "properties": item.record.as_dict()})
    return features


def geometry_points(geometry: dict) -> np.ndarray:
    def flatten(coordinates):
        if isinstance(coordinates[0], (float, int)):
            return [coordinates[:2]]
        return [point for part in coordinates for point in flatten(part)]
    return np.asarray(flatten(geometry["coordinates"]), dtype=np.float64)


def discover_vectors(workspace: BasinWorkspace, outlet_id=None) -> dict:
    roles = {"boundary": [], "gauges": [], "river": []}
    for path in sorted(workspace.region_shp.glob("*.shp")):
        with shapefile.Reader(str(path)) as reader:
            kind = reader.shapeType
        role = "boundary" if kind in (5, 15, 25) else "gauges" if kind in (1, 11, 21) else "river" if kind in (3, 13, 23) else None
        if role:
            roles[role].append(path)
    if not roles["boundary"]:
        raise FileNotFoundError(
            "缺少流域范围SHP，数据准备已停止。请将唯一一套 Polygon/"
            f"MultiPolygon 流域边界放入：{workspace.region_shp}\n"
            "边界至少需要同名的 .shp、.shx、.dbf、.prj 文件。"
        )
    if len(roles["boundary"]) > 1:
        raise ValueError(
            f"{workspace.region_shp} 中发现 {len(roles['boundary'])} 个面SHP；"
            "请只保留当前流域唯一的边界范围"
        )
    if not roles["gauges"]:
        raise FileNotFoundError(
            "缺少测站SHP，数据准备已停止。请将包含出口ID的唯一点SHP放入："
            f"{workspace.region_shp}"
        )
    if len(roles["gauges"]) > 1:
        raise ValueError(
            f"{workspace.region_shp} 中发现 {len(roles['gauges'])} 个点SHP；"
            "请只保留当前流域对应的测站文件"
        )
    if len(roles["river"]) > 1:
        raise ValueError("河线 SHP 多于一份，请仅保留本流域对应河线")
    boundary = read_vector(roles["boundary"][0])
    if len(boundary) != 1:
        raise ValueError("流域边界需为一条 Polygon/MultiPolygon 记录，不接受重叠流域集合")
    selected_id = outlet_id if outlet_id is not None else boundary[0]["properties"].get("ID")
    if selected_id is None:
        raise ValueError("边界没有 ID 字段，请显式提供 --outlet_id（LamaH 站点 ID）")
    gauges = read_vector(roles["gauges"][0])
    gauge_ids = [str(feature["properties"].get("ID")) for feature in gauges]
    if any(value in {"None", ""} for value in gauge_ids):
        raise ValueError("测站 SHP 的每条记录都必须包含非空 ID 字段")
    if len(set(gauge_ids)) != len(gauge_ids):
        raise ValueError("测站 SHP 含重复 ID")
    gauges = sorted(
        gauges,
        key=lambda feature: (
            0, int(feature["properties"]["ID"])
        ) if str(feature["properties"]["ID"]).isdigit() else (
            1, str(feature["properties"]["ID"])
        ),
    )
    selected = [feature for feature in gauges if str(feature["properties"].get("ID")) == str(selected_id)]
    if len(selected) != 1:
        raise ValueError(f"测站 SHP 中找不到唯一出口 ID={selected_id}；不按第一条测站或最大面积猜测")
    return {"boundary_path": roles["boundary"][0], "gauge_path": roles["gauges"][0],
            "river_path": roles["river"][0] if roles["river"] else None,
            "boundary": boundary[0], "gauges": gauges,
            "outlet": selected[0], "outlet_id": str(selected_id)}


@dataclass(frozen=True)
class DailySourceCatalog:
    dataset: str
    variable: str
    files_by_day: dict
    start: pd.Timestamp
    end: pd.Timestamp


def open_source_dataset(path):
    """按文件格式选择引擎，避免 Windows netCDF4 C 库的中文路径问题。"""
    import xarray as xr
    with Path(path).open("rb") as source:
        header = source.read(8)
    if header.startswith(b"CDF"):
        return xr.open_dataset(path, engine="scipy")
    if header == b"\x89HDF\r\n\x1a\n":
        return xr.open_dataset(path, engine="h5netcdf")
    raise ValueError(f"不是可识别的 NetCDF 文件：{path}")


def discover_daily_sources(source_root: Path) -> dict[str, DailySourceCatalog]:
    """从文件和时间轴确定实际覆盖，不信任目录名里的年份。"""
    import xarray as xr
    if not source_root.is_dir():
        raise FileNotFoundError(f"原始日气象数据路径无效：{source_root}")
    definitions = (("prec", "MSWEP", "precipitation"), ("pet", "GLEAM", "Ep"), ("t", "ERA5", "2m_temperature"))
    catalogs = {}
    for key, dataset, variable in definitions:
        folders = [folder for folder in source_root.iterdir() if folder.is_dir() and dataset.lower() in folder.name.lower()]
        if len(folders) != 1:
            raise FileNotFoundError(f"{source_root} 下需要唯一 {dataset} 数据文件夹；发现 {len(folders)} 个")
        files_by_day = {}
        for path in sorted(folders[0].glob("*.nc")):
            match = re.search(r"(\d{4})[.-](\d{2})[.-](\d{2})", path.stem)
            if match:
                dates = [pd.Timestamp("-".join(match.groups()))]
            else:
                with open_source_dataset(path) as data:
                    name = next((name for name in ("time", "valid_time") if name in data.variables), None)
                    if name is None:
                        raise ValueError(f"{path.name} 没有日期或时间坐标")
                    dates = pd.DatetimeIndex(np.asarray(data[name]).reshape(-1))
            for date in dates:
                day = pd.Timestamp(date).normalize()
                if day in files_by_day:
                    raise ValueError(f"{dataset} 日期重复：{day.date()}")
                files_by_day[day] = path
        if not files_by_day:
            raise FileNotFoundError(f"{dataset} 文件夹没有 NetCDF 日数据")
        catalogs[key] = DailySourceCatalog(dataset.lower(), variable, files_by_day, min(files_by_day), max(files_by_day))
    return catalogs


def common_period(catalogs: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = max(catalog.start for catalog in catalogs.values())
    end = min(catalog.end for catalog in catalogs.values())
    if start > end:
        raise ValueError("P/T/PET 没有共同时间区间")
    return start, end


def discover_discharge_sources(
    workspace: BasinWorkspace,
    vectors: dict,
) -> dict[str, Path]:
    """Return the observed SHP gauges, with one explicitly named outlet file."""
    pattern = re.compile(r"^ID_([^_]+?)(_outlet)?\.csv$")
    parsed: dict[str, tuple[Path, bool]] = {}
    unexpected = []
    for path in sorted(workspace.discharge_raw.glob("*.csv")):
        match = pattern.fullmatch(path.name)
        if match is None:
            unexpected.append(path.name)
            continue
        station_id, outlet_suffix = match.groups()
        if station_id in parsed:
            raise ValueError(
                f"站点 ID={station_id} 同时存在多个日流量文件："
                f"{parsed[station_id][0].name}, {path.name}"
            )
        parsed[station_id] = (path, outlet_suffix is not None)

    shp_ids = [
        str(feature["properties"]["ID"]) for feature in vectors["gauges"]
    ]
    shp_id_set = set(shp_ids)
    extra = sorted(set(parsed).difference(shp_id_set))
    marked_outlets = sorted(
        station_id for station_id, (_, is_outlet) in parsed.items() if is_outlet
    )
    errors = []
    if unexpected:
        errors.append("无法识别的流量文件名：" + ", ".join(unexpected))
    if extra:
        errors.append("日流量文件中的站点不在测站 SHP：" + ", ".join(extra))
    if marked_outlets != [str(vectors["outlet_id"])]:
        errors.append(
            "必须且只能将总出口命名为 "
            f"ID_{vectors['outlet_id']}_outlet.csv；当前出口标记={marked_outlets}"
        )
    for station_id, (path, is_outlet) in parsed.items():
        expected_name = (
            f"ID_{station_id}_outlet.csv" if is_outlet else f"ID_{station_id}.csv"
        )
        if path.name != expected_name:
            errors.append(f"流量文件名应为 {expected_name}，当前为 {path.name}")
    if errors:
        raise FileNotFoundError("\n- ".join(errors))
    return {
        station_id: parsed[station_id][0]
        for station_id in shp_ids
        if station_id in parsed
    }


def check_required_inputs(workspace: BasinWorkspace, source_root: Path, outlet_id=None) -> tuple[dict, dict, dict[str, Path]]:
    """集中显示输入错误。任何下载和训练之前都必须通过本检查。"""
    # 流域范围决定DEM下载、网格裁剪和后续空间处理，故首先校验矢量输入。
    try:
        vectors = discover_vectors(workspace, outlet_id)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "[矢量输入预检未通过；未下载、未处理、未训练]\n- " + str(error)
        ) from None
    except (ValueError, OSError) as error:
        raise ValueError(
            "[矢量输入预检未通过；未下载、未处理、未训练]\n- " + str(error)
        ) from None

    errors = []
    catalogs = None
    try:
        catalogs = discover_daily_sources(source_root)
    except (FileNotFoundError, ValueError, OSError) as error:
        errors.append(str(error))
    discharge_sources = None
    try:
        discharge_sources = discover_discharge_sources(workspace, vectors)
    except (FileNotFoundError, ValueError) as error:
        errors.append(str(error))
    if errors:
        raise FileNotFoundError("[输入预检未通过；未下载、未训练]\n- " + "\n- ".join(errors))
    return vectors, catalogs, discharge_sources
