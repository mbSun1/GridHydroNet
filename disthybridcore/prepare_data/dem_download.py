"""根据 SHP 的带缓冲范围下载缺少的 Copernicus GLO-30 原始瓦片。"""

import math
from pathlib import Path
import time

import rasterio
import requests
from pyproj import Transformer

from .basin_workspace import geometry_points


def required_dem_tiles(boundary: dict, buffer_m: float = 3000.0) -> list[str]:
    points = geometry_points(boundary["geometry"])
    left, bottom = points.min(axis=0) - buffer_m
    right, top = points.max(axis=0) + buffer_m
    west, south, east, north = Transformer.from_crs(3035, 4326, always_xy=True).transform_bounds(left, bottom, right, top, densify_pts=21)
    names = []
    for latitude in range(math.floor(south), math.ceil(north)):
        for longitude in range(math.floor(west), math.ceil(east)):
            lat = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}"
            lon = f"{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}"
            names.append(f"Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM")
    return names


def validate_dem_tile(path: Path) -> None:
    with rasterio.open(path) as source:
        if source.crs is None or source.count != 1 or source.width < 2:
            raise ValueError(f"DEM 瓦片无效：{path}")
        source.read(1, window=rasterio.windows.Window(0, 0, 2, 2))


def download_dem_tiles(workspace, vectors, buffer_m=3000.0) -> list[Path]:
    """复用已校验瓦片；流式下载，失败最多重试三次，绝不伪造 DEM。"""
    result = []
    cached = 0
    downloaded = 0
    names = required_dem_tiles(vectors["boundary"], buffer_m)
    with requests.Session() as session:
        for index, name in enumerate(names, 1):
            target = workspace.dem_tiles / f"{name}.tif"
            if target.exists():
                validate_dem_tile(target)
                cached += 1
                result.append(target)
                continue
            url = (
                "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
                f"{name}/{name}.tif"
            )
            temporary = target.with_suffix(".tif.part")
            for attempt in range(3):
                try:
                    print(f"[DEM下载] {index}/{len(names)}：{name}", flush=True)
                    downloaded_bytes = temporary.stat().st_size if temporary.exists() else 0
                    headers = (
                        {"Range": f"bytes={downloaded_bytes}-"}
                        if downloaded_bytes > 0
                        else {}
                    )
                    with session.get(
                        url,
                        headers=headers,
                        stream=True,
                        timeout=(60, 180),
                    ) as response:
                        response.raise_for_status()
                        write_mode = (
                            "ab"
                            if downloaded_bytes > 0 and response.status_code == 206
                            else "wb"
                        )
                        with temporary.open(write_mode) as output:
                            for block in response.iter_content(chunk_size=1024 * 1024):
                                if block:
                                    output.write(block)
                    validate_dem_tile(temporary)
                    temporary.replace(target)
                    downloaded += 1
                    break
                except (requests.RequestException, rasterio.errors.RasterioError) as error:
                    if attempt == 2:
                        raise RuntimeError(f"DEM 下载失败：{url}。检查网络/代理后重新运行；已有瓦片保留。") from error
                    time.sleep(2 * (attempt + 1))
            result.append(target)
    print(
        f"[DEM瓦片] 共{len(names)} | 缓存={cached} | 新下载={downloaded}",
        flush=True,
    )
    return result
