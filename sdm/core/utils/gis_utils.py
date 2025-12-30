import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling


def read_and_to_3857(path):
    dest_crs = CRS.from_epsg(3857)
    with rasterio.open(path) as src:
        src_crs = src.crs
        band1 = src.read(1, masked=True)  # первая полоса (например, пригодность 0..1)
        # Если уже в 3857 — просто вернуть как есть
        if src_crs == dest_crs:
            data = band1.filled(np.nan).astype("float32")
            transform = src.transform
            width, height = src.width, src.height
        else:
            # Считаем параметры целевой решетки
            transform, width, height = calculate_default_transform(
                src_crs, dest_crs, src.width, src.height, *src.bounds
            )
            # Готовим массив назначения
            data = np.full((height, width), np.nan, dtype="float32")
            reproject(
                source=band1.filled(np.nan),
                destination=data,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dest_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
    return data, transform, width, height