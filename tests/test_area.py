import math

import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import from_bounds

from sdm.utils.utils import get_geotiff_square, calculate_mask_area_km2

RESOLUTION_DEG = 30 / 3600.0  # 30 угловых секунд
SIZE_DEG = 1.0
GEOD = pyproj.Geod(ellps='WGS84')


def _make_tif(path, lat0, lon0=0.0, value=1.0, nodata_rows=None):
    """1x1-градусный растр с шагом 30'' на заданной широте, все ячейки = value."""
    n = round(SIZE_DEG / RESOLUTION_DEG)
    data = np.full((n, n), value, dtype=np.float32)
    if nodata_rows is not None:
        data[nodata_rows, :] = np.nan
    transform = from_bounds(lon0, lat0, lon0 + SIZE_DEG, lat0 + SIZE_DEG, n, n)
    with rasterio.open(
        str(path), 'w', driver='GTiff', height=n, width=n, count=1,
        dtype='float32', crs='EPSG:4326', transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    return str(path), transform, n


def _geodesic_bbox_area_km2(lon0, lat0, lon1, lat1):
    """Площадь прямоугольника по эллипсоиду WGS84, независимо от кода, который проверяем."""
    area_m2, _ = GEOD.polygon_area_perimeter(
        [lon0, lon1, lon1, lon0], [lat0, lat0, lat1, lat1],
    )
    return abs(area_m2) / 1_000_000.0


def test_equator_area_matches_independent_geodesic_calculation(tmp_path):
    path, _, _ = _make_tif(tmp_path / 'equator.tif', lat0=0.0)
    square, _ = get_geotiff_square(path, [0.5])
    expected = _geodesic_bbox_area_km2(0.0, 0.0, 1.0, 1.0)
    # ~12 300 км² для 1х1 градуса на экваторе
    assert square[0] == pytest.approx(expected, rel=0.01)
    assert square[0] == pytest.approx(12300, rel=0.03)


def test_area_at_60_degrees_matches_independent_geodesic_calculation(tmp_path):
    path, _, _ = _make_tif(tmp_path / 'lat60.tif', lat0=60.0)
    square, _ = get_geotiff_square(path, [0.5])
    expected = _geodesic_bbox_area_km2(0.0, 60.0, 1.0, 61.0)
    assert square[0] == pytest.approx(expected, rel=0.01)


def test_area_shrinks_with_latitude_roughly_like_cosine(tmp_path):
    """
    Поправка на широту: площадь ячейки на севере меньше, чем на юге.
    Для полосы 60-61° отношение к экватору близко к cos(60.5°) (широта середины полосы),
    а не ровно к cos(60°) — поэтому допуск шире, чем для прямых геодезических проверок выше.
    """
    path_eq, _, _ = _make_tif(tmp_path / 'eq.tif', lat0=0.0)
    path_60, _, _ = _make_tif(tmp_path / 'n60.tif', lat0=60.0)
    sq_eq, _ = get_geotiff_square(path_eq, [0.5])
    sq_60, _ = get_geotiff_square(path_60, [0.5])
    ratio = sq_60[0] / sq_eq[0]
    assert ratio == pytest.approx(math.cos(math.radians(60.5)), rel=0.01)
    assert ratio == pytest.approx(0.5, abs=0.02)  # качественно: "примерно половина", как в ТЗ


def test_nodata_cells_excluded_from_area_and_count(tmp_path):
    n = round(SIZE_DEG / RESOLUTION_DEG)
    nodata_rows = [0, 1, 2]  # первые 3 строки - NoData
    path, _, _ = _make_tif(tmp_path / 'with_nodata.tif', lat0=0.0, nodata_rows=nodata_rows)
    square, count = get_geotiff_square(path, [0.5])
    full_path, _, _ = _make_tif(tmp_path / 'full.tif', lat0=0.0)
    full_square, full_count = get_geotiff_square(full_path, [0.5])

    assert count[0] == full_count[0] - len(nodata_rows) * n
    assert square[0] < full_square[0]
    assert square[0] == pytest.approx(full_square[0] * (n - len(nodata_rows)) / n, rel=0.01)


def test_region_area_matches_sum_of_row_areas(tmp_path):
    """calculate_mask_area_km2 считает тем же способом, что и get_geotiff_square."""
    path, transform, n = _make_tif(tmp_path / 'mask.tif', lat0=30.0)
    mask = np.ones((n, n), dtype=bool)
    region_area = calculate_mask_area_km2(mask, transform)

    square, _ = get_geotiff_square(path, [0.5])
    assert region_area == pytest.approx(square[0], rel=0.001)


def test_region_area_excludes_false_cells(tmp_path):
    path, transform, n = _make_tif(tmp_path / 'mask_partial.tif', lat0=0.0)
    mask = np.ones((n, n), dtype=bool)
    mask[: n // 2, :] = False  # половина региона невалидна

    region_area = calculate_mask_area_km2(mask, transform)
    full_area = calculate_mask_area_km2(np.ones((n, n), dtype=bool), transform)
    assert region_area == pytest.approx(full_area / 2, rel=0.02)


def test_percent_of_region_times_region_area_equals_threshold_area(tmp_path):
    """
    Критерий приёмки: "% региона" * площадь региона = площадь при пороге (с точностью до округления).
    Используем неоднородный растр (часть региона ниже порога), чтобы проверка была содержательной.
    """
    n = round(SIZE_DEG / RESOLUTION_DEG)
    data = np.zeros((n, n), dtype=np.float32)
    data[: n // 3, :] = 1.0  # треть региона "пригодна" (значение выше порога)
    transform = from_bounds(0.0, 10.0, 1.0, 11.0, n, n)
    path = tmp_path / 'partial.tif'
    with rasterio.open(
        str(path), 'w', driver='GTiff', height=n, width=n, count=1,
        dtype='float32', crs='EPSG:4326', transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(data, 1)

    threshold = 0.5
    square, _ = get_geotiff_square(str(path), [threshold])
    region_mask = np.ones((n, n), dtype=bool)
    region_total_km2 = calculate_mask_area_km2(region_mask, transform)

    percent_of_region = round(square[0] / region_total_km2 * 100, 2)
    reconstructed_area = percent_of_region / 100 * region_total_km2

    assert reconstructed_area == pytest.approx(square[0], rel=0.01)
