import numpy as np
import pyproj
import pytest
from affine import Affine

from sdm.utils.utils import apply_decay_to_points, distance_to_points_km

GEOD = pyproj.Geod(ellps='WGS84')


def _independent_distance_km(lon1, lat1, lon2, lat2):
    """Независимый от кода, который проверяем, геодезический расчёт расстояния (км)."""
    _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return abs(dist_m) / 1000.0


def _grid_at(lat_center, resolution_deg, half_size_px):
    """Affine-трансформация для квадратной сетки, центр которой - на заданной широте."""
    size = half_size_px * 2 + 1
    center_offset = resolution_deg * (half_size_px + 0.5)
    transform = Affine(resolution_deg, 0, -center_offset, 0, -resolution_deg, lat_center + center_offset)
    return transform, size


def test_distance_matches_independent_haversine_on_both_axes():
    """
    На 45° с.ш. расстояние до точки присутствия совпадает с независимым геодезическим расчётом
    и по параллели (запад-восток), и по меридиану (север-юг) - при одинаковом числе ячеек
    эти расстояния разные (параллель короче меридиана на этой широте), поэтому недостаточно
    считать расстояние в пикселях с одним и тем же шагом по обеим осям (ТЗ 1.4).
    """
    resolution_deg = 30 / 3600.0  # 30"
    transform, size = _grid_at(45.0, resolution_deg, half_size_px=200)
    valid_mask = np.ones((size, size), dtype=bool)
    center = size // 2

    dist_km = distance_to_points_km(valid_mask, np.array([center]), np.array([center]), transform)
    center_lon, center_lat = transform * (center + 0.5, center + 0.5)

    offset = 120
    lon_e, lat_e = transform * (center + offset + 0.5, center + 0.5)
    expected_e = _independent_distance_km(center_lon, center_lat, lon_e, lat_e)
    assert dist_km[center, center + offset] == pytest.approx(expected_e, rel=0.01)

    lon_n, lat_n = transform * (center + 0.5, center - offset + 0.5)
    expected_n = _independent_distance_km(center_lon, center_lat, lon_n, lat_n)
    assert dist_km[center - offset, center] == pytest.approx(expected_n, rel=0.01)

    # cos(45°) ≈ 0.707: то же число ячеек даёт заметно меньшее расстояние по параллели
    assert expected_n > expected_e * 1.2


def test_ring_extent_in_km_is_the_same_on_both_axes_at_45n():
    """
    Кольцо 50-100 км вокруг одной точки присутствия на 45° с.ш.: кандидаты лежат в этом
    диапазоне и по меридиану, и по параллели, а физическая протяжённость кольца (км) по обеим
    осям одинакова (окружность), хотя число ячеек до границ кольца - разное (не квадратный пиксель).
    """
    resolution_deg = 30 / 3600.0
    transform, size = _grid_at(45.0, resolution_deg, half_size_px=250)
    valid_mask = np.ones((size, size), dtype=bool)
    center = size // 2

    dist_km = distance_to_points_km(valid_mask, np.array([center]), np.array([center]), transform)
    ring_mask = (dist_km >= 50) & (dist_km <= 100)

    row_hits = np.flatnonzero(ring_mask[:, center])   # кольцо по меридиану (столбец центра)
    col_hits = np.flatnonzero(ring_mask[center, :])   # кольцо по параллели (строка центра)
    assert row_hits.size > 0 and col_hits.size > 0

    row_km_span = dist_km[row_hits, center].max() - dist_km[row_hits, center].min()
    col_km_span = dist_km[center, col_hits].max() - dist_km[center, col_hits].min()
    assert row_km_span == pytest.approx(col_km_span, rel=0.05)

    # Число шагов до ~75 км разное на двух осях: по параллели ячейка "короче" на этой широте,
    # поэтому нужно больше шагов, чтобы пройти то же физическое расстояние.
    rows_going_up = dist_km[:center + 1, center][::-1]
    cols_going_right = dist_km[center, center:]
    row_steps_to_75 = int(np.argmin(np.abs(rows_going_up - 75)))
    col_steps_to_75 = int(np.argmin(np.abs(cols_going_right - 75)))
    assert col_steps_to_75 > row_steps_to_75


@pytest.mark.parametrize("resolution_deg, half_size_px", [(30 / 3600.0, 300), (5 / 60.0, 40)])
def test_ring_size_in_km_is_resolution_independent(resolution_deg, half_size_px):
    """То же кольцо 50-100 км в километрах на разных разрешениях сетки (30" и 5')."""
    transform, size = _grid_at(45.0, resolution_deg, half_size_px=half_size_px)
    valid_mask = np.ones((size, size), dtype=bool)
    center = size // 2

    dist_km = distance_to_points_km(valid_mask, np.array([center]), np.array([center]), transform)
    ring_mask = (dist_km >= 50) & (dist_km <= 100)
    row_hits = np.flatnonzero(ring_mask[:, center])
    assert row_hits.size > 0

    min_km, max_km = dist_km[row_hits, center].min(), dist_km[row_hits, center].max()
    # Допуск шире обычного: на грубой сетке (5') один шаг по широте - это уже ~9 км
    assert min_km == pytest.approx(50, abs=12)
    assert max_km == pytest.approx(100, abs=12)


def test_m_factor_buffer_is_circular_not_elliptical():
    """
    М-фактор (без барьеров рельефа/воды) с буфером 100 км: граница буфера - окружность
    в километрах, а не эллипс, хотя в пикселях её протяжённость по осям различается
    (ячейка не квадратная на этой широте).
    """
    resolution_deg = 30 / 3600.0
    transform, size = _grid_at(45.0, resolution_deg, half_size_px=250)
    center = size // 2
    rows, cols = np.array([center]), np.array([center])

    multiplier = apply_decay_to_points(
        raster_shape=(size, size), transform=transform,
        observation_rows=rows, observation_cols=cols,
        buffer_km=100, decay_type='buffer',
    )
    inside = multiplier > 0

    col_hits = np.flatnonzero(inside[center, :])
    row_hits = np.flatnonzero(inside[:, center])
    col_extent_px = col_hits.max() - col_hits.min()
    row_extent_px = row_hits.max() - row_hits.min()
    # В пикселях протяжённость разная (неквадратный пиксель)
    assert col_extent_px != row_extent_px

    dist_km = distance_to_points_km(np.ones((size, size), dtype=bool), rows, cols, transform)
    col_extent_km = dist_km[center, col_hits.max()] + dist_km[center, col_hits.min()]
    row_extent_km = dist_km[row_hits.max(), center] + dist_km[row_hits.min(), center]
    # А в километрах - одинаковая (окружность диаметром ~200 км, а не эллипс)
    assert col_extent_km == pytest.approx(row_extent_km, rel=0.1)
    assert col_extent_km == pytest.approx(200, rel=0.1)


def test_m_factor_cost_distance_buffer_is_also_circular():
    """
    То же самое, но по ветке cost-distance (Fast Marching, scikit-fmm) без реальных барьеров
    (уклон = 0 везде) - проверяет анизотропный dx=(dy_km, dx_km), переданный в skfmm.travel_time.
    """
    pytest.importorskip('skfmm')

    resolution_deg = 30 / 3600.0
    transform, size = _grid_at(45.0, resolution_deg, half_size_px=250)
    center = size // 2
    rows, cols = np.array([center]), np.array([center])
    flat_slope = np.zeros((size, size), dtype=np.float32)  # без барьеров, но включает ветку FMM

    multiplier = apply_decay_to_points(
        raster_shape=(size, size), transform=transform,
        observation_rows=rows, observation_cols=cols,
        buffer_km=100, decay_type='buffer', slope_data=flat_slope,
    )
    inside = multiplier > 0

    col_hits = np.flatnonzero(inside[center, :])
    row_hits = np.flatnonzero(inside[:, center])

    dist_km = distance_to_points_km(np.ones((size, size), dtype=bool), rows, cols, transform)
    col_extent_km = dist_km[center, col_hits.max()] + dist_km[center, col_hits.min()]
    row_extent_km = dist_km[row_hits.max(), center] + dist_km[row_hits.min(), center]
    assert col_extent_km == pytest.approx(row_extent_km, rel=0.1)
    assert col_extent_km == pytest.approx(200, rel=0.1)
