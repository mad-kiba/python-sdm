import numpy as np
from affine import Affine

from sdm.utils.utils import sample_background

# ~1 км на пиксель (приблизительно, для простоты: 1° широты ~= 111.32 км)
KM_TRANSFORM = Affine(1 / 111.32, 0, 0, 0, -1 / 111.32, 0)


def test_wide_ring_has_no_shortfall():
    """Обычный случай: кольцо достаточно широкое - добора не требуется."""
    valid = np.ones((30, 30), dtype=bool)
    rng = np.random.default_rng(1)
    *_, shortfall_info = sample_background(
        valid, {(15, 15)}, 50, rng, bg_pc=50,
        distance_min_km=0.5, distance_max_km=15, transform=KM_TRANSFORM,
    )
    assert shortfall_info['bg_shortfall_reason'] is None
    assert shortfall_info['bg_requested'] == 50
    assert shortfall_info['bg_random_actual'] + shortfall_info['bg_buffer_actual'] == 50


def test_narrow_ring_is_backfilled_with_random_background():
    """
    Узкое кольцо (только 4 ближайшие ячейки вокруг единственной точки присутствия, ~1 км
    по сторонам, но не по диагонали, ~1.4 км) не может дать нужное число псевдоотсутствий,
    но в большом регионе хватает случайного фона, чтобы добрать разницу: итог должен
    совпасть с запросом.
    """
    valid = np.ones((30, 30), dtype=bool)
    rng = np.random.default_rng(1)
    rows_bg, cols_bg, rows_random, cols_random, rows_buffer, cols_buffer, num_biased, shortfall_info = sample_background(
        valid, {(15, 15)}, 100, rng, bg_pc=50,
        distance_min_km=0.5, distance_max_km=1.1, transform=KM_TRANSFORM,
    )
    assert shortfall_info['bg_requested'] == 100
    assert shortfall_info['bg_shortfall_reason'] == 'ring_too_small'
    assert shortfall_info['bg_buffer_actual'] < 50  # кольцо дало меньше, чем половину запроса
    assert shortfall_info['bg_random_actual'] + shortfall_info['bg_buffer_actual'] == 100
    assert len(rows_bg) == 100


def test_tiny_region_shortfall_cannot_be_fully_backfilled():
    """Регион настолько мал, что даже с добором случайным фоном не хватает ячеек на весь запрос."""
    valid = np.ones((5, 5), dtype=bool)
    rng = np.random.default_rng(1)
    *_, shortfall_info = sample_background(
        valid, {(2, 2)}, 30, rng, bg_pc=50,
        distance_min_km=0.5, distance_max_km=1.1, transform=KM_TRANSFORM,
    )
    assert shortfall_info['bg_shortfall_reason'] == 'region_too_small'
    assert shortfall_info['bg_random_actual'] + shortfall_info['bg_buffer_actual'] < 30


def test_no_presence_points_for_buffer_is_reported():
    valid = np.ones((10, 10), dtype=bool)
    rng = np.random.default_rng(1)
    *_, shortfall_info = sample_background(
        valid, set(), 20, rng, bg_pc=50,
        distance_min_km=0.5, distance_max_km=2,
    )
    assert shortfall_info['bg_shortfall_reason'] == 'no_presence_for_buffer'
