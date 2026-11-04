import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sdm.utils.plots import create_beautiful_histogram
from sdm.utils.utils import build_histogram_stats, calculate_histogram_similarity, clean_nans_for_json, get_similarity_bins_num

CONTINUOUS = {'unit': 'мм', 'scale': 1, 'diff': 0}
CATEGORICAL = {'unit': 'да', 'kind': 'categorical', 'scale': 1, 'diff': 0}


@pytest.fixture
def background():
    return np.random.default_rng(1).normal(100.0, 20.0, 50000)


def test_identical_arrays_give_one(background):
    assert calculate_histogram_similarity(background, background) == 1.0


def test_disjoint_arrays_give_low_non_negative(background):
    top_decile = background[background >= np.percentile(background, 90)]
    d = calculate_histogram_similarity(top_decile, background)
    assert 0.0 <= d < 0.2


@pytest.mark.parametrize("factor", [0.001, 1000.0])
def test_scale_invariance(background, factor):
    # Именно этот тест ловит нормировку на плотность: тогда D зависит от ширины бина
    obs = np.random.default_rng(2).normal(120.0, 10.0, 400)
    assert calculate_histogram_similarity(obs * factor, background * factor) == pytest.approx(
        calculate_histogram_similarity(obs, background), abs=1e-3)


def test_narrow_layer_is_in_unit_interval(background):
    obs = np.random.default_rng(2).normal(120.0, 10.0, 400)
    assert 0.0 <= calculate_histogram_similarity(obs * 1e-3, background * 1e-3) <= 1.0


def test_empty_and_small_obs_give_none(background):
    assert calculate_histogram_similarity(np.array([]), background) is None
    assert calculate_histogram_similarity(background[:9], background) is None
    assert calculate_histogram_similarity(background[:100], np.array([])) is None


def test_constant_background_gives_number_not_tuple():
    same = calculate_histogram_similarity(np.full(20, 5.0), np.full(100, 5.0))
    other = calculate_histogram_similarity(np.full(20, 7.0), np.full(100, 5.0))
    assert same == 1.0
    assert other == 0.0


def test_obs_outside_background_range_are_not_lost():
    full = np.linspace(0.0, 1.0, 1000)
    # При диапазоне только по фону эти точки выпали бы, и пересечение оказалось бы завышено
    obs = np.concatenate([np.linspace(0.0, 1.0, 50), np.linspace(5.0, 6.0, 50)])
    assert calculate_histogram_similarity(obs, full) <= 0.51


@pytest.mark.parametrize("n_obs, bins", [(4, 10), (20, 10), (422, 41), (10000, 50)])
def test_bins_num(n_obs, bins):
    assert get_similarity_bins_num(n_obs) == bins


@pytest.mark.parametrize("shift, expected", [
    (0.0, 'same'), (0.5, 'higher'), (1.0, 'much_higher'), (-0.5, 'lower'), (-1.0, 'much_lower'),
])
def test_direction(background, shift, expected):
    obs = background + shift * background.std()
    metrics = build_histogram_stats(obs, background, CONTINUOUS)['metrics']
    assert metrics['std_diff'] == pytest.approx(shift, abs=0.02)
    assert metrics['direction'] == expected


def test_constant_background_has_no_direction():
    metrics = build_histogram_stats(np.full(20, 7.0), np.full(100, 5.0), CONTINUOUS)['metrics']
    assert metrics['std_diff'] is None and metrics['direction'] is None
    assert metrics['rel_width'] is None


def test_continuous_entry_format(background):
    obs = background[:422]
    entry = build_histogram_stats(obs, background, CONTINUOUS)
    assert entry['stats_version'] == 2
    assert entry['unit'] == 'мм' and entry['kind'] == 'continuous'
    assert entry['n_obs'] == 422 and entry['n_region'] == background.size and entry['bins'] == 41
    for part in ('obs', 'region'):
        assert {'mean', 'median', 'p5', 'p10', 'p90', 'p95', 'std', 'width_p10_p90'} <= set(entry[part])
    assert set(entry['metrics']) == {'schoener_d', 'rel_width', 'std_diff', 'direction', 'obs_in_region_pct'}
    assert entry['metrics']['rel_width'] == pytest.approx(1.0, abs=0.15)
    assert entry['metrics']['obs_in_region_pct'] == pytest.approx(90, abs=5)


def test_small_obs_entry(background):
    entry = build_histogram_stats(background[:5], background, CONTINUOUS)
    assert entry['bins'] is None
    assert entry['metrics']['schoener_d'] is None and entry['metrics']['rel_width'] is None
    assert entry['metrics']['direction'] is not None


def test_empty_obs_entry_is_json_safe(background):
    entry = clean_nans_for_json(build_histogram_stats(np.array([]), background, CONTINUOUS))
    assert entry['obs']['mean'] is None
    assert entry['metrics']['direction'] is None
    json.dumps(entry, allow_nan=False)


def test_categorical_entry():
    obs = np.array([1.0] * 40 + [0.0] * 60)
    full = np.array([1.0] * 150 + [0.0] * 850)
    entry = build_histogram_stats(obs, full, CATEGORICAL)
    assert entry['kind'] == 'categorical' and entry['bins'] == 2
    assert entry['obs'] == {'share': 0.4} and entry['region'] == {'share': 0.15}
    assert set(entry['metrics']) == {'schoener_d'}
    assert entry['metrics']['schoener_d'] == pytest.approx(0.75)


def test_rounding_has_no_float_noise():
    entry = build_histogram_stats(np.full(20, 0.3), np.linspace(0.0, 1.0, 100), CONTINUOUS)
    assert json.dumps(entry['obs']['median']) == '0.3'


@pytest.mark.parametrize("info, with_box", [(CONTINUOUS, False), (CONTINUOUS, True), (CATEGORICAL, False), (CATEGORICAL, True)])
def test_plot_returns_entry_and_box_is_optional(info, with_box):
    rng = np.random.default_rng(3)
    obs, full = rng.integers(0, 2, 100).astype(float), rng.integers(0, 2, 1000).astype(float)
    if info is CONTINUOUS:
        obs, full = rng.normal(5, 1, 100), rng.normal(4, 2, 1000)
    fig, ax = plt.subplots()
    entry = create_beautiful_histogram(ax, obs, 'layer', 50 if info is CONTINUOUS else 2, full, {'layer': info},
                                       with_stats_box=with_box)
    assert entry['kind'] == info.get('kind', 'continuous')
    assert (len(ax.texts) == 1) == with_box
    plt.close(fig)


def test_plot_box_with_undefined_metrics():
    fig, ax = plt.subplots()
    entry = create_beautiful_histogram(ax, np.array([1.0, 2.0]), 'layer', 50, np.linspace(0, 5, 100),
                                       {'layer': CONTINUOUS}, with_stats_box=True)
    assert entry['metrics']['schoener_d'] is None
    assert 'н/д' in ax.texts[0].get_text()
    plt.close(fig)


def test_plot_without_any_data():
    fig, ax = plt.subplots()
    assert create_beautiful_histogram(ax, np.array([]), 'layer', 50, np.array([]), {'layer': CONTINUOUS}) is None
    plt.close(fig)
