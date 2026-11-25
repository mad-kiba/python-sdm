import inspect

import numpy as np

from sdm.utils.utils import apply_decay_to_points, sample_background

SHAPE = (100, 100)


def sample(bias=None, strength=0.7, n_bg=200):
    valid = np.ones(SHAPE, dtype=bool)
    return sample_background(valid, {(50, 50)}, n_bg, np.random.default_rng(1), 100, 1, 2,
                             bias_weights_map=bias, bias_sampling_strength=strength)


def test_without_layer_no_points_are_biased():
    assert sample()[-1] == 0


def test_zero_strength_means_no_biased_points():
    assert sample(np.ones(SHAPE, dtype=np.float32), strength=0)[-1] == 0


def test_strength_sets_share_of_biased_points():
    result = sample(np.ones(SHAPE, dtype=np.float32), strength=0.7)
    assert result[-1] == int(len(result[2]) * 0.7)


def test_layer_without_observations_gives_no_biased_points():
    result = sample(np.zeros(SHAPE, dtype=np.float32))
    assert result[-1] == 0 and len(result[2]) > 0


def test_count_is_not_above_number_of_pixels_with_observations():
    bias = np.zeros(SHAPE, dtype=np.float32)
    bias[:2, :5] = 3.0
    assert sample(bias)[-1] == 10


def test_biased_points_prefer_observed_pixels():
    bias = np.zeros(SHAPE, dtype=np.float32)
    bias[:50] = 5.0
    rows = sample(bias, strength=1.0)[2]
    assert np.all(rows < 50)


def test_total_background_matches_plan_when_biased_part_is_underfilled():
    # Пикселей с данными намного меньше, чем запрошенная взвешенная доля (140 из 200) -
    # недостачу должна покрыть равномерная часть, а не удвоить её
    bias = np.zeros(SHAPE, dtype=np.float32)
    bias[:2, :5] = 3.0  # 10 пикселей с ненулевым весом
    result = sample(bias, strength=0.7, n_bg=200)
    assert result[-1] == 10
    assert len(result[2]) == 200


def test_m_factor_does_not_take_bias_layer():
    assert 'bias_data' not in inspect.signature(apply_decay_to_points).parameters
