import numpy as np

import create_bias_layers as cbl


def test_single_observation_survives_smoothing():
    # Округление до целых стирало всё, что давало в пике меньше 0.5, то есть до 12 наблюдений в пикселе
    counts = np.zeros((41, 41), dtype=np.uint32)
    counts[20, 20] = 1
    smoothed = cbl.smooth_counts(counts)
    assert smoothed.dtype == np.float32
    assert smoothed.max() > 0.03


def test_smoothing_keeps_total_number_of_observations():
    counts = np.zeros((101, 101), dtype=np.uint32)
    counts[50, 50] = 1000
    counts[30, 70] = 7
    assert abs(cbl.smooth_counts(counts).sum() - 1007) < 1


def test_layer_is_not_negative():
    counts = np.random.default_rng(1).integers(0, 5, (50, 50)).astype(np.uint32)
    assert cbl.smooth_counts(counts).min() >= 0


def test_layer_names_match_what_sdm_looks_for():
    assert cbl.get_layer_path('all', '30s', '30s').endswith('human_bias_alls.tif')
    assert cbl.get_layer_path('Aves', '30s', '30s').endswith('human_bias_Aves.tif')
    assert cbl.get_layer_path('Aves', '5m', '30s').endswith('human_bias_Aves_5m.tif')
