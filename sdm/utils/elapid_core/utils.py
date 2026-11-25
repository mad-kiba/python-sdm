# Адаптировано из elapid/utils.py (https://github.com/earth-chris/elapid)
# Copyright (c) 2021 Christopher Anderson, лицензия MIT - см. THIRD_PARTY_LICENSES.md
#
# В оригинальном utils.py рядом с нужными нам repeat_array/make_band_labels лежат
# совсем другие хелперы для работы с растрами и скачивания сэмплов, которые тянут
# geopandas/rasterio на уровне модуля. Здесь - только то, что реально использует
# features.py/models.py этого урезанного ядра.
"""Вспомогательные функции для адаптированного ядра MaxEnt."""

import multiprocessing as mp

import numpy as np

NCPUS = mp.cpu_count()


def repeat_array(x: np.ndarray, length: int = 1, axis: int = 0) -> np.ndarray:
    """Повторяет 1D numpy-массив вдоль оси заданное число раз.

    Args:
        x: n-мерный массив для повторения.
        length: сколько раз повторить массив.
        axis: ось, вдоль которой повторять (допустимы значения от 0 до n+1).

    Returns:
        (n+1)-мерный numpy-массив.
    """
    return np.expand_dims(x, axis=axis).repeat(length, axis=axis)


def n_digits(number) -> int:
    """Считает число значащих целых разрядов числа."""
    if number == 0:
        order = 1
    else:
        order = np.floor(np.log10(number)).astype(int) + 1
    return order


def make_band_labels(n_bands: int) -> list:
    """Создаёт список названий столбцов для n_bands признаков.

    Args:
        n_bands: общее число предикторов, под которые нужны подписи.

    Returns:
        labels: список названий столбцов.
    """
    n_zeros = n_digits(n_bands)
    labels = ["b{band_number:0{n_zeros}d}".format(band_number=i + 1, n_zeros=n_zeros) for i in range(n_bands)]
    return labels
