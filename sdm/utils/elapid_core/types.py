# Адаптировано из elapid/types.py (https://github.com/earth-chris/elapid)
# Copyright (c) 2021 Christopher Anderson, лицензия MIT - см. THIRD_PARTY_LICENSES.md
#
# В оригинале ArrayLike/CRSType/Vector собраны через `X | Y` (Python 3.10+) и включают
# geopandas.GeoDataFrame/pyproj.CRS - нашему ядру geopandas и pyproj не нужны вообще
# (мы передаём только numpy-массивы), поэтому CRSType/Vector убраны, а ArrayLike -
# без geopandas, через typing.Union для совместимости с Python 3.9.
"""Типы данных для адаптированного ядра MaxEnt."""

from typing import Any, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame]


def get_feature_types(return_string: bool = False) -> Union[list, str]:
    feature_types = "lqpht" if return_string else ["linear", "quadratic", "product", "hinge", "threshold"]
    return feature_types


def validate_feature_types(features: Union[str, list]) -> list:
    """Проверяет, что переданные типы признаков допустимы для MaxEnt.

    Args:
        features: список или строка, должны быть из ["linear", "quadratic", "product",
            "hinge", "threshold", "auto"] либо строка "lqphta".

    Returns:
        valid_features: список допустимых значений в стандартном формате.
    """
    valid_list = get_feature_types(return_string=False)
    valid_string = get_feature_types(return_string=True)
    valid_features = list()

    if type(features) is str:
        for feature in features:
            if feature == "a":
                return valid_list
            assert feature in valid_string, f"Недопустимый тип признака: {feature}"
            if feature == "l":
                valid_features.append("linear")
            elif feature == "q":
                valid_features.append("quadratic")
            elif feature == "p":
                valid_features.append("product")
            elif feature == "h":
                valid_features.append("hinge")
            elif feature == "t":
                valid_features.append("threshold")

    elif type(features) is list:
        for feature in features:
            if feature == "auto":
                return valid_list
            assert feature in valid_list, f"Недопустимый тип признака: {feature}"
            valid_features.append(feature)

    return valid_features


def validate_boolean(var: Any) -> bool:
    assert isinstance(var, bool), "Аргумент должен быть True/False"
    return var


def validate_numeric_scalar(var: Any) -> bool:
    assert isinstance(var, (int, float)), "Аргумент должен быть одним числом"
    return var
