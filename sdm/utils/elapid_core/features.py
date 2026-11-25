# Адаптировано из elapid/features.py (https://github.com/earth-chris/elapid)
# Copyright (c) 2021 Christopher Anderson, лицензия MIT - см. THIRD_PARTY_LICENSES.md
#
# Отличия от оригинала:
# - убраны compute_regularization/compute_lambdas - используются только в
#   glmnet-пути MaxentModel.fit(), которого в этом урезанном ядре нет;
# - `feature_types: str | list` (Python 3.10+) заменено на `Union[str, list]`;
# - импорты elapid.config/elapid.types/elapid.utils заменены на локальные модули
#   этого пакета (sdm/utils/elapid_core/).
"""Функции преобразования предикторов в составные признаки модели MaxEnt."""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer

from .config import MaxentConfig
from .types import ArrayLike, validate_boolean, validate_feature_types, validate_numeric_scalar
from .utils import make_band_labels, repeat_array


class FeaturesMixin:
    """Методы форматирования данных x и подписей."""

    def _format_covariate_data(self, x: ArrayLike):
        """Читает входные данные x и приводит их к единому формату массивов.

        Args:
            x: массив формы (n_samples, n_features).

        Returns:
            (continuous, categorical) - кортеж массивов с непрерывными и
                категориальными данными предикторов.
        """
        if isinstance(x, np.ndarray):
            if self.categorical_ is None:
                con = x
                cat = None
            else:
                con = x[:, self.continuous_]
                cat = x[:, self.categorical_]

        elif isinstance(x, pd.DataFrame):
            con = x[self.continuous_pd_].to_numpy()
            if len(self.categorical_pd_) > 0:
                cat = x[self.categorical_pd_].to_numpy()
            else:
                cat = None

        else:
            raise TypeError(f"Неподдерживаемый тип x: {type(x)}. Должен быть pd.DataFrame или np.array")

        return con, cat

    def _format_labels_and_dtypes(self, x: ArrayLike, categorical: list = None, labels: list = None) -> None:
        """Читает x и индексы категориальных столбцов/подписи, сохраняет для дальнейшего использования.

        Args:
            x: массив формы (n_samples, n_features).
            categorical: индексы столбцов x, которые являются категориальными.
            labels: подписи столбцов предикторов. Игнорируется, если x - pandas DataFrame.
        """
        if isinstance(x, np.ndarray):
            nrows, ncols = x.shape
            if categorical is None:
                continuous = list(range(ncols))
            else:
                continuous = list(set(range(ncols)).difference(set(categorical)))
            self.labels_ = labels or make_band_labels(ncols)
            self.categorical_ = categorical
            self.continuous_ = continuous

        elif isinstance(x, pd.DataFrame):
            x = x.drop(["geometry"], axis=1, errors="ignore")
            self.labels_ = labels or list(x.columns)

            self.continuous_pd_ = list(x.select_dtypes(exclude="category").columns)
            self.categorical_pd_ = list(x.select_dtypes(include="category").columns)

            all_columns = list(x.columns)
            self.continuous_ = [all_columns.index(item) for item in self.continuous_pd_ if item in all_columns]
            if len(self.categorical_pd_) != 0:
                self.categorical_ = [all_columns.index(item) for item in self.categorical_pd_ if item in all_columns]
            else:
                self.categorical_ = None


class LinearTransformer(MinMaxScaler):
    """Линейное преобразование признаков, масштабирует в диапазон 0-1."""

    def __init__(self, clamp: bool = MaxentConfig.clamp, feature_range: tuple = (0.0, 1.0)):
        self.clamp = clamp
        self.feature_range = feature_range
        super().__init__(clip=clamp, feature_range=feature_range)


class QuadraticTransformer(BaseEstimator, TransformerMixin):
    """Квадратичное преобразование признаков с масштабированием в 0-1."""

    def __init__(self, clamp: bool = MaxentConfig.clamp, feature_range: tuple = (0.0, 1.0)):
        self.clamp = clamp
        self.feature_range = feature_range
        self.estimator = None

    def fit(self, x: ArrayLike) -> "QuadraticTransformer":
        self.estimator = MinMaxScaler(clip=self.clamp, feature_range=self.feature_range)
        self.estimator.fit(np.array(x) ** 2)
        return self

    def transform(self, x: ArrayLike) -> np.ndarray:
        return self.estimator.transform(np.array(x) ** 2)

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        self.fit(x)
        return self.estimator.transform(np.array(x) ** 2)

    def inverse_transform(self, x: ArrayLike) -> np.ndarray:
        return self.estimator.inverse_transform(np.array(x)) ** 0.5


class ProductTransformer(BaseEstimator, TransformerMixin):
    """Попарное произведение столбцов признаков, с масштабированием в 0-1."""

    def __init__(self, clamp: bool = MaxentConfig.clamp, feature_range: tuple = (0.0, 1.0)):
        self.clamp = clamp
        self.feature_range = feature_range
        self.estimator = None

    def fit(self, x: ArrayLike) -> "ProductTransformer":
        self.estimator = MinMaxScaler(clip=self.clamp, feature_range=self.feature_range)
        self.estimator.fit(column_product(np.array(x)))
        return self

    def transform(self, x: ArrayLike) -> np.ndarray:
        return self.estimator.transform(column_product(np.array(x)))

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class ThresholdTransformer(BaseEstimator, TransformerMixin):
    """Бинарные пороги по равномерно расставленным интервалам для каждого предиктора."""

    def __init__(self, n_thresholds: int = MaxentConfig.n_threshold_features):
        self.n_thresholds = n_thresholds
        self.mins_ = None
        self.maxs_ = None
        self.threshold_indices_ = None

    def fit(self, x: ArrayLike) -> "ThresholdTransformer":
        x = np.array(x)
        self.mins_ = x.min(axis=0)
        self.maxs_ = x.max(axis=0)
        self.threshold_indices_ = np.linspace(self.mins_, self.maxs_, self.n_thresholds)
        return self

    def transform(self, x: ArrayLike) -> np.ndarray:
        x = np.array(x)
        xarr = repeat_array(x, len(self.threshold_indices_), axis=-1)
        tarr = repeat_array(self.threshold_indices_.transpose(), len(x), axis=0)
        thresh = (xarr > tarr).reshape(x.shape[0], -1)
        return thresh.astype(np.uint8)

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class HingeTransformer(BaseEstimator, TransformerMixin):
    """Hinge-преобразование предикторов."""

    def __init__(self, n_hinges: int = MaxentConfig.n_hinge_features):
        self.n_hinges = n_hinges
        self.mins_ = None
        self.maxs_ = None
        self.hinge_indices_ = None

    def fit(self, x: ArrayLike) -> "HingeTransformer":
        x = np.array(x)
        self.mins_ = x.min(axis=0)
        self.maxs_ = x.max(axis=0)
        self.hinge_indices_ = np.linspace(self.mins_, self.maxs_, self.n_hinges)
        return self

    def transform(self, x: ArrayLike) -> np.ndarray:
        x = np.array(x)
        xarr = repeat_array(x, self.n_hinges - 1, axis=-1)
        lharr = repeat_array(self.hinge_indices_[:-1].transpose(), len(x), axis=0)
        rharr = repeat_array(self.hinge_indices_[1:].transpose(), len(x), axis=0)
        lh = left_hinge(xarr, lharr, self.maxs_)
        rh = right_hinge(xarr, self.mins_, rharr)
        return np.concatenate((lh, rh), axis=2).reshape(x.shape[0], -1)

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """One-hot кодирование категориальных предикторов."""

    def __init__(self):
        self.estimators_ = None

    def fit(self, x: ArrayLike) -> "CategoricalTransformer":
        self.estimators_ = []
        x = np.array(x)
        if x.ndim == 1:
            estimator = OneHotEncoder(dtype=np.uint8, sparse_output=False)
            self.estimators_.append(estimator.fit(x.reshape(-1, 1)))
        else:
            nrows, ncols = x.shape
            for col in range(ncols):
                xsub = x[:, col].reshape(-1, 1)
                estimator = OneHotEncoder(dtype=np.uint8, sparse_output=False)
                self.estimators_.append(estimator.fit(xsub))
        return self

    def transform(self, x: ArrayLike) -> np.ndarray:
        x = np.array(x)
        if x.ndim == 1:
            estimator = self.estimators_[0]
            return estimator.transform(x.reshape(-1, 1))
        else:
            class_data = []
            nrows, ncols = x.shape
            for col in range(ncols):
                xsub = x[:, col].reshape(-1, 1)
                estimator = self.estimators_[col]
                class_data.append(estimator.transform(xsub))
            return np.concatenate(class_data, axis=1)

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class CumulativeTransformer(QuantileTransformer):
    """Процентильное преобразование для оценки кумулятивной пригодности."""

    def __init__(self):
        super().__init__(n_quantiles=100, output_distribution="uniform")


class MaxentFeatureTransformer(BaseEstimator, TransformerMixin, FeaturesMixin):
    """Преобразует предикторы в признаки формата MaxEnt."""

    def __init__(
        self,
        feature_types: Union[str, list] = MaxentConfig.feature_types,
        clamp: bool = MaxentConfig.clamp,
        n_hinge_features: int = MaxentConfig.n_hinge_features,
        n_threshold_features: int = MaxentConfig.n_threshold_features,
    ):
        """Вычисляет признаки по заданным типам MaxEnt (linear, quadratic, hinge и т.д.).

        Args:
            feature_types: список типов признаков MaxEnt для генерации.
            clamp: ограничивать значения признаков диапазоном min/max из обучения при предсказании.
            n_hinge_features: число узлов hinge-признаков.
            n_threshold_features: число пороговых признаков.
        """
        self.feature_types = feature_types
        self.clamp = clamp
        self.n_hinge_features = n_hinge_features
        self.n_threshold_features = n_threshold_features
        self.categorical_ = None
        self.continuous_ = None
        self.categorical_pd_ = None
        self.continuous_pd_ = None
        self.labels_ = None
        self.feature_names_ = None
        self.estimators_ = {
            "linear": None,
            "quadratic": None,
            "product": None,
            "threshold": None,
            "hinge": None,
            "categorical": None,
        }

    def fit(self, x: ArrayLike, categorical: list = None, labels: list = None) -> "MaxentFeatureTransformer":
        self.feature_types = validate_feature_types(self.feature_types)
        self.clamp = validate_boolean(self.clamp)
        self.n_hinge_features = validate_numeric_scalar(self.n_hinge_features)
        self.n_threshold_features = validate_numeric_scalar(self.n_threshold_features)

        self._format_labels_and_dtypes(x, categorical=categorical, labels=labels)
        con, cat = self._format_covariate_data(x)
        nrows, ncols = con.shape

        feature_names = []
        if "linear" in self.feature_types:
            estimator = LinearTransformer(clamp=self.clamp)
            estimator.fit(con)
            self.estimators_["linear"] = estimator
            feature_names += ["linear"] * estimator.n_features_in_

        if "quadratic" in self.feature_types:
            estimator = QuadraticTransformer(clamp=self.clamp)
            estimator.fit(con)
            self.estimators_["quadratic"] = estimator
            feature_names += ["quadratic"] * estimator.estimator.n_features_in_

        if "product" in self.feature_types:
            estimator = ProductTransformer(clamp=self.clamp)
            estimator.fit(con)
            self.estimators_["product"] = estimator
            feature_names += ["product"] * estimator.estimator.n_features_in_

        if "threshold" in self.feature_types:
            estimator = ThresholdTransformer(n_thresholds=self.n_threshold_features)
            estimator.fit(con)
            self.estimators_["threshold"] = estimator
            feature_names += ["threshold"] * (estimator.n_thresholds * ncols)

        if "hinge" in self.feature_types:
            estimator = HingeTransformer(n_hinges=self.n_hinge_features)
            estimator.fit(con)
            self.estimators_["hinge"] = estimator
            feature_names += ["hinge"] * ((estimator.n_hinges - 1) * 2 * ncols)

        if cat is not None:
            estimator = CategoricalTransformer()
            estimator.fit(cat)
            self.estimators_["categorical"] = estimator
            for est in estimator.estimators_:
                feature_names += ["categorical"] * len(est.categories_[0])

        self.feature_names_ = feature_names

        return self

    def transform(self, x: ArrayLike) -> np.ndarray:
        con, cat = self._format_covariate_data(x)
        features = []

        if "linear" in self.feature_types:
            features.append(self.estimators_["linear"].transform(con))

        if "quadratic" in self.feature_types:
            features.append(self.estimators_["quadratic"].transform(con))

        if "product" in self.feature_types:
            features.append(self.estimators_["product"].transform(con))

        if "threshold" in self.feature_types:
            features.append(self.estimators_["threshold"].transform(con))

        if "hinge" in self.feature_types:
            features.append(self.estimators_["hinge"].transform(con))

        if cat is not None:
            features.append(self.estimators_["categorical"].transform(cat))

        return np.concatenate(features, axis=1)

    def fit_transform(self, x: ArrayLike, categorical: list = None, labels: list = None) -> np.ndarray:
        self.fit(x, categorical=categorical, labels=labels)
        return self.transform(x)


# --- вспомогательные функции ---


def column_product(array: np.ndarray) -> np.ndarray:
    """Вычисляет попарное произведение столбцов 2D-массива."""
    nrows, ncols = array.shape

    if ncols == 1:
        return array
    else:
        products = []
        for xstart in range(0, ncols - 1):
            products.append(array[:, xstart].reshape(nrows, 1) * array[:, xstart + 1 :])
        return np.concatenate(products, axis=1)


def left_hinge(x: ArrayLike, mn, mx) -> np.ndarray:
    """Вычисляет значения левого hinge-преобразования."""
    rng = repeat_array(mx, mn.shape[-1], axis=1) - mn
    valid = rng > 0
    out = np.zeros_like(x - mn, dtype=np.float64)
    hinge = np.clip(np.divide((x - mn), rng, where=valid, out=out), 0, 1)
    return hinge


def right_hinge(x: ArrayLike, mn, mx) -> np.ndarray:
    """Вычисляет значения правого hinge-преобразования."""
    mnr = repeat_array(mn, mx.shape[-1], axis=1)
    rng = mx - mnr
    valid = rng > 0
    out = np.zeros_like(x - mnr, dtype=np.float64)
    hinge = np.clip(np.divide((x - mnr), rng, where=valid, out=out), 0, 1)
    return hinge


def compute_weights(y: ArrayLike, pbr: float = 100) -> np.ndarray:
    """Вычисляет веса образцов в формате MaxEnt.

    Args:
        y: массив формы (n_samples,) с бинарными значениями присутствие/фон (1/0).
        pbr: соотношение весов присутствие:фон. pbr=100 задаёт фону вес в 100 раз меньше присутствия.

    Returns:
        weights: массив весов образцов.
    """
    weights = np.array(y + (1 - y) * pbr)
    return weights
