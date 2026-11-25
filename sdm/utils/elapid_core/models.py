# Адаптировано из elapid/models.py (https://github.com/earth-chris/elapid)
# Copyright (c) 2021 Christopher Anderson, лицензия MIT - см. THIRD_PARTY_LICENSES.md
#
# Отличия от оригинала (см. также sdm/utils/elapid_core/__init__.py):
# - убран путь через glmnet (initialize_glmnet_model, ветка else в fit()) - у нас
#   glmnet нет, используется только sklearn-путь (он и в оригинале используется по
#   умолчанию при отсутствии glmnet);
# - убраны NicheEnvelopeModel, EnsembleModel, графические и partial_dependence-методы
#   SDMMixin (нужен только matplotlib.pyplot/sklearn.inspection.partial_dependence,
#   не нужны нашему пайплайну) - оставлен только permutation_importance_scores;
# - `feature_types: list | str`, `class_weights: str | float` (Python 3.10+) заменены
#   на `typing.Union[...]`;
# - ВАЖНО, исправление: в initialize_sklearn_model добавлен явный penalty='l1'.
#   В оригинале передаётся только l1_ratio=1.0 без penalty - на sklearn>=1.8 этого
#   достаточно для L1-регуляризации (новый единый API), но на более старых версиях
#   (проверено эмпирически на нашем сервере) l1_ratio без penalty='elasticnet' тихо
#   игнорируется, и фит получается обычным L2 без единой предупреждающей строки -
#   то есть другая, менее разреженная модель без видимой ошибки. penalty='l1' даёт
#   одинаковый (проверено) результат на обоих сериях версий sklearn.
"""Класс модели MaxEnt."""

from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import MaxentConfig
from .features import MaxentFeatureTransformer, compute_weights
from .types import ArrayLike, validate_feature_types
from .utils import NCPUS


class SDMMixin(ClassifierMixin):
    """Миксин для SDM-классификаторов.

    Наследуется от sklearn ClassifierMixin, чтобы тэги estimator/classifier были
    подключены через __sklearn_tags__. В дочерних классах этот миксин должен идти
    *перед* BaseEstimator в списке базовых классов, чтобы переопределение тэгов
    имело приоритет над BaseEstimator.__sklearn_tags__ при поиске по MRO.
    """

    classes_ = np.array([0, 1])

    def score(self, x: ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None) -> float:
        """Возвращает средний AUC на переданных тестовых данных и метках.

        Переопределяет ClassifierMixin.score (которая использует accuracy) на ROC AUC -
        естественную метрику для SDM-моделей присутствие/фон.
        """
        return roc_auc_score(y, self.predict(x), sample_weight=sample_weight)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def permutation_importance_scores(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike = None,
        n_repeats: int = 10,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """Считает перестановочную важность признаков: насколько падает качество модели
        при случайном перемешивании значений одного признака.

        Эта оценка не отражает внутреннюю предсказательную ценность признака сама по
        себе, а то, насколько он важен именно для этой обученной модели.

        Args:
            x: тестовые образцы, массив формы (n_samples, n_features).
            y: метки присутствие/фон, массив формы (n_samples,).
            sample_weight: веса образцов, массив формы (n_samples,).
            n_repeats: число повторов перестановки.
            n_jobs: число параллельных потоков, -1 - все ядра.

        Returns:
            importances: массив формы (n_features, n_repeats).
        """
        pi = permutation_importance(self, x, y, sample_weight=sample_weight, n_jobs=n_jobs, n_repeats=n_repeats)
        return pi.importances


class MaxentModel(SDMMixin, BaseEstimator):
    """Модель MaxEnt (Maximum Entropy) для моделирования распространения видов."""

    def __init__(
        self,
        feature_types: Union[list, str] = MaxentConfig.feature_types,
        tau: float = MaxentConfig.tau,
        transform: str = MaxentConfig.transform,
        clamp: bool = MaxentConfig.clamp,
        scorer: str = MaxentConfig.scorer,
        beta_multiplier: float = MaxentConfig.beta_multiplier,
        n_hinge_features: int = MaxentConfig.n_hinge_features,
        n_threshold_features: int = MaxentConfig.n_threshold_features,
        convergence_tolerance: float = MaxentConfig.tolerance,
        max_iter: int = MaxentConfig.max_iter,
        class_weights: Union[str, float] = MaxentConfig.class_weights,
        n_cpus: int = NCPUS,
        random_state: int = None,
    ):
        """Создаёт объект модели MaxEnt.

        Args:
            feature_types: типы признаков MaxEnt для обучения. Строка "lqphta" или список
                ["linear", "quadratic", "product", "hinge", "threshold", "auto"].
            tau: масштабный коэффициент распространённости вида для логистического вывода.
            transform: тип выходного преобразования модели. Из ["raw", "logistic", "cloglog"].
            clamp: ограничивать значения признаков диапазоном из обучения при предсказании.
            scorer: метрика sklearn для обучения модели.
            beta_multiplier: масштаб регуляризации. Выше значение - больше коэффициентов
                отбрасывается. (В этом урезанном, sklearn-only ядре - единственный
                содержательный параметр регуляризации, см. комментарий в config.py).
            n_hinge_features: число узлов hinge-признаков на каждый предиктор.
            n_threshold_features: число пороговых признаков на каждый предиктор.
            convergence_tolerance: порог сходимости модели.
            max_iter: предел итераций для sklearn-решателя liblinear.
            class_weights: способ взвешивания точек присутствия. "balanced" считает
                соотношение по частоте выборки, либо передайте число - соотношение
                весов присутствие:фон. В R-пакете `maxnet` по умолчанию 100.
                None - не взвешивать.
            n_cpus: число потоков при обучении.
            random_state: семя генератора случайных чисел для воспроизводимости.
        """
        self.feature_types = feature_types
        self.tau = tau
        self.transform = transform
        self.clamp = clamp
        self.scorer = scorer
        self.beta_multiplier = beta_multiplier
        self.n_hinge_features = n_hinge_features
        self.n_threshold_features = n_threshold_features
        self.convergence_tolerance = convergence_tolerance
        self.n_cpus = n_cpus
        self.max_iter = max_iter
        self.class_weights = class_weights
        self.random_state = random_state

        # вычисляется при обучении
        self.initialized_ = False
        self.estimator = None
        self.transformer = None
        self.beta_scores_ = None
        self.entropy_ = 0.0
        self.alpha_ = 0.0

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike = None,
        categorical: list = None,
        labels: list = None,
    ) -> "MaxentModel":
        """Обучает модель MaxEnt на предикторах и точках присутствия/фона.

        Args:
            x: массив формы (n_samples, n_features) с данными предикторов.
            y: массив формы (n_samples,) с бинарными значениями присутствие/фон (1/0).
            sample_weight: веса образцов, массив формы (n_samples,). Модифицируется
                параметром class_weights, если он не None.
            categorical: индексы категориальных столбцов.
            labels: подписи предикторов. Игнорируется, если x - pandas DataFrame.
        """
        self.alpha_ = 0.0
        self.entropy_ = 0.0

        y = format_occurrence_data(y)
        x_raw = x

        self.feature_types = validate_feature_types(self.feature_types)
        self.transformer = MaxentFeatureTransformer(
            feature_types=self.feature_types,
            clamp=self.clamp,
            n_hinge_features=self.n_hinge_features,
            n_threshold_features=self.n_threshold_features,
        )
        features = self.transformer.fit_transform(x, categorical=categorical, labels=labels)

        if self.class_weights is not None:
            pbr = len(y) / y.sum() if self.class_weights == "balanced" else self.class_weights
            class_weight = compute_weights(y, pbr=pbr)
            if sample_weight is None:
                sample_weight = class_weight
            else:
                sample_weight = sample_weight * class_weight

        C = estimate_C_from_betas(self.beta_multiplier)
        self.initialize_sklearn_model(C)
        self.estimator.fit(features, y, sample_weight=sample_weight)
        self.beta_scores_ = self.estimator.coef_[0]

        self.initialized_ = True

        # калибровка в стиле MaxEnt
        class_transform = self.get_params()["transform"]
        self.set_params(transform="raw")
        raw = self.predict(x_raw[y == 0])
        self.set_params(transform=class_transform)

        # alpha - нормирующая константа, обеспечивающая сумму f1(z) к 1
        self.alpha_ = maxent_alpha(raw)
        # энтропия - относительная энтропия f1(z) относительно фонового распределения f(z)
        self.entropy_ = maxent_entropy(raw)

        return self

    def predict(self, x: ArrayLike) -> ArrayLike:
        """Применяет обученную модель к набору предикторов.

        Args:
            x: массив формы (n_samples, n_features) с данными предикторов.

        Returns:
            predictions: массив формы (n_samples,) с предсказаниями модели.
        """
        if not self.initialized_:
            raise NotFittedError("Модель должна быть сначала обучена (.fit())")

        features = self.transformer.transform(x)
        engma = np.matmul(features, self.beta_scores_) + self.alpha_

        if self.transform == "raw":
            return maxent_raw_transform(engma)
        elif self.transform == "logistic":
            return maxent_logistic_transform(engma, self.entropy_, self.tau)
        elif self.transform == "cloglog":
            return maxent_cloglog_transform(engma, self.entropy_)

    def predict_proba(self, x: ArrayLike) -> ArrayLike:
        """Вычисляет вероятности предсказания для классов 0/1.

        Args:
            x: массив формы (n_samples, n_features) с данными предикторов.

        Returns:
            predictions: массив формы (n_samples, 2) с вероятностями предсказания.
        """
        ypred = self.predict(x).reshape(-1, 1)
        predictions = np.hstack((1 - ypred, ypred))
        return predictions

    def initialize_sklearn_model(self, C: float, fit_intercept: bool = True) -> None:
        """Создаёт sklearn LogisticRegression с L1-регуляризацией.

        Args:
            C: параметр силы регуляризации (обратный).
            fit_intercept: учитывать свободный член.
        """
        self.estimator = LogisticRegression(
            C=C,
            fit_intercept=fit_intercept,
            # penalty='l1' - для версий sklearn, где это основной способ задать L1 (см.
            # комментарий в начале файла). l1_ratio=1.0 - для версий sklearn, где penalty
            # уже объявлен устаревшим в пользу l1_ratio; без него новые версии sklearn
            # предупреждают о несогласованности (l1_ratio по умолчанию 0.0 означает L2).
            # Проверено эмпирически: вместе дают одинаковый результат на обеих линейках.
            penalty="l1",
            l1_ratio=1.0,
            solver="liblinear",
            tol=self.convergence_tolerance,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )


def maxent_alpha(raw: np.ndarray) -> float:
    """Вычисляет нормирующую константу alpha модели MaxEnt."""
    return -np.log(np.sum(raw))


def maxent_entropy(raw: np.ndarray) -> float:
    """Вычисляет показатель энтропии модели MaxEnt для масштабирования логистического вывода."""
    scaled = raw / np.sum(raw)
    return -np.sum(scaled * np.log(scaled))


def maxent_raw_transform(engma: np.ndarray) -> np.ndarray:
    """Считает «сырую» оценку пригодности MaxEnt."""
    return np.exp(engma)


def maxent_logistic_transform(engma: np.ndarray, entropy: float, tau: float = MaxentConfig.tau) -> np.ndarray:
    """Считает логистическую оценку пригодности MaxEnt."""
    logratio = np.exp(engma) * np.exp(entropy)
    return (tau * logratio) / ((1 - tau) + (tau * logratio))


def maxent_cloglog_transform(engma: np.ndarray, entropy: float) -> np.ndarray:
    """Считает cloglog-оценку пригодности MaxEnt."""
    return 1 - np.exp(-np.exp(engma) * np.exp(entropy))


def format_occurrence_data(y: ArrayLike) -> ArrayLike:
    """Приводит y к одномерному массиву uint8.

    Raises:
        ValueError: если передан массив с 2+ столбцами.
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if y.ndim > 1:
        if y.shape[1] > 1 or y.ndim > 2:
            raise ValueError(f"Передан y с несколькими столбцами, форма {y.shape}. Должен быть 1D или 1 столбец.")
        y = y.flatten()

    return y.astype("uint8")


def estimate_C_from_betas(beta_multiplier: float) -> float:
    """Переводит параметр регуляризации beta_multiplier формата MaxEnt в параметр C формата sklearn."""
    return 2 / (1 - np.exp(-beta_multiplier))
