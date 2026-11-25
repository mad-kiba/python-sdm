# Адаптировано из elapid/config.py (https://github.com/earth-chris/elapid)
# Copyright (c) 2021 Christopher Anderson, лицензия MIT - см. THIRD_PARTY_LICENSES.md
#
# Убраны RegularizationConfig, NicheEnvelopeConfig, EnsembleConfig - не нужны
# в урезанном sklearn-only ядре (см. sdm/utils/elapid_core/__init__.py).
"""Параметры конфигурации модели MaxEnt."""

from typing import Union


class MaxentConfig:
    # ограничивает диапазон значений признаков min/max
    clamp: bool = True

    # регуляризация beta отбрасывает коэффициенты признаков, если они не улучшают
    # качество модели. эти множители настраивают силу регуляризации.
    # увеличьте значение, чтобы отбросить больше коэффициентов, уменьшите, чтобы
    # оставить больше. beta_multiplier масштабирует регуляризацию для всех признаков.
    #
    # ВАЖНО: в оригинальном elapid здесь были ещё beta_hinge/beta_lqp/beta_threshold/
    # beta_categorical для раздельной настройки регуляризации по типу признака - но
    # они используются только в glmnet-пути (через compute_regularization), которого
    # в этом урезанном ядре нет. В sklearn-пути (единственном здесь) работает только
    # beta_multiplier - он единственный осмысленный параметр регуляризации.
    beta_multiplier: float = 1.5

    # какие типы признаков MaxEnt считать по умолчанию
    feature_types: list = ["linear", "hinge", "product"]

    # число hinge- и threshold-признаков на каждый предиктор
    n_hinge_features: int = 10
    n_threshold_features: int = 10

    # метрика обучения модели (из `sklearn.metrics`)
    scorer: str = "roc_auc"

    # масштабный коэффициент распространённости вида
    tau: float = 0.5
    transform: str = "cloglog"

    # порог сходимости модели
    tolerance: float = 2e-6

    # предел итераций для sklearn-решателя liblinear
    max_iter: int = 1000

    # метод взвешивания точек присутствия.
    # "balanced" нормирует веса по плотности выборки, либо укажите число -
    # соотношение вес(присутствие):вес(фон). В R-пакете `maxnet` используется 100.
    class_weights: Union[str, float] = 100
