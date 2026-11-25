# sdm/utils/models.py
# Библиотека PythonSDM для моделирования распространения видов
# - модель MaxEnt (обёртка над адаптированным ядром elapid, см. sdm/utils/elapid_core/)
# - модель GLM

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from .elapid_core.models import MaxentModel as _ElapidMaxentModel
from .i18n import DEFAULT_LANG, t

class GLM:
    """
    Обобщённая линейная модель (логистическая регрессия) для SDM.

    Признаки - линейные и квадратичные (как в MaxEnt): чисто линейная модель может описать
    только монотонный отклик вида на предиктор, а большинство экологических откликов
    унимодальны (вид предпочитает некоторый оптимум температуры/осадков и т.д.) - квадратичный
    член необходим, чтобы в принципе смочь описать такую форму (ter Braak & Looman, 1986).
    Попарные произведения признаков (interaction terms) не добавляются по умолчанию: при
    большом числе предикторов их число растёт квадратично и увеличивает риск переобучения
    и коллинеарности сильнее, чем это оправдано для автоматического конвейера без ручного
    отбора предикторов.

    Обучение полностью делегируется sklearn.linear_model.LogisticRegressionCV: сила L2-
    регуляризации (C) подбирается кросс-валидацией на обучающих данных, а не фиксируется
    заранее - более защищаемый выбор, чем произвольная константа, и стандартная практика
    для GLM/glmnet-подобных моделей в этой области.

    Args:
        add_quadratic (bool): Включает квадратичные признаки (x^2). По умолчанию True.
        Cs (int): Число значений C (обратная сила регуляризации) в сетке кросс-валидации.
        cv (int): Число фолдов кросс-валидации для подбора C.
        class_weight (str | dict): Передаётся в LogisticRegressionCV как есть.
            'balanced' компенсирует типичный для SDM сильный перекос фон/присутствие.
        max_iter (int): Максимум итераций L-BFGS на каждую подгонку внутри кросс-валидации.
        random_state (int, optional): Семя генератора случайных чисел (для разбиения фолдов).
    """
    def __init__(self, add_quadratic=True, Cs=10, cv=5, class_weight='balanced', max_iter=1000, random_state=None, lang=DEFAULT_LANG):
        self.add_quadratic = add_quadratic
        self.Cs = Cs
        self.cv = cv
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.lang = lang

        self.n_features_orig = None
        self._clf = None
        self._feature_importances_ = None
        self.selected_C_ = None

    def _expand_features(self, X):
        """Генерация признаков (линейные, при необходимости квадратичные) - подмножество MaxEnt._expand_features."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        features = [X]
        if self.add_quadratic:
            features.append(X ** 2)
        return np.hstack(features)

    def fit(self, X, y):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_orig = X.shape[1]
        X_ext = self._expand_features(X)

        # Квадратичный член (X**2) имеет совсем другой масштаб и распределение, чем сам X
        # (неотрицателен, другая дисперсия), даже если X уже стандартизован на уровне стека
        # предикторов. Без отдельной стандартизации расширенной матрицы признаков lbfgs сходится
        # медленно или не сходится вовсе, особенно на слабой регуляризации (больших C). Масштабируем
        # X_ext целиком и тем же scaler-ом в predict_proba, а не просто увеличиваем max_iter -
        # это устраняет саму причину, а не просто ждёт дольше с тем же плохо обусловленным входом.
        self._scaler = StandardScaler()
        X_ext_scaled = self._scaler.fit_transform(X_ext)

        print(t('log_glm_training', self.lang, cv=self.cv))
        self._clf = LogisticRegressionCV(
            Cs=self.Cs, cv=self.cv, solver='lbfgs', max_iter=self.max_iter,
            class_weight=self.class_weight, random_state=self.random_state, n_jobs=-1,
            # AUC, а не accuracy: при типичном для SDM перекосе фон/присутствие (1:5...1:20)
            # accuracy почти не отличает лучшую регуляризацию от "всегда предсказывать фон".
            # Это тот же показатель, которым оценивается финальная модель (self.auc), а не
            # новый дефолт sklearn (neg_log_loss), выбран сознательно, не оставлен как есть.
            scoring='roc_auc',
        )
        self._clf.fit(X_ext_scaled, y)
        # .C_ у LogisticRegressionCV - либо массив из одного значения (на класс, старое поведение
        # sklearn), либо уже скаляр (новое поведение, параметр use_legacy_attributes - его на старых
        # версиях sklearn вообще нет, поэтому не передаём его явно и читаем .C_ в форме, устойчивой
        # к обоим вариантам, а не подгоняем код под одну конкретную версию библиотеки).
        self.selected_C_ = float(np.ravel(self._clf.C_)[0])
        print(t('log_glm_training_done', self.lang, c=self.selected_C_))

        # Важности: |коэффициент|, агрегированные из расширенных признаков обратно к исходным
        # (тот же приём, что у MaxEnt.feature_importances_, но проще - без попарных произведений)
        w_abs = np.abs(self._clf.coef_[0])
        importances = w_abs[:self.n_features_orig].copy()
        if self.add_quadratic:
            importances += w_abs[self.n_features_orig:2 * self.n_features_orig]
        sum_imp = np.sum(importances)
        self._feature_importances_ = (importances / sum_imp) if sum_imp > 0 else importances

        return self

    @property
    def feature_importances_(self):
        if self._feature_importances_ is None:
            raise RuntimeError(t('log_err_model_not_fitted', self.lang))
        return self._feature_importances_

    def predict_proba(self, X):
        if self._clf is None:
            raise RuntimeError(t('log_err_model_not_fitted', self.lang))
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_orig:
            raise ValueError(t('log_err_unexpected_feature_count', self.lang, expected=self.n_features_orig, actual=X.shape[1]))
        return self._clf.predict_proba(self._scaler.transform(self._expand_features(X)))


class MaxEnt:
    """
    Модель MaxEnt - обёртка над адаптированным ядром elapid (sdm/utils/elapid_core/),
    настоящей реализацией MaxEnt (linear/hinge/threshold/product-признаки, L1-регуляризация,
    калибровка через alpha/entropy), а не упрощённой логистической регрессией с квадратами,
    как было раньше.

    Добавляет единственное, чего нет в самом elapid: .feature_importances_, нужный всему
    остальному пайплайну (sdm.py читает его без защиты от отсутствия для любого алгоритма).
    Считается через перестановочную важность (permutation importance) - элапид её уже
    умеет считать в разрезе исходных предикторов, без необходимости разбирать расширенные
    hinge/threshold-признаки обратно самостоятельно.

    Args:
        feature_types (list|str): Типы признаков MaxEnt. По умолчанию - то же, что в elapid:
            ["linear", "hinge", "product"].
        beta_multiplier (float): Сила регуляризации (выше - больше коэффициентов отбрасывается).
        class_weights (str|float): Взвешивание присутствие/фон, см. elapid_core/config.py.
        random_state (int, optional): Семя генератора случайных чисел.
        n_importance_repeats (int): Число повторов при расчёте перестановочной важности.
            Каждый повтор - это n_features дополнительных пересчётов predict_proba по всей
            обучающей выборке, так что при большом числе предикторов стоит иметь это в виду.
    """
    def __init__(self, feature_types=None, beta_multiplier=None, class_weights=None,
                 random_state=None, n_importance_repeats=10, lang=DEFAULT_LANG, **maxent_kwargs):
        self.random_state = random_state
        self.n_importance_repeats = n_importance_repeats
        self.lang = lang
        self._maxent_kwargs = dict(maxent_kwargs)
        if feature_types is not None:
            self._maxent_kwargs['feature_types'] = feature_types
        if beta_multiplier is not None:
            self._maxent_kwargs['beta_multiplier'] = beta_multiplier
        if class_weights is not None:
            self._maxent_kwargs['class_weights'] = class_weights

        self._model = None
        self.n_features_orig = None
        self._feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_orig = X.shape[1]

        self._model = _ElapidMaxentModel(random_state=self.random_state, **self._maxent_kwargs)
        print(t('log_maxent_training', self.lang, features=self._model.feature_types, beta=self._model.beta_multiplier))
        self._model.fit(X, y)
        print(t('log_maxent_training_done', self.lang))

        imp = self._model.permutation_importance_scores(X, y, n_repeats=self.n_importance_repeats).mean(axis=1)
        # Перестановочная важность - это разница в AUC при порче признака, в теории может
        # оказаться чуть отрицательной (шум) для совсем неважного предиктора - обнуляем,
        # отрицательной "важности" в выдаче быть не должно.
        imp = np.clip(imp, 0, None)
        sum_imp = np.sum(imp)
        self._feature_importances_ = (imp / sum_imp) if sum_imp > 0 else imp

        return self

    @property
    def feature_importances_(self):
        if self._feature_importances_ is None:
            raise RuntimeError(t('log_err_model_not_fitted', self.lang))
        return self._feature_importances_

    # Делегирующие свойства - чтобы sdm.py мог читать их прямо с self.model (как у GLM),
    # не обращаясь к "приватному" self.model._model.
    @property
    def feature_types(self):
        return self._model.feature_types if self._model is not None else self._maxent_kwargs.get('feature_types')

    @property
    def beta_multiplier(self):
        return self._model.beta_multiplier if self._model is not None else self._maxent_kwargs.get('beta_multiplier')

    @property
    def class_weights(self):
        return self._model.class_weights if self._model is not None else self._maxent_kwargs.get('class_weights')

    def predict_proba(self, X):
        if self._model is None:
            raise RuntimeError(t('log_err_model_not_fitted', self.lang))
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_orig:
            raise ValueError(t('log_err_unexpected_feature_count', self.lang, expected=self.n_features_orig, actual=X.shape[1]))
        return self._model.predict_proba(X)
