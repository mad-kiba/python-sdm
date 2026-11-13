# sdm/utils/models.py
# Библиотека PythonSDM для моделирования распространения видов
# - модель MaxEnt
# - модель GLM

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

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
    def __init__(self, add_quadratic=True, Cs=10, cv=5, class_weight='balanced', max_iter=1000, random_state=None):
        self.add_quadratic = add_quadratic
        self.Cs = Cs
        self.cv = cv
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

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

        print(f"Обучение GLM (логистическая регрессия, подбор C по {self.cv}-фолдовой кросс-валидации)...")
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
        print(f"Обучение GLM завершено. Выбранная сила регуляризации C = {self.selected_C_:.5g}")

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
            raise RuntimeError("Модель не была обучена. Вызовите метод .fit() сначала.")
        return self._feature_importances_

    def predict_proba(self, X):
        if self._clf is None:
            raise RuntimeError("Модель не была обучена. Вызовите метод .fit() сначала.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_orig:
            raise ValueError(f"Ожидалось {self.n_features_orig} исходных признаков, получено {X.shape[1]}.")
        return self._clf.predict_proba(self._scaler.transform(self._expand_features(X)))


class MaxEnt:
    """
    Реализация модели максимальной энтропии (MaxEnt), максимально приближенная
    к классическому алгоритму (генерация сложных признаков, L2-регуляризация
    и калибровка логистического выхода).

    Args:
        X_pres (np.ndarray): Массив предикторов в точках присутствия.
                             Форма: (n_pres, n_features).
        X_bg (np.ndarray): Массив предикторов в фоновых точках.
                           Форма: (n_bg, n_features).
        add_quadratic (bool): Включает генерацию квадратичных фичей (x^2).
        add_product (bool): Включает попарные произведения фичей (x_i * x_j).
        reg_lambda (float): Коэффициент L2-регуляризации для предотвращения переобучения.
    """
    def __init__(self, X_pres, X_bg, add_quadratic=True, add_product=False, reg_lambda=0.01):
        # Принудительно используем float64 для избежания проблем с точностью в scipy.optimize
        self.X_pres = np.asarray(X_pres, dtype=np.float64)
        self.X_bg = np.asarray(X_bg, dtype=np.float64)

        if self.X_pres.ndim == 1:
            self.X_pres = self.X_pres.reshape(-1, 1)
        if self.X_bg.ndim == 1:
            self.X_bg = self.X_bg.reshape(-1, 1)

        if self.X_pres.shape[1] != self.X_bg.shape[1]:
            raise ValueError("Количество признаков в X_pres и X_bg должно совпадать.")

        self.n_pres = self.X_pres.shape[0]
        self.n_bg = self.X_bg.shape[0]
        self.n_features_orig = self.X_pres.shape[1]
        
        self.add_quadratic = add_quadratic
        self.add_product = add_product
        self.reg_lambda = reg_lambda

        self.X_train = np.vstack((self.X_pres, self.X_bg))
        self.y_train = np.array([1] * self.n_pres + [0] * self.n_bg)
        self.n_samples = self.X_train.shape[0]

        self.weights = None
        self._feature_importances_ = None
        self.log_Z_ = 0.0
        self.entropy_ = 0.0
        self.feature_mapping = []

    def _expand_features(self, X):
        """Генерация сложных фичей (линейные, квадратичные, произведения)."""
        features = [X]
        if self.add_quadratic:
            features.append(X ** 2)
            
        if self.add_product:
            prods = []
            for i in range(self.n_features_orig):
                for j in range(i + 1, self.n_features_orig):
                    prods.append(X[:, i] * X[:, j])
            if prods:
                features.append(np.column_stack(prods))
                
        return np.hstack(features)

    def _get_feature_mapping(self):
        """Связывает индексы расширенных фичей с оригинальными для оценки важности."""
        mapping = []
        # Linear
        for i in range(self.n_features_orig):
            mapping.append([i])
        # Quadratic
        if self.add_quadratic:
            for i in range(self.n_features_orig):
                mapping.append([i])
        # Product
        if self.add_product:
            for i in range(self.n_features_orig):
                for j in range(i + 1, self.n_features_orig):
                    mapping.append([i, j])
        return mapping

    def _sigmoid(self, z, max_val=700):
        """Численно стабильная сигмоида."""
        z_clipped = np.clip(z, -max_val, max_val)
        return np.where(z_clipped >= 0, 1 / (1 + np.exp(-z_clipped)), np.exp(z_clipped) / (np.exp(z_clipped) + 1))

    def _objective_function(self, weights):
        """Целевая функция с расчетом аналитического градиента."""
        w = weights[:-1]
        b = weights[-1]
        
        z = np.dot(self.X_train_ext, w) + b
        p = self._sigmoid(z)

        epsilon = 1e-9
        p_clip = np.clip(p, epsilon, 1. - epsilon)

        # Кросс-энтропия
        loss = -np.mean(self.y_train * np.log(p_clip) + (1 - self.y_train) * np.log(1 - p_clip))
        
        # L2-регуляризация (Ridge)
        loss += self.reg_lambda * np.sum(w ** 2)

        # Расчет градиента
        error = p - self.y_train
        dw = np.dot(self.X_train_ext.T, error) / self.n_samples
        dw += 2 * self.reg_lambda * w
        db = np.mean(error)

        return loss, np.append(dw, db)

    def fit(self, optimizer='L-BFGS-B', maxiter=2000, tol=1e-5):
        print("Расширение признакового пространства (генерация сложных фичей)...")
        self.X_train_ext = self._expand_features(self.X_train)
        self.feature_mapping = self._get_feature_mapping()
        
        n_weights = self.X_train_ext.shape[1] + 1 # +1 для bias
        initial_weights = np.zeros(n_weights)
        
        print(f"Обучение весов MaxEnt ({n_weights-1} признаков)...")
        result = minimize(self._objective_function,
                          initial_weights,
                          method=optimizer,
                          jac=True, # Используем аналитический градиент (очень быстро!)
                          options={'maxiter': maxiter, 'gtol': tol})
        
        if not result.success:
            print(f"Предупреждение: Оптимизация не завершилась успешно: {result.message}")

        self.weights = result.x

        # --- Математика MaxEnt: Расчет Z и Энтропии на фоновых точках ---
        print("Калибровка распределения MaxEnt (расчет Z и энтропии)...")
        X_bg_ext = self._expand_features(self.X_bg)
        z_bg = np.dot(X_bg_ext, self.weights[:-1]) + self.weights[-1]
        
        # Нормировочная константа Z (log-sum-exp trick для избежания переполнения)
        max_z = np.max(z_bg)
        sum_exp = np.sum(np.exp(z_bg - max_z))
        self.log_Z_ = max_z + np.log(sum_exp)
        
        # Вероятностное распределение на фоне
        raw_bg = np.exp(z_bg - self.log_Z_)
        # Энтропия H
        self.entropy_ = -np.sum(raw_bg * np.log(raw_bg + 1e-15))

        # --- Агрегация важности признаков ---
        importances = np.zeros(self.n_features_orig)
        w_abs = np.abs(self.weights[:-1])
        
        for i, mapped_indices in enumerate(self.feature_mapping):
            for idx in mapped_indices:
                importances[idx] += w_abs[i] / len(mapped_indices)
        
        sum_imp = np.sum(importances)
        self._feature_importances_ = (importances / sum_imp) if sum_imp > 0 else importances
        
        print("Обучение завершено.")

    @property
    def feature_importances_(self):
        if self._feature_importances_ is None:
            raise RuntimeError("Модель не была обучена. Вызовите метод .fit() сначала.")
        return self._feature_importances_

    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_orig:
            raise ValueError(f"Ожидалось {self.n_features_orig} исходных признаков, получено {X.shape[1]}.")

        # 1. Расширяем новые данные так же, как при обучении
        X_ext = self._expand_features(X)
        
        # 2. Считаем линейный предиктор
        z = np.dot(X_ext, self.weights[:-1]) + self.weights[-1]
        
        # 3. Вычисляем итоговую вероятность (Logistic Output)
        # Математическое упрощение: 
        # P = (e^H * raw) / (1 + e^H * raw), где raw = e^(z - Z)
        # Эквивалентно P = 1 / (1 + e^-(H + z - Z))
        # Это позволяет избежать ошибки 'inf / inf = NaN', если z очень большое.
        
        eta = self.entropy_ + z - self.log_Z_
        prob = self._sigmoid(eta)

        return np.column_stack((1 - prob, prob))
