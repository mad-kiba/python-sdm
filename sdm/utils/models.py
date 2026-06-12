import numpy as np
from scipy.optimize import minimize

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
