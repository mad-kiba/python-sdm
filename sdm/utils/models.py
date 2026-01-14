import numpy as np
from scipy.optimize import minimize

class MaxEnt:
    """
    Минимальная реализация модели максимальной энтропии (MaxEnt)
    для моделирования ареала вида.

    Args:
        X_pres (np.ndarray): Массив предикторов в точках присутствия.
                             Форма: (n_pres, n_features).
        X_bg (np.ndarray): Массив предикторов в фоновых точках.
                           Форма: (n_bg, n_features).
                           Ожидается, что n_bg >> n_pres.
    """
    def __init__(self, X_pres, X_bg):
        self.X_pres = np.asarray(X_pres)
        self.X_bg = np.asarray(X_bg)

        if self.X_pres.ndim == 1:
            self.X_pres = self.X_pres.reshape(-1, 1)
        if self.X_bg.ndim == 1:
            self.X_bg = self.X_bg.reshape(-1, 1)

        if self.X_pres.shape[1] != self.X_bg.shape[1]:
            raise ValueError("Количество признаков в X_pres и X_bg должно совпадать.")

        self.n_pres = self.X_pres.shape[0]
        self.n_bg = self.X_bg.shape[0]
        self.n_features = self.X_pres.shape[1]

        # Создаем объединенный набор данных для обучения
        # y=1 для точек присутствия, y=0 для фоновых точек
        self.X_train = np.vstack((self.X_pres, self.X_bg))
        self.y_train = np.array([1] * self.n_pres + [0] * self.n_bg)

        # Веса признаков (инициализируем нулями)
        self.weights = np.zeros(self.n_features)

        # Важность признаков (будет заполнена после обучения)
        self._feature_importances_ = None

    def _sigmoid(self, z):
        """Логистическая функция (сигмоида)."""
        return 1 / (1 + np.exp(-z))

    def _predict_linear(self, X):
        """Линейная комбинация весов и признаков."""
        # Добавляем фиктивный признак для свободного члена (bias)
        # Если у вас уже есть столбец единиц в X, этот шаг можно пропустить
        # Но для минимальной реализации лучше добавить его явным образом
        X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.dot(X_with_bias, self.weights)

    def _predict_proba_internal(self, X):
        """Внутренний метод предсказания вероятности (без добавления bias)."""
        return self._sigmoid(np.dot(X, self.weights[:-1]) + self.weights[-1]) # bias - последний вес

    def _objective_function(self, weights):
        """
        Целевая функция (отрицательная логарифмическая правдоподобность)
        для минимизации.
        """
        X_train_with_bias = np.hstack((self.X_train, np.ones((self.X_train.shape[0], 1))))
        linear_output = np.dot(X_train_with_bias, weights)
        predictions = self._sigmoid(linear_output)

        # Добавляем небольшое значение к предсказаниям, чтобы избежать log(0)
        epsilon = 1e-9
        predictions = np.clip(predictions, epsilon, 1. - epsilon)

        # Лосс-функция (кросс-энтропия)
        loss = -np.mean(self.y_train * np.log(predictions) + (1 - self.y_train) * np.log(1 - predictions))
        return loss

    def fit(self, optimizer='L-BFGS-B', maxiter=1000, tol=1e-4):
        """
        Обучает модель максимальной энтропии, находя оптимальные веса признаков.

        Args:
            optimizer (str): Алгоритм оптимизации. По умолчанию 'lbfgs'.
                             Другие варианты: 'cg', 'newton-cg', 'nelder-mead' и др.
            maxiter (int): Максимальное количество итераций для оптимизатора.
            tol (float): Допустимая погрешность для остановки оптимизации.
        """
        print("Начало обучения модели MaxEnt...")
        
        # Добавляем фиктивный признак для свободного члена (bias) к обучающим данным
        X_train_with_bias = np.hstack((self.X_train, np.ones((self.X_train.shape[0], 1))))
        
        n_weights = self.n_features + 1 # +1 для bias
        
        # Инициализируем веса нулями.
        # Можно использовать другие стратегии инициализации, но для простоты - нули.
        initial_weights = np.zeros(n_weights)
        
        # Оптимизируем веса, минимизируя целевую функцию
        result = minimize(self._objective_function,
                          initial_weights,
                          method=optimizer,
                          options={'maxiter': maxiter, 'gtol': tol}) # gtol - градиентная толерантность
        
        if not result.success:
            print(f"Предупреждение: Оптимизация не завершилась успешно: {result.message}")

        self.weights = result.x
        print("Обучение завершено.")

        # Вычисление важности признаков
        # Для MaxEnt, важность признака можно аппроксимировать абсолютным значением его веса
        # или квадратом веса. Здесь используем абсолютное значение.
        # Bias не учитывается как отдельный признак важности
        self._feature_importances_ = np.abs(self.weights[:-1])

    @property
    def feature_importances_(self):
        """
        Возвращает важность каждого признака.

        Важность признака аппроксимируется абсолютным значением его веса
        после обучения модели.
        """
        if self._feature_importances_ is None:
            raise RuntimeError("Модель не была обучена. Вызовите метод .fit() сначала.")
        return self._feature_importances_

    def predict_proba(self, X):
        """
        Предсказывает вероятность присутствия вида в новых точках.

        Args:
            X (np.ndarray): Массив предикторов для новых точек.
                            Форма: (n_new_points, n_features).

        Returns:
            np.ndarray: Массив вероятностей присутствия для каждой точки.
                        Форма: (n_new_points,).
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features:
            raise ValueError(f"Ожидалось {self.n_features} признаков, но получено {X.shape[1]}.")

        # Добавляем столбец единиц для свободного члена (bias)
        X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

        # Линейная комбинация + сигмоида
        # Здесь мы используем weights[:-1] для признаков и weights[-1] для bias
        linear_model = np.dot(X, self.weights[:-1]) + self.weights[-1]
        probabilities = self._sigmoid(linear_model)
        
        probabilities_absence = 1 - probabilities

        return np.column_stack((probabilities_absence, probabilities))
    
    