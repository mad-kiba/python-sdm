import numpy as np
import rasterio
import math
import os
import rasterio.transform
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import distance_transform_edt
from scipy.stats import skew, kurtosis, pearsonr, chisquare, spearmanr
from scipy.spatial.distance import cosine
import numpy as np
import pyproj
from pyproj import Transformer, CRS
from shapely.geometry import shape
from rasterio.features import shapes
from scipy.ndimage import gaussian_filter


def apply_decay_to_points(
    input_tiff_path: str,
    observation_rows: np.ndarray,
    observation_cols: np.ndarray,
    buffer_km: float,
    decay_type: str = 'exponential',
    decay_rate: float = 0.1 # Коэффициент затухания (k)
):
    """
    Применяет пространственное затухание вокруг точек наблюдения.

    Все точки в пределах 'buffer_km' (если decay_type='buffer') или
    с учетом 'decay_rate' (для других типов затухания) будут изменены.
    Точки за пределами 'buffer_km' (для 'buffer' типа) или с нулевым
    значением затухания будут установлены в 0.

    Args:
        input_tiff_path (str): Путь к однослойному файлу GeoTIFF (EPSG:4326).
        observation_rows (np.ndarray): Массив индексов строк (y-координат) точек наблюдения (EPSG:4326).
        observation_cols (np.ndarray): Массив индексов столбцов (x-координат) точек наблюдения (EPSG:4326).
        buffer_km (float): Максимальное расстояние в километрах, вокруг которого будет применяться затухание.
                           Для decay_type='buffer' это жесткий буфер.
                           Для других типов используется как начальный радиус или параметр.
        decay_type (str): Тип затухания. Допустимые значения: 'buffer', 'exponential', 'inverse_quadratic', 'gaussian'.
        decay_rate (float): Коэффициент затухания (k).
                           - Для 'exponential': k в e^(-kd).
                           - Для 'inverse_quadratic': k в 1/(1 + kd^2).
                           - Для 'gaussian': sigma = 1 / (decay_rate * sqrt(2*pi)) или можно настроить как sigma.
                                              Здесь я сделаю так, что decay_rate напрямую будет sigma.

    Returns:
        np.ndarray: Новый растр с примененным затуханием.
    """

    if decay_type not in ['buffer', 'exponential', 'inverse_quadratic', 'gaussian']:
        raise ValueError("decay_type должен быть одним из: 'buffer', 'exponential', 'inverse_quadratic', 'gaussian'")

    with rasterio.open(input_tiff_path) as src:
        # Проверяем CRS растра
        if src.crs != CRS.from_epsg(4326):
            raise ValueError("Входной GeoTIFF должен быть в EPSG:4326")

        # Загружаем данные растра
        raster_data = src.read(1)
        transform = src.transform
        bounds = src.bounds

        # Создаем копию данных для модификации
        decayed_raster = np.copy(raster_data)

        # Определяем CRS для вычислений расстояний
        # Будем использовать WGS84 (EPSG:4326) и трансформер для геодезических расстояний
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)

        # Преобразуем строки и столбцы в координаты (x, y)
        # ИСПОЛЬЗОВАНИЕ АТРИБУТОВ AFFINE:
        obs_coords = []
        for row, col in zip(observation_rows, observation_cols):
            x = transform.a * col + transform.b * row + transform.c
            y = transform.d * col + transform.e * row + transform.f
            obs_coords.append((x, y))
        obs_lons = np.array([coord[0] for coord in obs_coords])
        obs_lats = np.array([coord[1] for coord in obs_coords])

        # Получаем координаты всех пикселей растра
        rows_all, cols_all = np.indices(raster_data.shape)
        #pixel_coords = [transform.xy(row, col) for row, col in zip(rows_all.flatten(), cols_all.flatten())]
        pixel_coords = [rasterio.transform.xy(transform, row, col) for row, col in zip(rows_all.flatten(), cols_all.flatten())]
        pixel_lons = np.array([coord[0] for coord in pixel_coords])
        pixel_lats = np.array([coord[1] for coord in pixel_coords])

        # Преобразуем массив пиксельных координат в 2D массив для удобства
        pixel_coords_2d = np.vstack((pixel_lons, pixel_lats)).T
        obs_coords_2d = np.vstack((obs_lons, obs_lats)).T

        # Вычисляем расстояния от каждого пикселя до КАЖДОЙ точки наблюдения
        # Это может быть ресурсоемко для больших растров и множества точек!
        # Для оптимизации можно сначала вычислить расстояния до ближайшей точки наблюдения.

        # ----- Оптимизация: Вычисление расстояния до ближайшей точки наблюдения -----
        # Создадим массив расстояний, где каждый элемент - расстояние от пикселя
        # до БЛИЖАЙШЕЙ точки наблюдения.

        # Инициализируем массив расстояний очень большим числом
        min_distances_km = np.full(raster_data.shape, np.inf, dtype=np.float64)
        print(10)
        # Проходим по каждой точке наблюдения и обновляем минимальное расстояние
        for i, obs_coord in enumerate(obs_coords_2d):
            # Вычисляем геодезическое расстояние от текущей точки наблюдения
            # до всех пикселей растра.
            # transformer.transform(lon1, lat1, lon2, lat2) возвращает (distance_in_meters, azimuth1, azimuth2)
            # Нам нужно расстояние в метрах.
            # Используем np.apply_along_axis для применения функции к каждой строке (пикселю)
            # Важно: pyproj.Transformer.transform ожидает (lon, lat)
            distances_meters = np.array([
                transformer.transform(obs_coord[0], obs_coord[1], p_lon, p_lat)[0]
                for p_lon, p_lat in pixel_coords_2d
            ])

            distances_km = distances_meters / 1000.0 # Переводим в километры

            # Изменяем форму массива расстояний к форме растра
            distances_km_reshaped = distances_km.reshape(raster_data.shape)

            # Обновляем минимальное расстояние: min_distances_km[j, k] = min(min_distances_km[j, k], distances_km_reshaped[j, k])
            min_distances_km = np.minimum(min_distances_km, distances_km_reshaped)

        # Теперь min_distances_km содержит расстояние до ближайшей точки наблюдения для каждого пикселя.
        # ----------------------------------------------------------------------------
        print(20)
        # Применение алгоритмов затухания
        if decay_type == 'buffer':
            # Жесткий буфер: все, что дальше buffer_km, становится 0
            decayed_raster[min_distances_km > buffer_km] = 0
            # Точки наблюдения, которые были изначально 1, должны остаться 1,
            # но мы работаем с модификацией всего растра, так что это поведение
            # уже учтено, если исходный растр содержал 1.
            # Если мы хотим, чтобы НА ВСЕХ ПОВЕРХНОСТЯХ, которые сейчас 0,
            # где есть точки наблюдения, но они дальше buffer, тоже стало 0,
            # то это сделано.

        elif decay_type == 'exponential':
            # Функция: f(d) = exp(-decay_rate * d)
            # Здесь d - это min_distances_km
            # Устанавливаем значение 0 там, где d > buffer_km (чтобы не экспоненциально уменьшать далеко)
            decay_factor = np.exp(-decay_rate * min_distances_km)
            # Применяем буфер, чтобы обрезать влияние дальше buffer_km
            decay_factor[min_distances_km > buffer_km] = 0
            # Результат: исходные значения растра * коэффициент затухания
            # Если исходные значения растра не 0 или 1, а представляют некоторую плотность,
            # то это будет корректно. Если это бинарный растр (1 - есть, 0 - нет),
            # то мы просто "размазываем" единицу.
            decayed_raster = raster_data * decay_factor

        elif decay_type == 'inverse_quadratic':
            # Функция: f(d) = 1 / (1 + decay_rate * d^2)
            decay_factor = 1 / (1 + decay_rate * (min_distances_km ** 2))
            decay_factor[min_distances_km > buffer_km] = 0
            decayed_raster = raster_data * decay_factor

        elif decay_type == 'gaussian':
            # Функция: f(d) = exp(-d^2 / (2 * sigma^2))
            # Здесь sigma = decay_rate (можно настроить)
            sigma = decay_rate
            decay_factor = np.exp(-(min_distances_km ** 2) / (2 * (sigma ** 2)))
            decay_factor[min_distances_km > buffer_km] = 0
            decayed_raster = raster_data * decay_factor
        print(30)
        # Теперь надо убедиться, что пиксели, которые были изначально 0
        # (и не попали под действие буфера), остаются 0.
        # Иначе, если исходный растр был бинарным (1 - есть, 0 - нет),
        # то умножение на коэффициент затухания (меньше 1)
        # может оставить не-ноль там, где не должно быть.
        #
        # Если исходный растр - это бинарный растр присутствия (1/0),
        # и мы хотим "размазать" эти единицы, то:
        # 1. Инициализируем `decayed_raster` нулями.
        # 2. Находим индексы, где `raster_data` == 1.
        # 3. Для этих индексов вычисляем `decay_factor`.
        # 4. Присваиваем `decayed_raster[indices] = decay_factor[indices]`.
        #
        # Давайте сделаем так: если исходный растр бинарный,
        # то мы будем "размазывать" эти единицы.
        print(40)
        if np.all(np.unique(raster_data) <= 1): # Предполагаем бинарный растр (0 или 1)
             # Если у нас точки наблюдения, которые должны быть 1,
             # а остальное - 0, и мы хотим "размазать" эти 1.
             # Мы уже вычислили `decay_factor`.
             # Теперь применим его только там, где `raster_data` == 1.
             # Но мы работаем с `min_distances_km`, поэтому лучше будет
             # так:
             # Создаем новый растр, заполненный нулями.
             # Для каждого пикселя:
             #   Если raster_data[px] == 1:
             #     decayed_raster[px] = decay_factor[px]
             #   Иначе:
             #     decayed_raster[px] = 0
             #
             # Можно сделать это эффективнее:
             # `decayed_raster = raster_data * decay_factor` уже сделано выше,
             # но это может дать дробные значения там, где было 0.
             #
             # Исправим:
             decayed_raster = np.zeros_like(raster_data, dtype=np.float32)
             # Находим индексы, где исходный растр был 1
             present_indices = (raster_data == 1)
             # Применяем рассчитанный фактор затухания только к этим пикселям
             decayed_raster[present_indices] = decay_factor[present_indices]

        # Убедимся, что значения не выходят за разумные пределы (например, 0-1, если это вероятность)
        # В зависимости от типа затухания, значения могут быть > 1 (если decay_rate очень мал)
        # или < 0 (не должно произойти).
        # Для 'inverse_quadratic' и 'exponential' значения будут от 0 до 1.
        # Для 'gaussian' тоже от 0 до 1.
        # Если растр исходный был с другими значениями, то здесь надо будет масштабировать.
        # Для простоты, будем считать, что результат должен быть в диапазоне [0, 1]
        # если исходный растр был бинарным.
        decayed_raster = np.clip(decayed_raster, 0, 1)
        print(50)

        # Обновляем метаданные растра
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "dtype": rasterio.float32, # Для дробных значений
            "nodata": 0.0 # Или другой подходящий NoData
        })
        
        print(60)
        
        with rasterio.open(input_tiff_path, 'w', **out_meta) as dst:
            dst.write(decayed_raster, 1)
        print(f"Результат затухания записан в: {input_tiff_path}")
    

# Вспомогательная функция предсказания по стеку батчами
def predict_suitability_for_stack(model, stack, valid_mask, batch_size=500_000):
    bands, H, W = stack.shape
    flat = stack.reshape(bands, -1).T  # (H*W, bands)
    suitability_flat = np.full(H * W, np.nan, dtype="float32")
    valid_idx = np.flatnonzero(valid_mask.ravel())
    for start in range(0, len(valid_idx), batch_size):
        end = start + batch_size
        sel = valid_idx[start:end]
        X_pred = flat[sel]
        pred = model.predict_proba(X_pred)[:, 1].astype("float32")
        suitability_flat[sel] = pred
    return suitability_flat.reshape(H, W)


def wrap_long_lines(text, max_len=60):
    """
    Wraps a string to a new line if its length exceeds max_len,
    breaking at the nearest space to the left.

    Args:
        text (str): The input string.
        max_len (int): The maximum length of a line.

    Returns:
        str: The wrapped string with newlines.
    """
    if len(text) <= max_len:
        return text

    # Find the last space within the max_len limit
    wrap_point = text.rfind(" ", 0, max_len)

    # If no space is found, break at max_len (though the prompt asks for nearest space left)
    # In a real-world scenario, you might want to handle this edge case differently,
    # e.g., by breaking mid-word or using a different strategy.
    if wrap_point == -1:
        wrap_point = max_len

    # Recursively wrap the rest of the text
    return text[:wrap_point].rstrip() + "\n" + wrap_long_lines(text[wrap_point:].lstrip(), max_len)


def read_and_to_3857(path):
    dest_crs = CRS.from_epsg(3857)
    with rasterio.open(path) as src:
        src_crs = src.crs
        band1 = src.read(1, masked=True)  # первая полоса (например, пригодность 0..1)
        # Если уже в 3857 — просто вернуть как есть
        if src_crs == dest_crs:
            data = band1.filled(np.nan).astype("float32")
            transform = src.transform
            width, height = src.width, src.height
        else:
            # Считаем параметры целевой решетки
            transform, width, height = calculate_default_transform(
                src_crs, dest_crs, src.width, src.height, *src.bounds
            )
            # Готовим массив назначения
            data = np.full((height, width), np.nan, dtype="float32")
            reproject(
                source=band1.filled(np.nan),
                destination=data,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dest_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
    return data, transform, width, height


def round_to_significant_figures(number: float, sig_digits: int = 4) -> float:
    """
    Округляет число до заданного количества значащих цифр.

    Args:
        number: Число, которое нужно округлить.
        sig_digits: Количество значащих цифр. По умолчанию 4.

    Returns:
        Округленное число.
    """
    
    if not isinstance(sig_digits, int) or sig_digits <= 0:
        print("Количество значащих цифр должно быть положительным целым числом.")
        raise ValueError("Количество значащих цифр должно быть положительным целым числом.")
    
    if number == 0:
        return 0.0
    
    number = float(number)
    # Определяем порядок величины числа
    # Это поможет нам понять, где находится первая значащая цифра
    # log10(abs(number)) дает степень 10, к которой примерно равно число.
    # Например, log10(340) ~ 2.54, log10(1029.6) ~ 3.01, log10(2244.6) ~ 3.35
    # floor(log10(abs(number))) дает порядок величины (3 -> 10^3, 2 -> 10^2)
    order_of_magnitude = math.floor(math.log10(abs(number)))
    
    # Вычисляем множитель для округления
    # Мы хотим, чтобы первая значащая цифра была на позиции единиц.
    # Например, для 340.0001:
    #   order_of_magnitude = floor(log10(340)) = 2
    #   sig_digits = 4
    #   power_for_rounding = 2 - (4 - 1) = 2 - 3 = -1
    #   multiplier = 10**(-1) = 0.1
    #   number * multiplier = 3400.0001 (это чтобы получить 3400...)
    #   round(3400.0001) = 3400
    #   3400 / 10 = 340

    # Для 1029.6492:
    #   order_of_magnitude = floor(log10(1029.6)) = 3
    #   sig_digits = 4
    #   power_for_rounding = 3 - (4 - 1) = 3 - 3 = 0
    #   multiplier = 10**0 = 1
    #   number * multiplier = 1029.6492
    #   round(1029.6492) = 1030 (округляем до 4 значащих цифр)
    #   1030 / 1 = 1030

    # Для 2244.6997:
    #   order_of_magnitude = floor(log10(2244.6)) = 3
    #   sig_digits = 4
    #   power_for_rounding = 3 - (4 - 1) = 3 - 3 = 0
    #   multiplier = 10**0 = 1
    #   number * multiplier = 2244.6997
    #   round(2244.6997) = 2245
    #   2245 / 1 = 2245

    power_for_rounding = order_of_magnitude - (sig_digits - 1)
    multiplier = 10 ** power_for_rounding
    
    # Применяем округление и возвращаем результат
    return round(number / multiplier) * multiplier


def calculate_histogram_similarity(data_obs, data_full, bins_num=50, sig_figs=4):
    """
    Вычисляет показатель подобия двух гистограмм.
    Возвращает значение от 0 до 1 (1 - максимальное сходство).
    """
    
    # 1. Рассчитываем гистограммы
    # Устанавливаем общий диапазон, чтобы бины были сопоставимы
    # Можно взять min/max из data_full, или задать общий диапазон, если известно.
    # Предположим, что data_full содержит полный диапазон значений.
    min_val = np.min(data_full)
    max_val = np.max(data_full)
    
    # Убеждаемся, что диапазон не нулевой
    if min_val == max_val:
        # Если все значения одинаковы, то гистограмма - это один пик.
        # Сравнение будет тривиальным.
        if np.all(data_obs == data_obs[0]) and np.all(data_full == data_full[0]):
            return 1.0, 0.0, 0.0 # Полное сходство, корреляция 1, Чи-квадрат 0
        else:
            # Разные, но одномерные распределения
            return 0.0, 0.0, 0.0 # Полное несходство, корреляция 0, Чи-квадрат большой
            
    # bins_range = (min_val, max_val)
    # Если range задан, то numpy.histogram возвращает тот же диапазон бинов.
    # Если range не задан, numpy.histogram сам определяет диапазон.
    # Важно: для сравнения двух гистограмм, бины должны быть одинаковыми.
    # Поэтому лучше использовать общий диапазон, который охватывает оба набора данных.
    # Или, если range задан, но data_obs выходит за его пределы, эти значения будут игнорироваться.
    # Лучше всего - взять min/max из data_full, если он действительно охватывает всё.
    
    # Используем одинаковые бины для обоих наборов данных
    bins = np.linspace(min_val, max_val, bins_num + 1)

    counts_obs, _ = np.histogram(data_obs, bins=bins)
    counts_full, _ = np.histogram(data_full, bins=bins)

    # 2. Нормализация гистограмм (преобразование в плотности вероятности)
    # Сумма всех значений гистограммы должна стать равной 1.
    # Ширина бина (bin_width) нужна для получения истинной плотности вероятности.
    bin_width = (max_val - min_val) / bins_num
    
    # Нормализуем так, чтобы сумма плотностей была равна 1
    density_obs = counts_obs / (np.sum(counts_obs) * bin_width) if np.sum(counts_obs) > 0 else np.zeros_like(counts_obs)
    density_full = counts_full / (np.sum(counts_full) * bin_width) if np.sum(counts_full) > 0 else np.zeros_like(counts_full)

    # На случай, если после нормализации остались очень малые значения (близкие к нулю),
    # которые могут вызвать проблемы с некоторыми метриками.
    # Можно использовать небольшое эпсилон, если это необходимо.
    epsilon = 1e-10
    density_obs = np.maximum(density_obs, epsilon)
    density_full = np.maximum(density_full, epsilon)
    
    # 3. Вычисление метрик сходства

    # А. Коэффициент корреляции Пирсона
    # Он возвращает значение от -1 до 1. Для неотрицательных данных он будет от 0 до 1.
    # 1 - идеальная линейная зависимость (прямая пропорциональность).
    # 0 - отсутствие линейной зависимости.
    try:
        # Убираем все бины, где оба значения - epsilon (практически 0)
        # Это поможет избежать ошибок, если одно распределение имеет больше нулевых бинов.
        mask = (density_obs > epsilon) | (density_full > epsilon)
        
        if np.sum(mask) < 2: # Нужно минимум 2 точки для корреляции
            correlation = 0.0
        else:
            corr_coeff, _ = pearsonr(density_obs[mask], density_full[mask])
            # Результат pearsonr может быть NaN, если данные очень скудные
            correlation = corr_coeff if not np.isnan(corr_coeff) else 0.0
            
    except Exception as e:
        print(f"Ошибка при расчете корреляции Пирсона: {e}")
        correlation = 0.0

    
    return round_to_significant_figures(correlation, sig_figs)


def get_predictor_stats(data: np.ndarray) -> dict:
    """
    Вычисляет основные статистические показатели для набора данных.
    """
    
    data = data[~np.isnan(data)]
    
    stats = {
        'mean': round_to_significant_figures(np.mean(data), 4),
        'median': round_to_significant_figures(np.median(data), 4),
        'min': round_to_significant_figures(np.min(data), 4),
        'max': round_to_significant_figures(np.max(data), 4),
        'p5': round_to_significant_figures(np.percentile(data, 5), 4),
        'p95': round_to_significant_figures(np.percentile(data, 95), 4),
        'width_obs': round_to_significant_figures(np.percentile(data, 90) - np.percentile(data, 10), 4),
        'std_dev': 0,
        'skewness': 0,
        'kurtosis': 0,
    }
    
    try:
        stats['std_dev'] = round_to_significant_figures(np.std(data), 4)  # Стандартное отклонение
    except Exception as e:
        print('Ошибка вычисления статистики std_dev: ' + str(e))
    
    try:
        stats['skewness'] = round_to_significant_figures(skew(data), 4)  # Стандартное отклонение
    except Exception as e:
        print('Ошибка вычисления статистики skewness: ' + str(e))
        
    try:
        stats['kurtosis'] = round_to_significant_figures(kurtosis(data), 4)  # Стандартное отклонение
    except Exception as e:
        print('Ошибка вычисления статистики kurtosis: ' + str(e))
    
    return stats


def format_float(value: float) -> str:
    """
    Форматирует число с плавающей точкой для отображения (убирает лишние нули).
    """
    return f"{value:.4f}".rstrip('0').rstrip('.')


def save_error(error_path, text):
    with open(error_path, 'w') as f: # записываем файл
        f.write(str(text))


# считает площадь подходящих местообитаний
def get_geotiff_square(filepath: str, threshold) -> dict:
    with rasterio.open(filepath) as src:
        # Получаем данные растра
        raster_data = src.read(1)
        geod = pyproj.Geod(ellps='WGS84')

        # Получаем трансформацию растра
        transform = src.transform

        # Получаем размеры растра
        height, width = raster_data.shape

        # Словарь для хранения рассчитанных площадей пикселей по широте
        # Ключ - широта, значение - площадь пикселя в кв.км
        pixel_area_by_latitude = {}

        # Словарь для хранения результатов
        out_list = {t: 0.0 for t in threshold} # Инициализируем нулями
        out_count = {t: 0.0 for t in threshold}

        # Итерируемся по каждой строке (широте) растра
        for i in range(height):
            # Получаем широту центра пикселя в этой строке
            # Проекция WGS84 использует координаты (долгота, широта)
            # transform.i_to_lat(i) даст широту верхней границы пикселя,
            # нам нужна середина для большей точности
            center_lat = transform.f + (i + 0.5) * transform.e

            # Рассчитываем площадь пикселя для данной широты, если она еще не была рассчитана
            if center_lat not in pixel_area_by_latitude:
                # Создаем один пиксель в этой строке для расчета площади
                # Берем первый столбец (j=0) для примера
                # Создаем координаты углов этого пикселя
                try:
                    # Получаем координаты углов пикселя (верхний левый, верхний правый, нижний правый, нижний левый)
                    # transform.xy(i, j) возвращает (x, y) = (долгота, широта)
                    ul_lon, ul_lat = transform * (0, i)         # Верхний левый
                    ur_lon, ur_lat = transform * (1, i)         # Верхний правый
                    lr_lon, lr_lat = transform * (1, i + 1)     # Нижний правый
                    ll_lon, ll_lat = transform * (0, i + 1)     # Нижний левый

                    # Создаем полигон из этих координат
                    # Важно: pyproj.Geod ожидает список долгот и список широт
                    pixel_polygon_coords = [(ul_lon, ul_lat), (ur_lon, ur_lat), (lr_lon, lr_lat), (ll_lon, ll_lat)]

                    # Используем pyproj.Geod для расчета площади
                    area_sq_meters, _ = geod.polygon_area_perimeter(
                        [coord[0] for coord in pixel_polygon_coords],
                        [coord[1] for coord in pixel_polygon_coords]
                    )
                    pixel_area_sq_km = abs(area_sq_meters) / 1_000_000.0
                    pixel_area_by_latitude[center_lat] = pixel_area_sq_km
                except Exception as e:
                    print(f"Ошибка при расчете площади пикселя для широты {center_lat}: {e}")
                    # В случае ошибки, можно пропустить эту строку или присвоить какое-то значение по умолчанию
                    continue

            # Если площадь для данной широты не была рассчитана (из-за ошибки), пропускаем
            if center_lat not in pixel_area_by_latitude:
                continue

            # Получаем площадь одного пикселя для текущей широты
            current_pixel_area = pixel_area_by_latitude[center_lat]

            # Итерируемся по каждому порогу
            for k in threshold:
                # Создаем маску для пикселей, значение которых выше порога в текущей строке
                mask_row = raster_data[i, :] > k

                # Считаем количество пикселей в этой строке, которые выше порога
                num_pixels_above_threshold = mask_row.sum()

                # Добавляем площадь этих пикселей к общему итогу для данного порога
                out_list[k] += num_pixels_above_threshold * current_pixel_area
                out_count[k] += num_pixels_above_threshold

    out_square = []
    out_num = []
    for k in threshold:
        out_square.append(round(out_list[k]))
        out_num.append(round(out_count[k]))
        
    return out_square, out_num
    


def save_geotiff(output_path, array2d, profile):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prof = profile.copy()
    with rasterio.open(output_path, "w", **prof) as dst:
        dst.write(array2d.astype("float32"), 1)
        

def inverse_scale(scaled_data, scale_params, info):
    if scale_params is None or "mean" not in scale_params or "scale" not in scale_params:
        return scaled_data
    method = scale_params.get("method", "standard")
    mean = scale_params["mean"]
    scale = scale_params["scale"]
    #print('Inverse scaling: ')
    #print('Scale: '+str(scale)+', mean: '+str(mean))
    diff = info.get('diff')
    dsca = info.get('scale')
    if diff:
        mean = mean - diff
    if method == "standard":
        data = scaled_data * scale + mean
        if (dsca):
            data = (scaled_data * scale + mean)/dsca
        return data
    else:
        return scaled_data


def extract_features_from_stack(stack, rows, cols):
    """Извлекает значения предикторов из стека по индексам пикселей.
       Возвращает X: (n_samples, n_bands)."""
    # stack: (bands, H, W)
    # fancy-indexing
    return stack[:, rows, cols].T  # (n_samples, n_bands)


# Сэмплирует n_bg фоновых пикселей, разделяя их на две части
def sample_background(valid_mask, presence_rc_set, n_bg, rng, bg_pc = 100,
                      distance_min_pixels = 1, distance_max_pixels = 1, text_filename = '', month = 0):
    """
    Сэмплирует n_bg фоновых пикселей, разделяя их на две части:
    1. 50% точек - случайно в пределах valid_mask (исключая точки присутствия).
    2. 50% точек - в пределах "огибающей" (буфера) вокруг точек присутствия,
       на расстоянии от distance_min_pixels до distance_max_pixels (в единицах растра).

    Args:
        valid_mask (np.ndarray): Булева маска, где True - пиксели, пригодные для моделирования (регион за вычетом морей).
        presence_rc_set (set): Множество кортежей (строка, столбец) точек присутствия вида.
        n_bg (int): Общее желаемое количество фоновых точек.
        rng (np.random.Generator): Объект генератора случайных чисел.
        distance_min_pixels (float): Минимальное расстояние в пикселях от точек присутствия для генерации фона.
        distance_max_pixels (float): Максимальное расстояние в пикселях от точек присутствия для генерации фона.

    Returns:
        tuple: Кортеж (rows_bg, cols_bg) - массивы строк и столбцов фоновых точек.
    """
    
    height, width = valid_mask.shape

    # --- Часть 1: Случайный фон по всей valid_mask ---
    n_bg_random = int(round(n_bg * bg_pc / 100))
    

    # Создаем маску для случайного фона, исключая точки присутствия
    random_bg_mask = valid_mask.copy()
    if presence_rc_set: # Проверяем, есть ли вообще точки присутствия
        # Преобразуем точки присутствия в линейные индексы
        # Добавляем проверку на выход за границы и на наличие точки в valid_mask
        pres_linear_indices = []
        for r, c in presence_rc_set:
            if 0 <= r < height and 0 <= c < width and valid_mask[r, c]:
                pres_linear_indices.append(r * width + c)
        
        if pres_linear_indices:
            pres_linear_indices = np.array(pres_linear_indices, dtype=np.int64)
            # Создаем булеву маску линейных индексов точек присутствия
            presence_linear_mask = np.zeros(valid_mask.size, dtype=bool)
            presence_linear_mask[pres_linear_indices] = True
            # Применяем маску к плоскому представлению random_bg_mask
            random_bg_mask.ravel()[presence_linear_mask] = False

    # Выбираем случайные фоновые точки
    candidates_random = np.flatnonzero(random_bg_mask)
    if candidates_random.size == 0:
        print("ВНИМАНИЕ: Нет доступных валидных пикселей для генерации случайного фона.")
        rows_random, cols_random = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    else:
        n_bg_random = min(n_bg_random, candidates_random.size)
        chosen_random = rng.choice(candidates_random, size=n_bg_random, replace=False)
        rows_random = chosen_random // width
        cols_random = chosen_random % width

    print(f"Сгенерировано фоновых точек: {n_bg_random}")

    # если не требуется брать точки с границ
    if bg_pc == 100:
        if month == 0:
            with open(text_filename, 'a') as f:
                f.write(f"\n{len(rows_random)},0")
        return rows_random, cols_random, rows_random, cols_random, [], []
    
    # --- Часть 2: Фон в "огибающей" (буфере) ---
    # Вычисляем, сколько точек нужно для буферной части
    n_bg_buffer_target = int(n_bg - len(rows_random)) #* (100-bg_pc) / 100
    
    print(f"Нужно сгенерировать точек псевдоприсутствия: {n_bg_buffer_target}")
    
    if n_bg_buffer_target <= 0:
        # Если уже набрали достаточно случайных точек, возвращаем их
        if month == 0:
            with open(text_filename, 'a') as f:
                f.write(f"\n{len(rows_random)},0")
        return rows_random, cols_random, rows_random, cols_random, [], []
    
    # Если точек присутствия нет, буферная часть не может быть сгенерирована
    if not presence_rc_set:
        print("ВНИМАНИЕ: Отсутствуют точки присутствия для генерации фона в огибающей.")
        if month == 0:
            with open(text_filename, 'a') as f:
                f.write(f"\n{len(rows_random)},0")
        return rows_random, cols_random, rows_random, cols_random, [], []
    
    # 1. Создаем массив расстояний до ближайшей точки присутствия
    # Инициализируем массив бесконечностью, а точки присутствия - нулем.
    distance_array_for_transform = np.full(valid_mask.shape, np.inf)
    
    # Заполняем массив расстояний, используя только валидные точки присутствия
    for r, c in presence_rc_set:
        if 0 <= r < height and 0 <= c < width and valid_mask[r, c]:
            distance_array_for_transform[r, c] = 0

    # Вычисляем расстояние от каждого пикселя до ближайшей точки присутствия (в единицах растра)
    # distance_transform_edt работает с матрицей, поэтому результат будет в "пикселях".
    distance_to_presence_pixels = distance_transform_edt(distance_array_for_transform)
    
    # 2. Создаем маску для фона в огибающей
    # Пиксели должны быть:
    #   - В пределах valid_mask
    #   - Расстояние до точек присутствия должно быть >= distance_min_pixels
    #   - Расстояние до точек присутствия должно быть <= distance_max_pixels
    buffer_mask = (distance_to_presence_pixels >= distance_min_pixels) & \
                  (distance_to_presence_pixels <= distance_max_pixels) & \
                  valid_mask
    
    # Теперь уберем из этой маски сами точки присутствия, чтобы не сэмплировать их как фон
    if presence_rc_set:
        pres_linear_indices = []
        for r, c in presence_rc_set:
            if 0 <= r < height and 0 <= c < width and valid_mask[r, c]:
                pres_linear_indices.append(r * width + c)

        if pres_linear_indices:
            pres_linear_indices = np.array(pres_linear_indices, dtype=np.int64)
            presence_linear_mask = np.zeros(valid_mask.size, dtype=bool)
            presence_linear_mask[pres_linear_indices] = True
            buffer_mask.ravel()[presence_linear_mask] = False
    
    # 3. Выбираем фоновые точки из буферной маски
    candidates_buffer = np.flatnonzero(buffer_mask)
    
    n_bg_buffer = 0
    rows_buffer, cols_buffer = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    if candidates_buffer.size > 0:
        n_bg_buffer = int(min(n_bg_buffer_target, candidates_buffer.size))
        chosen_buffer = rng.choice(candidates_buffer, size=n_bg_buffer, replace=False)
        rows_buffer = chosen_buffer // width
        cols_buffer = chosen_buffer % width
    else:
        print(f"ВНИМАНИЕ: Нет доступных валидных пикселей в радиусе ({distance_min_pixels} - {distance_max_pixels} пикселей) вокруг точек присутствия для генерации фона.")
    
    # Объединяем результаты
    all_rows = np.concatenate((rows_random, rows_buffer))
    all_cols = np.concatenate((cols_random, cols_buffer))
    
    # Если общее количество точек меньше n_bg (из-за ограничений),
    # нужно добрать недостающее количество из любой оставшейся доступной области.
    current_total = len(all_rows)
    remaining_to_sample = n_bg - current_total
    
    
    if remaining_to_sample > 0:
        print(f"ВНИМАНИЕ: Сгенерировано {current_total} фоновых точек, вместо желаемых {n_bg}.")

    # Перемешиваем финальный набор точек
    if len(all_rows) > 0:
        indices = rng.permutation(len(all_rows))
        all_rows = all_rows[indices]
        all_cols = all_cols[indices]
    
    if month == 0:
        with open(text_filename, 'a') as f:
            f.write(f"\n{len(rows_random)},{len(rows_buffer)}")
    
    return all_rows, all_cols, rows_random, cols_random, rows_buffer, cols_buffer


# Подсчёт метрики Бойса (Continuous Boyce Index - CBI)
def continuous_boyce_index(obs, fit, num_bins=100, window_width=0.1):
    """
    Вычисляет Continuous Boyce Index (CBI) для оценки моделей Presence-Background.
    Оценивает корреляцию между предсказанной пригодностью и частотой встреч вида.
    
    Args:
        obs (array-like): Вероятности предсказаний для подтвержденных точек присутствия.
        fit (array-like): Вероятности предсказаний для всех точек (ожидаемое распределение).
        num_bins (int): Количество шагов смещения скользящего окна.
        window_width (float): Ширина скользящего окна вероятности (например, 0.1).
        
    Returns:
        float: Значение индекса Бойса от -1 до 1.
    """
    obs = np.asarray(obs)
    fit = np.asarray(fit)
    
    obs = obs[~np.isnan(obs)]
    fit = fit[~np.isnan(fit)]
    
    if len(obs) == 0 or len(fit) == 0:
        return np.nan
        
    min_val = 0.0
    max_val = 1.0
    window_step = (max_val - min_val) / num_bins
    if window_width < window_step:
        window_width = window_step
        
    bin_starts = np.arange(min_val, max_val - window_width + window_step, window_step)
    
    f_ratios = []
    bin_medians = []
    total_obs = len(obs)
    total_fit = len(fit)
    
    for start in bin_starts:
        end = start + window_width
        obs_in_bin = np.sum((obs >= start) & (obs <= end))
        fit_in_bin = np.sum((fit >= start) & (fit <= end))
        
        if fit_in_bin > 0:
            p_i = obs_in_bin / total_obs
            e_i = fit_in_bin / total_fit
            f_ratios.append(p_i / e_i)
            bin_medians.append(start + window_width / 2.0)
            
    if len(f_ratios) < 2:
        return np.nan
        
    cbi, _ = spearmanr(bin_medians, f_ratios)
    return cbi