# sdm/utils/plots.py
# Библиотека PythonSDM для моделирования распространения видов
# - набор вспомогательных функций

import numpy as np
import rasterio
import math
import os
import json
import pyproj
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import distance_transform_edt
from scipy.stats import skew, kurtosis, pearsonr, spearmanr
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings


# Применяет пространственное затухание вокруг точек наблюдения.
def apply_decay_to_points(
    raster_shape: tuple,
    transform,
    observation_rows: np.ndarray,
    observation_cols: np.ndarray,
    buffer_km: float,
    decay_type: str = 'buffer',
    decay_rate: float = 0.1,
    slope_data: np.ndarray = None,
    elev_data: np.ndarray = None,
    water_data: np.ndarray = None,
    is_bird: bool = False,
    bias_data: np.ndarray = None,
    dclass: list = None,
    height_barrier: float = 500,
):
    """
    Создает матрицу пространственного затухания (множителей от 0.0 до 1.0) 
    вокруг точек наблюдения (M-фактор из фреймворка BAM).
    
    Может учитывать сопротивление рельефа (Friction Surface) с помощью 
    алгоритма Fast Marching (scikit-fmm).

    Args:
        raster_shape (tuple): Размеры растра (height, width).
        transform (affine.Affine): Матрица трансформации растра.
        observation_rows (np.ndarray): Массив индексов строк точек наблюдения.
        observation_cols (np.ndarray): Массив индексов столбцов точек наблюдения.
        buffer_km (float): Максимальное расстояние в километрах, вокруг которого будет применяться затухание.
        decay_type (str): Тип затухания ('buffer', 'exponential', 'inverse_quadratic', 'gaussian', 'linear', 'sigmoid').
        decay_rate (float): Коэффициент затухания (например, sigma для gaussian).
        slope_data (np.ndarray, optional): 2D массив с уклонами в градусах для учета рельефа.
        elev_data (np.ndarray, optional): 2D массив с абсолютной высотой (м) для учета климатических пределов.
        water_data (np.ndarray, optional): 2D массив с % открытой воды (0-100) для водных барьеров.
        is_bird (bool): Флаг для птиц (игнорируют водные барьеры).
        bias_data (np.ndarray, optional): 2D массив со слоем предвзятости наблюдений (human bias).
        dclass (list, optional): Список с названием класса животного (например, ['Amphibia']).

    Returns:
        np.ndarray: 2D матрица множителей (0.0 ... 1.0).
    """
    if decay_type not in ['buffer', 'exponential', 'inverse_quadratic', 'gaussian', 'linear', 'sigmoid']:
        raise ValueError("decay_type должен быть одним из: 'buffer', 'exponential', 'inverse_quadratic', 'gaussian', 'linear', 'sigmoid'")

    presence_mask = np.zeros(raster_shape, dtype=bool)
    presence_mask[observation_rows, observation_cols] = True
    
    # Оцениваем физический размер пикселя (в км) в центре растра для масштабирования
    geod = pyproj.Geod(ellps='WGS84')
    mid_row = raster_shape[0] // 2
    lon1, lat1 = transform * (0, mid_row)
    lon2, lat2 = transform * (1, mid_row)
    _, _, pixel_size_m = geod.inv(lon1, lat1, lon2, lat2)
    pixel_size_km = abs(pixel_size_m) / 1000.0

    use_fmm = any(data is not None for data in [slope_data, elev_data, water_data, bias_data])

    if use_fmm:
        try:
            import skfmm
            # skfmm ищет нулевой контур (границу между + и -). Задаем фон = 1, точки = -1.
            phi = np.ones(raster_shape, dtype=np.float32)
            phi[presence_mask] = -1.0
            
            # Базовая скорость перемещения
            speed = np.ones(raster_shape, dtype=np.float32)
            
            # 0. Штраф за предвзятость наблюдений (Human Bias)
            if bias_data is not None:
                # Нормализуем bias_data в диапазон [0, 1] для предсказуемости
                max_bias = np.nanmax(bias_data)
                if max_bias > 0:
                    normalized_bias = np.nan_to_num(bias_data / max_bias)
                    # Чем выше bias, тем ниже скорость. Используем экспоненциальное затухание.
                    speed *= np.exp(-normalized_bias * 2.0) # Коэффициент 2.0 можно настраивать

            # 1. Штраф за крутые уклоны (Горы обходятся "дорого")
            if slope_data is not None:
                speed *= np.exp(-np.clip(slope_data, 0, None) / 20.0)
                
            # 2. Штраф за высотные барьеры (Экстремальные перевалы и впадины)
            if elev_data is not None:
                pres_elevs = elev_data[observation_rows, observation_cols]
                if len(pres_elevs) > 0:
                    # Допуск: вид может подняться на 500м из-за потепления, или спуститься на 500м
                    max_elev = np.nanmax(pres_elevs) + height_barrier
                    min_elev = np.nanmin(pres_elevs) - height_barrier
                    # Если пиксель выше/ниже предела, скорость падает в 20 раз (непроходимая зона)
                    speed[elev_data > max_elev] *= 0.05
                    speed[elev_data < min_elev] *= 0.05
                    
            # 3. Обработка водных пространств (барьеры или коридоры)
            if water_data is not None:
                # Определяем классы, для которых вода - коридор
                water_corridor_classes = ['Amphibia', 'Pisces'] # Рыбы, если будут

                is_water_species = dclass and any(c in water_corridor_classes for c in dclass)

                if is_water_species:
                    # Для амфибий и рыб вода - это "хайвей". Увеличиваем скорость.
                    # Чем больше воды, тем выше скорость (плавный бонус).
                    speed *= (1.0 + (water_data / 100.0) * 4.0) # Бонус до 5x в полностью водных пикселях
                elif not is_bird:
                    # Для остальных наземных видов вода - барьер.
                    speed[water_data > 50] *= 0.001
            
            speed = np.maximum(speed, 0.001) # Защита от деления на ноль
            
            # Вычисляем Cost-Distance (Время в пути = эквивалент километров с учетом гор)
            tt = skfmm.travel_time(phi, speed, dx=pixel_size_km)
            min_distances_km = np.abs(tt.data)
            min_distances_km[presence_mask] = 0.0
        except ImportError:
            print("\nВНИМАНИЕ: Для учета рельефа/барьеров требуется библиотека 'scikit-fmm'. Выполните 'pip install scikit-fmm'.")
            print("Используется обычное евклидово расстояние.")
            min_distances_km = distance_transform_edt(~presence_mask) * pixel_size_km
    else:
        # Обычное евклидово расстояние в километрах
        min_distances_km = distance_transform_edt(~presence_mask) * pixel_size_km
        
    # Формируем матрицу множителей (от 0.0 до 1.0)
    multiplier = np.ones(raster_shape, dtype=np.float32)

    if decay_type == 'buffer':
        multiplier[min_distances_km > buffer_km] = 0.0
    elif decay_type == 'exponential':
        multiplier = np.exp(-decay_rate * min_distances_km)
        multiplier[min_distances_km > buffer_km] = 0.0
    elif decay_type == 'inverse_quadratic':
        multiplier = 1.0 / (1.0 + decay_rate * (min_distances_km ** 2))
        multiplier[min_distances_km > buffer_km] = 0.0
    elif decay_type == 'gaussian':
        sigma = decay_rate
        multiplier = np.exp(-(min_distances_km ** 2) / (2.0 * (sigma ** 2)))
        multiplier[min_distances_km > buffer_km] = 0.0
    elif decay_type == 'linear':
        multiplier = 1.0 - (min_distances_km / buffer_km)
        multiplier[min_distances_km > buffer_km] = 0.0
    elif decay_type == 'sigmoid':
        # Сигмоида: плавно падает от 1 до 0. 
        # Центр падения (0.5) находится на половине буфера.
        # Крутизна (steepness) настраивается так, чтобы на границе buffer_km значение было около 0.01
        steepness = 10.0 / buffer_km if buffer_km > 0 else 1.0
        mid_point = buffer_km / 2.0
        z = steepness * (min_distances_km - mid_point)
        z = np.clip(z, -700, 700) # Защита от RuntimeWarning: overflow encountered in exp
        multiplier = 1.0 / (1.0 + np.exp(z))
        multiplier[min_distances_km > buffer_km] = 0.0

    # Гарантируем, что в самих точках наблюдений множитель строго равен 1.0
    multiplier[presence_mask] = 1.0
    multiplier = np.clip(multiplier, 0.0, 1.0)
    
    return multiplier
    

# Генерирует комбинированную карту M-фактора для нескольких временных периодов
def generate_combined_m_factor_map(
    raster_shape, transform, observation_rows, observation_cols,
    m_factors_dict, decay_type, decay_rate, height_barrier, slope_data, elev_data, water_data, is_bird,
    current_multiplier, profile, output_tif_path
):
    """
    Генерирует комбинированную карту M-фактора для нескольких временных периодов
    и сохраняет её в GeoTIFF.
    """
    combined_m = np.zeros(raster_shape, dtype=np.float32)
    
    # Накладываем от большего к меньшему, чтобы перекрыть внутренние зоны.
    # Ожидается словарь формата {4: M_FACTOR_2100, 3: M_FACTOR_2070, 2: M_FACTOR_2040}
    for val, buffer_km in sorted(m_factors_dict.items(), reverse=True):
        if buffer_km > 0:
            m_mask = apply_decay_to_points(
                raster_shape=raster_shape, transform=transform, 
                observation_rows=observation_rows, observation_cols=observation_cols, 
                buffer_km=buffer_km, decay_type=decay_type, decay_rate=decay_rate, height_barrier=height_barrier,
                slope_data=slope_data, elev_data=elev_data, water_data=water_data, is_bird=is_bird
            )
            combined_m[m_mask > 0] = val
            
    # Поверх всего кладем текущий M-фактор (самый строгий)
    combined_m[current_multiplier > 0] = 1
    
    # Очищаем фон (заменяем нули на NoData)
    combined_m[combined_m == 0] = np.nan
    save_geotiff(output_tif_path, combined_m, profile)


# Предсказывает пригодность местообитаний (suitability) для всего стека предикторов по батчам.
def predict_suitability_for_stack(model, stack, valid_mask, batch_size=500_000):
    """
    Предсказывает пригодность местообитаний (suitability) для всего стека предикторов по батчам.

    Снижает потребление оперативной памяти при прогнозировании на больших растрах,
    разбивая матрицу признаков на части.

    Args:
        model: Обученная модель классификации (обязан иметь метод `predict_proba`).
        stack (np.ndarray): 3D массив предикторов формы (bands, H, W).
        valid_mask (np.ndarray): 2D булева маска валидных пикселей формы (H, W).
        batch_size (int, optional): Количество пикселей за один проход. По умолчанию 500_000.

    Returns:
        np.ndarray: 2D массив вероятностей (H, W), где невалидные пиксели - np.nan.
    """
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


# Обрезает строки ближайшим пробелом с заданной длинной.
def wrap_long_lines(text, max_len=60):
    """
    Обрезает строки ближайшим пробелом с заданной длинной.

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


# Считывает GeoTIFF файл и при необходимости репроецирует его в EPSG:3857 (Web Mercator).
def read_and_to_3857(path, resampling_method=Resampling.bilinear):
    """
    Считывает GeoTIFF файл и при необходимости репроецирует его в EPSG:3857 (Web Mercator).
    
    Используется для подготовки растровых данных к корректному отображению поверх
    веб-карт (например, OpenStreetMap) без геометрических искажений.
    
    Args:
        path (str): Путь к исходному файлу GeoTIFF.
        resampling_method: Метод ресэмплинга (по умолчанию bilinear). Для категориальных масок нужно Resampling.nearest.
        
    Returns:
        tuple: Кортеж (data, transform, width, height), где:
            - data (np.ndarray): 2D массив значений растра (заполненный np.nan вместо NoData).
            - transform (affine.Affine): Аффинная матрица трансформации.
            - width (int): Ширина растра в пикселях.
            - height (int): Высота растра в пикселях.
    """
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
                resampling=resampling_method,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
    return data, transform, width, height


# Округляет число до заданного количества значащих цифр.
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
    
    if math.isnan(number) or math.isinf(number):
        return float(number)
    
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


# Вычисляет показатель подобия двух гистограмм.
def calculate_histogram_similarity(data_obs, data_full, bins_num=50, sig_figs=4, bins_range=None):
    """
    Вычисляет показатель подобия двух распределений (гистограмм).
    
    Использует коэффициент корреляции Пирсона между нормализованными плотностями
    вероятности двух наборов данных на одинаковой сетке интервалов.
    
    Args:
        data_obs (np.ndarray): Массив значений в точках наблюдений.
        data_full (np.ndarray): Массив значений на всем доступном фоне (ландшафте).
        bins_num (int, optional): Количество интервалов разбиения (бинов). По умолчанию 50.
        sig_figs (int, optional): Количество значащих цифр для округления результата. По умолчанию 4.
        bins_range (tuple, optional): Кортеж (min, max) для ускорения.
        
    Returns:
        float: Значение от 0.0 до 1.0 (1.0 - максимальное сходство, 0.0 - сходства нет).
    """

    if len(data_obs) == 0 or len(data_full) == 0:
        return 0.0
    
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
            # Проверка на нулевую дисперсию, чтобы избежать ConstantInputWarning
            if np.std(density_obs[mask]) < 1e-8 or np.std(density_full[mask]) < 1e-8:
                correlation = 0.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr_coeff, _ = pearsonr(density_obs[mask], density_full[mask])
                    correlation = corr_coeff if not np.isnan(corr_coeff) else 0.0
            
    except Exception as e:
        print(f"Ошибка при расчете корреляции Пирсона: {e}")
        correlation = 0.0

    # посоветовали использовать Schoener's D для оценки похожести гистограмм
    schoeners_d = 1.0 - 0.5 * np.sum(np.abs(density_obs - density_full))
    return round_to_significant_figures(schoeners_d, sig_figs)


# Рассчитывает многомерную экологическую пластичность (Niche Breadth).
def calculate_niche_breadth_pca(X_pres: np.ndarray, X_bg: np.ndarray, top_indices: list = None) -> float:
    """
    Рассчитывает многомерную экологическую пластичность (Niche Breadth) 
    на основе площади минимального выпуклого многоугольника (MCP) 
    в пространстве первых двух главных компонент (PCA).
    
    Args:
        X_pres (np.ndarray): Матрица предикторов в точках присутствия.
        X_bg (np.ndarray): Матрица предикторов для фоновых точек (весь доступный ландшафт).
        top_indices (list, optional): Индексы колонок топ-предикторов. Если None, используются все.
        
    Returns:
        float: Значение пластичности от 0.0 (экстремальный специалист) до 1.0+ (генералист).
    """
    if len(X_pres) < 3 or len(X_bg) < 3:
        return 0.0 # Для построения 2D полигона нужно минимум 3 точки
        
    if top_indices is not None:
        X_pres_sub = X_pres[:, top_indices]
        X_bg_sub = X_bg[:, top_indices]
    else:
        X_pres_sub = X_pres
        X_bg_sub = X_bg
        
    # 1. Стандартизация (Z-score) на основе доступного фона
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_bg_sub, X_pres_sub]))
    X_bg_scaled = scaler.transform(X_bg_sub)
    X_pres_scaled = scaler.transform(X_pres_sub)
    
    # 2. PCA (сжатие до 2-х измерений)
    pca = PCA(n_components=2)
    pca.fit(X_bg_scaled) # Обучаем PCA на фоне (чтобы оси отражали макроклимат)
    
    bg_pca = pca.transform(X_bg_scaled)
    pres_pca = pca.transform(X_pres_scaled)
    
    # 3. Вычисление площадей Convex Hull (Выпуклых оболочек)
    try:
        area_bg = ConvexHull(bg_pca).volume # В 2D пространстве volume возвращает площадь
        area_pres = ConvexHull(pres_pca).volume
        return float(area_pres / area_bg) if area_bg > 0 else 0.0
    except Exception:
        return 0.0


# Вычисляет основные статистические показатели для набора данных.
def get_predictor_stats(data: np.ndarray) -> dict:
    """
    Вычисляет основные статистические показатели для одномерного набора данных.
    
    Рассчитывает среднее, медиану, минимумы/максимумы, процентили (5, 95), 
    стандартное отклонение, асимметрию и эксцесс. Автоматически игнорирует NaN.
    
    Args:
        data (np.ndarray): Входной массив числовых данных.
        
    Returns:
        dict: Словарь со статистическими показателями.
    """
    
    # Оптимизация памяти: избегаем копирования 9Мб+ массива, если он уже очищен от NaN
    if np.isnan(data).any():
        data = data[~np.isnan(data)]

    if data.size == 0:
        return {
            'mean': np.nan, 'median': np.nan, 'min': np.nan, 'max': np.nan,
            'p5': np.nan, 'p95': np.nan, 'width_obs': np.nan,
            'std_dev': np.nan, 'skewness': np.nan, 'kurtosis': np.nan,
        }
    
    # Оптимизация: вычисляем все процентили за один проход по массиву 
    p_vals = np.percentile(data, [5, 10, 90, 95])
    
    stats = {
        'mean': round_to_significant_figures(np.mean(data), 4),
        'median': round_to_significant_figures(np.median(data), 4),
        'min': round_to_significant_figures(np.min(data), 4),
        'max': round_to_significant_figures(np.max(data), 4),
        'p5': round_to_significant_figures(p_vals[0], 4),
        'p95': round_to_significant_figures(p_vals[3], 4),
        'width_obs': round_to_significant_figures(p_vals[2] - p_vals[1], 4),
        'std_dev': 0,
        'skewness': 0,
        'kurtosis': 0,
    }
    
    try:
        std_val = np.std(data)
        stats['std_dev'] = round_to_significant_figures(std_val, 4)  # Стандартное отклонение
    except Exception as e:
        print('Ошибка вычисления статистики std_dev: ' + str(e))
        std_val = 0.0
    
    try:
        if std_val < 1e-8:
            stats['skewness'] = 0.0
            stats['kurtosis'] = 0.0
        else:
            stats['skewness'] = round_to_significant_figures(skew(data), 4)  # Асимметрия (Skewness)
            stats['kurtosis'] = round_to_significant_figures(kurtosis(data), 4)  # Эксцесс (Kurtosis)
    except Exception as e:
        print('Ошибка вычисления статистики skew/kurtosis: ' + str(e))
    
    return stats


# Форматирует число с плавающей точкой для отображения (убирает лишние нули).
def format_float(value: float) -> str:
    """
    Форматирует число с плавающей точкой для компактного отображения.
    
    Округляет до 4 знаков после запятой и удаляет незначащие нули в конце.
    
    Args:
        value (float): Исходное число.
        
    Returns:
        str: Отформатированная строка.
    """
    return f"{value:.4f}".rstrip('0').rstrip('.')


# Сохраняет текст ошибки в указанный текстовый файл.
def save_error(error_path, text):
    """
    Сохраняет текст ошибки в указанный текстовый файл.
    
    Args:
        error_path (str): Путь к файлу лога ошибки.
        text (str|Exception): Текст ошибки или объект исключения.
    """
    with open(error_path, 'w') as f: # записываем файл
        f.write(str(text))


# Логирование ошибок
def handle_model_error(error_obj, error_filename, model_data, json_filename, context_msg=None):
    """
    Обрабатывает ошибку моделирования: выводит в консоль, сохраняет в файл лога,
    обновляет статус в JSON словаре и возвращает словарь для прерывания пайплайна.
    """
    if context_msg:
        print(context_msg)
    print(str(error_obj))
    save_error(error_filename, error_obj)
    if model_data is not None:
        model_data['status'] = 'error'
        model_data['error'] = str(error_obj)
        save_json(model_data, json_filename)
    return {'status': 'terminated', 'error': str(error_obj), 'code': 401}


# Вычисляет физическую площадь (в кв. км) и количество пикселей, превышающих заданные пороги вероятности.
def get_geotiff_square(filepath: str, threshold) -> dict:
    """
    Вычисляет физическую площадь (в кв. км) и количество пикселей, превышающих заданные пороги вероятности.

    Использует эллипсоид WGS84 (pyproj.Geod) для точного расчета площади каждого пикселя
    в зависимости от его широты (компенсация искажений проекции EPSG:4326).

    Args:
        filepath (str): Путь к растровому файлу GeoTIFF с вероятностями.
        threshold (list of float): Список пороговых значений (от 0 до 1).

    Returns:
        tuple: (out_square, out_num) - списки с площадью (км²) и количеством пикселей для каждого порога.
    """
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


# Сохраняет 2D массив numpy в формате GeoTIFF.
def save_geotiff(output_path, array2d, profile):
    """
    Сохраняет 2D массив numpy в формате GeoTIFF.
    
    Создает необходимые директории, если они не существуют, и применяет
    переданный профиль метаданных (CRS, трансформацию, NoData).
    
    Args:
        output_path (str): Целевой путь для сохранения файла.
        array2d (np.ndarray): 2D массив данных растра.
        profile (dict): Словарь с метаданными Rasterio.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prof = profile.copy()
    with rasterio.open(output_path, "w", **prof) as dst:
        dst.write(array2d.astype("float32"), 1)


# Выполняет обратное масштабирование значений предиктора в оригинальные единицы.
def inverse_scale(scaled_data, scale_params, info):
    """
    Выполняет обратное масштабирование значений предиктора в оригинальные единицы.
    
    Используется для перевода стандартизированных данных (например, умноженных на 10 
    или со смещением) обратно в понятные физические величины (градусы, миллиметры).
    
    Args:
        scaled_data (np.ndarray): Массив масштабированных данных.
        scale_params (dict): Параметры масштабирования, полученные из JSON конфигурации.
        info (dict): Метаданные предиктора из справочника (содержит scale, diff).
        
    Returns:
        np.ndarray: Массив значений в оригинальных физических единицах.
    """
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
        # Оптимизация производительности: считаем итоговые скаляры ДО умножения огромных матриц!
        # Это избавляет Numpy от выделения 4 лишних массивов в оперативной памяти сервера.
        if dsca:
            final_scale = scale / dsca
            final_mean = mean / dsca
        else:
            final_scale = scale
            final_mean = mean
        return scaled_data * final_scale + final_mean
    else:
        return scaled_data


# Рекурсивно заменяет NaN и Infinity на None (null в JSON)
def clean_nans_for_json(obj):
    """
    Рекурсивно заменяет NaN и Infinity на None (null в JSON),
    чтобы избежать синтаксических ошибок при парсинге клиентами. 
    NB: В языке Python стандартный модуль json по умолчанию допускает запись значений NaN (Not a Number), 
    Infinity и -Infinity. Однако официальный стандарт формата JSON (RFC 8259) не поддерживает эти значения.
    
    Args:
        obj: Словарь, список или примитив для очистки.
        
    Returns:
        Очищенный объект.
    """
    if isinstance(obj, dict):
        return {k: clean_nans_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_nans_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return [clean_nans_for_json(v) for v in obj.tolist()]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj


# Извлекает значения предикторов из 3D-стека по координатам (индексам) пикселей.
def extract_features_from_stack(stack, rows, cols):
    """
    Извлекает значения предикторов из 3D-стека по координатам (индексам) пикселей.

    Args:
        stack (np.ndarray): 3D массив предикторов формы (bands, H, W).
        rows (np.ndarray): Массив целочисленных индексов строк.
        cols (np.ndarray): Массив целочисленных индексов столбцов.

    Returns:
        np.ndarray: 2D матрица признаков формы (n_samples, n_bands).
    """
    # stack: (bands, H, W)
    # fancy-indexing
    return stack[:, rows, cols].T  # (n_samples, n_bands)


# Сэмплирует n_bg фоновых пикселей, разделяя их на две части
def sample_background(valid_mask, presence_rc_set, n_bg, rng, bg_pc = 100,
                      distance_min_pixels = 1, distance_max_pixels = 1, bias_weights_map=None, bias_scale_params=None,
                      bias_sampling_strength=1.0):
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
        bias_weights_map (np.ndarray, optional): 2D массив с весами для сэмплирования (слой human bias).
        bias_scale_params (dict, optional): Словарь с 'mean' и 'scale' для обратного масштабирования bias_weights_map.
        bias_sampling_strength (float, optional): Доля (0.0-1.0) случайных точек, которые будут сэмплироваться с учетом bias.

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

        if bias_weights_map is not None and bias_sampling_strength > 0 and candidates_random.size > 0:
            print(f"Применяется гибридное взвешенное сэмплирование (сила = {bias_sampling_strength * 100}%)...")
            
            n_biased = int(n_bg_random * bias_sampling_strength)
            n_uniform = n_bg_random - n_biased

            # --- Часть 1: Взвешенное сэмплирование ---
            chosen_biased = np.array([], dtype=np.int64)
            if n_biased > 0 and candidates_random.size > 0:
                weights = bias_weights_map.ravel()[candidates_random]
                
                # 1. Обратное масштабирование, если параметры переданы
                if bias_scale_params and 'mean' in bias_scale_params and 'scale' in bias_scale_params:
                    print("Выполняется обратное масштабирование весов предвзятости...")
                    mean = bias_scale_params['mean']
                    scale = bias_scale_params['scale']
                    weights = weights * scale + mean
                
                # 2. Обеспечиваем неотрицательность и сглаживаем
                # Устанавливаем отрицательные значения в 0, так как они означают плотность наблюдений ниже средней
                weights[weights < 0] = 0
                # Логарифмическое преобразование для сглаживания экстремальных пиков
                weights = np.log1p(weights.astype(np.float64)) # log1p(x) = log(1+x)

                total_weight = np.sum(weights)
                if total_weight > 0 and not np.isnan(total_weight):
                    probabilities = weights / total_weight
                    # Защита от ситуации, когда n_biased > len(candidates_random)
                    size_to_sample = min(n_biased, len(candidates_random))
                    try:
                        chosen_biased = rng.choice(candidates_random, size=size_to_sample, replace=False, p=probabilities)
                    except ValueError as e:
                        # Эта ошибка может возникнуть, если сумма вероятностей не равна 1.0 из-за ошибок округления.
                        # В этом случае нормализуем еще раз.
                        print(f"Предупреждение при взвешенном сэмплировании: {e}. Повторная нормализация вероятностей.")
                        probabilities /= np.sum(probabilities)
                        chosen_biased = rng.choice(candidates_random, size=size_to_sample, replace=False, p=probabilities)
                else: # Если все веса нулевые, эта часть будет пустой
                    n_uniform += n_biased # Добавляем недостающие точки к равномерной выборке

            # --- Часть 2: Равномерное случайное сэмплирование ---
            chosen_uniform = np.array([], dtype=np.int64)
            if n_uniform > 0 and candidates_random.size > 0:
                # Исключаем уже выбранные взвешенные точки, чтобы не было дублей
                remaining_candidates = np.setdiff1d(candidates_random, chosen_biased, assume_unique=True)
                if remaining_candidates.size > 0:
                    n_uniform = min(n_uniform, remaining_candidates.size)
                    chosen_uniform = rng.choice(remaining_candidates, size=n_uniform, replace=False)

            chosen_random = np.concatenate((chosen_biased, chosen_uniform))

        else:
            # Стандартное равномерное сэмплирование
            print("Применяется стандартное равномерное сэмплирование фона (bias слой не используется или сила=0)...")
            chosen_random = rng.choice(candidates_random, size=n_bg_random, replace=False)

        rows_random = chosen_random // width
        cols_random = chosen_random % width

    print(f"Сгенерировано фоновых точек: {n_bg_random}")

    # если не требуется брать точки с границ
    if bg_pc == 100:
        return rows_random, cols_random, rows_random, cols_random, [], []
    
    # --- Часть 2: Фон в "огибающей" (буфере) ---
    # Вычисляем, сколько точек нужно для буферной части
    n_bg_buffer_target = int(n_bg - len(rows_random)) #* (100-bg_pc) / 100
    
    print(f"Нужно сгенерировать точек псевдоприсутствия: {n_bg_buffer_target}")
    
    if n_bg_buffer_target <= 0:
        # Если уже набрали достаточно случайных точек, возвращаем их
        return rows_random, cols_random, rows_random, cols_random, [], []
    
    # Если точек присутствия нет, буферная часть не может быть сгенерирована
    if not presence_rc_set:
        print("ВНИМАНИЕ: Отсутствуют точки присутствия для генерации фона в огибающей.")
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


# Сохранение данных модели в json-файл
def save_json(data_dict, filepath):
    """
    Сохраняет словарь в формате JSON, предварительно очистив его от NaN/Infinity.
    """
    try:
        cleaned_data = clean_nans_for_json(data_dict)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Ошибка сохранения JSON: {e}")