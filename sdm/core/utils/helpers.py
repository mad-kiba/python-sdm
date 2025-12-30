import numpy as np
import math
import os
import rasterio

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


def get_predictor_stats(data: np.ndarray) -> dict:
    """
    Вычисляет основные статистические показатели для набора данных.
    """
    
    stats = {
        'mean': round_to_significant_figures(np.mean(data), 4),
        'median': round_to_significant_figures(np.median(data), 4),
        'min': round_to_significant_figures(np.min(data), 4),
        'max': round_to_significant_figures(np.max(data), 4),
        'p5': round_to_significant_figures(np.percentile(data, 5), 4),
        'p95': round_to_significant_figures(np.percentile(data, 95), 4)
    }
    
    return stats


def format_float(value: float) -> str:
    """
    Форматирует число с плавающей точкой для отображения (убирает лишние нули).
    """
    return f"{value:.4f}".rstrip('0').rstrip('.')


def save_geotiff(output_path, array2d, profile):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prof = profile.copy()
    with rasterio.open(output_path, "w", **prof) as dst:
        dst.write(array2d.astype("float32"), 1)
        

def inverse_scale(scaled_data, scale_params):
    if scale_params is None or "mean" not in scale_params or "scale" not in scale_params:
        return scaled_data
    method = scale_params.get("method", "standard")
    mean = scale_params["mean"]
    scale = scale_params["scale"]
    if method == "standard":
        return scaled_data * scale + mean
    else:
        return scaled_data