return
# todo

# examples/example_basic.py

import os
from sdm import perform_sdm_modeling # Импортируем основную функцию

# --- Параметры ---
PRESENCE_DATA = 'path/to/your/occurrences.csv'
PREDICTOR_CURRENT = 'path/to/input_raw_predictors'
FUTURE_PREDICTORS = {
    '2021-2040': 'path/to/input_raw_predictors/2021-2040',
    '2041-2060': 'path/to/input_raw_predictors/2041-2060',
    # ... другие периоды
}
OUTPUT_DIR = 'sdm_results/my_first_model'
REGION_BBOX = (10.0, 50.0, 20.0, 55.0) # min_lon, min_lat, max_lon, max_lat
RESOLUTION = 0.1 # Градусов
MODEL_TYPE = 'maxent' # Или другой тип, если вы его поддерживаете

# Создаем выходную директорию, если она не существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Запуск моделирования ---
try:
    results = perform_sdm_modeling(
        presence_data_path=PRESENCE_DATA,
        predictor_dir=PREDICTOR_CURRENT,
        future_predictor_dirs=FUTURE_PREDICTORS,
        output_dir=OUTPUT_DIR,
        region_bbox=REGION_BBOX,
        resolution_deg=RESOLUTION,
        model_type=MODEL_TYPE,
        # Дополнительные параметры для генерации фона, если нужно
        bg_mult=10,
        bg_distance_min_pixels=5,
        bg_distance_max_pixels=20,
        bg_pc=100
    )

    print("\nМоделирование успешно завершено. Результаты сохранены в:", OUTPUT_DIR)
    print("Сгенерированные файлы:")
    for key, val in results.items():
        print(f"- {key}: {val}")

except FileNotFoundError as e:
    print(f"Ошибка: Не найден файл или директория - {e}")
except Exception as e:
    print(f"Произошла ошибка во время моделирования: {e}")
