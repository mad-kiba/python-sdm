import numpy as np
import os
import glob
import rasterio
from affine import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
from rasterio.transform import array_bounds
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt

# --- Функция для обработки одного GeoTIFF файла ---
def process_single_geotiff(input_filepath: str, output_dir: str, target_crs: str, target_resolution_deg: float, bbox: tuple[float, float, float, float]):
    """
    Проверяет CRS и разрешение GeoTIFF, при необходимости репроектирует и изменяет разрешение,
    затем обрезает по заданному bbox, гарантируя точное совпадение границ и разрешения.

    Args:
        input_filepath (str): Путь к исходному GeoTIFF файлу.
        output_dir (str): Директория для сохранения обработанного файла.
        target_crs (str): Желаемая система координат.
        target_resolution_deg (float): Желаемое разрешение в градусах.
        bbox (tuple[float, float, float, float]): Границы обрезки (min_lon, min_lat, max_lon, max_lat).

    Returns:
        str: Путь к созданному GeoTIFF файлу, или None в случае ошибки.
    """
    temp_output_filepath = None # Переменная для временного файла

    try:
        with rasterio.open(input_filepath) as src:
            print(f"\nОбработка файла: {os.path.basename(input_filepath)}")
            print(f"  Исходный CRS: {src.crs}")
            print(f"  Исходное разрешение: {src.res[0]:.6f} x {src.res[1]:.6f} градусов")
            print(f"  Исходные границы: {src.bounds}")

            src_crs = src.crs
            
            # 1. Определяем ТОЧНЫЕ границы и трансформацию для целевого выходного растра.
            # Мы хотим, чтобы выходной растр ТОЧНО соответствовал bbox и разрешению.
            # Для этого calculate_default_transform должен работать от "целевого" представления.
            
            # Преобразуем bbox в целевую CRS, если это еще не сделано
            target_bounds_for_crop = rasterio.warp.transform_bounds(src_crs, target_crs, *bbox)

            # Рассчитываем трансформацию, размеры и разрешение, которые будут ТОЧНО соответствовать bbox и target_resolution_deg
            final_transform, final_width, final_height = calculate_default_transform(
                src_crs, # CRS исходного растра
                target_crs, # Целевая CRS
                src.width, src.height, *src.bounds, # Исходные размеры и границы
                resolution=target_resolution_deg, # Желаемое разрешение
                bounds=target_bounds_for_crop # И, самое главное, целевые границы
            )
            
            # Корректируем final_transform, чтобы он точно начинался с target_bounds_for_crop
            # Этот шаг часто необходим для точного выравнивания сетки пикселей
            # Но сначала проверим, совпадают ли границы напрямую
            if not np.allclose(list(final_transform * (final_width, final_height)) + list(final_transform * (0, 0)), 
                               [target_bounds_for_crop[2], target_bounds_for_crop[0], target_bounds_for_crop[3], target_bounds_for_crop[1]]):
                # Если автоматический расчет не дал идеального выравнивания,
                # явно задаем трансформацию, начиная с target_bounds_for_crop
                final_transform = Affine(
                    target_resolution_deg, 0.0, target_bounds_for_crop[0],
                    0.0, -target_resolution_deg, target_bounds_for_crop[3]
                )
                # Пересчитываем размеры, чтобы они соответствовали bbox и новому разрешению
                final_width = int(np.ceil((target_bounds_for_crop[2] - target_bounds_for_crop[0]) / target_resolution_deg))
                final_height = int(np.ceil((target_bounds_for_crop[3] - target_bounds_for_crop[1]) / target_resolution_deg))


            # Определение имени выходного файла
            base_name = os.path.splitext(os.path.basename(input_filepath))[0]
            output_filename = f"{base_name}.tif"
            final_output_filepath = os.path.join(output_dir, output_filename)

            # 2. Создаем новый GeoTIFF файл и репроектируем/обрезаем данные
            with rasterio.open(
                final_output_filepath,
                'w',
                driver='GTiff',
                height=final_height,
                width=final_width,
                count=1,
                dtype=src.dtypes[0], # Используем тип данных из исходного файла
                crs=target_crs,
                transform=final_transform,
                nodata=src.nodata, # Сохраняем nodata значение, если оно есть
                compress='LZW',
                tiled=True
            ) as final_dst:
                # Выполняем репроецирование с учетом точной целевой трансформации и границ
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(final_dst, 1),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=final_dst.transform,
                    dst_crs=final_dst.crs,
                    resampling=ResamplingEnum.bilinear, # Используем билинейную для изменения разрешения
                    bounds=target_bounds_for_crop # Указываем границы, по которым нужно обрезать
                )
            
            print(f"  Сохранен файл: {final_output_filepath}")
            # Читаем сохраненный файл, чтобы вывести его точные параметры
            with rasterio.open(final_output_filepath) as final_saved_src:
                print(f"    CRS: {final_saved_src.crs}, Разрешение: {final_saved_src.res[0]:.6f}, Границы: {final_saved_src.bounds}")

            return final_output_filepath

    except rasterio.errors.RasterioIOError as e:
        print(f"  Ошибка чтения/записи файла {os.path.basename(input_filepath)}: {e}")
        return None
    except Exception as e:
        print(f"  Непредвиденная ошибка при обработке {os.path.basename(input_filepath)}: {e}")
        # Попытка удалить временный файл, если он существует
        if temp_output_filepath and os.path.exists(temp_output_filepath):
            os.remove(temp_output_filepath)
        return None


def clip_rasters(RAW_RASTER_DIR, OUTPUT_RASTER_DIR, IN_MIN_LAT, IN_MIN_LON, IN_MAX_LAT, IN_MAX_LON, MODEL_FUTURE):
    if IN_MIN_LAT==0:
        return
    
    print("\n-- Обработка предикторов")
    #print(RAW_RASTER_DIR)
    #print(OUTPUT_RASTER_DIR)
    
    TARGET_RESOLUTION_DEG = 30 / 3600.0  
    TARGET_CRS = "EPSG:4326"
    BBOX = (IN_MIN_LAT, IN_MIN_LON, IN_MAX_LAT, IN_MAX_LON)
    
    os.makedirs(OUTPUT_RASTER_DIR, exist_ok=True)
    
    processed_files_count = 0

    for root, _, files in os.walk(RAW_RASTER_DIR):
        if MODEL_FUTURE == 0 and root != RAW_RASTER_DIR:
            continue  # обрабатывать только файлы в корневой папке
    
        # Относительный путь текущей подпапки относительно корня RAW_RASTER_DIR
        rel_dir = os.path.relpath(root, RAW_RASTER_DIR)
        # Для корня rel_dir == '.', поэтому избегаем добавления '.' в путь
        output_subdir = os.path.join(OUTPUT_RASTER_DIR, '' if rel_dir == '.' else rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
        
        for filename in files:
            if filename.lower().endswith((".tif", ".tiff")):
                input_filepath = os.path.join(root, filename)
    
                base_name = os.path.splitext(os.path.basename(input_filepath))[0]
                output_filename = f"{base_name}.tif"
    
                # Итоговый путь включает подпапку (или её отсутствие, если файл лежит в корне)
                final_output_filepath = os.path.join(output_subdir, output_filename)
    
                if os.path.isfile(input_filepath) and not os.path.isfile(final_output_filepath):
                    print()
                    print(str(processed_files_count)+' = '+input_filepath)
                    result_path = process_single_geotiff(
                        input_filepath,
                        output_subdir,  # сохраняем в соответствующую подпапку (или корень, если MODEL_FUTURE == 0)
                        TARGET_CRS,
                        TARGET_RESOLUTION_DEG,
                        BBOX
                    )
    
                    if result_path:
                        processed_files_count += 1
            
    
    print(f"-- Обработка предикторов завершена")
    print(f"Обработано файлов: {processed_files_count}")
    print(f"Результаты сохранены в папку: '{OUTPUT_RASTER_DIR}'")


def load_raster_stack(raster_dir, predictors = 'all'):
    """Считывает все GeoTIFF из папки и строит стек (bands, H, W).
       Возвращает: stack(float32), valid_mask(bool), transform, crs, profile, band_names(list)"""
    
    all_tifs = sorted(glob.glob(os.path.join(raster_dir, "*.tif")))
    
    # фильтруем по входящему списку предикторов
    if predictors.strip().lower() == 'all':
        tifs = sorted(glob.glob(os.path.join(raster_dir, "*.tif")))
    else:
        predictor_files = [f"{p.strip()}.tif" for p in predictors.split(',')]
        tifs = [f_path for f_path in all_tifs if os.path.basename(f_path) in predictor_files]
    
    if not tifs:
        expected_files_str = ", ".join(predictor_files)
        raise FileNotFoundError(f"В папке {raster_dir} не найдены файлы, соответствующие предикторам: {expected_files_str}. "
                                f"Найдены только: {', '.join([os.path.basename(f) for f in all_tifs]) if all_tifs else 'ни одного'}")
    
    band_arrays = []
    band_names = []
    ref_transform = None
    ref_width = ref_height = None
    ref_crs = None
    
    for i, fp in enumerate(tifs):
        with rasterio.open(fp) as ds:
            arr = ds.read(1, masked=True).astype("float32")  # masked -> маскирует nodata
            arr = np.ma.filled(arr, np.nan)                  # превращаем masked в np.nan
            if i == 0:
                ref_transform = ds.transform
                ref_width, ref_height = ds.width, ds.height
                ref_crs = ds.crs
            else:
                # Проверки согласованности
                if ds.transform != ref_transform or ds.width != ref_width or ds.height != ref_height:
                    raise ValueError(f"Растр {fp} не согласован по геометрии с первым растром")
                if ds.crs != ref_crs:
                    raise ValueError(f"Растр {fp} имеет другой CRS: {ds.crs} vs {ref_crs}")
            band_arrays.append(arr)
            band_names.append(os.path.splitext(os.path.basename(fp))[0])
    stack = np.stack(band_arrays, axis=0)  # shape: (bands, H, W)
    # Маска валидных пикселей: валиден, если нет NaN во всех слоях
    valid_mask = np.all(~np.isnan(stack), axis=0)
    # Профиль для сохранения результата
    profile = {
        "driver": "GTiff",
        "height": ref_height,
        "width": ref_width,
        "count": 1,
        "dtype": "float32",
        "crs": ref_crs,
        "transform": ref_transform,
        "compress": "lzw",
        "nodata": np.nan
    }
    return stack, valid_mask, ref_transform, ref_crs, profile, band_names


def points_to_pixel_indices(lons, lats, transform, width, height):
    """Преобразует координаты (lon, lat) в индексы пикселей (row, col).
       Возвращает row, col и маску тех, кто внутри границ растра."""
    # rasterio.transform.rowcol даёт целые индексы (row, col) для x,y
    from rasterio.transform import rowcol
    rows, cols = rowcol(transform, lons, lats, op=float)  # float -> потом приведём к int с проверкой
    rows = np.floor(rows).astype(int)
    cols = np.floor(cols).astype(int)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return rows, cols, inside


def extract_features_from_stack(stack, rows, cols):
    """Извлекает значения предикторов из стека по индексам пикселей.
       Возвращает X: (n_samples, n_bands)."""
    # stack: (bands, H, W)
    # fancy-indexing
    return stack[:, rows, cols].T  # (n_samples, n_bands)


# Сэмплирует n_bg фоновых пикселей, разделяя их на две части
def sample_background(valid_mask, presence_rc_set, n_bg, rng, bg_pc = 100, distance_min_pixels = 1, distance_max_pixels = 1, text_filename = ''):
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
    if bg_pc==100:
        with open(text_filename, 'a') as f:
            f.write(f"\n{len(rows_random)}")
        return rows_random, cols_random
    
    # --- Часть 2: Фон в "огибающей" (буфере) ---
    # Вычисляем, сколько точек нужно для буферной части
    n_bg_buffer_target = int(n_bg - len(rows_random)) #* (100-bg_pc) / 100
    
    print(f"Нужно сгенерировать точек псевдоприсутствия: {n_bg_buffer_target}")
    
    if n_bg_buffer_target <= 0:
        # Если уже набрали достаточно случайных точек, возвращаем их
        with open(text_filename, 'a') as f:
            f.write(f"\n{len(rows_random)}")
        return rows_random, cols_random
    
    # Если точек присутствия нет, буферная часть не может быть сгенерирована
    if not presence_rc_set:
        print("ВНИМАНИЕ: Отсутствуют точки присутствия для генерации фона в огибающей.")
        with open(text_filename, 'a') as f:
            f.write(f"\n{len(rows_random)}")
        return rows_random, cols_random
    
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
        print(f"ВНИМАНИЕ: Сгенерировано {current_total} фоновых точек, вместо желаемых {n_bg}. Попытка добрать {remaining_to_sample} из оставшейся территории.")

        # Создаем маску для оставшихся кандидатов:
        # Это должны быть валидные пиксели, которые еще не были выбраны.
        final_selection_mask = valid_mask.copy()

        # Исключаем точки присутствия
        if presence_rc_set:
            pres_linear_indices = []
            for r, c in presence_rc_set:
                if 0 <= r < height and 0 <= c < width and valid_mask[r, c]:
                    pres_linear_indices.append(r * width + c)
            
            if pres_linear_indices:
                pres_linear_indices = np.array(pres_linear_indices, dtype=np.int64)
                presence_linear_mask = np.zeros(valid_mask.size, dtype=bool)
                presence_linear_mask[pres_linear_indices] = True
                final_selection_mask.ravel()[presence_linear_mask] = False

        # Исключаем точки, которые уже были выбраны (из случайной и буферной частей)
        already_chosen_linear = []
        if len(rows_random) > 0:
            already_chosen_linear.extend(rows_random * width + cols_random)
        if len(rows_buffer) > 0:
            already_chosen_linear.extend(rows_buffer * width + cols_buffer)
        
        if already_chosen_linear:
            already_chosen_linear = np.array(already_chosen_linear, dtype=np.int64)
            # Убедимся, что индексы в пределах размера массива
            valid_already_chosen_linear = already_chosen_linear[(already_chosen_linear >= 0) & (already_chosen_linear < final_selection_mask.size)]
            
            if len(valid_already_chosen_linear) > 0:
                already_chosen_mask = np.zeros(final_selection_mask.size, dtype=bool)
                already_chosen_mask[valid_already_chosen_linear] = True
                final_selection_mask.ravel()[already_chosen_mask] = False

        final_candidates = np.flatnonzero(final_selection_mask)
        
        if final_candidates.size > 0:
            n_to_add = min(remaining_to_sample, final_candidates.size)
            chosen_extra = rng.choice(final_candidates, size=n_to_add, replace=False)
            rows_extra = chosen_extra // width
            cols_extra = chosen_extra % width
            all_rows = np.concatenate((all_rows, rows_extra))
            all_cols = np.concatenate((all_cols, cols_extra))

    # Перемешиваем финальный набор точек
    if len(all_rows) > 0:
        indices = rng.permutation(len(all_rows))
        all_rows = all_rows[indices]
        all_cols = all_cols[indices]
        
    with open(text_filename, 'a') as f:
            f.write(f"\n{len(rows_random)},{len(rows_buffer)}")

    return all_rows, all_cols