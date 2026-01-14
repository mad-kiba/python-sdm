import os
import rasterio
import numpy as np
from affine import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
from rasterio.transform import array_bounds
from rasterio.crs import CRS


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


def clip_rasters(RAW_RASTER_DIR, OUTPUT_RASTER_DIR, IN_MIN_LAT, IN_MIN_LON, IN_MAX_LAT, IN_MAX_LON, MODEL_FUTURE, IN_RESOLUTION):
    if IN_MIN_LAT==0:
        return
    
    print("\n-- Обработка предикторов")
    #print(RAW_RASTER_DIR)
    #print(OUTPUT_RASTER_DIR)
    
    if IN_RESOLUTION == '30s':
        TARGET_RESOLUTION_DEG = 30 / 3600.0
    if IN_RESOLUTION == '1m':
        TARGET_RESOLUTION_DEG = 2 * 30 / 3600.0
    if IN_RESOLUTION == '5m':
        TARGET_RESOLUTION_DEG = 10 * 30 / 3600.0    
        
    TARGET_CRS = "EPSG:4326"
    BBOX = (IN_MIN_LAT, IN_MIN_LON, IN_MAX_LAT, IN_MAX_LON)
    
    os.makedirs(OUTPUT_RASTER_DIR, exist_ok=True)
    
    processed_files_count = 0

    for root, _, files in os.walk(RAW_RASTER_DIR):
        # пропускаем обработку предикторов будущего, если это не требуется
        if MODEL_FUTURE == 0:
            if '2021-2040' in root:
                continue
            if '2041-2060' in root:
                continue
            if '2061-2080' in root:
                continue
            if '2081-2100' in root:
                continue
    
        # Относительный путь текущей подпапки относительно корня RAW_RASTER_DIR
        rel_dir = os.path.relpath(root, RAW_RASTER_DIR)
        # Для корня rel_dir == '.', поэтому избегаем добавления '.' в путь
        output_subdir = os.path.join(OUTPUT_RASTER_DIR, '' if rel_dir == '.' else rel_dir)
        os.makedirs(output_subdir, exist_ok=True)
        
        #print(f"\nТекущая папка: {root}")
        
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



def pixel_indices_to_points(rows, cols, transform, width, height):
    """
    Преобразует индексы пикселей (row, col) обратно в координаты (lon, lat).

    Args:
        rows (np.ndarray): Массив индексов строк пикселей.
        cols (np.ndarray): Массив индексов столбцов пикселей.
        transform (affine.Affine): Объект Transform из rasterio, описывающий
                                   геопривязку растра.
        width (int): Ширина растра (количество столбцов).
        height (int): Высота растра (количество строк).

    Returns:
        tuple: Кортеж, содержащий:
            - lons (np.ndarray): Массив долгот, соответствующих индексам пикселей.
            - lats (np.ndarray): Массив широт, соответствующих индексам пикселей.
            - inside (np.ndarray): Булев массив, указывающий, какие пиксели
                                   находятся в пределах границ растра.
    """
    # Создаем массивы для хранения результатов
    lons = np.empty(len(rows))
    lats = np.empty(len(rows))
    inside = np.empty(len(rows), dtype=bool)
    
    # Итерируемся по каждому индексу пикселя
    for i, (row, col) in enumerate(zip(rows, cols)):
        # Проверяем, находится ли пиксель в пределах границ растра
        if 0 <= row < height and 0 <= col < width:
            # Используем метод `transform` для получения координат (x, y)
            # x соответствует долготе, y - широте
            lon, lat = transform * (col, row) # Обратите внимание на порядок: col для x, row для y
            lons[i] = lon
            lats[i] = lat
            inside[i] = True
        else:
            # Если пиксель вне границ, присваиваем NaN или другое значение по умолчанию
            lons[i] = np.nan
            lats[i] = np.nan
            inside[i] = False
    
    return lons, lats, inside



