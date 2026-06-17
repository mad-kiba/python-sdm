# sdm/utils/data_loader.py
# Библиотека PythonSDM для моделирования распространения видов
# - набор функций для предобработки и загрузки наблюдательных данных

import os
import pandas as pd
import numpy as np
import json
import glob
import rasterio
import re
import shutil
from scipy.ndimage import distance_transform_edt

from .plots import plot_geotiff_with_osm
from .utils import clean_nans_for_json


# Основная функция загрузки, фильтрации и подготовки данных о встречаемости вида.
def load_species_occurrence_data(IN_ID, IN_CSV, IN_CSV_ADDITIONAL, CSV_FILENAME, CSV_FILENAME_ADD,
                                CSV_FILTERED_FILENAME, MONTH_FILENAME,
                                IN_MIN_LON, IN_MIN_LAT, IN_MAX_LON, IN_MAX_LAT, 
                                ALLOWED_COORD_UNCERTAIN, MINIMUM_YEAR_ALLOWED):
    """
    Основная функция загрузки, фильтрации и подготовки данных о встречаемости вида.
    
    Обрабатывает входные CSV файлы (в том числе сырые дампы строк), объединяет дополнительные
    наблюдения, стандартизирует координаты, фильтрует мусорные точки (GBIF), применяет 
    пространственные и временные фильтры, а также генерирует статистику по месяцам.
    
    Args:
        IN_ID (int): Идентификатор задачи/модели.
        IN_CSV (str): Путь к файлу CSV или сырая строка (дамп) с данными наблюдений.
        IN_CSV_ADDITIONAL (str): Сырая строка с дополнительными данными наблюдений.
        CSV_FILENAME (str): Путь для сохранения основного рабочего CSV файла.
        CSV_FILENAME_ADD (str): Путь для сохранения дополнительного CSV файла.
        CSV_FILTERED_FILENAME (str): Путь для сохранения отфильтрованного CSV.
        MONTH_FILENAME (str): Путь для сохранения JSON со статистикой по месяцам.
        TEXT_FILENAME (str): Путь для сохранения текстовой статистики.
        IN_MIN_LON (float): Минимальная долгота (Bounding Box).
        IN_MIN_LAT (float): Минимальная широта (Bounding Box).
        IN_MAX_LON (float): Максимальная долгота (Bounding Box).
        IN_MAX_LAT (float): Максимальная широта (Bounding Box).
        ALLOWED_COORD_UNCERTAIN (float): Максимально допустимая неопределенность координат (в метрах).
        MINIMUM_YEAR_ALLOWED (int): Минимально допустимый год наблюдения.
        
    Returns:
        dict: Словарь с результатами: колонки координат, очищенный DataFrame, финальные
              точки присутствия (occ), статус и извлеченная таксономия (species, kingdom, class).
              
    Raises:
        ValueError: При ошибках чтения файлов, пустых данных или отсутствии достаточного числа валидных точек.
    """
    
    try:
        # Защита от OSError (File name too long) при передаче длинных сырых строк
        is_file_path = len(IN_CSV) < 2048 and os.path.isfile(IN_CSV)
        if is_file_path:
            shutil.copyfile(IN_CSV, CSV_FILENAME)
        else:
            if not IN_CSV:
                print('file is empty')
                raise ValueError('Входной файл пустой.')
            with open(CSV_FILENAME, 'w', encoding='utf-8') as f: # записываем дамп
                f.write(IN_CSV)
    except Exception as e:
        print('file read/copy error')
        raise ValueError('Ошибка обработки входного файла: ' + str(e))
        
    df = detect_and_read_csv(CSV_FILENAME)
    df = standardize_coord_names(df)
    
    # вычисляем информацию о виде
    species = ''
    if 'species' in df.columns:
        if (len(df['species'].unique())==1):
            species = df['species'].unique()[0]
            print(f"Определён вид: {species}")
    
    kingdom = ['']
    dclass = ['']
    if 'kingdom' in df.columns and 'class' in df.columns:
        temp_df = df.query("`kingdom`!='' and `class`!=''")
        temp_df = temp_df.dropna(subset=['kingdom', 'class'])
        kingdom = temp_df['kingdom'].unique()
        dclass =   temp_df['class'].unique()
        print(f"Вычислено царство {kingdom} и класс {dclass}")
    
    print(f"Всего загружено записей: {len(df)}")
    
    if IN_CSV_ADDITIONAL: # Безопасная проверка на пустую строку и на None
        with open(CSV_FILENAME_ADD, 'w', encoding='utf-8') as f: # Явно указываем UTF-8 для избежания краша на кириллице
            f.write(IN_CSV_ADDITIONAL)
            f.close()
            
        df2 = detect_and_read_csv(CSV_FILENAME_ADD)
        df2 = standardize_coord_names(df2)
        
        columns_df = df.columns
        columns_df2 = df2.columns
        common_columns_in_both = list(set(columns_df) & set(columns_df2))
        df2_filtered = df2[common_columns_in_both]
        df = pd.concat([df, df2_filtered], ignore_index=True)
        
        print(f"Записей после дозагрузки: {len(df)}, из них дозагружено: {len(df2)}")
    
    # записываем в файл со статистикой общее число входных наблюдений до фильтрации
    total_obs_in_csv = len(df)
    
    # вычисление полей с координатами
    LAT_COL = 'lat'
    LON_COL = 'lon'
    
    if not LAT_COL in df.columns:
        print('csv parse error')
        raise ValueError('Ошибка обработки csv. Проверьте, что у входных данных корректный формат. Колонки с координатами должны называться lat, lon. Ячейки должны разделяться символом табуляции (при экспорте из Excel используйте формат текстовый файл с табуляцией).')
    
    # 2.1) Фильтрация мусорных данных из GBIF
    print(f"-- 2.1. Фильтрация мусорных данных из GBIF ({IN_ID})")
    if 'coordinateUncertaintyInMeters' in df.columns:
        df['coordinateUncertaintyInMeters'] = df['coordinateUncertaintyInMeters'].fillna(0).astype(float).astype(int) 
        df = df[df['coordinateUncertaintyInMeters']<ALLOWED_COORD_UNCERTAIN]
        
    if 'collectionCode' in df.columns:
        df = df[df['collectionCode']!='EOA']
    #print(df[LAT_COL])
    #print(df[LON_COL])
    
    # Эффективная и безопасная очистка координат (без дублирования DataFrame)
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors='coerce')
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors='coerce')
    df = df.dropna(subset=[LAT_COL, LON_COL])
    
    print(f"Осталось записей после фильтрации: {len(df)}")

    # 2.3) фильтрация по координатам
    df = df[df[LAT_COL]>=IN_MIN_LAT]
    df = df[df[LAT_COL]<=IN_MAX_LAT]
    df = df[df[LON_COL]>=IN_MIN_LON]
    df = df[df[LON_COL]<=IN_MAX_LON]
    
    # 2.3) группировка по месяцам для таблички встреч
    print(f"-- 2.2. Группировка по месяцам ({IN_ID})")
    
    MONTH_COL = ''
    counts_dict = {}
    if 'year' in df.columns:
        year_numeric = pd.to_numeric(df['year'], errors='coerce')
        df_coord_filtered = df[year_numeric > MINIMUM_YEAR_ALLOWED].copy()
        
        month_col = ''
        if 'month' in df_coord_filtered.columns:
            # MONTH_FILENAME
            MONTH_COL = 'month'
            df_cleaned = df_coord_filtered.dropna(subset=['year', 'month'])
            df_cleaned['year'] = df_cleaned['year'].astype(int)
            df_cleaned['month'] = df_cleaned['month'].astype(int)
            
            df_cleaned['year_month'] = df_cleaned['year'].astype(str) + '-' + df_cleaned['month'].astype(str).str.zfill(2)
            
            monthly_counts = df_cleaned.groupby('year_month').size()
            counts_dict = monthly_counts.to_dict()
            with open(MONTH_FILENAME, 'w', encoding='utf-8') as f:
                json.dump(clean_nans_for_json(counts_dict), f, ensure_ascii=False, indent=4) # indent=4 для читаемости
    else:
        df_coord_filtered = df
    
    # 2.4) финальные присустсвия
    print(f"-- 2.3. Финальные присутствия ({IN_ID})")
    occ = load_occurrences(df_coord_filtered, LON_COL, LAT_COL, MONTH_COL)
    print("\n-- Обработка наблюдений")
    print(f"Осталось записей финально CSV: {len(occ)}")
    
    df_coord_filtered.to_csv(CSV_FILTERED_FILENAME, index=False)
    
    if len(occ)==0:
        print('Not enough points')
        raise ValueError('Во входных данных нет наблюдений. Проверьте источник.')
    
    if len(occ)<10:
        print('Less than 10 points')
        raise ValueError(f"Недостаточно точек. Должно быть не менее 10, сейчас: {len(occ)}.")
    
    return {
        'LAT_COL': LAT_COL, 'LON_COL': LON_COL, 'df': df, 'occ': occ, 'status': 'done', 
        'species': species, 'kingdom': kingdom, 'dclass': dclass, 
        'total_obs_in_csv': total_obs_in_csv, 'monthly_counts': counts_dict}


# Вспомогательная функция для приведения имен координат к единому стандарту.
def standardize_coord_names(d):
    """
    Приводит названия колонок с координатами в DataFrame к единому стандарту ('lat' и 'lon').
    
    Args:
        d (pandas.DataFrame): Входной DataFrame, содержащий колонки с координатами.
        
    Returns:
        pandas.DataFrame: DataFrame со стандартизированными названиями колонок.
    """
    if 'Latitude' in d.columns:
        d = d.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
    elif 'latitude' in d.columns:
        d = d.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    elif 'decimalLatitude' in d.columns:
        d = d.rename(columns={'decimalLatitude': 'lat', 'decimalLongitude': 'lon'})
    return d


# Загружает DataFrame с наблюдениями, очищает и фильтрует некорректные координаты.
def load_occurrences(df, lon_col, lat_col, month_col=''): 
    """
    Загружает DataFrame с наблюдениями, очищает и фильтрует некорректные координаты.
    
    Удаляет строки с пропущенными координатами и отфильтровывает точки, 
    выходящие за пределы допустимых географических значений (-180..180, -90..90).
    
    Args:
        df (pandas.DataFrame): Исходный DataFrame с данными наблюдений.
        lon_col (str): Название колонки с долготой.
        lat_col (str): Название колонки с широтой.
        month_col (str, optional): Название колонки с месяцем наблюдения. По умолчанию ''.
        
    Returns:
        pandas.DataFrame: Отфильтрованный DataFrame, содержащий только валидные координаты.
        
    Raises:
        ValueError: Если в DataFrame отсутствуют указанные колонки координат.
    """
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"В CSV нет столбцов {lon_col}/{lat_col}")
    
    df.loc[:, lat_col] = df[lat_col].astype(float)
    df.loc[:, lon_col] = df[lon_col].astype(float)
    if month_col=='':
        df = df[[lon_col, lat_col]]
    else:
        df = df[[lon_col, lat_col, month_col]]
    df = df.dropna(subset=[lon_col, lat_col])
    # Базовая фильтрация координат
    df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180) & (df[lat_col] >= -90) & (df[lat_col] <= 90)]
    df = df.reset_index(drop=True)
    return df


# Автоматически определяет разделитель CSV файла на основе первой строки и читает файл с помощью pandas.
def detect_and_read_csv(filename):
    """
    Автоматически определяет разделитель CSV файла на основе первой строки
    и читает файл с помощью pandas.

    Args:
        filename (str): Путь к CSV файлу.

    Returns:
        pandas.DataFrame: DataFrame, содержащий данные из CSV файла.
    """
    delimiters = [',', ';', '\t']
    detected_separator = '\t'  # Значение по умолчанию

    with open(filename, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    # Проверяем, какой из стандартных разделителей чаще встречается в первой строке
    # и считаем, что это и есть основной разделитель.
    # Мы гарантируем, что в файле минимум две колонки, т.е. есть хотя бы один разделитель.
    separator_counts = {delim: first_line.count(delim) for delim in delimiters}

    # Находим разделитель с максимальным количеством вхождений
    # Если несколько разделителей имеют одинаковое максимальное количество,
    # то приоритет будет у того, что раньше в списке delimiters (',', ';', '\t')
    # Например, если первая строка ";;;", то ';' будет выбран.
    # Если первая строка ",;,", то ',' будет выбран.
    detected_separator = max(separator_counts, key=separator_counts.get)

    # Если max вернул 0 (что маловероятно при наличии минимум одного разделителя),
    # то мы остаемся на значении по умолчанию '\t'.
    if separator_counts[detected_separator] == 0:
        detected_separator = '\t'

    print(f"Определен разделитель: '{detected_separator}'") # Опционально: для отладки

    df = pd.read_csv(filename, sep=detected_separator, index_col=False, on_bad_lines='skip', low_memory=False)
    return df


# Основная функция загрузки слоёв GeoTIFF с предикторами окружающей среды.
def load_environmental_predictors(raster_dir, predictors = 'all', period='current', interval='', scales='', 
                                  bio_info='', base_period='1981-2010'):
    """
    Загружает слои GeoTIFF с предикторами окружающей среды и формирует 3D стек (матрицу).
    
    Автоматически определяет статические и динамические предикторы, выравнивает их,
    заменяет значения NoData на NaN, а также выполняет экстраполяцию значений
    прибрежной зоны на водные пространства для устранения артефактов на границах суши.
    
    Args:
        raster_dir (str): Корневая директория с растрами предикторов.
        predictors (str, optional): Строка с названиями нужных предикторов через запятую, 
                                    либо 'all' для загрузки всех. По умолчанию 'all'.
        period (str, optional): Период моделирования ('current', 'future', 'monthly'). По умолчанию 'current'.
        interval (str, optional): Дополнительный интервал/сценарий (например, сценарий SSP для будущего).
        scales (dict, optional): Словарь с параметрами масштабирования предикторов.
        bio_info (dict, optional): Словарь с метаданными предикторов (единицы измерения, лимиты).
        base_period (str, optional): Базовый период для замены в названиях файлов (например, '1981-2010').
        
    Returns:
        tuple: (stack, valid_mask, ref_transform, ref_crs, profile, band_names, band_paths)
            stack (np.ndarray): 3D массив (стек) со значениями предикторов (bands, H, W).
            valid_mask (np.ndarray): 2D булева маска валидных пикселей.
            
    Raises:
        FileNotFoundError: Если необходимые растры не найдены в директории.
        ValueError: Если найденные растры имеют различное разрешение, размеры или CRS.
    """
    
    # У нас предикторы делятся на статические и динамические.
    # Для моделирования настоящего - объединяем их.
    static_subdir = os.path.join(raster_dir, "static")
    if period =='current':
        dynamic_subdir = os.path.join(raster_dir, "dynamic_current")
    if period == 'future':
        dynamic_subdir = os.path.join(raster_dir, "dynamic_predictable/"+interval)
        # важно, чтобы все предикторы для будущего назывались аналогично настоящему
    if period == 'monthly':
        dynamic_subdir = os.path.join(raster_dir, "dynamic_monthly/"+str(interval))
    
    # Собираем файлы из подпапки "static"
    static_tifs = glob.glob(os.path.join(static_subdir, "*.tif"))
    
    # Собираем файлы из подпапки динамичных подпапок
    dynamic_tifs = glob.glob(os.path.join(dynamic_subdir, "*.tif"))

    # Получаем список среднегодовых предикторов для их исключения при помесячном прогнозе
    dynamic_current_basenames = [os.path.basename(f) for f in glob.glob(os.path.join(raster_dir, "dynamic_current", "*.tif"))]
    
    # Объединяем оба списка
    all_available_tifs = []
    all_available_tifs.extend(static_tifs)
    all_available_tifs.extend(dynamic_tifs)
        
    print(f"Входной путь для предикторов: {raster_dir}")
    print(f"Статические предикторы: {static_subdir}")
    print(f"Динамические предикторы: {dynamic_subdir}")
    
    desired_tifs_ordered = []
    not_found_tifs = []
    
    # фильтруем по входящему списку предикторов
    if predictors.strip().lower() == 'all':
        # Если 'all', используем все найденные файлы
        desired_filenames_no_ext = [os.path.splitext(os.path.basename(f))[0] for f in all_available_tifs]
        desired_tifs_ordered = all_available_tifs # Изначальный порядок из glob
    else:
        # Создаем список желаемых имен файлов (без .tif)
        predictor_names_no_ext = [p.strip() for p in predictors.split(',')]
        
        # Создаем полный список ожидаемых имен файлов (.tif)
        expected_full_filenames = [f"{name}.tif" for name in predictor_names_no_ext]
        
        # Фильтруем все доступные файлы, чтобы остались только те, что в списке ожидаемых
        # и сохраняем их в порядке, заданном в predictors
        
        for expected_filename in expected_full_filenames:
            # Если это помесячное моделирование, игнорируем среднегодовые предикторы
            if period == 'monthly' and expected_filename in dynamic_current_basenames:
                continue
                
            found = False
            # Ищем файл в уже собранном списке all_available_tifs
            for available_file_path in all_available_tifs:
                if os.path.basename(available_file_path) == expected_filename:
                    desired_tifs_ordered.append(available_file_path)
                    found = True
                    break # Переходим к следующему ожидаемому файлу
                
                if period == 'future':
                    # хак для разных имён файлов предикторов CHELSA
                    subrep = os.path.basename(available_file_path)
                    # Универсальное удаление любой GCM модели и сценария SSP (например, _mpi-esm1-2-hr_ssp126)
                    # Это делает загрузчик совместимым с UKESM1, GFDL-ESM4, IPSL и др.
                    subrep = re.sub(r'_[a-zA-Z0-9\-]+_ssp[0-9]{3}', '', subrep)
                    subrep = subrep.replace(interval.split('/')[0], base_period)
                    
                    if subrep == expected_filename:
                        desired_tifs_ordered.append(available_file_path)
                        found = True
                        break
            if found == False:
                not_found_tifs.append(expected_filename)
                
        if period == 'monthly': # добавим помесячные предикторы
            for path in dynamic_tifs:
                if path not in desired_tifs_ordered:
                    desired_tifs_ordered.append(path)
    
        
    tifs = desired_tifs_ordered
    
    if len(not_found_tifs)>0:
        print('----------------')
        print('Доступные предикторы:')
        print(all_available_tifs)
        print('Нужные предикторы:')
        print(predictors)
        print('Загруженные предикторы:')
        print(tifs)
        print('Не найденные предикторы:')
        print(not_found_tifs)
        print('----------------')
    
    
    if not tifs:
        raise FileNotFoundError(f"В папке {raster_dir} не найдены файлы, соответствующие предикторам. "
                                f"Найдены только: {', '.join([os.path.basename(f) for f in all_available_tifs]) if all_available_tifs else 'ни одного'}")
    
    band_arrays = []
    band_names = []
    band_paths = []
    ref_transform = None
    ref_width = ref_height = None
    ref_crs = None
    
    # создаём jpg для каждого слоя, если заданы масштабы (=для текущего периода)
    if scales:
        for i, fp in enumerate(tifs):
            fpjpg = fp.replace('.tif', '.jpg')
            band_paths.append(fpjpg)
            
            if not os.path.exists(fpjpg):
                band = os.path.splitext(os.path.basename(fp))[0]
                if scales.get(band):
                    mean = scales[band]['mean']
                    sc = scales[band]['scale']
                else:
                    mean = 0
                    sc = 1
                plot_geotiff_with_osm(fp, fpjpg, mean, sc, band, bio_info)
    
    for i, fp in enumerate(tifs):
        with rasterio.open(fp) as ds:
            arr = ds.read(1, masked=True)
            arr = np.ma.filled(arr.astype("float32"), np.nan) # Обязательно превращаем NoData в NaN
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
    
    # Экстраполяция предикторов на водные пространства (Nearest Neighbor)
    print("Экстраполяция значений предикторов на акватории (Nearest Neighbor)...")
    MAX_EXTRAPOLATE_PIXELS = 50 # Лимит экстраполяции от берега (50 точек растра, т.е. примерно 50 км при step=30")

    for i in range(stack.shape[0]):
        nan_mask = np.isnan(stack[i])
        if nan_mask.any() and not nan_mask.all():
            # Для каждого NaN-пикселя находим координаты ближайшего валидного пикселя (суши)
            distances, indices = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
            # Применяем экстраполяцию только к тем пикселям моря, которые близко к берегу
            close_enough = nan_mask & (distances <= MAX_EXTRAPOLATE_PIXELS)
            stack[i][close_enough] = stack[i][tuple(indices)][close_enough]
    
    # Маска валидных пикселей: валиден, если нет NaN во всех слоях
    valid_mask = ~np.isnan(stack[0])
    # Теперь валидна не вся карта BBox, а только суша + 50 точек прибрежной зоны

    # Профиль для сохранения результата
    
    #print('Форма растров: ')
    #print(stack.shape)
    
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
    
    return stack, valid_mask, ref_transform, ref_crs, profile, band_names, band_paths


