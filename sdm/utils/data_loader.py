import os
import pandas as pd
import numpy as np
import json
import glob
import rasterio

def load_occurrences(df, lon_col, lat_col, month_col=''): 
    """Загружает CSV с наблюдениями, фильтрует некорректные координаты."""
    df.loc[:, lat_col] = df[lat_col].astype(float)
    df.loc[:, lon_col] = df[lon_col].astype(float)
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"В CSV нет столбцов {lon_col}/{lat_col}")
    if month_col=='':
        df = df[[lon_col, lat_col]]
    else:
        df = df[[lon_col, lat_col, month_col]]
    df = df.dropna(subset=[lon_col, lat_col])
    # Базовая фильтрация координат
    df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180) & (df[lat_col] >= -90) & (df[lat_col] <= 90)]
    df = df.reset_index(drop=True)
    return df


def load_species_occurrence_data(IN_ID, IN_CSV, IN_CSV_ADDITIONAL, CSV_FILENAME, CSV_FILENAME_ADD, MONTH_FILENAME, TEXT_FILENAME,
                                IN_MIN_LON, IN_MIN_LAT, IN_MAX_LON, IN_MAX_LAT, jobs):
    
    try:
        if (os.path.isfile(IN_CSV)): # если это путь к файлу, читаем файл, иначе считаем дампом csv
            with open(IN_CSV, 'r') as file: # читаем исходный файл
                IN_CSV = file.read()
        else:
            print('На вход пришёл набор csv')
        if IN_CSV == '':
            print('file is empty')
            raise ValueError('Входной файл пустой.')
    except Exception as e:
        print('file read error')
        raise ValueError('Ошибка чтения файла: ' + str(e))
    
    with open(CSV_FILENAME, 'w') as f: # записываем файл
        f.write(IN_CSV)
        
    df = pd.read_csv(CSV_FILENAME, sep="\t", index_col=False, on_bad_lines='skip', low_memory=False)
    species = ''
    if 'species' in df.columns:
        if (len(df['species'].unique())==1):
            species = df['species'].unique()[0]
            print(f"Определён вид: {species}")
    
    print(f"Всего загружено записей: {len(df)}")
    
    if IN_CSV_ADDITIONAL != '': # если есть дополнительные записи - добавим их
        with open(CSV_FILENAME_ADD, 'w') as f: # записываем файл
            f.write(IN_CSV_ADDITIONAL)
            f.close()
            
        df2 = pd.read_csv(CSV_FILENAME_ADD, sep="\t", index_col=False, on_bad_lines='skip', low_memory=False, )
        
        columns_df = df.columns
        columns_df2 = df2.columns
        common_columns_in_both = list(set(columns_df) & set(columns_df2))
        df2_filtered = df2[common_columns_in_both]
        df = pd.concat([df, df2_filtered], ignore_index=True)
        
        print(f"Записей после дозагрузки: {len(df)}, из них дозагружено: {len(df2)}")
            
    
    # вычисление полей с координатами
    LAT_COL = 'lat'
    LON_COL = 'lon'
    
    if 'Latitude' in df.columns:
        LAT_COL = 'Latitude'
        LON_COL = 'Longitude'
        
    if 'latitude' in df.columns:
        LAT_COL = 'latitude'
        LON_COL = 'longitude'
    
    if 'decimalLatitude' in df.columns:
        LAT_COL = 'decimalLatitude'
        LON_COL = 'decimalLongitude'
    
    if not LAT_COL in df.columns:
        print('csv parse error')
        raise ValueError('Ошибка обработки csv. Проверьте, что у входных данных корректный формат.')
    
    # 2.1) Фильтрация мусорных данных из GBIF
    print(f"-- 2.1. Фильтрация мусорных данных из GBIF ({IN_ID})")
    if 'coordinateUncertaintyInMeters' in df.columns:
        df['coordinateUncertaintyInMeters'] = df['coordinateUncertaintyInMeters'].fillna(0).astype(float).astype(int) 
        df = df[df['coordinateUncertaintyInMeters']<1000]
        
    if 'collectionCode' in df.columns:
        df = df[df['collectionCode']!='EOA']
    #print(df[LAT_COL])
    #print(df[LON_COL])
    
    print(f"Осталось записей после фильтрации: {len(df)}")
    
    # 2.2) группировка по месяцам
    print(f"-- 2.2. Группировка по месяцам ({IN_ID})")
    # здесь где-то перепутаны координаты!!!
    MONTH_COL = ''
    if 'year' in df.columns:
        df_coord_filtered = df[df['year']>2010]
        df_coord_filtered = df_coord_filtered[df_coord_filtered[LAT_COL].astype(float)>IN_MIN_LON]
        df_coord_filtered = df_coord_filtered[df_coord_filtered[LAT_COL]<IN_MAX_LON]
        df_coord_filtered = df_coord_filtered[df_coord_filtered[LON_COL]>IN_MIN_LAT]
        df_coord_filtered = df_coord_filtered[df_coord_filtered[LON_COL]<IN_MAX_LAT]
        
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
                json.dump(counts_dict, f, ensure_ascii=False, indent=4) # indent=4 для читаемости
    
    # 2.3) финальные присустсвия
    print(f"-- 2.3. Финальные присутствия ({IN_ID})")
    occ = load_occurrences(df, LON_COL, LAT_COL, MONTH_COL)
    print("\n-- Обработка наблюдений")
    print(f"Осталось записей финально CSV: {len(occ)}")
    
    with open(TEXT_FILENAME, 'a') as f:
        f.write(f"{len(occ)}")
    
    if len(occ)==0:
        print('Not enough points')
        raise ValueError('Во входных данных нет наблюдений. Проверьте источник.')
    
    if len(occ)<10:
        print('Less than 10 points')
        raise ValueError(f"Недостаточно точек. Должно быть не менее 10, сейчас: {len(occ)}.")
    
    return {'LAT_COL': LAT_COL, 'LON_COL': LON_COL, 'df': df, 'occ': occ, 'status': 'done', 'species': species}


def load_environmental_predictors(raster_dir, predictors = 'all', period='current'):
    """Считывает все GeoTIFF из папки и строит стек (bands, H, W).
       Возвращает: stack(float32), valid_mask(bool), transform, crs, profile, band_names(list)"""
    
    # У нас предикторы делятся на статические и динамические.
    # Для моделирования настоящего - объединяем их.
    static_subdir = os.path.join(raster_dir, "static")
    if period=='current':
        dynamic_subdir = os.path.join(raster_dir, "dynamic_current")
    else:
        dynamic_subdir = os.path.join(raster_dir, "dynamic_predictable/"+period)
    
    # Собираем файлы из подпапки "static"
    static_tifs = glob.glob(os.path.join(static_subdir, "*.tif"))
    
    # Собираем файлы из подпапки "dynamic_current"
    dynamic_tifs = glob.glob(os.path.join(dynamic_subdir, "*.tif"))
    
    # Объединяем оба списка
    all_available_tifs = []
    all_available_tifs.extend(static_tifs)
    all_available_tifs.extend(dynamic_tifs)
        
    print(f"Входной путь для предикторов: {raster_dir}")
    print(f"Статические предикторы: {static_subdir}")
    print(f"Динамические предикторы: {dynamic_subdir}")
    
    #print('----------------')
    #print('Все предикторы:')
    #print(all_available_tifs)
    #print('Нужные предикторы:')
    #print(predictors)
    #print('----------------')
    
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
        desired_tifs_ordered = []
        
        for expected_filename in expected_full_filenames:
            found = False
            # Ищем файл в уже собранном списке all_available_tifs
            for available_file_path in all_available_tifs:
                if os.path.basename(available_file_path) == expected_filename:
                    desired_tifs_ordered.append(available_file_path)
                    found = True
                    break # Переходим к следующему ожидаемому файлу
    
    tifs = desired_tifs_ordered
    
    
    if not tifs:
        raise FileNotFoundError(f"В папке {raster_dir} не найдены файлы, соответствующие предикторам. "
                                f"Найдены только: {', '.join([os.path.basename(f) for f in all_available_tifs]) if all_available_tifs else 'ни одного'}")
    
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


