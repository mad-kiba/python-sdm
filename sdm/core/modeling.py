# python-sdm/core/modeling.py
import os
import pandas as pd
import numpy as np
import json
import math
import glob
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from rasterio.transform import xy

from .data_loading import load_occurrences
from .preprocessing import clip_rasters, load_raster_stack, points_to_pixel_indices
from .preprocessing import sample_background, extract_features_from_stack
from .utils.helpers import inverse_scale, save_geotiff
from .utils.plot_utils import draw_map, create_beautiful_histogram


def run_sdm(IN_ID, IN_CSV, PREDICTORS, IN_MIN_LAT, IN_MIN_LON, IN_MAX_LAT, IN_MAX_LON, IN_RESOLUTION, MODEL_FUTURE, IN_MODEL,
            BG_MULT, BG_DISTANCE_MIN, BG_DISTANCE_MAX, BG_PC, jobs):
    print("-- Регион для моделирования: ")
    print("("+str(IN_MIN_LAT)+","+str(IN_MIN_LON)+"), ("+str(IN_MAX_LAT)+","+str(IN_MAX_LON)+")")
    
    RANDOM_SEED = 42
    
    OUTPUT_SUITABILITY_TIF = "output/suitability_"+str(IN_ID)+".tif"  # куда сохранить карту пригодности
    OUTPUT_SUITABILITY_JPG = "output/suitability_"+str(IN_ID)+".jpg"
    OUTPUT_HISTOGRAMS_DIR = "output/gistos"
    OUTPUT_PREDICTIONS_DIR = "output/predictions"
    
    OUTPUT_FUTURE_DIR = os.path.join(OUTPUT_PREDICTIONS_DIR, str(IN_ID))
    
    RAW_RASTER_DIR = "input_predictors"
    
    SCALES_FILE = os.path.join(RAW_RASTER_DIR, 'predictors_scales.json')
    
    OUTPUT_RASTER_DIR = "output_predictors/("+str(IN_MIN_LAT)+","+str(IN_MIN_LON)+"), ("+str(IN_MAX_LAT)+","+str(IN_MAX_LON)+")"
    RASTER_DIR = OUTPUT_RASTER_DIR # папка с GeoTIFF-предикторами
    
    # Сколько фоновых точек генерировать: мин(10000, 10 * N_presence)
    MAX_BG = 10000
    
    # начали
    np.random.seed(RANDOM_SEED)
    
    text_filename = 'output/texts/'+str(IN_ID)+'.txt'
    pred_filename = 'output/texts/'+str(IN_ID)+'_pred.txt'
    stack_filename = 'output/texts/'+str(IN_ID)+'_stack.txt'
    month_filename = 'output/texts/'+str(IN_ID)+'_month.txt'
    csv_filename = 'output/texts/'+str(IN_ID)+'.csv'
    
    # 1) Подготовка предикторов к нужным координатам
    print("\n-- 1. Подготовка предикторов ")
    clip_rasters(RAW_RASTER_DIR, OUTPUT_RASTER_DIR, IN_MIN_LAT, IN_MIN_LON, IN_MAX_LAT, IN_MAX_LON, MODEL_FUTURE)
    
    # 2) Загрузка стека предикторов
    print("\n-- 2. Загрузка предикторов ")
    stack, valid_mask, transform, crs, profile, band_names = load_raster_stack(RASTER_DIR, PREDICTORS)
    bands, H, W = stack.shape
    
    print(f"\n-- Загружено предикторов: {bands} | Размер: {H} x {W} | CRS: {crs}")
    print("Слои:", band_names)
    
    with open(text_filename, 'a') as f:
        f.write(f"{bands} | Размер: {H} x {W} | CRS: {crs}")
        f.write(f"\n{band_names}")
    
    # 3) Загрузка присутствий
    print("\n-- 3. Загрузка наблюдений")
    if (os.path.isfile(IN_CSV)): # если это путь к файлу, читаем файл, иначе считаем дампом csv
        with open(IN_CSV, 'r') as file: # читаем исходный файл
            IN_CSV = file.read()
    with open(csv_filename, 'a') as f: # записываем в архив
        f.write(IN_CSV)
    df = pd.read_csv(csv_filename, sep="\t", index_col=False, on_bad_lines='skip', low_memory=False)
    
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
        jobs[IN_ID]['status'] = 'error'
        jobs[IN_ID]['error'] = 'csv parse error'
        return {"error": "csv parse error", "status": "terminated"}, 401
    
    # 3.1) Фильтрация мусорных данных из GBIF
    print("-- 3.1. Фильтрация мусорных данных из GBIF")
    if 'coordinateUncertaintyInMeters' in df.columns:
        df['coordinateUncertaintyInMeters'] = df['coordinateUncertaintyInMeters'].fillna(0).astype(int)
        df = df[df['coordinateUncertaintyInMeters']<1000]
        
    if 'collectionCode' in df.columns:
        df = df[df['collectionCode']!='EOA']
    #print(df[LAT_COL])
    #print(df[LON_COL])
    
    # 3.2) группировка по месяцам
    print("-- 3.2. Группировка по месяцам")
    # здесь где-то перепутаны координаты!!!
    df_coord_filtered = df[df['year']>2010]
    df_coord_filtered = df_coord_filtered[df_coord_filtered[LAT_COL].astype(float)>IN_MIN_LON]
    df_coord_filtered = df_coord_filtered[df_coord_filtered[LAT_COL]<IN_MAX_LON]
    df_coord_filtered = df_coord_filtered[df_coord_filtered[LON_COL]>IN_MIN_LAT]
    df_coord_filtered = df_coord_filtered[df_coord_filtered[LON_COL]<IN_MAX_LAT]
    
    
    if 'month' in df_coord_filtered.columns:
        # month_filename
        df_cleaned = df_coord_filtered.dropna(subset=['year', 'month'])
        df_cleaned['year'] = df_cleaned['year'].astype(int)
        df_cleaned['month'] = df_cleaned['month'].astype(int)
        
        df_cleaned['year_month'] = df_cleaned['year'].astype(str) + '-' + df_cleaned['month'].astype(str).str.zfill(2)
        
        monthly_counts = df_cleaned.groupby('year_month').size()
        counts_dict = monthly_counts.to_dict()
        with open(month_filename, 'w', encoding='utf-8') as f:
            json.dump(counts_dict, f, ensure_ascii=False, indent=4) # indent=4 для читаемости
    
    # 3.3) финальные присустсвия
    print("-- 3.3. Финальные присутствия")
    occ = load_occurrences(df, LON_COL, LAT_COL)
    print("\n-- Обработка наблюдений")
    print(f"Всего записей в CSV: {len(occ)}")
    
    with open(text_filename, 'a') as f:
        f.write(f"\n{len(occ)}")
    
    if len(occ)==0:
        print('Not enough points')
        jobs[IN_ID]['status'] = 'error'
        jobs[IN_ID]['error'] = 'not enough points'
        return {"error": "not enough points", "status": "terminated"}, 401
    
    
    
    
    # 4) Привязка присутствий к пикселям растра и фильтрация по маске валидности
    print("\n-- 4. Привязка присутствий к пикселям растра и фильтрация по маске валидности")
    rows, cols, inside = points_to_pixel_indices(occ[LON_COL].values, occ[LAT_COL].values,
                                                 transform, W, H)
    # Фильтруем те, что внутри растра
    rows, cols = rows[inside], cols[inside]
    # И те, что попадают на валидные пиксели (без NaN во всех слоях)
    valid_here = valid_mask[rows, cols]
    rows, cols = rows[valid_here], cols[valid_here]
    print(f"Присутствий внутри валидной области: {len(rows)}")
    
    with open(text_filename, 'a') as f:
        f.write(f"\n{len(rows)}")
        
    if len(rows)==0:
        print('Not enough points in region')
        jobs[IN_ID]['status'] = 'error'
        jobs[IN_ID]['error'] = 'not enough points in region'
        return {"error": "not enough points in region", "status": "terminated"}, 401
    
    
    # 4.1) создаём полные растры для всего спектра слоёв-предикторов
    print("-- 4.1. Создаём полные растры для всего спектра слоёв-предикторов")
    rows_grid, cols_grid = np.indices((H, W))
    
    # Преобразуем их в одномерные массивы
    rows_full_flat = rows_grid.flatten()
    cols_full_flat = cols_grid.flatten()
    
    # 4.1.1) Фильтруем эти полные индексы по маске валидности
    # valid_mask[rows_full_flat, cols_full_flat] вернет булеву маску для каждого пикселя
    # True, если пиксель валиден, False - если NaN
    valid_pixels_mask = valid_mask[rows_full_flat, cols_full_flat]
    
    # Применяем булеву маску, чтобы получить только валидные индексы
    rows_full = rows_full_flat[valid_pixels_mask]
    cols_full = cols_full_flat[valid_pixels_mask]
    
    
    
    # 5) Дедупликация по пикселю (30″ клетка) — оставляем по одному наблюдению на клетку
    print("\n-- 5. Дедупликация по пикселю — оставляем по одному наблюдению на клетку")
    pres_rc = pd.DataFrame({"r": rows, "c": cols}).drop_duplicates().values
    rows_p = pres_rc[:, 0]
    cols_p = pres_rc[:, 1]
    n_presence = len(rows_p)
    if n_presence < 20:
        print("Внимание: очень мало уникальных присутствий в пределах растра.")
    print(f"Уникальных присутствий (по пикселю): {n_presence}")
    
    with open(text_filename, 'a') as f:
        f.write(f"\n{n_presence}")
    
    
    # 6) Генерация фоновых точек и точек псевдоотсутствия
    print("\n-- 6. Генерация фоновых точек и точек псевдоотсутствия")
    # 6.1) если нужно генерировать точки псевдоотсутствия, но параметры заданы на авто
    if BG_PC!=100 and BG_DISTANCE_MIN==0:
        print("Нужно генерировать точки псевдоприсутствия, и параметры огибающих заданы на авто. Определяем их.")
        if len(df['kingdom'].unique())==1 and len(df['class'].unique())<=1:
            # значения по умолчанию
            BG_DISTANCE_MIN = 5
            BG_DISTANCE_MAX = 10
            
            # вычисляем параметры
            if df['class'].unique()==['Aves']: # Птицы
                BG_DISTANCE_MIN = 20
                BG_DISTANCE_MAX = 50
                
            if df['class'].unique()==['Mammalia']: # Млекопитающие
                BG_DISTANCE_MIN = 10
                BG_DISTANCE_MAX = 20
                
            if df['class'].unique()==['Amphibia']: # Амфибии
                BG_DISTANCE_MIN = 10
                BG_DISTANCE_MAX = 20
                
            if df['class'].unique()==['Squamata'] or df['class'].unique()==['Testudines']: # Рептилии
                BG_DISTANCE_MIN = 10
                BG_DISTANCE_MAX = 20
        else:
            BG_PC = 100
    
    print("\n-- Генерация фоновых точек и точек псевдоотсутствия")
    print(f"Вычисленные параметры точек: BG_PC={BG_PC}, BG_DISTANCE_MIN={BG_DISTANCE_MIN}, BG_DISTANCE_MAX={BG_DISTANCE_MAX}")
    
    
    # 6.2) Генерация фоновых точек
    rng = np.random.default_rng(RANDOM_SEED)
    n_bg = min(MAX_BG, BG_MULT * n_presence)
    
    BG_ABS_PC = 100 - BG_PC
    with open(text_filename, 'a') as f:
        f.write(f"\n{BG_PC},{BG_ABS_PC},{BG_DISTANCE_MIN},{BG_DISTANCE_MAX},{BG_MULT}")
        f.write(f"\n{IN_MIN_LAT},{IN_MIN_LON},{IN_MAX_LAT},{IN_MAX_LON},{IN_RESOLUTION},{IN_MODEL}")
    
    
    rows_bg, cols_bg = sample_background(valid_mask, set(map(tuple, pres_rc)), n_bg, rng, BG_PC, BG_DISTANCE_MIN, BG_DISTANCE_MAX, text_filename)
    #print(f"Сэмплировано фоновых точек: {len(rows_bg)}")
    
    
    # 7) Извлечение признаков
    print("\n-- 7. Извлечение признаков")
    X_pres = extract_features_from_stack(stack, rows_p, cols_p)
    X_bg = extract_features_from_stack(stack, rows_bg, cols_bg)
    X_orig = extract_features_from_stack(stack, rows, cols)
    X_full = extract_features_from_stack(stack, rows_full, cols_full)
    X = np.vstack([X_pres, X_bg])
    y = np.hstack([np.ones(len(X_pres), dtype=int), np.zeros(len(X_bg), dtype=int)])
    print(f"Матрица признаков: {X.shape}, классы: {np.bincount(y)}")
    
    np.savetxt(stack_filename, X_orig, delimiter=";", fmt="%d")
    
    
    
    # 8) постройка гистограмм
    print("\n-- 8. Постройка гистограмм")
    num_predictors = len(band_names) # Получаем точное количество предикторов
    #print(SCALES_FILE)
    
    with open(SCALES_FILE, 'r') as f:
        scales_config = json.load(f)
        
    # Динамически определяем количество строк и столбцов для сетки
    # Делаем сетку максимально приближенной к квадрату
    cols_num = int(math.ceil(math.sqrt(num_predictors))) # Количество столбцов
    rows_num = int(math.ceil(num_predictors / cols_num))       # Количество строк
    
    # Создаем фигуру с учетом динамических размеров
    fig, axes = plt.subplots(rows_num, cols_num, figsize=(cols_num * 5.5, rows_num * 4)) # Увеличенный размер для лучшего размещения
    
    # Если у нас только один предиктор, axes будет не массивом, а одним объектом Axes
    if num_predictors == 1:
        axes = np.array([axes])
    elif num_predictors == 0:
        axes = np.array([]) # Пустой массив, если нет предикторов
        
    # Регулируем количество бинов, если оно больше, чем количество уникальных значений (что маловероятно, но для безопасности)
    bins_num = 50
    if bins_num > len(np.unique(X_pres)):
        bins_num = len(np.unique(X_pres))
        print(f"Количество бинов было уменьшено до {bins_num}, так как оно превышало количество уникальных значений.")
    
    # --- Сохранение каждой гистограммы в отдельный файл ---
    # Пересоздаем фигуру и оси для сохранения, чтобы они были независимы от plt.show()
    # Это важно, чтобы сохранить чистые изображения без лишних элементов, добавленных plt.show()
    # (хотя в данном случае plt.show() уже показал, но для чистоты процесса сохранения)
    #print(num_predictors)
    # Нужно заново пройтись по данным, чтобы сохранить каждую гистограмму отдельно
    for i, band_name in enumerate(band_names):
        # Создаем новую фигуру для каждого графика
        fig_single, ax_single = plt.subplots(1, 1, figsize=(7, 5)) # Размер одного графика
        # Получаем масштабированные данные (они уже в X_pres)
        scaled_data_for_plot = X_pres[:, i]
        scaled_data_for_plot_full = X_full[:, i] 
        # Получаем параметры масштабирования для текущего предиктора
        # Убедитесь, что band_name соответствует ключам в scales_config
        scale_params = scales_config.get(band_name)
        # Применяем обратное преобразование, если параметры найдены
        
        layer_data = ''
        
        title = ''
        if (len(df['species'].unique())==1):
            title = 'Вид: '+df['species'].unique()[0]
        
        if scale_params:
            data_for_plot_original_scale = inverse_scale(scaled_data_for_plot, scale_params)
            data_for_plot_original_scale_full = inverse_scale(scaled_data_for_plot_full, scale_params)
            create_beautiful_histogram(ax_single, data_for_plot_original_scale, band_name, bins_num, data_for_plot_original_scale_full, title)
        else:
            print(f"Предупреждение: Параметры масштабирования не найдены для '{band_name}'. Отображаются масштабированные значения.")
            create_beautiful_histogram(ax_single, scaled_data_for_plot, band_name, bins_num, scaled_data_for_plot_full, title)
        
        # Создаем имя файла
        # Заменяем недопустимые символы, если есть в band_name
        safe_band_name = band_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        dir_path = os.path.join(OUTPUT_HISTOGRAMS_DIR, str(IN_ID))
        os.makedirs(dir_path, exist_ok=True)
        output_filename = os.path.join(OUTPUT_HISTOGRAMS_DIR, str(IN_ID), f"{safe_band_name}.png")
        # Сохраняем фигуру
        plt.savefig(output_filename, dpi=300, bbox_inches='tight') # dpi для качества, bbox_inches='tight' для обрезки лишних полей
        print(f"Сохранена гистограмма: {i} - {output_filename}")
        plt.close(fig_single) # Закрываем фигуру, чтобы освободить память
    plt.close(fig)
    
    print(f"Все гистограммы сохранены в папку: '{OUTPUT_HISTOGRAMS_DIR}\{IN_ID}'")
    
    archive_name = "histos.zip"
    archive_path = os.path.join(dir_path, archive_name)
    
    # 8.1. Получаем список всех файлов в папке для упаковки в архив
    files_to_zip = glob.glob(os.path.join(dir_path, "*.png"))
    
    # 8.2. Проверяем, есть ли вообще файлы для упаковки, пакуем
    if not files_to_zip:
        print(f"В папке {OUTPUT_HISTOGRAMS_DIR}\{IN_ID} нет файлов для упаковки.")
    else:
        # 3. Создаем ZIP-архив
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_zip:
                # Добавляем файл в архив. os.path.basename гарантирует,
                # что в архиве будут только имена файлов, а не полные пути.
                zipf.write(file_path, os.path.basename(file_path))
        
        print(f"Все файлы из '{OUTPUT_HISTOGRAMS_DIR}\{IN_ID}' успешно упакованы в '{archive_path}'.")
    
    
    
    # 9) Разделение на train/test
    print("\n-- 9. Разделение на train/test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    
    
    # 10) Обучение модели
    print("\n-- 10. Обучение модели")
    
    clf = RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        max_depth=10
    )
    
    xgbm = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=500,        # Количество деревьев
        learning_rate=0.05,      # Скорость обучения
        max_depth=10,             # Максимальная глубина деревьев
        subsample=0.8,           # Доля объектов для обучения каждого дерева
        colsample_bytree=0.8,    # Доля признаков для обучения каждого дерева
        random_state=RANDOM_SEED,
        n_jobs=-1,               # Использовать все доступные ядра CPU
        #use_label_encoder=False,
        eval_metric='auc',
        tree_method='hist'       # Хорошо работает с большими данными
    )
    
    
    if (IN_MODEL=='RandomForest'):
        model = clf
        
    if (IN_MODEL=='XGBoost'):
        model = xgbm
    
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC (holdout): {auc:.3f}")
    
    with open(text_filename, 'a') as f:
        f.write(f"\n{auc:.3f}")
        
        if (len(df['species'].unique())==1):
            title = df['species'].unique()[0]
            f.write(f"\n{title}")
        else:
            f.write(f"\nне определён")
    
    # Важность переменных
    importances = model.feature_importances_
    print("Важность предикторов:")
    for name, imp in sorted(zip(band_names, importances), key=lambda x: -x[1]):
        print(f"  {name:30s} {imp:.4f}")
        with open(pred_filename, 'a') as f:
            f.write(f"\n_{name:30s}:{imp:.4f}")
        
        
    # 11) Прогноз на всю область и сохранение карты пригодности
    print("\n-- 11. Прогноз на всю область и сохранение карты пригодности")
    # Предсказываем только на валидных пикселях
    valid_idx = np.flatnonzero(valid_mask.ravel())
    flat = stack.reshape(bands, -1).T  # (H*W, bands)
    
    suitability_flat = np.full(H * W, np.nan, dtype="float32")

    # Чтобы не упереться в память, делаем батчами
    batch = 500_000
    for start in range(0, len(valid_idx), batch):
        end = start + batch
        sel = valid_idx[start:end]
        X_pred = flat[sel]
        pred = model.predict_proba(X_pred)[:, 1].astype("float32")
        suitability_flat[sel] = pred
        
    suitability = suitability_flat.reshape(H, W)
    save_geotiff(OUTPUT_SUITABILITY_TIF, suitability, profile)
    print(f"Карта пригодности сохранена: {OUTPUT_SUITABILITY_TIF}")
    
    mask_high_suitability05 = suitability > 0.05
    count_high_suitability05 = np.sum(mask_high_suitability05)
    
    mask_high_suitability50 = suitability > 0.5
    count_high_suitability50 = np.sum(mask_high_suitability50)
    
    mask_high_suitability95 = suitability > 0.95
    count_high_suitability95 = np.sum(mask_high_suitability95)
    
    with open(text_filename, 'a') as f:
        f.write(f"\nCHS05:{count_high_suitability05}")
        f.write(f"\nCHS50:{count_high_suitability50}")
        f.write(f"\nCHS95:{count_high_suitability95}")
    
    # (Опционально) можно сохранить также использованные точки присутствия в пиксельных координатах
    # или вернуть их центры в географических координатах:
    xs, ys = xy(transform, rows_p, cols_p, offset="center")
    used_occ_df = pd.DataFrame({"lon": xs, "lat": ys})
    used_occ_df.to_csv(os.path.join(os.path.dirname(OUTPUT_SUITABILITY_TIF), "used_presences_"+str(IN_ID)+".csv"), index=False)
    print("Сохранены использованные присутствия (уникальные по пикселю): used_presences_"+str(IN_ID)+".csv")
    
    
    
    # 12) дальше рисуем картинку
    print("\n-- 12. Рисуем карту")
    title = ''
    if (len(df['species'].unique())==1):
        title = 'Карта вероятности присутствия вида '+df['species'].unique()[0]
    draw_map(OUTPUT_SUITABILITY_TIF, OUTPUT_SUITABILITY_JPG, title)
    
    
    
    
    
    # 13) если это стандартный регион - делаем с нашей моделью прогноз на будущее
    if MODEL_FUTURE==1:
        print("\n-- 13. Приступаю к прогнозу будущего")
        # Пути
        TRAIN_DIR = os.path.join(OUTPUT_RASTER_DIR, "1970-2000")      # где лежат обучающие предикторы
        FUTURE_ROOT_DIR = os.path.join(OUTPUT_RASTER_DIR)   # где лежат папки периодов 2021-2040, ...
        os.makedirs(OUTPUT_FUTURE_DIR, exist_ok=True)
        
        PREDICTORS_STD = 'Consensus_reduced_class_12,roughness_std3x3,slope_deg,wc2.1_30s_bio_1,'+\
                          'wc2.1_30s_bio_10,wc2.1_30s_bio_11,wc2.1_30s_bio_12,wc2.1_30s_bio_13,'+\
                          'wc2.1_30s_bio_14,wc2.1_30s_bio_15,wc2.1_30s_bio_16,wc2.1_30s_bio_17,'+\
                          'wc2.1_30s_bio_18,wc2.1_30s_bio_19,wc2.1_30s_bio_2,wc2.1_30s_bio_3,'+\
                          'wc2.1_30s_bio_4,wc2.1_30s_bio_5,wc2.1_30s_bio_6,wc2.1_30s_bio_7,wc2.1_30s_bio_8,'+\
                          'wc2.1_30s_bio_9,wc2.1_30s_elev'
        
        # 13.1) Загружаем обучающий стек предикторов (1970-2000) и обучаем модель на уже подготовленных точках
        stack_train, valid_mask_train, transform_train, crs_train, profile_train, band_names_train = \
            load_raster_stack(TRAIN_DIR, PREDICTORS_STD)
        
        if isinstance(PREDICTORS_STD, str):
            PREDICTORS_STD_EXP = [p.strip() for p in PREDICTORS_STD.split(',') if p.strip()]
        
        # Извлекаем признаки в обучающих точках
        X_pres = extract_features_from_stack(stack_train, rows_p, cols_p)
        X_bg   = extract_features_from_stack(stack_train, rows_bg, cols_bg)
        
        X = np.vstack([X_pres, X_bg])
        y = np.hstack([np.ones(len(X_pres), dtype=int), np.zeros(len(X_bg), dtype=int)])
        
        
        xgbm_std = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,        # Количество деревьев
            learning_rate=0.05,      # Скорость обучения
            max_depth=10,             # Максимальная глубина деревьев
            subsample=0.8,           # Доля объектов для обучения каждого дерева
            colsample_bytree=0.8,    # Доля признаков для обучения каждого дерева
            random_state=RANDOM_SEED,
            n_jobs=-1,               # Использовать все доступные ядра CPU
            #use_label_encoder=False,
            eval_metric='auc',
            tree_method='hist'       # Хорошо работает с большими данными
        )
        
        xgbm_std.fit(X, y)
        
        
        # 13.2) Прогноз на всю область и сохранение карты пригодности по текущему периоду
        # Чтобы не упереться в память, делаем батчами
        suitability = predict_suitability_for_stack(xgbm_std, stack_train, valid_mask_train, batch_size=500_000)
        
        OUTPUT_SUITABILITY_TIF = OUTPUT_FUTURE_DIR + "/1970-2000.tif"
        save_geotiff(OUTPUT_SUITABILITY_TIF, suitability, profile)
        print(f"Карта пригодности сохранена: {OUTPUT_SUITABILITY_TIF}")
        
        title = ''
        if (len(df['species'].unique())==1):
            title = 'Карта вероятности присутствия вида '+df['species'].unique()[0]+"\nТекущий период (базовые климатические переменные)"
        OUTPUT_SUITABILITY_JPG = OUTPUT_FUTURE_DIR + "/1970-2000.jpg"
        draw_map(OUTPUT_SUITABILITY_TIF, OUTPUT_SUITABILITY_JPG, title)
        print(f"Карта пригодности сохранена: {OUTPUT_SUITABILITY_JPG}")
        os.remove(OUTPUT_SUITABILITY_TIF)
        
        
        
        # 13.3) Прогноз для будущих периодов/сценариев
        for period in sorted(d for d in os.listdir(FUTURE_ROOT_DIR)
                             if os.path.isdir(os.path.join(FUTURE_ROOT_DIR, d))):
            period_dir = os.path.join(FUTURE_ROOT_DIR, period)
        
            for scenario in sorted(d for d in os.listdir(period_dir)
                                   if os.path.isdir(os.path.join(period_dir, d))):
                scen_dir = os.path.join(period_dir, scenario)
                print(f"\nПрогноз: {period} / {scenario}")
        
                # Загружаем будущие предикторы строго в порядке PREDICTORS_STD;
                # если load_raster_stack не гарантирует порядок, переупорядочим по именам
                stack_fut, valid_mask_fut, transform_fut, crs_fut, profile_fut, band_names_fut = \
                    load_raster_stack(scen_dir, PREDICTORS_STD)
        
                # Проверка и переупорядочивание при необходимости
                if set(band_names_fut) != set(PREDICTORS_STD_EXP):
                    print(f"Пропуск {period}/{scenario}: набор слоёв не совпадает с обучающим.")
                    continue
                if list(band_names_fut) != list(PREDICTORS_STD_EXP):
                    # Переупорядочим ось слоёв под порядок PREDICTORS_STD
                    idx = [band_names_fut.index(b) for b in PREDICTORS_STD_EXP]
                    stack_fut = stack_fut[idx, :, :]
                    band_names_fut = [band_names_fut[i] for i in idx]
        
                # (Необязательно) Проверка совместимости CRS/разрешения
                if crs_fut != crs_train:
                    print(f"Внимание: CRS отличается у {period}/{scenario} (train={crs_train}, fut={crs_fut}).")
        
                suitability_f = predict_suitability_for_stack(xgbm_std, stack_fut, valid_mask_fut, batch_size=500_000)
        
                out_name = f"{period}-{scenario}.tif"
                out_path = os.path.join(OUTPUT_FUTURE_DIR, out_name)
                save_geotiff(out_path, suitability_f, profile_fut)
                print(f"Сохранено: {out_path}")
                
                mask_high_suitability05 = suitability_f > 0.05
                count_high_suitability05 = np.sum(mask_high_suitability05)
                
                mask_high_suitability50 = suitability_f > 0.5
                count_high_suitability50 = np.sum(mask_high_suitability50)
                
                mask_high_suitability95 = suitability_f > 0.95
                count_high_suitability95 = np.sum(mask_high_suitability95)
                
                out_name_img = f"{period}-{scenario}.{count_high_suitability05}.{count_high_suitability50}.{count_high_suitability95}.jpg"
                out_path_img = os.path.join(OUTPUT_FUTURE_DIR, out_name_img)
                #print(out_path)
                #print(out_path_img)
                
                title = ''
                if (len(df['species'].unique())==1):
                    title = 'Карта вероятности присутствия вида '+df['species'].unique()[0]+"\nПериод: "+period+" (сценарий "+scenario+")"
                
                draw_map(out_path, out_path_img, title)
                os.remove(out_path)
                
                
        print(f"\nВсе прогнозы сохранены в папку: '{OUTPUT_FUTURE_DIR}'")

        archive_name = "futures.zip"
        archive_path = os.path.join(OUTPUT_FUTURE_DIR, archive_name)
        
        # 13.4) Получаем список всех файлов в папке для упаковки в архив
        files_to_zip = glob.glob(os.path.join(OUTPUT_FUTURE_DIR, "*"))
        
        # 13.5) Проверяем, есть ли вообще файлы для упаковки, пакуем
        if not files_to_zip:
            print(f"В папке {OUTPUT_FUTURE_DIR} нет файлов для упаковки.")
        else:
            # 3. Создаем ZIP-архив
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files_to_zip:
                    # Добавляем файл в архив. os.path.basename гарантирует,
                    # что в архиве будут только имена файлов, а не полные пути.
                    zipf.write(file_path, os.path.basename(file_path))
        
        print(f"Все файлы из '{OUTPUT_FUTURE_DIR}' успешно упакованы в '{archive_path}'.")

    
    print("\n-- Моделирование завершено")
    
    jobs[IN_ID]['status'] = 'done'
    return {'result': 'Ok'}, 200


# Вспомогательная функция предсказания по стеку батчами
def predict_suitability_for_stack(trained_clf, stack, valid_mask, batch_size=500_000):
    bands, H, W = stack.shape
    flat = stack.reshape(bands, -1).T  # (H*W, bands)
    suitability_flat = np.full(H * W, np.nan, dtype="float32")
    valid_idx = np.flatnonzero(valid_mask.ravel())
    for start in range(0, len(valid_idx), batch_size):
        end = start + batch_size
        sel = valid_idx[start:end]
        X_pred = flat[sel]
        pred = trained_clf.predict_proba(X_pred)[:, 1].astype("float32")
        suitability_flat[sel] = pred
    return suitability_flat.reshape(H, W)