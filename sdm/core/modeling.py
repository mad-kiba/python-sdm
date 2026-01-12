# python-sdm/core/modeling.py
import os
import pandas as pd
import numpy as np
import json
import math
import glob
import zipfile
import time
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from rasterio.transform import xy
from scipy.optimize import minimize

from .data_loading import load_occurrences
from .preprocessing import clip_rasters, load_raster_stack, points_to_pixel_indices, pixel_indices_to_points
from .preprocessing import sample_background, extract_features_from_stack
from .utils.helpers import inverse_scale, save_geotiff
from .utils.plot_utils import draw_map, create_beautiful_histogram, create_animated_gif, create_avi_from_images


def run_sdm(IN_ID, IN_CSV, PREDICTORS, IN_MIN_LAT, IN_MIN_LON, IN_MAX_LAT, IN_MAX_LON, IN_RESOLUTION, MODEL_FUTURE, IN_MODEL,
            BG_MULT, BG_DISTANCE_MIN, BG_DISTANCE_MAX, BG_PC, DO_GISTO, jobs):
    print(f"-- Регион для моделирования ({IN_ID}): ")
    print("("+str(IN_MIN_LAT)+","+str(IN_MIN_LON)+"), ("+str(IN_MAX_LAT)+","+str(IN_MAX_LON)+"), step: "+IN_RESOLUTION)
    
    if IN_MIN_LAT==0 and IN_MAX_LAT==0:
        jobs[IN_ID]['status'] = 'done'
        return {'result': 'Ok'}, 200
    
    RANDOM_SEED = 42
    
    OUTPUT_SUITABILITY_TIF = "output/suitability_"+str(IN_ID)+".tif"  # куда сохранить карту пригодности
    OUTPUT_SUITABILITY_JPG = "output/suitability_"+str(IN_ID)+".jpg"
    OUTPUT_HISTOGRAMS_DIR = "output/gistos"
    OUTPUT_PREDICTIONS_DIR = "output/predictions"
    
    OUTPUT_FUTURE_DIR = os.path.join(OUTPUT_PREDICTIONS_DIR, str(IN_ID))
    
    RAW_RASTER_DIR = "input_predictors"
    
    SCALES_FILE = os.path.join(RAW_RASTER_DIR, 'predictors_scales.json')
    
    OUTPUT_RASTER_DIR = "output_predictors/"+IN_RESOLUTION+"/("+str(IN_MIN_LAT)+","+str(IN_MIN_LON)+"), ("+str(IN_MAX_LAT)+","+str(IN_MAX_LON)+")"
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
    
    # для запуска в многопоточном режиме
    j = jobs.get(IN_ID)
    if not j:
        jobs[IN_ID] = {'status': 'queued', 'file': None, 'error': None}
    
    # 1) Подготовка предикторов к нужным координатам
    print(f"\n-- 1. Подготовка предикторов ({IN_ID})")
    clip_rasters(RAW_RASTER_DIR, OUTPUT_RASTER_DIR, IN_MIN_LAT, IN_MIN_LON, IN_MAX_LAT, IN_MAX_LON, MODEL_FUTURE, IN_RESOLUTION)
    
    

    
    # 2) Загрузка присутствий
    print(f"\n-- 2. Загрузка наблюдений ({IN_ID})")
    if (os.path.isfile(IN_CSV)): # если это путь к файлу, читаем файл, иначе считаем дампом csv
        try:
            with open(IN_CSV, 'r') as file: # читаем исходный файл
                IN_CSV = file.read()
        except Exception as e:
            print('file read error')
            jobs[IN_ID]['status'] = 'error'
            jobs[IN_ID]['error'] = 'Ошибка чтения файла. Проверьте, что файл не пустой.'
            return {"error": "Ошибка чтения файла. Проверьте, что файл не пустой.", "status": "terminated"}, 401
    
    with open(csv_filename, 'w') as f: # записываем в архив
        f.write(IN_CSV)
    
    df = pd.read_csv(csv_filename, sep="\t", index_col=False, on_bad_lines='skip', low_memory=False)
    if (len(df['species'].unique())==1):
        species = df['species'].unique()[0]
        print(f"Определён вид: {species}")
    
    print(f"Всего загружено записей: {len(df)}")
    
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
        jobs[IN_ID]['status'] = 'error'
        jobs[IN_ID]['error'] = 'Ошибка обработки csv. Проверьте, что у входных данных корректный формат.'
        return {"error": "Ошибка обработки csv. Проверьте, что у входных данных корректный формат.", "status": "terminated"}, 401
    
    # 3.1) Фильтрация мусорных данных из GBIF
    print(f"-- 2.1. Фильтрация мусорных данных из GBIF ({IN_ID})")
    if 'coordinateUncertaintyInMeters' in df.columns:
        df['coordinateUncertaintyInMeters'] = df['coordinateUncertaintyInMeters'].fillna(0).astype(float).astype(int) 
        df = df[df['coordinateUncertaintyInMeters']<1000]
        
    if 'collectionCode' in df.columns:
        df = df[df['collectionCode']!='EOA']
    #print(df[LAT_COL])
    #print(df[LON_COL])
    
    print(f"Осталось записей после фильтрации: {len(df)}")
    
    # 3.2) группировка по месяцам
    print(f"-- 2.2. Группировка по месяцам ({IN_ID})")
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
    print(f"-- 2.3. Финальные присутствия ({IN_ID})")
    occ = load_occurrences(df, LON_COL, LAT_COL)
    print("\n-- Обработка наблюдений")
    print(f"Осталось записей финально CSV: {len(occ)}")
    
    with open(text_filename, 'a') as f:
        f.write(f"{len(occ)}")
    
    if len(occ)==0:
        print('Not enough points')
        jobs[IN_ID]['status'] = 'error'
        jobs[IN_ID]['error'] = 'Во входных данных нет наблюдений. Проверьте источник.'
        return {"error": "Во входных данных нет наблюдений. Проверьте источник.", "status": "terminated"}, 401
    
    if len(occ)<10:
        print('Less than 10 points')
        jobs[IN_ID]['status'] = 'error'
        jobs[IN_ID]['error'] = f"Недостаточно точек. Должно быть не менее 10, сейчас: {len(occ)}."
        return {"error": f"Недостаточно точек. Должно быть не менее 10, сейчас: {len(occ)}.", "status": "terminated"}, 401
    
    
    # 3) Загрузка стека предикторов
    print(f"\n-- 3. Загрузка предикторов ({IN_ID})")
    stack, valid_mask, transform, crs, profile, band_names = load_raster_stack(RASTER_DIR, PREDICTORS)
    bands, H, W = stack.shape
    
    print(f"\n-- Загружено предикторов: {bands} | Размер: {H} x {W} | CRS: {crs}")
    print("Слои:", band_names)
    
    with open(text_filename, 'a') as f:
        f.write(f"\n{bands} | Размер: {H} x {W} | CRS: {crs}")
        f.write(f"\n{band_names}")
    
    
    # 4) Привязка присутствий к пикселям растра и фильтрация по маске валидности
    print(f"\n-- 4. Привязка присутствий к пикселям растра и фильтрация по маске валидности ({IN_ID})")
    rows, cols, inside = points_to_pixel_indices(occ[LON_COL].values, occ[LAT_COL].values, transform, W, H)
    # Фильтруем те, что внутри растра
    rows, cols = rows[inside], cols[inside]
    # И те, что попадают на валидные пиксели (без NaN во всех слоях)
    valid_here = valid_mask[rows, cols]
    rows, cols = rows[valid_here], cols[valid_here]
    
    print(f"Присутствий внутри валидной области: {len(rows)}")
    
    with open(text_filename, 'a') as f:
        f.write(f"\n{len(rows)}")
    
    if len(rows)<10:
        print('Not enough points in region')
        jobs[IN_ID]['status'] = 'error'
        jobs[IN_ID]['error'] = f"Внутри области моделирования недостаточно точек. Должно быть не менее 10, сейчас: {len(rows)}."
        return {"error": f"Внутри области моделирования недостаточно точек. Должно быть не менее 10, сейчас: {len(rows)}.", "status": "terminated"}, 401
    
    
    # 4.1) создаём полные растры для всего спектра слоёв-предикторов
    print(f"-- 4.1. Создаём полные растры для всего спектра слоёв-предикторов ({IN_ID})")
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
    print(f"\n-- 5. Дедупликация по пикселю — оставляем по одному наблюдению на клетку ({IN_ID})")
    pres_rc = pd.DataFrame({"r": rows, "c": cols}).drop_duplicates().values
    rows_p = pres_rc[:, 0]
    cols_p = pres_rc[:, 1]
    n_presence = len(rows_p)
    if n_presence < 20:
        print("Внимание: очень мало уникальных присутствий в пределах растра.")
    print(f"Уникальных присутствий (по пикселю): {n_presence}")
    
    if n_presence<5:
        print('Not enough unique points in region')
        jobs[IN_ID]['status'] = 'error'
        jobs[IN_ID]['error'] = f"Внутри области моделирования очень мало уникальных присутствий. Должно быть не менее 5, сейчас: {n_presence}."
        return {"error": f"Внутри области моделирования очень мало уникальных присутствий. Должно быть не менее 5, сейчас: {n_presence}.", "status": "terminated"}, 401
    
    rows_coord, cols_coord, inside = pixel_indices_to_points(rows_p, cols_p, transform, W, H)
    
    with open(text_filename, 'a') as f:
        f.write(f"\n{n_presence}")
    
    
    # 6) Генерация фоновых точек и точек псевдоотсутствия
    print(f"\n-- 6. Генерация фоновых точек и точек псевдоотсутствия ({IN_ID})")
    # 6.1) если нужно генерировать точки псевдоотсутствия, но параметры заданы на авто
    if BG_PC!=100 and BG_DISTANCE_MIN==0:
        print("Нужно генерировать точки псевдоприсутствия, и параметры огибающих заданы на авто. Определяем их.")
        if len(df['kingdom'].unique())==1 and len(df['class'].unique())<=1:
            # значения по умолчанию
            BG_DISTANCE_MIN = 10
            BG_DISTANCE_MAX = 20
            
            # вычисляем параметры
            if df['class'].unique()==['Aves']: # Птицы
                BG_DISTANCE_MIN = 50
                BG_DISTANCE_MAX = 100
                
            if df['class'].unique()==['Mammalia']: # Млекопитающие
                BG_DISTANCE_MIN = 20
                BG_DISTANCE_MAX = 50
                
            if df['class'].unique()==['Amphibia']: # Амфибии
                BG_DISTANCE_MIN = 20
                BG_DISTANCE_MAX = 50
                
            if df['class'].unique()==['Squamata'] or df['class'].unique()==['Testudines']: # Рептилии
                BG_DISTANCE_MIN = 20
                BG_DISTANCE_MAX = 50
        else:
            BG_PC = 100
    
    print(f"\n-- Генерация фоновых точек и точек псевдоотсутствия ({IN_ID})")
    print(f"Вычисленные параметры точек: BG_PC={BG_PC}, BG_DISTANCE_MIN={BG_DISTANCE_MIN}, BG_DISTANCE_MAX={BG_DISTANCE_MAX}")
    
    
    # 6.2) Генерация фоновых точек
    if (IN_MODEL=='MaxEnt'):
        BG_MULT = 100
        BG_ABS_PC = 0
        BG_PC = 100
    else:
        BG_ABS_PC = 100 - BG_PC
        
    with open(text_filename, 'a') as f:
        f.write(f"\n{BG_PC},{BG_ABS_PC},{BG_DISTANCE_MIN},{BG_DISTANCE_MAX},{BG_MULT}")
        f.write(f"\n{IN_MIN_LAT},{IN_MIN_LON},{IN_MAX_LAT},{IN_MAX_LON},{IN_RESOLUTION},{IN_MODEL}")
            
    rng = np.random.default_rng(RANDOM_SEED)
    n_bg = min(MAX_BG, BG_MULT * n_presence)
        
    
    rows_bg, cols_bg = sample_background(valid_mask, set(map(tuple, pres_rc)), n_bg, rng, BG_PC, BG_DISTANCE_MIN, BG_DISTANCE_MAX, text_filename)
    #print(f"Сэмплировано фоновых точек: {len(rows_bg)}")
    
    
    # 7) Извлечение признаков
    print(f"\n-- 7. Извлечение признаков ({IN_ID})")
    X_pres = extract_features_from_stack(stack, rows_p, cols_p)
    X_bg = extract_features_from_stack(stack, rows_bg, cols_bg)
    X_orig = extract_features_from_stack(stack, rows, cols)
    X_full = extract_features_from_stack(stack, rows_full, cols_full)
    X = np.vstack([X_pres, X_bg])
    y = np.hstack([np.ones(len(X_pres), dtype=int), np.zeros(len(X_bg), dtype=int)])
    print(f"Матрица признаков: {X.shape}, классы: {np.bincount(y)}")
    
    np.savetxt(stack_filename, X_orig, delimiter=";", fmt="%d")
    
    
    
    # 8) постройка гистограмм
    if DO_GISTO == 1:
        print(f"\n-- 8. Постройка гистограмм ({IN_ID})")
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
    print(f"\n-- 9. Разделение на train/test ({IN_ID})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    
    
    # 10) Обучение модели
    print(f"\n-- 10. Обучение модели ({IN_ID})")
    
    if (IN_MODEL=='MaxEnt'):
        model = MaxEnt(X_pres=X_pres, X_bg=X_bg)
        model.fit(
            maxiter=500,
            tol=1e-5
        )
    
    if (IN_MODEL=='RandomForest'):
        model = RandomForestClassifier(
            n_estimators=500,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            class_weight="balanced_subsample",
            max_depth=10
        )
        model.fit(X_train, y_train)
        
    if (IN_MODEL=='XGBoost'):
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,        # Количество деревьев
            learning_rate=0.05,      # Скорость обучения
            max_depth=10,             # Максимальная глубина деревьев
            subsample=0.8,           # Доля объектов для обучения каждого дерева
            colsample_bytree=0.8,    # Доля признаков для обучения каждого дерева
            random_state=RANDOM_SEED,
            n_jobs=-1,               # Использовать все доступные ядра CPU
            eval_metric='auc',
            tree_method='hist'       # Хорошо работает с большими данными
        )
        
        model.fit(X_train, y_train)
    
    
    y_prob = model.predict_proba(X_test)[:, 1]
    #print(model.predict_proba(X_test))
    
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
    if (IN_MODEL=='MaxEnt'):
        importances = model.weights
    else:
        importances = model.feature_importances_
    
    print("Важность предикторов:")
    for name, imp in sorted(zip(band_names, importances), key=lambda x: -x[1]):
        print(f"  {name:30s} {imp:.4f}")
        with open(pred_filename, 'a') as f:
            f.write(f"\n_{name:30s}:{imp:.4f}")
        
        
    # 11) Прогноз на всю область и сохранение карты пригодности
    print(f"\n-- 11. Прогноз на всю область и сохранение карты пригодности ({IN_ID})")
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
    print(f"\n-- 12. Рисуем карту ({IN_ID})")
    title = ''
    if (len(df['species'].unique())==1):
        title = 'Карта вероятности присутствия вида '+df['species'].unique()[0]+f" ({IN_ID})"
    adtitle = f"\nМодель: {IN_MODEL}, шаг: {IN_RESOLUTION}, уник. точек: {n_presence}, ROC-AUC: {auc:.3f}";
    title = title + adtitle
    draw_map(OUTPUT_SUITABILITY_TIF, OUTPUT_SUITABILITY_JPG, title, rows_coord, cols_coord)
    
    
    
    
    
    # 13) если это стандартный регион - делаем с нашей моделью прогноз на будущее
    if MODEL_FUTURE==1 and IN_MODEL!='MaxEnt':
        print(f"\n-- 13. Приступаю к прогнозу будущего ({IN_ID})")
        # Пути
        FUTURE_ROOT_DIR = os.path.join(OUTPUT_RASTER_DIR, 'dynamic_predictable')   # где лежат папки периодов 2021-2040, ...
        
        os.makedirs(OUTPUT_FUTURE_DIR, exist_ok=True)
        
        PREDICTORS_STD = PREDICTORS
        
        # 13.1) Загружаем обучающий стек предикторов (1970-2000) и обучаем модель на уже подготовленных точках
        stack_train, valid_mask_train, transform_train, crs_train, profile_train, band_names_train = \
            load_raster_stack(RASTER_DIR, PREDICTORS_STD)
        print('Предикторы для обучения модели будущего загружены')
        
        if isinstance(PREDICTORS_STD, str):
            PREDICTORS_STD_EXP = [p.strip() for p in PREDICTORS_STD.split(',') if p.strip()]
        
        # Извлекаем признаки в обучающих точках
        X_pres = extract_features_from_stack(stack_train, rows_p, cols_p)
        X_bg   = extract_features_from_stack(stack_train, rows_bg, cols_bg)
        
        X = np.vstack([X_pres, X_bg])
        y = np.hstack([np.ones(len(X_pres), dtype=int), np.zeros(len(X_bg), dtype=int)])
        
        model.fit(X, y)
        
        
        # 13.2) Прогноз на всю область и сохранение карты пригодности по текущему периоду
        # Чтобы не упереться в память, делаем батчами
        suitability = predict_suitability_for_stack(model, stack_train, valid_mask_train, batch_size=500_000)
        
        OUTPUT_SUITABILITY_TIF = OUTPUT_FUTURE_DIR + "/1970-2000.tif"
        save_geotiff(OUTPUT_SUITABILITY_TIF, suitability, profile)
        print(f"Карта пригодности сохранена: {OUTPUT_SUITABILITY_TIF}")
        
        title = ''
        if (len(df['species'].unique())==1):
            title = 'Карта вероятности присутствия вида '+df['species'].unique()[0]+f" ({IN_ID})\nТекущий период (базовые климатические переменные)"
        OUTPUT_SUITABILITY_JPG = OUTPUT_FUTURE_DIR + "/1970-2000.jpg"
        draw_map(OUTPUT_SUITABILITY_TIF, OUTPUT_SUITABILITY_JPG, title, rows_coord, cols_coord)
        print(f"Карта пригодности сохранена: {OUTPUT_SUITABILITY_JPG}")
        #os.remove(OUTPUT_SUITABILITY_TIF) # пока не удаляем tif для будущего
        
        
        
        
        # 13.3) Прогноз для будущих периодов/сценариев
        future_imgs = {}
        for period in sorted(d for d in os.listdir(FUTURE_ROOT_DIR)
                             if os.path.isdir(os.path.join(FUTURE_ROOT_DIR, d))):
            period_dir = os.path.join(FUTURE_ROOT_DIR, period)
        
            for scenario in sorted(d for d in os.listdir(period_dir)
                                   if os.path.isdir(os.path.join(period_dir, d))):
                scen_dir = os.path.join(period, scenario)
                print(f"\nПрогноз: {period} / {scenario}")
                
                # Загружаем будущие предикторы строго в порядке PREDICTORS_STD;
                # если load_raster_stack не гарантирует порядок, переупорядочим по именам
                stack_fut, valid_mask_fut, transform_fut, crs_fut, profile_fut, band_names_fut = \
                    load_raster_stack(RASTER_DIR, PREDICTORS_STD, scen_dir)
                
                # Проверка и переупорядочивание при необходимости
                if set(band_names_fut) != set(PREDICTORS_STD_EXP):
                    print(f"Пропуск {period}/{scenario}: набор слоёв не совпадает с обучающим.")
                    continue
                #if list(band_names_fut) != list(PREDICTORS_STD_EXP):
                #    # Переупорядочим ось слоёв под порядок PREDICTORS_STD
                #    idx = [band_names_fut.index(b) for b in PREDICTORS_STD_EXP]
                #    stack_fut = stack_fut[idx, :, :]
                #    band_names_fut = [band_names_fut[i] for i in idx]
                
                # (Необязательно) Проверка совместимости CRS/разрешения
                if crs_fut != crs_train:
                    print(f"Внимание: CRS отличается у {period}/{scenario} (train={crs_train}, fut={crs_fut}).")
                
                suitability_f = predict_suitability_for_stack(model, stack_fut, valid_mask_fut, batch_size=500_000)
                
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
                print(f"Карта пригодности сохранена: {out_name_img}")
                
                # записываем в список прогнозов будущего
                if scenario not in future_imgs:
                    future_imgs[scenario] = []
                future_imgs[scenario].append(out_path_img)
                if period=='2081-2100':
                    future_imgs[scenario].append(out_path_img)
                
                #print(out_path)
                #print(out_path_img)
                
                title = ''
                if (len(df['species'].unique())==1):
                    title = 'Карта вероятности присутствия вида '+df['species'].unique()[0]+f" ({IN_ID})\nПериод: "+period+" (сценарий "+scenario+")"
                
                draw_map(out_path, out_path_img, title, rows_coord, cols_coord)
                if scenario!='SSP245_EC-Earth3-Veg':
                    os.remove(out_path) # пока не удаляем tif для будущего
        
        print(f"\nСоздаём анимацию:")
        for k in future_imgs:
            output_gif_path = os.path.join(OUTPUT_FUTURE_DIR, k+".gif")
            output_avi_path = os.path.join(OUTPUT_FUTURE_DIR, k+".avi")
            
            create_animated_gif(future_imgs[k], output_gif_path, duration=600)
            create_avi_from_images(future_imgs[k], output_avi_path, 2)
            
        
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
    
    