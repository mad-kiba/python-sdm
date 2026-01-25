# sdm/sdm.py

import os
import traceback
import json
import math
import glob
import zipfile
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from rasterio.transform import xy
#import libpysal - нужно для расчётов Moran's I, сейчас не используется
#import esda

# Импорт функций из utils
from .utils.preprocessing import clip_rasters, points_to_pixel_indices, pixel_indices_to_points
from .utils.data_loader import load_species_occurrence_data, load_environmental_predictors
from .utils.utils import sample_background, extract_features_from_stack, inverse_scale
from .utils.utils import save_geotiff, predict_suitability_for_stack
from .utils.plots import create_beautiful_histogram, draw_map, create_animated_gif, create_avi_from_images
from .utils.models import MaxEnt

class PythonSDM:
    def __init__(self, config):
        
        for attribute_name, attribute_value in config.items():
            setattr(self, attribute_name, attribute_value)
        
        
        print(f"-- Регион для моделирования ({self.IN_ID}): ")
        print("("+str(self.IN_MIN_LAT)+","+str(self.IN_MIN_LON)+"), ("+str(self.IN_MAX_LAT)+","+str(self.IN_MAX_LON)+"), step: "+self.IN_RESOLUTION)
        
        if self.IN_MIN_LAT==0 and self.IN_MAX_LAT==0:
            self.JOBS[self.IN_ID]['status'] = 'done'
            return {'result': 'Ok', 'code': 200}
        
        self.RANDOM_SEED = 42
        
        self.OUTPUT_SUITABILITY_TIF = "output/suitability/"+str(self.IN_ID)+"/suitability_"+str(self.IN_ID)+".tif"  # куда сохранить карту пригодности
        self.OUTPUT_SUITABILITY_JPG = "output/suitability/"+str(self.IN_ID)+"/suitability_"+str(self.IN_ID)+".jpg"
        self.OUTPUT_HISTOGRAMS_DIR = "output/gistos"
        self.OUTPUT_PREDICTIONS_DIR = "output/predictions"
        self.OUTPUT_SEASONS_DIR = "output/seasons"
        
        self.OUTPUT_FUTURE_DIR = os.path.join(self.OUTPUT_PREDICTIONS_DIR, str(self.IN_ID))
        self.OUTPUT_SEASONS_DIR = os.path.join(self.OUTPUT_SEASONS_DIR, str(self.IN_ID))
        
        self.RAW_RASTER_DIR = "input_predictors"
        
        self.SCALES_FILE = os.path.join(self.RAW_RASTER_DIR, 'predictors_scales.json')
        
        self.OUTPUT_RASTER_DIR = "output_predictors/"+self.IN_RESOLUTION+"/("+str(self.IN_MIN_LAT)+","\
                                +str(self.IN_MIN_LON)+"), ("+str(self.IN_MAX_LAT)+","+str(self.IN_MAX_LON)+")"
        self.RASTER_DIR = self.OUTPUT_RASTER_DIR # папка с GeoTIFF-предикторами
        
        if self.SCENARIOS == 'all':
            self.SCENARIOS = 'SSP126_EC-Earth3-Veg,SSP245_EC-Earth3-Veg,SSP370_EC-Earth3-Veg,SSP585_EC-Earth3-Veg'
        
        # Сколько фоновых точек генерировать: мин(10000, 10 * N_presence)
        self.MAX_BG = 10000
        
        # начали
        np.random.seed(self.RANDOM_SEED)
        
        self.TEXT_FILENAME = 'output/texts/'+str(self.IN_ID)+'/'+str(self.IN_ID)+'.txt'
        self.PRED_FILENAME = 'output/texts/'+str(self.IN_ID)+'/'+str(self.IN_ID)+'_pred.txt'
        self.STACK_FILENAME = 'output/texts/'+str(self.IN_ID)+'/'+str(self.IN_ID)+'_stack.txt'
        self.MONTH_FILENAME = 'output/texts/'+str(self.IN_ID)+'/'+str(self.IN_ID)+'_month.txt'
        self.CSV_FILENAME = 'output/texts/'+str(self.IN_ID)+'/'+str(self.IN_ID)+'.csv'
        self.CSV_FILENAME_ADD = 'output/texts/'+str(self.IN_ID)+'/'+str(self.IN_ID)+'_add.csv'
        self.GISTO_STATS = 'output/texts/'+str(self.IN_ID)+'/'+str(self.IN_ID)+'_gistos.js'
        self.FUTURE_SUITS = 'output/texts/'+str(self.IN_ID)+'/'+str(self.IN_ID)+'_futures.js'
        
        os.makedirs('output/texts/'+str(self.IN_ID)+'/', exist_ok=True)
        os.makedirs('output/suitability/'+str(self.IN_ID)+'/', exist_ok=True)
        
        # для запуска в многопоточном режиме
        j = self.JOBS.get(self.IN_ID)
        if not j:
            self.JOBS[self.IN_ID] = {'status': 'queued', 'file': None, 'error': None}
    
    
    def prepare_predictors(self):
        # 1) Подготовка предикторов к нужным координатам
        print(f"\n-- 1. Подготовка предикторов ({self.IN_ID})")
        clip_rasters(self.RAW_RASTER_DIR, self.OUTPUT_RASTER_DIR, self.IN_MIN_LAT, self.IN_MIN_LON,
                     self.IN_MAX_LAT, self.IN_MAX_LON, self.MODEL_FUTURE, self.IN_RESOLUTION)
        
    
    def load_occurences(self):
        # 2) Загрузка присутствий
        print(f"\n-- 2. Загрузка наблюдений ({self.IN_ID})")
        
        try:
            ret = load_species_occurrence_data(self.IN_ID, self.IN_CSV, self.IN_CSV_ADDITIONAL, self.CSV_FILENAME, self.CSV_FILENAME_ADD,
                                               self.MONTH_FILENAME, self.TEXT_FILENAME,
                                               self.IN_MIN_LON, self.IN_MIN_LAT, self.IN_MAX_LON, self.IN_MAX_LAT, self.JOBS)
        except Exception as e:
            # если не будут возвращаться тексты ошибок исключений, раскомментировать две строчки ниже:
            print(e)
            return {'status': 'terminated', 'error': str(e), 'code': 401}
        
        self.LAT_COL = ret['LAT_COL']
        self.LON_COL = ret['LON_COL']
        
        self.df = ret['df']
        self.occ = ret['occ']
        
        self.source_occ = ret['occ']
        self.species = ret['species']
    
    
    def load_predictors(self):
        # 3) Загрузка стека предикторов
        print(f"\n-- 3. Загрузка предикторов ({self.IN_ID})")
        try:
            self.stack, self.valid_mask, self.transform, self.crs, self.profile, self.band_names = \
                load_environmental_predictors(self.RASTER_DIR, self.PREDICTORS)
            self.bands, self.H, self.W = self.stack.shape
        except Exception as e:
            print(e)
            return {'status': 'terminated', 'error': str(e), 'code': 401}
        
        print(f"\n-- Загружено предикторов: {self.bands} | Размер: {self.H} x {self.W} | CRS: {self.crs}")
        print("Слои:", self.band_names)
        
        with open(self.TEXT_FILENAME, 'a') as f:
            f.write(f"\n{self.bands} | Размер: {self.H} x {self.W} | CRS: {self.crs}")
            f.write(f"\n{self.band_names}")
    
    
    def prepare_data(self, month = 0):
        # 4) Привязка присутствий к пикселям растра и фильтрация по маске валидности
        print(f"\n-- 4. Привязка присутствий к пикселям растра и фильтрация по маске валидности ({self.IN_ID})")
        if (month!=0):
            self.occ = self.source_occ.dropna()
            self.occ.loc[:, 'month'] = self.occ['month'].astype(int)
            self.occ = self.occ[(self.occ['month'])==month]
        else:
            self.occ = self.source_occ
        
        rows, cols, inside = points_to_pixel_indices(self.occ[self.LON_COL].values, self.occ[self.LAT_COL].values,\
                                                     self.transform, self.W, self.H)
        # Фильтруем те, что внутри растра
        rows, cols = rows[inside], cols[inside]
        # И те, что попадают на валидные пиксели (без NaN во всех слоях)
        valid_here = self.valid_mask[rows, cols]
        rows, cols = rows[valid_here], cols[valid_here]
        
        print(f"Присутствий внутри валидной области: {len(rows)}")
        
        if month==0:
            with open(self.TEXT_FILENAME, 'a') as f:
                f.write(f"\n{len(rows)}")
        
        if len(rows)<10 and month=='':
            print('Not enough points in region')
            return {'status': 'terminated', 'error': f"Внутри области моделирования недостаточно точек. Должно быть не менее 10, сейчас: {len(rows)}.", 'code': 401}
        
        # 4.1) создаём полные растры для всего спектра слоёв-предикторов
        print(f"-- 4.1. Создаём полные растры для всего спектра слоёв-предикторов ({self.IN_ID})")
        rows_grid, cols_grid = np.indices((self.H, self.W))
        
        # Преобразуем их в одномерные массивы
        rows_full_flat = rows_grid.flatten()
        cols_full_flat = cols_grid.flatten()
        
        # 4.2) Фильтруем эти полные индексы по маске валидности
        # valid_mask[rows_full_flat, cols_full_flat] вернет булеву маску для каждого пикселя
        # True, если пиксель валиден, False - если NaN
        valid_pixels_mask = self.valid_mask[rows_full_flat, cols_full_flat]
        
        # Применяем булеву маску, чтобы получить только валидные индексы
        self.rows_full = rows_full_flat[valid_pixels_mask]
        self.cols_full = cols_full_flat[valid_pixels_mask]
        
        self.rows = rows
        self.cols = cols
        
        
    def deduplicate_data(self, month = 0):
        # 5) Дедупликация по пикселю (30″ клетка) — оставляем по одному наблюдению на клетку
        print(f"\n-- 5. Дедупликация по пикселю — оставляем по одному наблюдению на клетку ({self.IN_ID})")
        self.pres_rc = pd.DataFrame({"r": self.rows, "c": self.cols}).drop_duplicates().values
        rows_p = self.pres_rc[:, 0]
        cols_p = self.pres_rc[:, 1]
        n_presence = len(rows_p)
        if n_presence < 20:
            print("Внимание: очень мало уникальных присутствий в пределах растра.")
        print(f"Уникальных присутствий (по пикселю): {n_presence}")
        
        if n_presence<5 and month==0:
            print('Not enough unique points in region')
            return {'status': 'terminated', 'error': f"Внутри области моделирования очень мало уникальных присутствий. Должно быть не менее 5, сейчас: {n_presence}.", 'code': 401}
        
        self.rows_coord, self.cols_coord, inside = pixel_indices_to_points(rows_p, cols_p, self.transform, self.W, self.H)
        
        if month==0:
            with open(self.TEXT_FILENAME, 'a') as f:
                f.write(f"\n{n_presence}")
            
        self.n_presence = n_presence
        self.rows_p = rows_p
        self.cols_p = cols_p


    def generate_bg_pa(self, month = 0):
        # 6) Генерация фоновых точек и точек псевдоотсутствия
        print(f"\n-- 6. Генерация фоновых точек и точек псевдоотсутствия ({self.IN_ID})")
        # 6.1) если нужно генерировать точки псевдоотсутствия, но параметры заданы на авто
        if self.BG_PC!=100 and self.BG_DISTANCE_MIN==0:
            
            dkingdom = ['']
            dclass = ['']
            
            if 'kingdom' in self.df.columns and 'class' in self.df.columns:
                temp_df = self.df.query("`kingdom`!='' and `class`!=''")
                temp_df = temp_df.dropna(subset=['kingdom', 'class'])
                dkingdom = temp_df['kingdom'].unique()
                dclass =   temp_df['class'].unique()
            
            print(f"Вычислено царство {dkingdom} и класс {dclass}")
            
            print("Нужно генерировать точки псевдоприсутствия, и параметры огибающих заданы на авто. Определяем их.")
            if len(dkingdom)==1 and len(dclass)<=1:
                # значения по умолчанию
                self.BG_DISTANCE_MIN = 10
                self.BG_DISTANCE_MAX = 20
                
                # вычисляем параметры
                if dclass==['Aves']: # Птицы
                    self.BG_DISTANCE_MIN = 50
                    self.BG_DISTANCE_MAX = 100
                    
                if dclass==['Mammalia']: # Млекопитающие
                    self.BG_DISTANCE_MIN = 20
                    self.BG_DISTANCE_MAX = 50
                    
                if dclass==['Amphibia']: # Амфибии
                    self.BG_DISTANCE_MIN = 20
                    self.BG_DISTANCE_MAX = 50
                    
                if dclass==['Squamata'] or dclass==['Testudines']: # Рептилии
                    self.BG_DISTANCE_MIN = 20
                    self.BG_DISTANCE_MAX = 50
            else:
                self.BG_PC = 100
        
        print(f"\n-- Генерация фоновых точек и точек псевдоотсутствия ({self.IN_ID})")
        print(f"Вычисленные параметры точек: BG_PC={self.BG_PC},"+\
              f"BG_DISTANCE_MIN={self.BG_DISTANCE_MIN}, BG_DISTANCE_MAX={self.BG_DISTANCE_MAX}")
        
        
        # 6.2) Генерация фоновых точек
        if (self.IN_MODEL=='MaxEnt'):
            self.BG_MULT = 100
            self.BG_ABS_PC = 0
            self.BG_PC = 100
        else:
            self.BG_ABS_PC = 100 - self.BG_PC
        
        if month==0:
            with open(self.TEXT_FILENAME, 'a') as f:
                f.write(f"\n{self.BG_PC},{self.BG_ABS_PC},{self.BG_DISTANCE_MIN},{self.BG_DISTANCE_MAX},{self.BG_MULT}")
                f.write(f"\n{self.IN_MIN_LAT},{self.IN_MIN_LON},{self.IN_MAX_LAT},{self.IN_MAX_LON},{self.IN_RESOLUTION},{self.IN_MODEL}")
                
        rng = np.random.default_rng(self.RANDOM_SEED)
        n_bg = min(self.MAX_BG, self.BG_MULT * self.n_presence)
        
        self.rows_bg, self.cols_bg = sample_background(self.valid_mask, set(map(tuple, self.pres_rc)), n_bg,
                                                       rng, self.BG_PC, self.BG_DISTANCE_MIN, self.BG_DISTANCE_MAX,
                                                       self.TEXT_FILENAME, month)
                
        print(f"Сэмплировано фоновых точек: {len(self.rows_bg)}")

    
    def extract_features(self):
        # 7) Извлечение признаков
        print(f"\n-- 7. Извлечение признаков ({self.IN_ID})")
        self.X_pres = extract_features_from_stack(self.stack, self.rows_p, self.cols_p)
        self.X_bg = extract_features_from_stack(self.stack, self.rows_bg, self.cols_bg)
        self.X_orig = extract_features_from_stack(self.stack, self.rows, self.cols)
        self.X_full = extract_features_from_stack(self.stack, self.rows_full, self.cols_full)
        self.X = np.vstack([self.X_pres, self.X_bg])
        self.y = np.hstack([np.ones(len(self.X_pres), dtype=int), np.zeros(len(self.X_bg), dtype=int)])
        print(f"Матрица признаков: {self.X.shape}, классы: {np.bincount(self.y)}")
        
        np.savetxt(self.STACK_FILENAME, self.X_orig, delimiter=";", fmt="%d")
    

    def draw_gistos(self):
        # 8) постройка гистограмм
        if self.DO_GISTO == 1:
            print(f"\n-- 8. Постройка гистограмм ({self.IN_ID})")
            num_predictors = len(self.band_names) # Получаем точное количество предикторов
            #print(SCALES_FILE)
            
            with open(self.SCALES_FILE, 'r') as f:
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
            if bins_num > len(np.unique(self.X_pres)):
                bins_num = len(np.unique(self.X_pres))
                print(f"Количество бинов было уменьшено до {bins_num}, так как оно превышало количество уникальных значений.")
            
            gistos_info = {}
            
            # --- Сохранение каждой гистограммы в отдельный файл ---
            # Пересоздаем фигуру и оси для сохранения, чтобы они были независимы от plt.show()
            # Это важно, чтобы сохранить чистые изображения без лишних элементов, добавленных plt.show()
            # (хотя в данном случае plt.show() уже показал, но для чистоты процесса сохранения)
            #print(num_predictors)
            # Нужно заново пройтись по данным, чтобы сохранить каждую гистограмму отдельно
            for i, band_name in enumerate(self.band_names):
                # Создаем новую фигуру для каждого графика
                fig_single, ax_single = plt.subplots(1, 1, figsize=(7, 5)) # Размер одного графика
                # Получаем масштабированные данные (они уже в X_pres)
                scaled_data_for_plot = self.X_pres[:, i]
                scaled_data_for_plot_full = self.X_full[:, i] 
                # Получаем параметры масштабирования для текущего предиктора
                # Убедитесь, что band_name соответствует ключам в scales_config
                scale_params = scales_config.get(band_name)
                # Применяем обратное преобразование, если параметры найдены
                
                layer_data = ''
                
                title = ''
                if self.species!='':
                    title = 'Вид: '+self.species
                
                if scale_params:
                    data_for_plot_original_scale = inverse_scale(scaled_data_for_plot, scale_params)
                    data_for_plot_original_scale_full = inverse_scale(scaled_data_for_plot_full, scale_params)
                    gist = create_beautiful_histogram(ax_single, data_for_plot_original_scale, band_name, bins_num,
                                                      data_for_plot_original_scale_full, title)
                else:
                    print(f"Предупреждение: Параметры масштабирования не найдены для '{band_name}'. Отображаются масштабированные значения.")
                    gist = create_beautiful_histogram(ax_single, scaled_data_for_plot, band_name, bins_num,
                                                      scaled_data_for_plot_full, title)
                
                gistos_info[band_name] = gist
                
                # Создаем имя файла
                # Заменяем недопустимые символы, если есть в band_name
                safe_band_name = band_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                dir_path = os.path.join(self.OUTPUT_HISTOGRAMS_DIR, str(self.IN_ID))
                os.makedirs(dir_path, exist_ok=True)
                output_filename = os.path.join(self.OUTPUT_HISTOGRAMS_DIR, str(self.IN_ID), f"{safe_band_name}.png")
                # Сохраняем фигуру
                plt.savefig(output_filename, dpi=300, bbox_inches='tight') # dpi для качества, bbox_inches='tight' для обрезки лишних полей
                print(f"Сохранена гистограмма: {i} - {output_filename}")
                plt.close(fig_single) # Закрываем фигуру, чтобы освободить память
            plt.close(fig)
            
            with open(self.GISTO_STATS, 'a') as f:
                json.dump(gistos_info, f, ensure_ascii=False, indent=4)
            
            print(f"Все гистограммы сохранены в папку: '{self.OUTPUT_HISTOGRAMS_DIR}\{self.IN_ID}'")
            
            archive_name = "histos.zip"
            archive_path = os.path.join(dir_path, archive_name)
            
            # 8.1. Получаем список всех файлов в папке для упаковки в архив
            files_to_zip = glob.glob(os.path.join(dir_path, "*.png"))
            
            # 8.2. Проверяем, есть ли вообще файлы для упаковки, пакуем
            if not files_to_zip:
                print(f"В папке {self.OUTPUT_HISTOGRAMS_DIR}\{self.IN_ID} нет файлов для упаковки.")
            else:
                # 3. Создаем ZIP-архив
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in files_to_zip:
                        # Добавляем файл в архив. os.path.basename гарантирует,
                        # что в архиве будут только имена файлов, а не полные пути.
                        zipf.write(file_path, os.path.basename(file_path))
                
                print(f"Все файлы из '{self.OUTPUT_HISTOGRAMS_DIR}\{self.IN_ID}' успешно упакованы в '{archive_path}'.")


    def split_train_test(self):
        # 9) Разделение на train/test
        print(f"\n-- 9. Разделение на train/test ({self.IN_ID})")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=self.RANDOM_SEED
        )
        
        
    def train_model(self, month = 0):
        # 10) Обучение модели
        print(f"\n-- 10. Обучение модели ({self.IN_ID})")
        
        if (self.IN_MODEL=='MaxEnt'):
            self.model = MaxEnt(X_pres=self.X_pres, X_bg=self.X_bg)
            self.model.fit(
                maxiter=500,
                tol=1e-5
            )
        
        if (self.IN_MODEL=='RandomForest'):
            self.model = RandomForestClassifier(
                n_estimators=500,
                n_jobs=-1,
                random_state=self.RANDOM_SEED,
                class_weight="balanced_subsample",
                max_depth=10
            )
            self.model.fit(self.X_train, self.y_train)
            
        if (self.IN_MODEL=='XGBoost'):
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=500,        # Количество деревьев
                learning_rate=0.05,      # Скорость обучения
                max_depth=10,             # Максимальная глубина деревьев
                subsample=0.8,           # Доля объектов для обучения каждого дерева
                colsample_bytree=0.8,    # Доля признаков для обучения каждого дерева
                random_state=self.RANDOM_SEED,
                n_jobs=-1,               # Использовать все доступные ядра CPU
                eval_metric='auc',
                tree_method='hist'       # Хорошо работает с большими данными
            )
            
            self.model.fit(self.X_train, self.y_train)
        
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        #print(model.predict_proba(X_test))
        
        self.auc = roc_auc_score(self.y_test, y_prob)
        print(f"ROC AUC (holdout): {self.auc:.3f}")
        
        # Если это основной прогон - записываем auc
        if month==0:
            with open(self.TEXT_FILENAME, 'a') as f:
                f.write(f"\n{self.auc:.3f}")
                
                if self.species!='':
                    title = self.species
                    f.write(f"\n{title}")
                else:
                    f.write(f"\nне определён")
        
        # Важность переменных
        if (self.IN_MODEL=='MaxEnt'):
            importances = self.model.weights
        else:
            importances = self.model.feature_importances_
        
        if month==0:
            print("Важность предикторов:")
            for name, imp in sorted(zip(self.band_names, importances), key=lambda x: -x[1]):
                print(f"  {name:30s} {imp:.4f}")
                with open(self.PRED_FILENAME, 'a') as f:
                    f.write(f"\n_{name:30s}:{imp:.4f}")


    def predict_current(self, month = 0):
        # 11) Прогноз на всю область и сохранение карты пригодности
        print(f"\n-- 11. Прогноз на всю область и сохранение карты пригодности ({self.IN_ID})")

        self.suitability = predict_suitability_for_stack(self.model, self.stack, self.valid_mask, batch_size=500_000)
        
        if (month!=0):
            self.OUTPUT_SUITABILITY_TIF = "output/suitability/"+str(self.IN_ID)+"/suitability_"+str(self.IN_ID)+"_"+str(month)+".tif"
            self.OUTPUT_SUITABILITY_JPG = "output/seasons/"+str(self.IN_ID)+"/cur_"+str(month)+".jpg"
        else:
            self.OUTPUT_SUITABILITY_TIF = "output/suitability/"+str(self.IN_ID)+"/suitability_"+str(self.IN_ID)+".tif"
            self.OUTPUT_SUITABILITY_TIF_ORIG = "output/suitability/"+str(self.IN_ID)+"/suitability_"+str(self.IN_ID)+".tif"
            self.OUTPUT_SUITABILITY_JPG = "output/suitability/"+str(self.IN_ID)+"/suitability_"+str(self.IN_ID)+".jpg"
        
        save_geotiff(self.OUTPUT_SUITABILITY_TIF, self.suitability, self.profile)
        print(f"Карта пригодности сохранена: {self.OUTPUT_SUITABILITY_TIF}")
        
        mask_high_suitability05 = self.suitability > 0.05
        hi_sui05 = np.sum(mask_high_suitability05)
        
        mask_high_suitability50 = self.suitability > 0.5
        hi_sui50 = np.sum(mask_high_suitability50)
        
        mask_high_suitability95 = self.suitability > 0.95
        hi_sui95 = np.sum(mask_high_suitability95)
        
        if month==0:
            with open(self.TEXT_FILENAME, 'a') as f:
                f.write(f"\nCHS05:{hi_sui05}")
                f.write(f"\nCHS50:{hi_sui50}")
                f.write(f"\nCHS95:{hi_sui95}")
        
        # (Опционально) можно сохранить также использованные точки присутствия в пиксельных координатах
        # или вернуть их центры в географических координатах:
        xs, ys = xy(self.transform, self.rows_p, self.cols_p, offset="center")
        used_occ_df = pd.DataFrame({"lon": xs, "lat": ys})
        used_occ_df.to_csv(os.path.join(os.path.dirname(self.OUTPUT_SUITABILITY_TIF),
                                        "used_presences_"+str(self.IN_ID)+".csv"), index=False)
        print("Сохранены использованные присутствия (уникальные по пикселю): used_presences_"+str(self.IN_ID)+".csv")
    
    
    def calculate_moransi(self):
        print("\nРасчёт коэффициента Moran's I: ")
        #moran_results =  calculate_morans_i_for_suitability(self.suitability_flat, self.rows_p, self.cols_p,
        #                                                    self.W, self.H, self.transform, self.crs)
        
        num_cells = self.suitability_flat.shape[0]
        cell_coords = np.array([[r, c] for r in range(self.H) for c in range(self.W)])
        
        k = 5 # Количество ближайших соседей
        W_knn = libpysal.weights.KNN.from_array(cell_coords, k=k)
        moran = esda.Moran(self.suitability_flat, W_knn)
        
        print(f"Значение Moran's I: {moran.I}")
        print(f"Ожидаемое значение E(I): {moran.EI}")
        #print(f"Дисперсия Var(I): {moran.VI}")
        print(f"Z-score: {moran.z_sim}") # Z-score на основе симуляций
        print(f"P-value: {moran.p_sim}") # P-value на основе симуляций
    
    
    def draw_map_current(self, month = 0):
        # 12) дальше рисуем картинку
        print(f"\n-- 12. Рисуем карту ({self.IN_ID})")
        title = ''
        if self.species!='':
            title = 'Карта вероятности присутствия вида '+self.species+f" ({self.IN_ID})"
        adtitle = f"\nМодель: {self.IN_MODEL}, шаг: {self.IN_RESOLUTION}, уник. точек: {self.n_presence}, ROC-AUC: {self.auc:.3f}";
        title = title + adtitle
        
        if month!=0:
            title = title + ", месяц: "+str(month)
        
        if self.n_presence>5:
            #print('---Tif:'+self.OUTPUT_SUITABILITY_TIF)
            draw_map(self.OUTPUT_SUITABILITY_TIF, self.OUTPUT_SUITABILITY_JPG, title, self.rows_coord, self.cols_coord)
        else:
            #print('---Tif:'+self.OUTPUT_SUITABILITY_TIF_ORIG)
            draw_map(self.OUTPUT_SUITABILITY_TIF_ORIG, self.OUTPUT_SUITABILITY_JPG, title, self.rows_coord, self.cols_coord, 1)
        
        if month!=0:
            self.monthly_imgs.append(self.OUTPUT_SUITABILITY_JPG)
            if self.OUTPUT_SUITABILITY_TIF_ORIG!=self.OUTPUT_SUITABILITY_TIF:
                if os.path.exists(self.OUTPUT_SUITABILITY_TIF):
                    os.remove(self.OUTPUT_SUITABILITY_TIF)
    
    
    def predict_future(self):
        # 13) если это стандартный регион - делаем с нашей моделью прогноз на будущее
        if self.MODEL_FUTURE==1 and self.IN_MODEL!='MaxEnt':
            print(f"\n-- 13. Приступаю к прогнозу будущего ({self.IN_ID})")
            # Пути
            FUTURE_ROOT_DIR = os.path.join(self.OUTPUT_RASTER_DIR, 'dynamic_predictable')   # где лежат папки периодов 2021-2040, ...
            
            os.makedirs(self.OUTPUT_FUTURE_DIR, exist_ok=True)
            
            if isinstance(self.PREDICTORS, str):
                PREDICTORS_EXP = [p.strip() for p in self.PREDICTORS.split(',') if p.strip()]
            
            
            OUTPUT_SUITABILITY_TIF = self.OUTPUT_FUTURE_DIR + "/1970-2000.tif"
            save_geotiff(OUTPUT_SUITABILITY_TIF, self.suitability, self.profile)
            print(f"Карта пригодности сохранена: {OUTPUT_SUITABILITY_TIF}")
            
            title = ''
            if self.species!='':
                title = 'Карта вероятности присутствия вида '+self.species+\
                        f" ({self.IN_ID})\nТекущий период (базовые климатические переменные)"
            OUTPUT_SUITABILITY_JPG = self.OUTPUT_FUTURE_DIR + "/1970-2000.jpg"
            draw_map(OUTPUT_SUITABILITY_TIF, OUTPUT_SUITABILITY_JPG, title, self.rows_coord, self.cols_coord)
            print(f"Карта пригодности сохранена: {OUTPUT_SUITABILITY_JPG}")
            #os.remove(OUTPUT_SUITABILITY_TIF) # пока не удаляем tif для будущего
            
            # области пригодности
            mask_high_suitability05 = self.suitability > 0.05
            hi_sui05 = np.sum(mask_high_suitability05)
            
            mask_high_suitability50 = self.suitability > 0.5
            hi_sui50 = np.sum(mask_high_suitability50)
            
            mask_high_suitability95 = self.suitability > 0.95
            hi_sui95 = np.sum(mask_high_suitability95)
            try:
                future_stats = {}
                future_stats['1970-2000'] = []
                future_stats['1970-2000'].append({'n05': hi_sui05, 'n50': hi_sui50, 'n95': hi_sui95})
            except Exception as e:
                print('Ошибка')
                print(str(e))
            
            
            # 13.1) Прогноз для будущих периодов/сценариев
            future_imgs = {}
            #print(self.SCENARIOS.split(','))
            for period in sorted(d for d in os.listdir(FUTURE_ROOT_DIR)
                                 if os.path.isdir(os.path.join(FUTURE_ROOT_DIR, d))):
                period_dir = os.path.join(FUTURE_ROOT_DIR, period)
                #print(period)
            
                for scenario in sorted(d for d in os.listdir(period_dir)
                                       if os.path.isdir(os.path.join(period_dir, d))):
                    #print(scenario)
                    if scenario in self.SCENARIOS.split(','):
                        scen_dir = os.path.join(period, scenario)
                        print(f"\nПрогноз: {period} / {scenario}")
                        
                        # Загружаем будущие предикторы строго в порядке self.PREDICTORS;
                        # если load_raster_stack не гарантирует порядок, переупорядочим по именам
                        stack_fut, valid_mask_fut, transform_fut, crs_fut, profile_fut, band_names_fut = \
                            load_environmental_predictors(self.RASTER_DIR, self.PREDICTORS, scen_dir)
                        
                        suitability_f = predict_suitability_for_stack(self.model, stack_fut, valid_mask_fut, batch_size=500_000)
                        
                        out_name = f"{period}-{scenario}.tif"
                        out_path = os.path.join(self.OUTPUT_FUTURE_DIR, out_name)
                        save_geotiff(out_path, suitability_f, profile_fut)
                        print(f"Сохранено: {out_path}")
                        
                        mask_high_suitability05 = suitability_f > 0.05
                        hi_sui05 = np.sum(mask_high_suitability05)
                        
                        mask_high_suitability50 = suitability_f > 0.5
                        hi_sui50 = np.sum(mask_high_suitability50)
                        
                        mask_high_suitability95 = suitability_f > 0.95
                        hi_sui95 = np.sum(mask_high_suitability95)
                        
                        out_name_img = f"{period}-{scenario}.{hi_sui05}.{hi_sui50}.{hi_sui95}.jpg"
                        out_path_img = os.path.join(self.OUTPUT_FUTURE_DIR, out_name_img)
                        print(f"Карта пригодности сохранена: {out_name_img}")
                        
                        # записываем в список прогнозов будущего
                        if scenario not in future_imgs:
                            future_imgs[scenario] = []
                        
                        if scenario not in future_stats:
                            future_stats[scenario] = []
                        
                        future_imgs[scenario].append(out_path_img)
                        future_stats[scenario].append({'n05': hi_sui05, 'n50': hi_sui50, 'n95': hi_sui95})
                        
                        if period=='2081-2100': # дублируем последний слайд, чтобы была пауза в анимации
                            future_imgs[scenario].append(out_path_img)
                        
                        #print(out_path)
                        #print(out_path_img)
                        
                        title = ''
                        if self.species!='':
                            title = 'Карта вероятности присутствия вида '+self.species+\
                                    f" ({self.IN_ID})\nПериод: "+period+" (сценарий "+scenario+")"
                        
                        draw_map(out_path, out_path_img, title, self.rows_coord, self.cols_coord)
                        if scenario!='SSP370_EC-Earth3-Veg':
                            os.remove(out_path) # пока не удаляем tif для будущего
            
            try:
                with open(self.FUTURE_SUITS, 'a') as f:
                    json.dump(future_stats, f, ensure_ascii=False, default=str)
            except Exception as e:
                print(str(e))
            
            print(f"\nСоздаём анимацию:")
            try:
                for k in future_imgs:
                    output_gif_path = os.path.join(self.OUTPUT_FUTURE_DIR, k+".gif")
                    output_mp4_path = os.path.join(self.OUTPUT_FUTURE_DIR, k+".mp4")
                    
                    create_animated_gif(future_imgs[k], output_gif_path, duration=600)
                    create_avi_from_images(future_imgs[k], output_mp4_path, 2)
            except Exception as e:
                print("Ошибка создания анимации: " + str(e))
                
            
            print(f"\nВсе прогнозы сохранены в папку: '{self.OUTPUT_FUTURE_DIR}'")
    
            archive_name = "futures.zip"
            archive_path = os.path.join(self.OUTPUT_FUTURE_DIR, archive_name)
            
            # 13.4) Получаем список всех файлов в папке для упаковки в архив
            files_to_zip = glob.glob(os.path.join(self.OUTPUT_FUTURE_DIR, "*"))
            
            # 13.5) Проверяем, есть ли вообще файлы для упаковки, пакуем
            if not files_to_zip:
                print(f"В папке {self.OUTPUT_FUTURE_DIR} нет файлов для упаковки.")
            else:
                # 3. Создаем ZIP-архив
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in files_to_zip:
                        # Добавляем файл в архив. os.path.basename гарантирует,
                        # что в архиве будут только имена файлов, а не полные пути.
                        zipf.write(file_path, os.path.basename(file_path))
            
            print(f"Все файлы из '{self.OUTPUT_FUTURE_DIR}' успешно упакованы в '{archive_path}'.")
            
    
    def predict_monthly(self):
        # 14) сезонный прогноз
        if self.DO_SEASON == 1 and 'month' in self.df.columns:
            print(f"\n-- 14. Приступаю к помесячному моделированию")
            self.monthly_imgs = []
            os.makedirs("output/seasons/"+str(self.IN_ID), exist_ok=True)
            
            try:
                for month in range(1, 13):
                    print(f"\n---- Прогноз для месяца {month}")
                    self.prepare_data(month)
                    self.deduplicate_data(month)
                    if self.n_presence>5:
                        self.generate_bg_pa(month)
                        self.extract_features()
                        self.split_train_test()
                        self.train_model(month)
                        self.predict_current(month)
                    self.OUTPUT_SUITABILITY_JPG = "output/seasons/"+str(self.IN_ID)+"/cur_"+str(self.IN_ID)+"_"+str(month)+".jpg"
                    self.draw_map_current(month)
            except Exception as e:
                print(e)
                return {'status': 'terminated', 'error': str(e), 'code': 401}
            
            
            # 14.1) Анимания сезонности
            print(f"\nСоздаём сезонности:")
            try:
                output_gif_path = os.path.join(self.OUTPUT_SEASONS_DIR, "monthly_"+str(self.IN_ID)+".gif")
                output_mp4_path = os.path.join(self.OUTPUT_SEASONS_DIR, "monthly_"+str(self.IN_ID)+".mp4")
                
                create_animated_gif(self.monthly_imgs, output_gif_path, duration=600)
                create_avi_from_images(self.monthly_imgs, output_mp4_path, 2)
            except Exception as e:
                print("Ошибка создания анимации: " + str(e))
                
                
            # 14.2) Упаковываем сезонность в архив
            print("\n-- Упаковка сезонных прогнозов")
            archive_name = "seasons.zip"
            archive_path = os.path.join(self.OUTPUT_SEASONS_DIR, archive_name)
            #print("Путь к архиву: "+archive_path)
            files_to_zip = glob.glob(os.path.join(self.OUTPUT_SEASONS_DIR, "*"))
            #print("Файлы для упаковки: ")
            #print(files_to_zip)
            
            if not files_to_zip:
                print(f"В папке {self.OUTPUT_SEASONS_DIR} нет файлов для упаковки.")
            else:
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in files_to_zip:
                        zipf.write(file_path, os.path.basename(file_path))
                        
            
            print("-- Конец помесячного моделирования")


