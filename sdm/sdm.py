# to optimize

# sdm/sdm.py
# Библиотека PythonSDM для моделирования распространения видов

import os
import traceback
import json
import math
import glob
import zipfile
import time
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, precision_score
from rasterio.transform import xy
from sklearn.calibration import CalibratedClassifierCV
#import libpysal - нужно для расчётов Moran's I, сейчас не используется
#import esda

# Импорт функций из utils
from .utils.preprocessing import clip_rasters, points_to_pixel_indices, pixel_indices_to_points
from .utils.data_loader import load_species_occurrence_data, load_environmental_predictors
from .utils.utils import sample_background, extract_features_from_stack, inverse_scale, continuous_boyce_index
from .utils.utils import save_geotiff, predict_suitability_for_stack, save_error, get_geotiff_square, clean_nans_for_json
from .utils.utils import apply_decay_to_points, calculate_niche_breadth_pca
from .utils.plots import create_beautiful_histogram, draw_map, create_animated_gif, create_avi_from_images, plot_roc_auc_curve, draw_m_factor_map
from .utils.models import MaxEnt
from .utils.predictors_info import get_predictors_info

class PythonSDM:
    def __init__(self, config): # 0) Загрузка входных параметров
        
        for attribute_name, attribute_value in config.items(): # заполняем входящие параметры
            setattr(self, attribute_name, attribute_value)
            
            
        # для запуска в многопоточном режиме
        j = self.JOBS.get(self.IN_ID)
        self.ERROR_FILENAME = f"output/texts/{self.IN_ID}/{self.IN_ID}_error.txt"
        if not j:
            self.JOBS[self.IN_ID] = {'status': 'queued', 'file': None, 'error': None}
            

        # если нулевые параметры - это запрос на данные из старого расчёта
        if self.IN_MIN_LAT==0 and self.IN_MAX_LAT==0:
            #print('Get old query: ' + str(self.IN_ID))
            if os.path.exists(self.ERROR_FILENAME):
                with open(self.ERROR_FILENAME, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    print('Ошибка:' + file_content)
                    self.JOBS[self.IN_ID]['status'] = 'error'
                    self.JOBS[self.IN_ID]['error'] = file_content
                    return {'status': 'error', 'error': file_content, 'code': 401}
            else:
                self.JOBS[self.IN_ID]['status'] = 'done'
                self.is_old_query = True
                return {'result': 'Ok', 'code': 200}
            return
        else:
            # это полноценный запуск новой модели
            print(f"-- Регион для моделирования ({self.IN_ID}): ")
            print(f"({self.IN_MIN_LON},{self.IN_MIN_LAT}), ({self.IN_MAX_LON},{self.IN_MAX_LAT}), step: {self.IN_RESOLUTION}")
        
        
        
        self.RANDOM_SEED = 42
        
        self.OUTPUT_SUITABILITY_TIF = f"output/suitability/{self.IN_ID}/suitability_{self.IN_ID}.tif"  # куда сохранить карту пригодности
        self.OUTPUT_SUITABILITY_JPG = f"output/suitability/{self.IN_ID}/suitability_{self.IN_ID}.jpg"
        self.OUTPUT_HISTOGRAMS_DIR = "output/gistos"
        self.OUTPUT_PREDICTIONS_DIR = "output/predictions"
        self.OUTPUT_PAST_DIR = "output/past"
        self.OUTPUT_SEASONS_DIR = "output/seasons"
        
        self.OUTPUT_ROC_AUC_JPG = f"output/aucs/{self.IN_ID}/roc-auc.jpg"
        
        self.OUTPUT_FUTURE_DIR = os.path.join(self.OUTPUT_PREDICTIONS_DIR, str(self.IN_ID))
        self.OUTPUT_PAST_DIR = os.path.join(self.OUTPUT_PAST_DIR, str(self.IN_ID))
        self.OUTPUT_SEASONS_DIR = os.path.join(self.OUTPUT_SEASONS_DIR, str(self.IN_ID))
        
        self.RAW_RASTER_DIR = "input_predictors"
        self.PREDICTORS_JPEGS = "output/predictors_jpegs"
        
        
        self.SCALES_FILE = os.path.join(self.RAW_RASTER_DIR, 'predictors_scales.json')
        
        self.OUTPUT_RASTER_DIR = f"output_predictors/{self.IN_RESOLUTION}/({self.IN_MIN_LON},{self.IN_MIN_LAT}), ({self.IN_MAX_LON},{self.IN_MAX_LAT})"
        self.RASTER_DIR = self.OUTPUT_RASTER_DIR # папка с GeoTIFF-предикторами
        
        if self.SCENARIOS == 'all':
            #self.SCENARIOS = 'SSP126_EC-Earth3-Veg,SSP245_EC-Earth3-Veg,SSP370_EC-Earth3-Veg,SSP585_EC-Earth3-Veg' # old for WorldClim
            self.SCENARIOS = 'ssp126,ssp370,ssp585' # new for CHELSA
        
        # Сколько фоновых точек генерировать: мин(10000, 10 * N_presence)
        self.MAX_BG = 10000

        # какую неопределённость координат (в метрах из DWCA/GBIF) считаем допустимой
        self.ALLOWED_COORD_UNCERTAIN = 1000
        
        # начиная с какого года используем данные
        #self.MINIMUM_YEAR_ALLOWED = 1980
        self.MINIMUM_YEAR_ALLOWED = 2000

        # какой метод затухания распространения вида использовать (M-фактор из BAM-фреймворка)
        self.M_FACTOR_DECAY_TYPE = 'sigmoid'

        # начали
        np.random.seed(self.RANDOM_SEED)
        
        self.TEXT_FILENAME = f"output/texts/{self.IN_ID}/{self.IN_ID}.txt"
        self.PRED_FILENAME = f"output/texts/{self.IN_ID}/{self.IN_ID}_pred.txt"
        self.MONTH_FILENAME = f"output/texts/{self.IN_ID}/{self.IN_ID}_month.txt"
        self.CSV_FILENAME = f"output/texts/{self.IN_ID}/{self.IN_ID}.csv"
        self.CSV_FILENAME_ADD = f"output/texts/{self.IN_ID}/{self.IN_ID}_add.csv"
        self.CSV_FILTERED_FILENAME = f"output/texts/{self.IN_ID}/{self.IN_ID}_filtered.csv"
        self.GISTO_STATS = f"output/texts/{self.IN_ID}/{self.IN_ID}_gistos.js"
        self.FUTURE_SUITS = f"output/texts/{self.IN_ID}/{self.IN_ID}_futures.js"
        
        base_dirs = ['texts', 'suitability', 'aucs', 'gistos', 'predictions', 'past', 'seasons', 'predictors_jpegs']
        for d in base_dirs:
            # Создаем корневую директорию
            os.makedirs(f"output/{d}/", exist_ok=True)
            # Создаем директорию конкретно для этой задачи (там, где это применимо)
            if d in ['texts', 'suitability', 'aucs']:
                os.makedirs(f"output/{d}/{self.IN_ID}/", exist_ok=True)
        
        self.bio_info = get_predictors_info()
    
    
    def prepare_predictors(self): # 1) Подготовка предикторов к нужным координатам
        print(f"\n-- 1. Подготовка предикторов ({self.IN_ID})")
        try:
            clip_rasters(self.RAW_RASTER_DIR, self.OUTPUT_RASTER_DIR, self.IN_MIN_LAT, self.IN_MIN_LON,
                         self.IN_MAX_LAT, self.IN_MAX_LON, self.MODEL_FUTURE, self.MODEL_PAST, self.MODEL_SEASON, self.IN_RESOLUTION)
        except Exception as e:
            print('Ошибка подготовки предикторов:')
            print(e)
            save_error(self.ERROR_FILENAME, e)
            return {'status': 'terminated', 'error': str(e), 'code': 401}
        
    
    def load_occurrences(self): # 2) Загрузка присутствий
        print(f"\n-- 2. Загрузка наблюдений ({self.IN_ID})")
        
        try:
            ret = load_species_occurrence_data(self.IN_ID, self.IN_CSV, self.IN_CSV_ADDITIONAL,
                                               self.CSV_FILENAME, self.CSV_FILENAME_ADD, self.CSV_FILTERED_FILENAME,
                                               self.MONTH_FILENAME, self.TEXT_FILENAME,
                                               self.IN_MIN_LON, self.IN_MIN_LAT, self.IN_MAX_LON, self.IN_MAX_LAT,
                                               self.ALLOWED_COORD_UNCERTAIN, self.MINIMUM_YEAR_ALLOWED)
        except Exception as e:
            # если не будут возвращаться тексты ошибок исключений, раскомментировать две строчки ниже:
            print(e)
            save_error(self.ERROR_FILENAME, e)
            return {'status': 'terminated', 'error': str(e), 'code': 401}
        
        self.LAT_COL = ret['LAT_COL']
        self.LON_COL = ret['LON_COL']
        
        self.df = ret['df']
        self.occ = ret['occ']
        
        self.source_occ = ret['occ']
        self.species = ret['species']
        self.kingdom = ret['kingdom']
        self.dclass = ret['dclass']
    
    
    def load_predictors(self, period = 'current', month = ''): # 3) Загрузка стека предикторов
        print(f"\n-- 3. Загрузка предикторов ({self.IN_ID})")
        try:
            if os.path.exists(self.SCALES_FILE):
                with open(self.SCALES_FILE, 'r') as f:
                    self.scales_config = json.load(f)
            else:
                # тут подделать скейлы
                self.scales_config = {}
                print('Файл масштабов не найден')
            
            if period == 'current':
                self.stack, self.valid_mask, self.transform, self.crs, self.profile, self.band_names, self.band_paths = \
                    load_environmental_predictors(self.RASTER_DIR, self.PREDICTORS, scales = self.scales_config, bio_info = self.bio_info)
            elif period == 'monthly':
                self.stack, self.valid_mask, self.transform, self.crs, self.profile, self.band_names, self.band_paths = \
                    load_environmental_predictors(self.RASTER_DIR, self.PREDICTORS, period = 'monthly', interval = month, bio_info = self.bio_info)
            else:
                raise ValueError(f"Неподдерживаемый период загрузки предикторов: {period}")
            
            self.bands, self.H, self.W = self.stack.shape
        except Exception as e:
            print(e)
            save_error(self.ERROR_FILENAME, e)
            return {'status': 'terminated', 'error': str(e), 'code': 401}
        
        print(f"\n-- Загружено предикторов: {self.bands} | Размер: {self.H} x {self.W} | CRS: {self.crs}")
        print("Слои:", self.band_names)
        
        if period == 'current':
            with open(self.TEXT_FILENAME, 'a') as f:
                f.write(f"\n{self.bands} | Размер: {self.H} x {self.W} | CRS: {self.crs}")
                f.write(f"\n{self.band_names}")
    
    
    def prepare_data(self, month = 0): # 4) Привязка присутствий к пикселям растра и фильтрация по маске валидности
        print(f"\n-- 4. Привязка присутствий к пикселям растра и фильтрация по маске валидности ({self.IN_ID})")
        if (month!=0):
            self.occ = self.source_occ.dropna().copy()
            self.occ.loc[:, 'month'] = self.occ['month'].astype(int)
            self.occ = self.occ[(self.occ['month'])==month]
        else:
            self.occ = self.source_occ
        
        rows, cols, inside = points_to_pixel_indices(self.occ[self.LON_COL].values, self.occ[self.LAT_COL].values,\
                                                     self.transform, self.W, self.H)
        # Фильтруем те, что внутри растра
        rows, cols = rows[inside], cols[inside]

        # Обязательно фильтруем точки, попавшие в зоны NoData (океан за пределами 50 пикс).
        # С новой экстраполяцией в data_loader это безопасно и спасает деревья решений от NaN-ошибок.
        valid_here = self.valid_mask[rows, cols]
        rows, cols = rows[valid_here], cols[valid_here]
        
        print(f"Присутствий внутри валидной области: {len(rows)}")
        
        if month==0:
            with open(self.TEXT_FILENAME, 'a') as f:
                f.write(f"\n{len(rows)}")
        
        if len(rows)<10 and month==0:
            print('Not enough points in region')
            save_error(self.ERROR_FILENAME, f"Внутри области моделирования недостаточно точек. Должно быть не менее 10, сейчас: {len(rows)}.")
            return {'status': 'terminated', 'error': f"Внутри области моделирования недостаточно точек. Должно быть не менее 10, сейчас: {len(rows)}.", 'code': 401}
        
        # 4.1) создаём полные растры для всего спектра слоёв-предикторов
        print(f"-- 4.1. Создаём полные растры для всего спектра слоёв-предикторов ({self.IN_ID})")
        self.rows_full, self.cols_full = np.nonzero(self.valid_mask)
        
        self.rows = rows
        self.cols = cols
        
        
    def deduplicate_data(self, month = 0): # 5) Дедупликация по пикселю (30″ клетка) — оставляем по одному наблюдению на клетку
        print(f"\n-- 5. Дедупликация по пикселю — оставляем по одному наблюдению на клетку ({self.IN_ID})")
        
        # Оставляем строго 1 точку на 1 уникальный пиксель растра
        # Spatial thinning не делаем. Метод не показал пользы, см. модели №50201, 50204, 50203
        df_pres = pd.DataFrame({"r": self.rows, "c": self.cols})
        df_pixel_thinned = df_pres.drop_duplicates(subset=["r", "c"])
        
        self.pres_rc = df_pixel_thinned[["r", "c"]].values
        
        rows_p = self.pres_rc[:, 0]
        cols_p = self.pres_rc[:, 1]

        n_presence = len(rows_p)
        if n_presence < 20:
            print("Внимание: очень мало уникальных присутствий в пределах растра.")
        print(f"Уникальных присутствий по точкам растра: {n_presence}")

        
        if n_presence<5 and month==0:
            print('Not enough unique points in region')
            save_error(self.ERROR_FILENAME, f"Внутри области моделирования очень мало уникальных присутствий. Должно быть не менее 5, сейчас: {n_presence}.")
            return {'status': 'terminated', 'error': f"Внутри области моделирования очень мало уникальных присутствий. Должно быть не менее 5, сейчас: {n_presence}.", 'code': 401}
        
        self.pres_lons, self.pres_lats, inside = pixel_indices_to_points(rows_p, cols_p, self.transform, self.W, self.H)
        
        if month==0:
            with open(self.TEXT_FILENAME, 'a') as f:
                f.write(f"\n{n_presence}")
            
        self.n_presence = n_presence
        self.rows_p = rows_p
        self.cols_p = cols_p


    def generate_bg_pa(self, month = 0): # 6) Генерация фоновых точек и точек псевдоотсутствия, а также фактора мобильности
        print(f"\n-- 6. Генерация фоновых точек и точек псевдоотсутствия ({self.IN_ID})")
        
        # границы распространения вида, фактор мобильности по умолчанию
        rec_mf_cur = 100
        rec_mf_2040 = 200
        rec_mf_2070 = 300
        rec_mf_2100 = 400
        
        # 6.1) если нужно генерировать точки псевдоотсутствия, но параметры заданы на авто
        if self.BG_PC!=100 and self.BG_DISTANCE_MIN==0:
            print("Нужно генерировать точки псевдоприсутствия, и параметры огибающих заданы на авто. Определяем их.")

            # Оцениваем физический размер пикселя в километрах для текущего разрешения растра
            if self.IN_RESOLUTION == '30s':
                pixel_size_km = 1.0
            elif self.IN_RESOLUTION == '1m':
                pixel_size_km = 2.0
            elif self.IN_RESOLUTION == '5m':
                pixel_size_km = 10.0
            else:
                pixel_size_km = 1.0
                

            if len(self.kingdom)==1 and len(self.dclass)<=1:
                # значения по умолчанию В КИЛОМЕТРАХ
                bg_min_km = 10.0
                bg_max_km = 20.0
                
                # вычисляем параметры
                if self.dclass==['Aves']: # Птицы
                    bg_min_km = 50.0
                    bg_max_km = 100.0
                    
                    # Зона M (доступность). Базовый радиус: 500 км.
                    # Скорость расширения (дельта): ~150 км за 30 лет
                    rec_mf_cur = 500
                    rec_mf_2040 = 650
                    rec_mf_2070 = 800
                    rec_mf_2100 = 950
                    
                if self.dclass==['Mammalia']: # Млекопитающие
                    bg_min_km = 20.0
                    bg_max_km = 50.0
                    
                    # Скорость расширения: ~30 км за 30 лет (усредненно для грызунов и хищников)
                    rec_mf_cur = 200
                    rec_mf_2040 = 230
                    rec_mf_2070 = 260
                    rec_mf_2100 = 290
                    
                if self.dclass==['Amphibia']: # Амфибии
                    bg_min_km = 10.0
                    bg_max_km = 50.0
                    
                    # Крайне низкая мобильность. Базовая зона M узкая (50 км). 
                    # Расширение: ~5 км за 30 лет
                    # В будущем здесь стоит прикрутить Cost-Distance (Friction Surface)
                    # с высоким "штрафом" за отсутствие влажности и водоемов.
                    rec_mf_cur = 50
                    rec_mf_2040 = 55
                    rec_mf_2070 = 60
                    rec_mf_2100 = 65
                    
                if self.dclass==['Squamata'] or self.dclass==['Testudines']: # Рептилии
                    bg_min_km = 10.0
                    bg_max_km = 50.0
                    
                    # Расширение: ~10 км за 30 лет
                    rec_mf_cur = 100
                    rec_mf_2040 = 110
                    rec_mf_2070 = 120
                    rec_mf_2100 = 130
                
                if self.kingdom==['Plantae']: # Растения
                    bg_min_km = 10.0
                    bg_max_km = 50.0
                    
                    # Растения мигрируют медленно (семена/вегетативно), но имеют Long-Distance Dispersal.
                    # Расширение: ~15 км за 30 лет.
                    rec_mf_cur = 100
                    rec_mf_2040 = 110
                    rec_mf_2070 = 120
                    rec_mf_2100 = 130
                    #self.M_FACTOR_DECAY_TYPE = 'inverse_quadratic' # Тяжелые хвосты (Long-Distance Dispersal)
                
                if self.kingdom==['Fungi']: # Грибы
                    bg_min_km = 10.0
                    bg_max_km = 50.0
                    
                    # Споры разлетаются далеко, но успешное укоренение требует времени.
                    # Расширение: ~30 км за 30 лет.
                    rec_mf_cur = 100
                    rec_mf_2040 = 130
                    rec_mf_2070 = 160
                    rec_mf_2100 = 190
                    #self.M_FACTOR_DECAY_TYPE = 'exponential' # Экспоненциальное затухание облака спор
                
                if self.dclass==['Insecta']: # Насекомые
                    bg_min_km = 20.0
                    bg_max_km = 100.0
                    
                    # Высокая мобильность, быстро следуют за потеплением климата
                    rec_mf_cur = 200
                    rec_mf_2040 = 250
                    rec_mf_2070 = 300
                    rec_mf_2100 = 350
                    #self.M_FACTOR_DECAY_TYPE = 'gaussian' # Диффузное активное расселение
                
                
                # Переводим физические километры в пиксели растра (шаги сетки)
                self.BG_DISTANCE_MIN = int(max(1, np.round(bg_min_km / pixel_size_km)))
                self.BG_DISTANCE_MAX = int(max(2, np.round(bg_max_km / pixel_size_km)))
            else:
                self.BG_PC = 100
        
        # если фактор мобильности задан на авто
        if self.M_FACTOR_CUR == 0:
            self.M_FACTOR_CUR = rec_mf_cur
        if self.M_FACTOR_2040 == 0:
            self.M_FACTOR_2040 = rec_mf_2040
        if self.M_FACTOR_2070 == 0:
            self.M_FACTOR_2070 = rec_mf_2070
        if self.M_FACTOR_2100 == 0:
            self.M_FACTOR_2100 = rec_mf_2100
        
        print(f"Текущие факторы мобильности: {self.M_FACTOR_CUR}, {self.M_FACTOR_2040}, {self.M_FACTOR_2070}, {self.M_FACTOR_2100}")
        
        print(f"\n-- Генерация фоновых точек и точек псевдоотсутствия ({self.IN_ID})")
        print(f"Вычисленные параметры точек: BG_PC={self.BG_PC},"+\
              f"BG_DISTANCE_MIN={self.BG_DISTANCE_MIN}, BG_DISTANCE_MAX={self.BG_DISTANCE_MAX}")
        
        try:
            # 6.2) Генерация фоновых точек
            if (self.IN_MODEL=='MaxEnt'): # какие-то значения для вывода, реально будет self.MAX_BG
                self.BG_MULT = 20
                self.BG_ABS_PC = 0 # не генерируем точки псевдоотсутствия
                self.BG_PC = 100
            else:
                self.BG_ABS_PC = 100 - self.BG_PC
            
            if month==0:
                with open(self.TEXT_FILENAME, 'a') as f:
                    f.write(f"\n{self.BG_PC},{self.BG_ABS_PC},{self.BG_DISTANCE_MIN},{self.BG_DISTANCE_MAX},{self.BG_MULT}")
                    f.write(f"\n{self.IN_MIN_LON},{self.IN_MIN_LAT},{self.IN_MAX_LON},{self.IN_MAX_LAT},{self.IN_RESOLUTION},{self.IN_MODEL}")
                    
            rng = np.random.default_rng(self.RANDOM_SEED)
            
            if self.IN_MODEL == 'MaxEnt':
                # MaxEnt требует интеграции по всему ландшафту.
                # Обычно используют 10 000 точек (self.MAX_BG) независимо от n_presence.
                n_bg = self.MAX_BG
            else:
                # Для RF и XGBoost сохраняем пропорцию классов
                n_bg = min(self.MAX_BG, int(self.BG_MULT * self.n_presence))
            
            
            self.rows_bg, self.cols_bg, self.rows_random, self.cols_random, self.rows_buffer, self.cols_buffer = sample_background(self.valid_mask,
                                                           set(map(tuple, self.pres_rc)), n_bg,
                                                           rng, self.BG_PC, self.BG_DISTANCE_MIN, self.BG_DISTANCE_MAX,
                                                           self.TEXT_FILENAME, month)
                    
            print(f"Сэмплировано фоновых точек + точек псевдоотсутствия: {len(self.rows_bg)}")
        except Exception as e:
            print('Ошибка генерации фоновых точек:')
            print(e)

    
    def extract_features(self): # 7) Извлечение признаков
        print(f"\n-- 7. Извлечение признаков ({self.IN_ID})")
        self.X_pres = extract_features_from_stack(self.stack, self.rows_p, self.cols_p)
        self.X_bg = extract_features_from_stack(self.stack, self.rows_bg, self.cols_bg)
        self.X_orig = extract_features_from_stack(self.stack, self.rows, self.cols)
        self.X_full = extract_features_from_stack(self.stack, self.rows_full, self.cols_full)
        self.X = np.vstack([self.X_pres, self.X_bg])
        self.y = np.hstack([np.ones(len(self.X_pres), dtype=int), np.zeros(len(self.X_bg), dtype=int)])
        print(f"Матрица признаков: {self.X.shape}, классы: {np.bincount(self.y)}")


    def draw_gistos(self): # 8) постройка гистограмм
        if self.DO_GISTO == 1:
            print(f"\n-- 8. Постройка гистограмм ({self.IN_ID})")
            num_predictors = len(self.band_names) # Получаем точное количество предикторов
            
            if num_predictors == 0:
                print("Нет предикторов для построения гистограмм.")
                return
            
            gistos_info = {}
            
            # --- Сохранение каждой гистограммы в отдельный файл ---
            # Оптимизация: Создаем фигуру ОДИН РАЗ вне цикла. 
            # Создание фигуры в Matplotlib - очень "дорогая" операция для CPU.
            fig_single, ax_single = plt.subplots(1, 1, figsize=(7, 5))
            
            for i, band_name in enumerate(self.band_names):
                ax_single.clear() # Очищаем оси перед новым графиком
                # Получаем масштабированные данные (они уже в X_pres)
                scaled_data_for_plot = self.X_pres[:, i]
                scaled_data_for_plot_full = self.X_full[:, i]
                
                # Индивидуальная регулировка количества бинов (защита для категориальных слоев)
                bins_num = 50
                unique_vals = len(np.unique(scaled_data_for_plot))
                if unique_vals > 0 and bins_num > unique_vals:
                    bins_num = unique_vals
                    print(f"Количество бинов для '{band_name}' уменьшено до {bins_num} по числу уникальных значений.")
                
                # Получаем параметры масштабирования для текущего предиктора
                # Убедитесь, что band_name соответствует ключам в scales_config
                scale_params = self.scales_config.get(band_name)
                # Применяем обратное преобразование, если параметры найдены
                #print('Scale params for name: ' + band_name)
                #print(scale_params)
                
                layer_data = ''
                
                title = ''
                if self.species!='':
                    title = 'Вид: '+self.species
                
                if scale_params:
                    try:
                        data_for_plot_original_scale = inverse_scale(scaled_data_for_plot, scale_params, self.bio_info[band_name])
                        data_for_plot_original_scale_full = inverse_scale(scaled_data_for_plot_full, scale_params, self.bio_info[band_name])
                        
                        gist = create_beautiful_histogram(ax_single, data_for_plot_original_scale, band_name, bins_num,
                                                          data_for_plot_original_scale_full, self.bio_info, title)
                    except Exception as e:
                        print('Ошибка построения гистограммы:')
                        print(e)
                else:
                    print(f"Предупреждение: Параметры масштабирования не найдены для '{band_name}'. Отображаются масштабированные значения.")
                    gist = create_beautiful_histogram(ax_single, scaled_data_for_plot, band_name, bins_num,
                                                      scaled_data_for_plot_full, self.bio_info, title)
                
                gistos_info[band_name] = gist
                
                # Создаем имя файла
                # Заменяем недопустимые символы, если есть в band_name
                safe_band_name = band_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                dir_path = os.path.join(self.OUTPUT_HISTOGRAMS_DIR, str(self.IN_ID))
                os.makedirs(dir_path, exist_ok=True)
                output_filename = os.path.join(self.OUTPUT_HISTOGRAMS_DIR, str(self.IN_ID), f"{safe_band_name}.png")
                
                fig_single.tight_layout()
                plt.savefig(output_filename, dpi=100) 
                print(f"Сохранена гистограмма: {i} - {output_filename}")
                # plt.close(fig_single) - БОЛЬШЕ НЕ ЗАКРЫВАЕМ ВНУТРИ ЦИКЛА
            
            plt.close(fig_single) # Закрываем один раз после завершения всех предикторов
            clean_gistos_info = clean_nans_for_json(gistos_info)
            with open(self.GISTO_STATS, 'a') as f:
                json.dump(clean_gistos_info, f, ensure_ascii=False, indent=4)
            
            print(f"Все гистограммы сохранены в папку: '{self.OUTPUT_HISTOGRAMS_DIR}/{self.IN_ID}'")
            
            archive_name = "histos.zip"
            archive_path = os.path.join(dir_path, archive_name)
            
            # 8.1. Получаем список всех файлов в папке для упаковки в архив
            files_to_zip = glob.glob(os.path.join(dir_path, "*.png"))
            
            # 8.2. Проверяем, есть ли вообще файлы для упаковки, пакуем
            if not files_to_zip:
                print(f"В папке {self.OUTPUT_HISTOGRAMS_DIR}/{self.IN_ID} нет файлов для упаковки.")
            else:
                # 3. Создаем ZIP-архив
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in files_to_zip:
                        # Добавляем файл в архив. os.path.basename гарантирует,
                        # что в архиве будут только имена файлов, а не полные пути.
                        zipf.write(file_path, os.path.basename(file_path))
                
                print(f"Все файлы из '{self.OUTPUT_HISTOGRAMS_DIR}/{self.IN_ID}' успешно упакованы в '{archive_path}'.")


    def split_train_test(self): # 9) Разделение на train/test
        print(f"\n-- 9. Разделение на train/test ({self.IN_ID})")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=self.RANDOM_SEED
        )
        
        
    def train_model(self, month = 0):
        # 10) Обучение модели
        print(f"\n-- 10. Обучение валидационной модели (на {len(self.X_train)} точках) ({self.IN_ID})")
        
        # --- 1. ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ОЦЕНКИ (На Train выборке) ---
        # Эта модель используется ТОЛЬКО для получения честных метрик. Она не видит 20% карты.
        try:
            if (self.IN_MODEL=='MaxEnt'):
                # Выделяем присутствия и фон из обучающей выборки для честного тестирования MaxEnt
                X_train_pres = self.X_train[self.y_train == 1]
                X_train_bg = self.X_train[self.y_train == 0]
                eval_model = MaxEnt(X_pres=X_train_pres, X_bg=X_train_bg)
                eval_model.fit(maxiter=500, tol=1e-5)
                
            elif (self.IN_MODEL=='RandomForest'):
                eval_model = RandomForestClassifier(
                    n_estimators=500, n_jobs=-1, random_state=self.RANDOM_SEED, 
                    class_weight="balanced_subsample", max_depth=10
                )
                eval_model.fit(self.X_train, self.y_train)
                
            elif (self.IN_MODEL=='XGBoost'):
                eval_model = xgb.XGBClassifier(
                    objective='binary:logistic', n_estimators=500, learning_rate=0.05, 
                    max_depth=10, subsample=0.8, colsample_bytree=0.8, 
                    random_state=self.RANDOM_SEED, n_jobs=-1, eval_metric='auc', tree_method='hist'
                )
                eval_model.fit(self.X_train, self.y_train)
        except Exception as e:
            print('Ошибка обучения валидационной модели')
            print(str(e))
            save_error(self.ERROR_FILENAME, e)
            return {'status': 'terminated', 'error': str(e), 'code': 401}
            
        print('Обучение валидационной модели завершено')
        
        print(f"\n-- 10А. Вычисление честных метрик на отложенных блоках ({self.IN_ID})")
        
        try:
            # вычисление вероятностей на отложенной (тестовой) выборке
            y_prob = eval_model.predict_proba(self.X_test)[:, 1]

            # вычисление Continuous Boyce Index (CBI)
            obs_prob = y_prob[self.y_test == 1]
            self.boyce_index = continuous_boyce_index(obs_prob, y_prob)
            print(f"Continuous Boyce Index (CBI): {self.boyce_index:.3f}")
            
            # вычисление оптимального порога threshold
            # Пересчитываем fpr, tpr, thresholds для всех возможных пороков, чтобы найти оптимальный
            fpr_all, tpr_all, thresholds_all = roc_curve(self.y_test, y_prob)
        
            # Метрика для поиска оптимального порога: максимизация sum_sens_spec
            sum_sens_spec = tpr_all + (1 - fpr_all) # TPR = sensitivity, 1 - FPR = specificity
            best_idx = np.argmax(sum_sens_spec)
            self.optimal_threshold = thresholds_all[best_idx]
            
            self.auc = roc_auc_score(self.y_test, y_prob)
            print(f"ROC AUC (holdout): {self.auc:.3f}")
            
            # строим график ROC-AUC
            fpr, tpr, thresholds = roc_curve(self.y_test, y_prob)
            plot_roc_auc_curve(fpr, tpr, self.auc, self.OUTPUT_ROC_AUC_JPG, self.IN_ID)
            
            # вычисление других метрик
            y_pred = (y_prob >= self.optimal_threshold).astype(int)
            cm = confusion_matrix(self.y_test, y_pred)
            self.TN, self.FP, self.FN, self.TP = cm.ravel()
            self.auc = roc_auc_score(self.y_test, y_prob)
            self.kappa = cohen_kappa_score(self.y_test, y_pred)
            
            self.sensitivity = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0
            self.specificity = self.TN / (self.TN + self.FP) if (self.TN + self.FP) > 0 else 0
            
            self.tss = self.sensitivity + self.specificity - 1
            
            print(f"TSS: {self.tss:.4f}")
            print(f"Cohen's Kappa: {self.kappa:.4f}")
            
            self.fdr = self.FP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0
            self.for_rate = self.FN / (self.TN + self.FN) if (self.TN + self.FN) > 0 else 0
            self.ppv = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0
            self.npv = self.TN / (self.TN + self.FN) if (self.TN + self.FN) > 0 else 0
            
            self.bias_score = (self.TP + self.FP) / (self.TP + self.FN) if (self.TP + self.FN) > 0 else np.inf
            
            self.csi = self.TP / (self.TP + self.FN + self.FP) if (self.TP + self.FN + self.FP) > 0 else 0
            
            self.accuracy = accuracy_score(self.y_test, y_pred)
            self.misclassification_rate = 1 - self.accuracy
            
            
        except Exception as e:
            print('Ошибка оценки качества модели')
            print(e)
            
        # --- 2. ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ (На ВСЕХ данных) ---
        # Эта модель пойдет в predict_current для отрисовки полной и точной карты
        print(f"\n-- 10Б. Обучение финальной боевой модели на 100% данных ({self.IN_ID})")
        try:
            if (self.IN_MODEL=='MaxEnt'):
                self.model = MaxEnt(X_pres=self.X_pres, X_bg=self.X_bg)
                self.model.fit(maxiter=500, tol=1e-5)
            elif (self.IN_MODEL=='RandomForest'):
                self.model = RandomForestClassifier(
                    n_estimators=500, n_jobs=-1, random_state=self.RANDOM_SEED, 
                    class_weight="balanced_subsample", max_depth=10
                )
                self.model.fit(self.X, self.y) # Обучаем на X и y!
            elif (self.IN_MODEL=='XGBoost'):
                self.model = xgb.XGBClassifier(
                    objective='binary:logistic', n_estimators=500, learning_rate=0.05, 
                    max_depth=10, subsample=0.8, colsample_bytree=0.8, 
                    random_state=self.RANDOM_SEED, n_jobs=-1, eval_metric='auc', tree_method='hist'
                )
                self.model.fit(self.X, self.y) # Обучаем на X и y!
        except Exception as e:
            print('Ошибка обучения финальной модели')
            print(e)
            save_error(self.ERROR_FILENAME, e)
            return {'status': 'terminated', 'error': str(e), 'code': 401}
        
        # --- Важность переменных и Экологическая пластичность (Niche Breadth) ---
        # Наша новая версия MaxEnt теперь тоже поддерживает свойство feature_importances_
        importances = self.model.feature_importances_
        
        # Находим индексы Топ-5 самых важных предикторов
        top5_idx = np.argsort(importances)[::-1][:5]
        
        self.pca_breadth = calculate_niche_breadth_pca(self.X_pres, self.X_bg, top_indices=top5_idx)
        print(f"Многомерная экологическая пластичность (PCA Niche Breadth, top-5): {self.pca_breadth:.4f}")
        
        
        # Если это основной прогон - записываем метрики
        if month==0:
            with open(self.TEXT_FILENAME, 'a') as f:
                f.write(f"\n{self.auc:.3f},{self.tss:.3f},{self.kappa:.3f},{self.TN:.3f},{self.FP:.3f},{self.TP:.3f},{self.FN:.3f},{self.optimal_threshold:.3f},")
                f.write(f"{self.sensitivity:.3f},{self.specificity:.3f},{self.fdr:.3f},{self.for_rate:.3f},{self.ppv:.3f},{self.npv:.3f},")
                f.write(f"{self.bias_score:.3f},{self.csi:.3f},{self.accuracy:.3f},{self.misclassification_rate:.3f},{self.boyce_index:.3f},{self.pca_breadth:.4f}")
                
                if self.species!='':
                    title = self.species
                    f.write(f"\n{title}")
                else:
                    f.write(f"\nне определён")
        
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
        
        # --- BAM-фреймворк: Применение M-фактора (мобильность с учетом рельефа) ---
        if self.M_FACTOR_CUR != -1:
            try:
                print(f"Применяем фактор мобильности (M-фактор): {self.M_FACTOR_CUR} км...")
                slope_data = None
                if 'slope_deg' in self.band_names:
                    slope_idx = self.band_names.index('slope_deg')
                    slope_scaled = self.stack[slope_idx]
                    # Выполняем обратное масштабирование, чтобы передать в FMM реальные градусы уклона
                    slope_params = self.scales_config.get('slope_deg')
                    slope_info = self.bio_info.get('slope_deg', {})
                    slope_data = inverse_scale(slope_scaled, slope_params, slope_info)
                    print("  Слой 'slope_deg' найден! Алгоритм будет учитывать рельеф при расчете доступности.")
                else:
                    print("  Слой 'slope_deg' не найден. Будет использовано стандартное евклидово расстояние.")
                    
                elev_data = None
                if 'wc2.1_30s_elev' in self.band_names:
                    elev_idx = self.band_names.index('wc2.1_30s_elev')
                    elev_scaled = self.stack[elev_idx]
                    elev_params = self.scales_config.get('wc2.1_30s_elev')
                    elev_info = self.bio_info.get('wc2.1_30s_elev', {})
                    elev_data = inverse_scale(elev_scaled, elev_params, elev_info)
                    print("  Слой высоты найден! Горы будут физическим барьером.")
                    
                water_data = None
                if 'Consensus_reduced_class_12' in self.band_names:
                    water_idx = self.band_names.index('Consensus_reduced_class_12')
                    water_scaled = self.stack[water_idx]
                    water_params = self.scales_config.get('Consensus_reduced_class_12')
                    water_info = self.bio_info.get('Consensus_reduced_class_12', {})
                    water_data = inverse_scale(water_scaled, water_params, water_info)
                    print("  Слой открытой воды найден! Моря станут преградой.")
                    
                is_bird = (self.dclass == ['Aves'])

                # Генерируем матрицу множителей (0.0 ... 1.0)
                m_factor_multiplier = apply_decay_to_points(
                    raster_shape=(self.H, self.W),
                    transform=self.transform,
                    observation_rows=self.rows_p,
                    observation_cols=self.cols_p,
                    buffer_km=self.M_FACTOR_CUR,
                    decay_type=self.M_FACTOR_DECAY_TYPE, # 'buffer' делает жесткую обрезку (1 внутри, 0 снаружи)
                    slope_data=slope_data,
                    elev_data=elev_data,
                    water_data=water_data,
                    is_bird=is_bird
                )
                
                # Умножаем оригинальную пригодность (Абиотика) на M-фактор (Мобильность)
                self.suitability = self.suitability * m_factor_multiplier
            except Exception as e:
                print(f"Ошибка применения фактора мобильности: {e}")
        else:
            print("Фактор мобильности (M-фактор) отключен пользователем (-1).")
        # -------------------------------------------------------------------------
        
        if (month!=0):
            self.OUTPUT_SUITABILITY_TIF = "output/suitability/"+str(self.IN_ID)+"/suitability_"+str(self.IN_ID)+"_"+str(month)+".tif"
            self.OUTPUT_SUITABILITY_JPG = "output/seasons/"+str(self.IN_ID)+"/cur_"+str(month)+".jpg"
        else:
            self.OUTPUT_SUITABILITY_TIF = "output/suitability/"+str(self.IN_ID)+"/suitability_"+str(self.IN_ID)+".tif"
            self.OUTPUT_SUITABILITY_TIF_ORIG = "output/suitability/"+str(self.IN_ID)+"/suitability_"+str(self.IN_ID)+".tif"
            self.OUTPUT_SUITABILITY_JPG = "output/suitability/"+str(self.IN_ID)+"/suitability_"+str(self.IN_ID)+".jpg"
        
        save_geotiff(self.OUTPUT_SUITABILITY_TIF, self.suitability, self.profile)
        print(f"Карта пригодности сохранена: {self.OUTPUT_SUITABILITY_TIF}")
        
        # Сохраняем саму маску M-фактора отдельным файлом (для просмотра в QGIS)
        if month == 0 and self.M_FACTOR_CUR != -1 and 'm_factor_multiplier' in locals():
            try:
                M_FACTOR_TIF = os.path.join(os.path.dirname(self.OUTPUT_SUITABILITY_TIF), f"m_factor_mask_{self.IN_ID}.tif")
                
                # --- Генерация комбинированной карты M-фактора для всех периодов ---
                combined_m = np.zeros((self.H, self.W), dtype=np.float32)
                
                # Накладываем от большего к меньшему, чтобы перекрыть внутренние зоны. Значения: 4: 2100, 3: 2070, 2: 2040, 1: Текущий
                if getattr(self, 'M_FACTOR_2100', -1) > 0:
                    m2100 = apply_decay_to_points(
                        raster_shape=(self.H, self.W), transform=self.transform, 
                        observation_rows=self.rows_p, observation_cols=self.cols_p, 
                        buffer_km=self.M_FACTOR_2100, decay_type=self.M_FACTOR_DECAY_TYPE, 
                        slope_data=slope_data, elev_data=elev_data, water_data=water_data, is_bird=is_bird
                    )
                    combined_m[m2100 > 0] = 4
                if getattr(self, 'M_FACTOR_2070', -1) > 0:
                    m2070 = apply_decay_to_points(
                        raster_shape=(self.H, self.W), transform=self.transform, 
                        observation_rows=self.rows_p, observation_cols=self.cols_p, 
                        buffer_km=self.M_FACTOR_2070, decay_type=self.M_FACTOR_DECAY_TYPE, 
                        slope_data=slope_data, elev_data=elev_data, water_data=water_data, is_bird=is_bird
                    )
                    combined_m[m2070 > 0] = 3
                if getattr(self, 'M_FACTOR_2040', -1) > 0:
                    m2040 = apply_decay_to_points(
                        raster_shape=(self.H, self.W), transform=self.transform, 
                        observation_rows=self.rows_p, observation_cols=self.cols_p, 
                        buffer_km=self.M_FACTOR_2040, decay_type=self.M_FACTOR_DECAY_TYPE, 
                        slope_data=slope_data, elev_data=elev_data, water_data=water_data, is_bird=is_bird
                    )
                    combined_m[m2040 > 0] = 2
                
                # Поверх всего кладем текущий M-фактор (самый строгий)
                combined_m[m_factor_multiplier > 0] = 1
                
                combined_m[combined_m == 0] = np.nan
                save_geotiff(M_FACTOR_TIF, combined_m, self.profile)
                
                # Отрисовка JPEG карты M-фактора
                M_FACTOR_JPG = os.path.join(os.path.dirname(self.OUTPUT_SUITABILITY_TIF), f"m_factor_map_{self.IN_ID}.jpg")
                title_m = f"Зоны доступности вида (M-фактор) ({self.IN_ID})"
                draw_m_factor_map(M_FACTOR_TIF, M_FACTOR_JPG, title_m, self.pres_lons, self.pres_lats, id=self.IN_ID)
                print(f"Карта расширения M-фактора сохранена: {M_FACTOR_JPG}")
                
            except Exception: pass
        
        threshold_list = [0.05, 0.25, 0.5, 0.75, 0.95, self.optimal_threshold, self.optimal_threshold/2]
        self.gsq, self.gsc = get_geotiff_square(self.OUTPUT_SUITABILITY_TIF, threshold_list)
        
        if month==0:
            with open(self.TEXT_FILENAME, 'a') as f:
                f.write(f"\nSHSLOW:{self.gsq[6]}")
                f.write(f"\nSHSOPT:{self.gsq[5]}")
                f.write(f"\nSHS05:{self.gsq[0]}")
                f.write(f"\nSHS25:{self.gsq[1]}")
                f.write(f"\nSHS50:{self.gsq[2]}")
                f.write(f"\nSHS75:{self.gsq[3]}")
                f.write(f"\nSHS95:{self.gsq[4]}")
                f.write(f"\nCHSLOW:{self.gsc[6]}")
                f.write(f"\nCHSOPT:{self.gsc[5]}")
                f.write(f"\nCHS05:{self.gsc[0]}")
                f.write(f"\nCHS25:{self.gsc[1]}")
                f.write(f"\nCHS50:{self.gsc[2]}")
                f.write(f"\nCHS75:{self.gsc[3]}")
                f.write(f"\nCHS95:{self.gsc[4]}")
        
        if month==0:
            # Сохраним использованные точки присутствия в географических координатах:
            xs, ys = xy(self.transform, self.rows_p, self.cols_p, offset="center")
            used_occ_df = pd.DataFrame({"decimalLongitude": xs, "decimalLatitude": ys})
            used_occ_df.loc[:, 'kingdom'] = self.kingdom[0]
            used_occ_df.loc[:, 'class'] = self.dclass[0]
            used_occ_df.loc[:, 'species'] = self.species
            used_occ_df.to_csv(os.path.join(os.path.dirname(self.OUTPUT_SUITABILITY_TIF),
                                            "used_presences_"+str(self.IN_ID)+".csv"), index=False, sep="\t")
            print("Сохранены использованные присутствия (уникальные по пикселю): used_presences_"+str(self.IN_ID)+".csv")
            
            xs, ys = xy(self.transform, self.rows_random, self.cols_random, offset="center")
            used_rand_df = pd.DataFrame({"decimalLongitude": xs, "decimalLatitude": ys})
            #used_rand_df.loc[:, 'kingdom'] = self.kingdom[0]
            #used_rand_df.loc[:, 'class'] = self.dclass[0]
            #used_rand_df.loc[:, 'species'] = self.species
            used_rand_df.to_csv(os.path.join(os.path.dirname(self.OUTPUT_SUITABILITY_TIF),
                                            "used_randoms_"+str(self.IN_ID)+".csv"), index=False, sep="\t")
            print("Сохранены использованные присутствия (уникальные по пикселю): used_randoms_"+str(self.IN_ID)+".csv")
            
            xs, ys = xy(self.transform, self.rows_buffer, self.cols_buffer, offset="center")
            used_buff_df = pd.DataFrame({"decimalLongitude": xs, "decimalLatitude": ys})
            #used_buff_df.loc[:, 'kingdom'] = self.kingdom[0]
            #used_buff_df.loc[:, 'class'] = self.dclass[0]
            #used_buff_df.loc[:, 'species'] = self.species
            used_buff_df.to_csv(os.path.join(os.path.dirname(self.OUTPUT_SUITABILITY_TIF),
                                            "used_buffer_"+str(self.IN_ID)+".csv"), index=False, sep="\t")
            print("Сохранены использованные присутствия (уникальные по пикселю): used_buffer_"+str(self.IN_ID)+".csv")
    
    
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
        
        subopt = self.optimal_threshold/2
        adt1 = f"\nSопт = "+str(self.gsq[5])+f" кв.км (p>{self.optimal_threshold:.3f}), "
        adt2 = f"Sсубопт = "+str(self.gsq[6])+f" кв.км (p>{subopt:.3f})"
        title = title + adt1 + adt2
        
        try:
            if self.n_presence>5:
                #print('---Tif:'+self.OUTPUT_SUITABILITY_TIF)
                draw_map(self.OUTPUT_SUITABILITY_TIF, self.OUTPUT_SUITABILITY_JPG, title, self.pres_lons, self.pres_lats, id=self.IN_ID)
            else:
                #print('---Tif:'+self.OUTPUT_SUITABILITY_TIF_ORIG)
                draw_map(self.OUTPUT_SUITABILITY_TIF_ORIG, self.OUTPUT_SUITABILITY_JPG, title, self.pres_lons, self.pres_lats, 1, id=self.IN_ID)
        except Exception as e:
            print('Ошибка рисования карты')
            print(str(e))
        
        if month!=0:
            self.monthly_imgs.append(self.OUTPUT_SUITABILITY_JPG)
            if self.OUTPUT_SUITABILITY_TIF_ORIG!=self.OUTPUT_SUITABILITY_TIF:
                if os.path.exists(self.OUTPUT_SUITABILITY_TIF):
                    os.remove(self.OUTPUT_SUITABILITY_TIF)
        
        print(f"Карта присутствия сохранена в формате JPEG: {self.OUTPUT_SUITABILITY_JPG}")
    
    
    def predict_future(self):
        # 13) если это стандартный регион - делаем с нашей моделью прогноз на будущее
        if self.MODEL_FUTURE==1 and self.IN_MODEL!='MaxEnt':
            print(f"\n-- 13. Приступаю к прогнозу будущего ({self.IN_ID})")
            # Пути
            FUTURE_ROOT_DIR = os.path.join(self.OUTPUT_RASTER_DIR, 'dynamic_predictable')   # где лежат папки периодов 2021-2040, ...
            
            os.makedirs(self.OUTPUT_FUTURE_DIR, exist_ok=True)
            
            if isinstance(self.PREDICTORS, str):
                PREDICTORS_EXP = [p.strip() for p in self.PREDICTORS.split(',') if p.strip()]
            
            
            OUTPUT_SUITABILITY_TIF = self.OUTPUT_FUTURE_DIR + "/1981-2010.tif"
            save_geotiff(OUTPUT_SUITABILITY_TIF, self.suitability, self.profile)
            print(f"Карта пригодности сохранена: {OUTPUT_SUITABILITY_TIF}")
            
            threshold_list = [0.05, 0.25, 0.5, 0.75, 0.95, self.optimal_threshold, self.optimal_threshold/2]
            gsq, gsc = get_geotiff_square(OUTPUT_SUITABILITY_TIF, threshold_list)
            
            title = ''
            if self.species!='':
                title = 'Карта вероятности присутствия вида '+self.species+\
                        f" ({self.IN_ID})\nТекущий период (базовые климатические переменные)"
            
            subopt = self.optimal_threshold/2
            adt1 = f"\nSопт = "+str(gsq[5])+f" кв.км (p>{self.optimal_threshold:.3f}), "
            adt2 = f"Sсубопт = "+str(gsq[6])+f" кв.км (p>{subopt:.3f})"
            title = title + adt1 + adt2
                
            OUTPUT_SUITABILITY_JPG = self.OUTPUT_FUTURE_DIR + "/1981-2010.jpg"
            draw_map(OUTPUT_SUITABILITY_TIF, OUTPUT_SUITABILITY_JPG, title, self.pres_lons, self.pres_lats, id=self.IN_ID)
            print(f"Карта пригодности сохранена: {OUTPUT_SUITABILITY_JPG}")
            #os.remove(OUTPUT_SUITABILITY_TIF) # пока не удаляем tif для будущего
            
            
            future_stats = {}
            future_stats['1981-2010'] = []
            future_stats['1981-2010'].append({'n05': gsc[0], 'n50': gsc[2], 'n95': gsc[4], 'n25': gsc[1], 'n75': gsc[3], 'nopt': gsc[5], 'nlow': gsc[6],
                                              's05': gsq[0], 's50': gsq[2], 's95': gsq[4], 's25': gsq[1], 's75': gsq[3], 'sopt': gsq[5], 'slow': gsq[6],})
            
            
            # 13.1) Прогноз для будущих периодов/сценариев
            future_imgs = {}
            #print(self.SCENARIOS.split(','))
            for period in sorted(d for d in os.listdir(FUTURE_ROOT_DIR)
                                 if os.path.isdir(os.path.join(FUTURE_ROOT_DIR, d))):
                period_dir = os.path.join(FUTURE_ROOT_DIR, period)
                print(period)
            
                for scenario in sorted(d for d in os.listdir(period_dir)
                                       if os.path.isdir(os.path.join(period_dir, d))):
                    print(scenario)
                    if scenario in self.SCENARIOS.split(','):
                        scen_dir = os.path.join(period, scenario)
                        print(f"\nПрогноз: {period} / {scenario}")
                        
                        # Загружаем будущие предикторы строго в порядке self.PREDICTORS;
                        stack_fut, valid_mask_fut, transform_fut, crs_fut, profile_fut, band_names_fut, band_paths = \
                            load_environmental_predictors(self.RASTER_DIR, self.PREDICTORS, 'future', scen_dir, '', self.bio_info)
                        
                        # делаем прогноз для будущего с той же моделью, с которой делали текущее
                        try:
                            suitability_f = predict_suitability_for_stack(self.model, stack_fut, valid_mask_fut, batch_size=500_000)
                        except Exception as e:
                            print('Ошибка прогноза будущего')
                            print(str(e))
                            
                        # --- BAM-фреймворк: Применение M-фактора для будущего ---
                        future_m_factor = self.M_FACTOR_CUR
                        if '2040' in period: future_m_factor = self.M_FACTOR_2040
                        elif '2070' in period: future_m_factor = self.M_FACTOR_2070
                        elif '2100' in period: future_m_factor = self.M_FACTOR_2100
                        
                        if future_m_factor != -1:
                            slope_data_fut = None
                            if 'slope_deg' in band_names_fut:
                                slope_idx_fut = band_names_fut.index('slope_deg')
                                slope_scaled_fut = stack_fut[slope_idx_fut]
                                slope_params_fut = self.scales_config.get('slope_deg')
                                slope_info_fut = self.bio_info.get('slope_deg', {})
                                slope_data_fut = inverse_scale(slope_scaled_fut, slope_params_fut, slope_info_fut)
                                
                            elev_data_fut = None
                            if 'wc2.1_30s_elev' in band_names_fut:
                                elev_idx_fut = band_names_fut.index('wc2.1_30s_elev')
                                elev_data_fut = inverse_scale(stack_fut[elev_idx_fut], self.scales_config.get('wc2.1_30s_elev'), self.bio_info.get('wc2.1_30s_elev', {}))
                                
                            water_data_fut = None
                            if 'Consensus_reduced_class_12' in band_names_fut:
                                water_idx_fut = band_names_fut.index('Consensus_reduced_class_12')
                                water_data_fut = inverse_scale(stack_fut[water_idx_fut], self.scales_config.get('Consensus_reduced_class_12'), self.bio_info.get('Consensus_reduced_class_12', {}))
                            
                            try:
                                m_factor_multiplier_fut = apply_decay_to_points(
                                    raster_shape=(self.H, self.W),
                                    transform=transform_fut,
                                    observation_rows=self.rows_p,
                                    observation_cols=self.cols_p,
                                    buffer_km=future_m_factor,
                                    decay_type=self.M_FACTOR_DECAY_TYPE,
                                    slope_data=slope_data_fut,
                                    elev_data=elev_data_fut,
                                    water_data=water_data_fut,
                                    is_bird=(self.dclass == ['Aves'])
                                )
                                suitability_f = suitability_f * m_factor_multiplier_fut
                            except Exception as e:
                                print(f"Ошибка применения фактора мобильности для будущего ({period}): {e}")
                        else:
                            print(f"Фактор мобильности для будущего ({period}) отключен (-1).")
                        # --------------------------------------------------------
                        
                        out_name = f"{period}-{scenario}.tif"
                        out_path = os.path.join(self.OUTPUT_FUTURE_DIR, out_name)
                        save_geotiff(out_path, suitability_f, profile_fut)
                        print(f"Сохранено: {out_path}")
                        
                        threshold_list = [0.05, 0.25, 0.5, 0.75, 0.95, self.optimal_threshold, self.optimal_threshold/2]
                        gsq, gsc = get_geotiff_square(out_path, threshold_list)
                        
                        out_name_img = f"{period}-{scenario}.jpg"
                        out_path_img = os.path.join(self.OUTPUT_FUTURE_DIR, out_name_img)
                        print(f"Карта пригодности сохранена: {out_name_img}")
                        
                        # записываем в список прогнозов будущего
                        if scenario not in future_imgs:
                            future_imgs[scenario] = []
                        
                        if scenario not in future_stats:
                            future_stats[scenario] = {}
                        
                        future_imgs[scenario].append(out_path_img)
                        future_stats[scenario][period] = {'n05': gsc[0], 'n50': gsc[2], 'n95': gsc[4], 'n25': gsc[1], 'n75': gsc[3], 'nopt': gsc[5], 'nlow': gsc[6],
                                              's05': gsq[0], 's50': gsq[2], 's95': gsq[4], 's25': gsq[1], 's75': gsq[3], 'sopt': gsq[5], 'slow': gsq[6]}
                        
                        if period=='2071-2100': # дублируем последний слайд, чтобы была пауза в анимации
                            future_imgs[scenario].append(out_path_img)
                        
                        #print(out_path)
                        #print(out_path_img)
                        
                        title = ''
                        if self.species!='':
                            title = 'Карта вероятности присутствия вида '+self.species+\
                                    f" ({self.IN_ID})\nПериод: "+period+" (сценарий "+scenario+")"
                            
                        adt1 = f"\nSопт = "+str(gsq[5])+f" кв.км, "
                        adt2 = f"Sсубопт = "+str(gsq[6])+f" кв.км"
                        title = title + adt1 + adt2
                        
                        draw_map(out_path, out_path_img, title, self.pres_lons, self.pres_lats, id=self.IN_ID)
                        if scenario!='SSP370_EC-Earth3-Veg' and scenario!='ssp370':
                            os.remove(out_path) # пока не удаляем tif для будущего
            
            try:
                clean_futures = clean_nans_for_json(future_stats)
                with open(self.FUTURE_SUITS, 'a') as f:
                    json.dump(clean_futures, f, ensure_ascii=False, default=str)
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
    
            # Больше не упаковываем файлы в архив, передаём как есть
            #archive_name = "futures.zip"
            #archive_path = os.path.join(self.OUTPUT_FUTURE_DIR, archive_name)
            
            # 13.4) Получаем список всех файлов в папке для упаковки в архив
            #files_to_zip = glob.glob(os.path.join(self.OUTPUT_FUTURE_DIR, "*"))
            
            # 13.5) Проверяем, есть ли вообще файлы для упаковки, пакуем
            #if not files_to_zip:
            #    print(f"В папке {self.OUTPUT_FUTURE_DIR} нет файлов для упаковки.")
            #else:
            #    # 3. Создаем ZIP-архив
            #    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            #        for file_path in files_to_zip:
            #            # Добавляем файл в архив. os.path.basename гарантирует,
            #            # что в архиве будут только имена файлов, а не полные пути.
            #            zipf.write(file_path, os.path.basename(file_path))
            #print(f"Все файлы из '{self.OUTPUT_FUTURE_DIR}' успешно упакованы в '{archive_path}'.")
            
            print()
    
    
    def predict_past(self):
        # 14) если это стандартный регион - делаем с нашей моделью прогноз прошлого
        if self.MODEL_PAST==1 and self.IN_MODEL!='MaxEnt':
            print(f"\n-- 14. Приступаю к прогнозу прошлого ({self.IN_ID})")
            # Пути
            PAST_ROOT_DIR = os.path.join(self.OUTPUT_RASTER_DIR, 'dynamic_past')   # где лежат папки периодов 2021-2040, ...
            
            os.makedirs(self.OUTPUT_PAST_DIR, exist_ok=True)
    
    
    def predict_monthly(self):
        # 15) сезонный прогноз
        if self.MODEL_SEASON == 1 and 'month' in self.df.columns:
            print(f"\n-- 15. Приступаю к помесячному моделированию")
            self.monthly_imgs = []
            os.makedirs("output/seasons/"+str(self.IN_ID), exist_ok=True)
            
            try:
                for month in range(1, 13):
                    print(f"\n---- Прогноз для месяца {month}")
                    
                    MONTHLY_ROOT_DIR = os.path.join(self.OUTPUT_RASTER_DIR, 'dynamic_monthly')   # где лежат папки месяцев 1, 2, ...
                    period_dir = os.path.join(MONTHLY_ROOT_DIR, str(month))
                    
                    
                    # а почему тут нет загрузки предикторов?
                    self.load_predictors('monthly', month)
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
                save_error(self.ERROR_FILENAME, e)
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
