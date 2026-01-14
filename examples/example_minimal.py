import os
import sys

# эти три строчки нужны для возможности подключить import sdm
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from sdm import PythonSDM


# устанавливаем параметры для запуска модели
config = {
        'IN_ID': 1,
        'IN_CSV': 'data/falco_peregrinus.csv',
        #'IN_CSV': 'data/null.csv',
        'PREDICTORS': 'all', # используем все доступные предикторы
        'IN_MIN_LAT': 78.0,
        'IN_MIN_LON': 50.0,
        'IN_MAX_LAT': 90.0,
        'IN_MAX_LON': 56.0,
        'IN_RESOLUTION': '30s',
        'MODEL_FUTURE': 0, # 0 = не прогнозируем будущее
        'IN_MODEL': 'XGBoost',
        #'IN_MODEL': 'MaxEnt',
        'BG_MULT': 20,
        'BG_PC': 50,
        'BG_DISTANCE_MIN': 0, # указывается в шагах сетки, 0 = пытается вычислить автоматически исходя из систематики
        'BG_DISTANCE_MAX': 0, # указывается в шагах сетки
        #'DO_GISTO': 1, # нужно ли рисовать гистограммы
        'DO_GISTO': 0, 
        'JOBS': {}
}


sdm_instance = PythonSDM(config)        # 0) инициализация модели
sdm_instance.prepare_predictors()       # 1) предварительная подготовка предикторов
sdm_instance.load_occurences()          # 2) загрузка наблюдений
sdm_instance.load_predictors()          # 3) загрузка предикторов
sdm_instance.prepare_data()             # 4) привязка присутствий к пикселям растра и обработка
sdm_instance.deduplicate_data()         # 5) дедупликация данных
sdm_instance.generate_bg_pa()           # 6) генерация фотоновых точек и псевдоотсутствия
sdm_instance.extract_features()         # 7) извлечение признаков
sdm_instance.draw_gistos()              # 8) постройка гистограмм
sdm_instance.split_train_test()         # 9)
sdm_instance.train_model()              #10)
sdm_instance.predict_current()          #11)
sdm_instance.draw_map_current()         #12)
sdm_instance.predict_future()           #13)


