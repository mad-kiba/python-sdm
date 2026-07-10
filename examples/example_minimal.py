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
        #'IN_CSV': 'data/Ommatotriton_Ommatotriton_vittatus.csv',
        'IN_CSV': 'data/falco_peregrinus.csv',
        'IN_CSV_ADDITIONAL': '',
        'PREDICTORS': 'all', # используем все доступные предикторы
        'IN_MIN_LAT': 42.0,
        'IN_MIN_LON': 35.0,
        'IN_MAX_LAT': 93.0,
        'IN_MAX_LON': 59.0,
        'IN_RESOLUTION': '5m',
        'MODEL_FUTURE': 0, # 0 = не прогнозируем будущее
        'MODEL_SEASON': 0, # 0 = не делаем помесячный прогноз
        'MODEL_PAST': 0, # 0 = не прогнозируем прошлое
        'IN_MODEL': 'XGBoost',
        #'IN_MODEL': 'MaxEnt',
        'BG_MULT': 20,
        'BG_PC': 50,
        'BG_DISTANCE_MIN': 0, # указывается в шагах сетки, 0 = пытается вычислить автоматически исходя из систематики
        'BG_DISTANCE_MAX': 0, # указывается в шагах сетки
        #'DO_GISTO': 1, # нужно ли рисовать гистограммы
        'DO_GISTO': 0,
        'M_FACTOR_DECAY_TYPE': 'sigmoid',
        'M_FACTOR_DECAY_RATE': 0.1,
        'M_FACTOR_HEIGHT_BARRIER': 500,
        'SCENARIOS': 'all',
        'JOBS': {}
}


sdm_instance = PythonSDM(config)        # 0) инициализация модели
sdm_instance.prepare_predictors()       # 1) предварительная подготовка предикторов
sdm_instance.load_occurrences()         # 2) загрузка наблюдений
sdm_instance.load_predictors()          # 3) загрузка предикторов
sdm_instance.prepare_data()             # 4) привязка присутствий к пикселям растра и обработка
sdm_instance.deduplicate_data()         # 5) дедупликация данных
sdm_instance.generate_bg_pa()           # 6) генерация фотоновых точек и псевдоотсутствия
sdm_instance.extract_features()         # 7) извлечение признаков
sdm_instance.draw_gistos()              # 8) постройка гистограмм
sdm_instance.split_train_test()         # 9) разделение выборки на учебную и тестовую
sdm_instance.train_model()              #10) обучение модели
sdm_instance.predict_current()          #11) предсказание на текущем временном периоде
sdm_instance.calculate_metrics()        #12) вычисление метрик
sdm_instance.draw_map_current()         #13) рисование карты
sdm_instance.predict_future()           #14) предсказание будущего
#sdm_instance.predict_past()            #15) предсказание прошлого
sdm_instance.predict_monthly()          #16) помесячные предсказания
sdm_instance.set_done()                 #17) завершение модели

print('Моделирование завершено')


