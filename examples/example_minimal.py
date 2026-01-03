import os
import sys

# эти три строчки нужны для возможности подключить import sdm
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

import sdm


IN_ID = 1 # TODO: автоинкремент
#IN_CSV = 'data/crocodylus_niloticus.csv'
#IN_CSV = 'data/crocodylus_palaeindicus.csv'
IN_CSV = 'data/falco_peregrinus.csv'
IN_MODEL = 'XGBoost'
IN_MIN_LAT = 78.0
IN_MIN_LON = 50.0
IN_MAX_LAT = 90.0
IN_MAX_LON = 56.0
IN_RESOLUTION = '30s' 

# параметры для генерации фоновых точек
BG_MULT = 20
BG_PC = 50
BG_DISTANCE_MIN = 0 # указывается в шагах сетки, если 0 - пытается вычислить автоматически исходя из систематики
BG_DISTANCE_MAX = 0 # указывается в шагах сетки

PREDICTORS = 'all'

MODEL_FUTURE = 0 # прогнозируем будущее
JOBS = {}

sdm.run_sdm(IN_ID, IN_CSV, PREDICTORS, IN_MIN_LAT, IN_MIN_LON, IN_MAX_LAT, IN_MAX_LON,
        IN_RESOLUTION, MODEL_FUTURE, IN_MODEL, BG_MULT, BG_DISTANCE_MIN, BG_DISTANCE_MAX, BG_PC, JOBS)

