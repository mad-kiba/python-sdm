import os
import sys

# эти три строчки нужны для возможности подключить import sdm
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

import sdm 


IN_ID = 1 # добавить сюда автоинкремент
IN_CSV = 'data/726.csv'
IN_MODEL = 'XGBoost'
IN_MIN_LAT = 78.0
IN_MIN_LON = 50.0
IN_MAX_LAT = 90.0
IN_MAX_LON = 56.0
IN_RESOLUTION = '30s' # request.form.get('resolution', type=float)

# параметры для генерации фоновых точек
BG_MULT = 20
BG_PC = 50
BG_DISTANCE_MIN = 20
BG_DISTANCE_MAX = 50

#PREDICTORS = request.form.get('predictors')
PREDICTORS = 'all'
OUTPUT_SUITABILITY_JPG = "output/suitability_"+str(IN_ID)+".jpg"
OUTPUT_SUITABILITY_TIF = "output/suitability_"+str(IN_ID)+".tif"
OUTPUT_HISTOGRAMS_DIR = "output/gistos"
OUTPUT_HISTOGRAMS_ZIP = os.path.join(OUTPUT_HISTOGRAMS_DIR, str(IN_ID), "histos.zip")

OUTPUT_PREDICTIONS_DIR = "output/predictions"
OUTPUT_FUTURE_DIR = os.path.join(OUTPUT_PREDICTIONS_DIR, str(IN_ID))
OUTPUT_FUTURE_ZIP = os.path.join(OUTPUT_FUTURE_DIR, "futures.zip")

MODEL_FUTURE = 0 # прогнозируем будущее

sdm.run_sdm(IN_ID, IN_CSV, PREDICTORS, IN_MIN_LAT, IN_MIN_LON, IN_MAX_LAT, IN_MAX_LON,
        IN_RESOLUTION, MODEL_FUTURE, IN_MODEL, BG_MULT, BG_DISTANCE_MIN, BG_DISTANCE_MAX, BG_PC)

