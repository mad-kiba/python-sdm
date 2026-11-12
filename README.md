# Python-SDM
Species Distribution Modelling (SDM) library for Python.

Библиотека для моделирования распространения видов (SDM) для Python.
Доступна онлайн по адресу: https://wingeds.world/sdm

Суть SDM - поиск закономерностей между местами наблюдения вида и условиями окружающей среды в этих местах.
Примеры источников наблюдательных данных: GBIF.org, iNaturalist.org, eBird.org, wingeds.world, plantarium.su и др.
Условия среды: WorldClim (температура, осадки), SRTM (рельеф, высоты), OSM (речные и дорожные сети) и др.

Подробнее про алгоритм SDM: https://habr.com/ru/articles/983756/

Для запуска SDM обычно нужно провести большую работу по сбору и подготовке этих данных.
Библиотека Python-SDM развёрнута онлайн вместе с большим набором предикторов, приведённых к одному пространственному разрешению. 
Модель можно запустить здесь и сейчас: птицы, млекопитающие, рептилии, насекомые, растения - пробуйте всё. Адрес: https://wingeds.world/sdm

# Установка и запуск локальной версии
**Для запуска Python-SDM локально нужно:**
1. Скачать библиотеку в отдельную папку, например, `python_sdm`.
   
2. Установить Python:
```
yum install python3 python3-pip libxcb mesa-libGL # CentOS, RHEL
# или
apt install python3 python3-pip libxcb1 libgl1 # Ubuntu, Debian
```
3. Установить дополнительные библиотеки:
```
pip install pandas numpy scipy matplotlib scikit-learn xgboost opencv-python rasterio contextily pyproj Pillow affine joblib
```

**Технически библиотека готова к запуску. 
Дальше собираем исходные данные: слои-предикторы и наблюдательные данные:**

4. Создать папку `python_sdm\input_predictors` для предикторов. В ней может быть до пяти подпапок:
- `python_sdm\input_predictors\static` - предикторы в этой папке считаются статическими, неизменными по времени. Соответственно, модель может их использовать и для текущего времени, и для будущего (например, высота над уровнем моря - WorldClim height);
- `python_sdm\input_predictors\dynamic_current` - предикторы в этой папке считаются динамическими (например, WorldClim BIOxx). Если вы моделируете будущее, файлы с аналогичным названием должны лежать в следующей папке:
- `python_sdm\input_predictors\dynamic_predictable` - в этой папке лежат предикторы будущего (например, WorldClim BIOxx SSP370). Структура: подпапка на каждый период (например, `2041-2060`), а внутри неё - подпапка на каждый сценарий (например, `ssp370`), то есть `dynamic_predictable\2041-2060\ssp370\`;
- `python_sdm\input_predictors\dynamic_past` - в этой папке лежат предикторы прошлого (например, CHELSA TraCE21k). Структура проще: подпапка на каждый период без сценариев (например, `dynamic_past\02-180\`), и отдельная обязательная подпапка `dynamic_past\learning\` с предикторами "настоящего" в том же формате, что и остальные периоды прошлого - на них обучается отдельная модель для прогноза прошлого;
- `python_sdm\input_predictors\dynamic_monthly` - если включён помесячный прогноз (`MODEL_SEASON: 1`), сюда кладутся предикторы по месяцам: подпапка на каждый месяц (`1` … `12`).
Предикторы должны быть в формате GeoTIFF. Если в GeoTIFF несколько слоёв, используется только первый из них.
Для минимального запуска можно скачать `WorldClim elevation` (https://www.worldclim.org/data/worldclim21.html - масштаб `30 seconds`) и распаковать в папку `python_sdm\input_predictors\static`, а `Bioclimatic variables` скачать и распаковать в папку `python_sdm\input_predictors\dynamic_current`

5. Скопировать содержание папки `python_sdm\examples` (папку `data` и файл `example_minimal.py`) в корень текущей папки `python_sdm` (там же, где лежат папки `sdm и input_predictors`).

6. В этой папке выполнить команду:
```
python example_minimal.py
```
В результате будет запущена SDM для сапсана на территории Алтая. 

7. В папку `python_sdm\data` можно загрузить наблюдательные данные по любому интересующему виду. Наблюдательные данные можно взять на GBIF.org. Несколько минимальных примеров уже лежит в папке `examples\data`
