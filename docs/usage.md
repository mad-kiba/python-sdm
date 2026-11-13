# Использование модуля

Эта страница описывает публичный API библиотеки `sdm` — класс `PythonSDM` и его методы — как самостоятельный Python-интерфейс. Он не требует ни HTTP-сервера, ни фронтенда: всё, что нужно, — это словарь конфигурации и последовательный вызов методов, как в [examples/example_minimal.py](../examples/example_minimal.py).

## Минимальный пример

```python
from sdm import PythonSDM

config = {
    'IN_ID': 1,
    'IN_CSV': 'data/falco_peregrinus.csv',
    'IN_CSV_ADDITIONAL': '',
    'PREDICTORS': 'all',               # использовать все доступные предикторы
    'IN_MIN_LAT': 50.0,
    'IN_MIN_LON': 78.0,
    'IN_MAX_LAT': 56.0,
    'IN_MAX_LON': 90.0,
    'IN_RESOLUTION': '30s',
    'MODEL_FUTURE': 0,                 # 0 = не прогнозировать будущее
    'MODEL_SEASON': 0,                 # 0 = не делать помесячный прогноз
    'MODEL_PAST': 0,                   # 0 = не прогнозировать прошлое
    'IN_MODEL': 'XGBoost',
    'BG_MULT': 20,
    'BG_PC': 50,
    'BG_DISTANCE_MIN': 0,
    'BG_DISTANCE_MAX': 0,
    'DO_GISTO': 0,
    'M_FACTOR_DECAY_TYPE': 'sigmoid',
    'M_FACTOR_DECAY_RATE': 0.1,
    'M_FACTOR_HEIGHT_BARRIER': 500,
    'USE_BIAS_LAYER_FOR_BG': 1,
    'BIAS_SAMPLING_STRENGTH': 0.7,
    'M_FACTOR_CUR': -1,
    'M_FACTOR_2040': -1,
    'M_FACTOR_2070': -1,
    'M_FACTOR_2100': -1,
    'SCENARIOS': 'all',
    'JOBS': {},                        # обычный скрипт может передать пустой словарь
}

sdm_instance = PythonSDM(config)
sdm_instance.prepare_predictors()
sdm_instance.load_occurrences()
sdm_instance.load_predictors()
sdm_instance.prepare_data()
sdm_instance.deduplicate_data()
sdm_instance.generate_bg_pa()
sdm_instance.extract_features()
sdm_instance.draw_gistos()
sdm_instance.split_train_test()
sdm_instance.train_model()
sdm_instance.predict_current()
sdm_instance.calculate_metrics()
sdm_instance.draw_map_current()
sdm_instance.predict_future()
sdm_instance.predict_monthly()
sdm_instance.set_done()
```

## Шаги пайплайна

Методы `PythonSDM` вызываются строго в этом порядке (каждый опирается на результат предыдущего):

| № | Метод | Что делает |
|---|---|---|
| 0 | `PythonSDM(config)` | инициализация: разбор конфигурации, подготовка путей вывода, запись начального статуса в JOBS/JSON |
| 1 | `prepare_predictors()` | обрезка/выравнивание исходных растровых слоёв под заданный регион и разрешение |
| 2 | `load_occurrences()` | загрузка и фильтрация точек наблюдения вида (точность координат, год, дубликаты по систематике и т.п.) |
| 3 | `load_predictors()` | загрузка подготовленных растров-предикторов в память |
| 4 | `prepare_data()` | привязка точек присутствия к пикселям растра, извлечение значений предикторов в этих точках |
| 5 | `deduplicate_data()` | дедупликация наблюдений, попавших в один и тот же пиксель |
| 6 | `generate_bg_pa()` | генерация фоновых точек / точек псевдо-отсутствия с учётом доступной территории (M-фактор) и, опционально, поправки на смещение выборки |
| 7 | `extract_features()` | формирование итоговой матрицы признаков (присутствие + фон) для обучения |
| 8 | `draw_gistos()` | (опционально, см. `DO_GISTO`) построение гистограмм по каждому предиктору и расчёт статистик индекса Шёнера — см. [condition_profiles.md](condition_profiles.md) |
| 9 | `split_train_test()` | разбиение выборки на обучающую и тестовую |
| 10 | `train_model()` | обучение выбранной модели (`IN_MODEL`) |
| 11 | `predict_current()` | построение поверхности пригодности для текущего периода |
| 12 | `calculate_metrics()` | расчёт метрик качества (ROC AUC, TSS, каппа Коэна, continuous Boyce index) |
| 13 | `draw_map_current()` | отрисовка карты текущего периода (GeoTIFF + JPEG) |
| 14 | `predict_future()` | (опционально, см. `MODEL_FUTURE`) прогноз на будущие периоды/сценарии |
| 15 | `predict_past()` | (опционально, см. `MODEL_PAST`; не вызывается в минимальном примере) прогноз в прошлое |
| 16 | `predict_monthly()` | (опционально, см. `MODEL_SEASON`) помесячный прогноз |
| 17 | `set_done()` | финализация: статус модели помечается как завершённый |

Шаги 14, 15 и 16 выполняются как метод, но реально строят прогноз только если в конфигурации включены соответствующие флаги (`MODEL_FUTURE`/`MODEL_PAST`/`MODEL_SEASON`); иначе это no-op.

## Основные параметры конфигурации

| Параметр | Назначение |
|---|---|
| `IN_ID` | идентификатор запуска; определяет пути вывода (`output/.../{IN_ID}/...`) |
| `IN_CSV`, `IN_CSV_ADDITIONAL` | путь к CSV с наблюдениями (основной набор и, опционально, дополнительный) |
| `PREDICTORS` | `'all'` — использовать все доступные предикторы, либо список конкретных имён |
| `IN_MIN_LAT`, `IN_MIN_LON`, `IN_MAX_LAT`, `IN_MAX_LON` | границы региона моделирования |
| `IN_RESOLUTION` | пространственное разрешение: `'30s'`, `'1m'` или `'5m'` (угловые секунды/минуты) |
| `IN_MODEL` | алгоритм: `'RandomForest'`, `'XGBoost'`, `'GLM'` или `'MaxEnt'` — см. ниже |
| `MODEL_FUTURE`, `MODEL_PAST`, `MODEL_SEASON` | флаги `0`/`1`: строить ли прогноз на будущее / в прошлое / помесячно |
| `SCENARIOS` | `'all'` или список climate-сценариев (например `ssp126,ssp370,ssp585`) для прогноза будущего |
| `BG_MULT` | множитель количества фоновых точек относительно числа точек присутствия |
| `BG_PC` | доля точек, которые будут сгенерированы как настоящий фон (background), а не псевдо-отсутствие — остальное (`100 - BG_PC`) — псевдо-отсутствие |
| `BG_DISTANCE_MIN`, `BG_DISTANCE_MAX` | минимальная/максимальная дистанция (в шагах сетки) для генерации псевдо-отсутствия; `0` — подбирается автоматически исходя из систематики вида |
| `M_FACTOR_DECAY_TYPE` | функция затухания доступной территории от буфера наблюдений к нулю: `'buffer'`, `'linear'`, `'sigmoid'`, `'exponential'`, `'inverse_quadratic'`, `'gaussian'` |
| `M_FACTOR_DECAY_RATE` | скорость затухания для выбранной функции |
| `M_FACTOR_HEIGHT_BARRIER` | порог перепада высот (м), работающий как барьер при расчёте доступной территории через cost-distance (Fast Marching) |
| `M_FACTOR_CUR`, `M_FACTOR_2040`, `M_FACTOR_2070`, `M_FACTOR_2100` | радиус буфера доступной территории (км) для текущего периода и для будущих срезов; `-1` — не ограничивать явным радиусом |
| `USE_BIAS_LAYER_FOR_BG` | `0`/`1`: использовать ли поправку на смещение выборки наблюдений (target-group background) при генерации фона |
| `BIAS_SAMPLING_STRENGTH` | сила этой поправки (0 — чисто случайный фон, 1 — полностью пропорционально плотности наблюдений) |
| `DO_GISTO` | `0`/`1`: строить ли гистограммы и статистики по предикторам (шаг 8) |
| `JOBS` | словарь для отслеживания статуса запуска (`{'status': ..., 'file': ..., 'error': ...}`); для одиночного скрипта достаточно передать `{}` — это внутренний механизм отчёта о статусе, не обязательный компонент самого модуля |

## Выбор модели (`IN_MODEL`)

- **`'RandomForest'`**, **`'XGBoost'`** — ансамбли деревьев (scikit-learn / xgboost). Единственные модели, поддерживающие прогноз на будущее и в прошлое (см. [faq.md](faq.md) — почему).
- **`'GLM'`** — логистическая регрессия с линейными и квадратичными признаками; сила L2-регуляризации подбирается кросс-валидацией по ROC AUC. Квадратичный член нужен, чтобы описать унимодальный отклик вида на предиктор (ter Braak & Looman, 1986), а не только монотонный.
- **`'MaxEnt'`** — адаптировано из модельного ядра библиотеки [elapid](https://github.com/earth-chris/elapid) (linear/hinge/threshold/product-признаки, логистическая регрессия с L1-регуляризацией, калибровка выхода raw/logistic/cloglog); геопространственная обвязка elapid (растры/векторы) не используется — её роль в пайплайне PythonSDM выполняет сам модуль.

`GLM` и `MaxEnt` обучаются и оцениваются только для текущего периода.

## Результаты

Основные выходные файлы (пути строятся на основе `IN_ID`):

- `output/suitability/{IN_ID}/suitability_{IN_ID}.tif` и `.jpg` — карта пригодности территории (GeoTIFF и превью JPEG) для текущего периода;
- `output/texts/{IN_ID}/{IN_ID}.json` — основной JSON с результатами: параметры модели (`model_params`), поправка на смещение выборки (`bias_correction`), метрики качества, статистики по предикторам (`histogram_stats`, если включён `DO_GISTO`), пути ко всем файлам результата;
- `output/aucs/{IN_ID}/roc-auc.jpg` — график ROC-кривой;
- `output/gistos/{IN_ID}/` — гистограммы по предикторам (если включён `DO_GISTO`);
- `output/predictions/{IN_ID}/`, `output/past/{IN_ID}/`, `output/seasons/{IN_ID}/` — прогнозы на будущее, в прошлое и по месяцам соответственно (если включены).

Подробный формат поля `histogram_stats` описан в [condition_profiles.md](condition_profiles.md). Формат полей `model_params`/`bias_correction` в том виде, в каком их использует конкретный HTTP-сервер/фронтенд автора, описан в [frontend_model_params_bias.md](frontend_model_params_bias.md) — это документация одного из вариантов использования JSON-результата, а не требование самого модуля.
