# sdm/utils/i18n.py
# Словари переводов подписей, выводимых на изображения (карты, графики, гистограммы).
#
# Чтобы добавить новый язык:
#   1. добавить его код в SUPPORTED_LANGUAGES (и, если нужно, алиасы в LANG_ALIASES);
#   2. добавить его перевод для каждого ключа в TRANSLATIONS;
#   3. добавить его в MONTH_NAMES.
# Если перевод для конкретного ключа на новом языке не добавлен, используется DEFAULT_LANG.

DEFAULT_LANG = 'ru'
SUPPORTED_LANGUAGES = ('ru', 'en')

# Альтернативные написания языка во входных параметрах (регистронезависимо).
LANG_ALIASES = {
    'ru': 'ru',
    'rus': 'ru',
    'russian': 'ru',
    'en': 'en',
    'eng': 'en',
    'english': 'en',
}


def normalize_lang(lang) -> str:
    """
    Приводит значение параметра языка подписей к одному из SUPPORTED_LANGUAGES.
    Неизвестное или пустое значение приводится к DEFAULT_LANG.
    """
    if not lang:
        return DEFAULT_LANG
    key = str(lang).strip().lower()
    return LANG_ALIASES.get(key, DEFAULT_LANG)


MONTH_NAMES = {
    'ru': {
        1: 'январь', 2: 'февраль', 3: 'март', 4: 'апрель', 5: 'май', 6: 'июнь',
        7: 'июль', 8: 'август', 9: 'сентябрь', 10: 'октябрь', 11: 'ноябрь', 12: 'декабрь',
    },
    'en': {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December',
    },
}


def month_name(month: int, lang: str = DEFAULT_LANG) -> str:
    lang = normalize_lang(lang)
    names = MONTH_NAMES.get(lang, MONTH_NAMES[DEFAULT_LANG])
    return names.get(month, str(month))


TRANSLATIONS = {
    # --- Карта пригодности условий (draw_map) ---
    'map_title_with_species': {
        'ru': 'Карта пригодности условий для вида {species} ({id})',
        'en': 'Habitat suitability map for {species} ({id})',
    },
    'map_title_default': {
        'ru': 'Карта пригодности условий для вида',
        'en': 'Habitat suitability map',
    },
    'map_colorbar_suitability': {
        'ru': 'Пригодность условий (относительная)',
        'en': 'Habitat suitability (relative)',
    },
    'map_model_line': {
        'ru': '\nМодель: {model}, шаг: {resolution}, уник. точек: {n_presence}',
        'en': '\nModel: {model}, step: {resolution}, unique points: {n_presence}',
    },
    'map_month_suffix': {
        'ru': ', месяц: {month}',
        'en': ', month: {month}',
    },
    'map_insufficient_obs': {
        'ru': '\nНедостаточно наблюдений для обучения модели в этом периоде',
        'en': '\nNot enough observations to train a model for this period',
    },
    'map_auc_line': {
        'ru': ', ROC-AUC: {auc:.3f}',
        'en': ', ROC AUC: {auc:.3f}',
    },
    'map_area_optimal_with_threshold': {
        'ru': '\nSопт = {area} км² (пригодность>{thr:.3f}), ',
        'en': '\nSopt = {area} km² (suitability>{thr:.3f}), ',
    },
    'map_area_soft_with_threshold': {
        'ru': 'Sмягк = {area} км² (пригодность>{thr:.3f})',
        'en': 'Ssoft = {area} km² (suitability>{thr:.3f})',
    },
    'map_area_optimal': {
        'ru': '\nSопт = {area} км², ',
        'en': '\nSopt = {area} km², ',
    },
    'map_area_soft': {
        'ru': 'Sмягк = {area} км²',
        'en': 'Ssoft = {area} km²',
    },
    'map_future_baseline_period': {
        'ru': '\nТекущий период (базовые климатические переменные)',
        'en': '\nCurrent period (baseline climate variables)',
    },
    'map_period_scenario': {
        'ru': '\nПериод: {period} (сценарий {scenario})',
        'en': '\nPeriod: {period} (scenario {scenario})',
    },
    'map_period': {
        'ru': '\nПериод: {period_label}',
        'en': '\nPeriod: {period_label}',
    },
    'map_footer_biodiv': {
        'ru': 'https://wingeds.world/biodiv - карта биоразнообразия Центральной Азии',
        'en': 'https://wingeds.world/biodiv - Central Asia biodiversity map',
    },
    'map_footer_sdm': {
        'ru': 'https://wingeds.world/sdm/{id} - модель распространения вида (SDM)',
        'en': 'https://wingeds.world/sdm/{id} - species distribution model (SDM)',
    },
    'period_present': {
        'ru': 'Наст. вр.',
        'en': 'Present',
    },
    'period_past_baseline': {
        'ru': 'Современность (базовый период)',
        'en': 'Present (baseline period)',
    },

    # --- Карта M-фактора (draw_m_factor_map) ---
    'mfactor_title_default': {
        'ru': 'Зоны доступности вида (M-фактор)',
        'en': 'Species accessible area (M-factor)',
    },
    'mfactor_title_with_id': {
        'ru': 'Зоны доступности вида (M-фактор) ({id})',
        'en': 'Species accessible area (M-factor) ({id})',
    },
    'mfactor_colorbar_label': {
        'ru': 'Прогноз расширения доступной зоны',
        'en': 'Projected expansion of the accessible area',
    },

    # --- ROC-кривая (plot_roc_auc_curve) ---
    'roc_legend_curve': {
        'ru': 'ROC-кривая (AUC = {auc:.3f})',
        'en': 'ROC curve (AUC = {auc:.3f})',
    },
    'roc_legend_random': {
        'ru': 'Случайное угадывание',
        'en': 'Random guess',
    },
    'roc_xlabel': {
        'ru': 'False Positive Rate (Ложноположительные)',
        'en': 'False Positive Rate',
    },
    'roc_ylabel': {
        'ru': 'True Positive Rate (Истинно положительные)',
        'en': 'True Positive Rate',
    },
    'roc_title': {
        'ru': 'Качество модели: ROC-кривая (id={id})',
        'en': 'Model performance: ROC curve (id={id})',
    },

    # --- Гистограммы по предикторам (create_beautiful_histogram) ---
    'hist_no_data': {
        'ru': 'Нет данных для отображения',
        'en': 'No data to display',
    },
    'hist_legend_background': {
        'ru': 'Фон (весь слой)',
        'en': 'Background (whole layer)',
    },
    'hist_legend_presence': {
        'ru': 'Присутствия',
        'en': 'Presence',
    },
    'hist_legend_mean': {
        'ru': 'Среднее ({value})',
        'en': 'Mean ({value})',
    },
    'hist_legend_median': {
        'ru': 'Медиана ({value})',
        'en': 'Median ({value})',
    },
    'hist_legend_percentiles': {
        'ru': '5%/95%',
        'en': '5%/95%',
    },
    'hist_title_prefix': {
        'ru': 'Гистограмма отклика для предиктора:\n',
        'en': 'Response histogram for predictor:\n',
    },
    'hist_species_line': {
        'ru': 'Вид: {species}',
        'en': 'Species: {species}',
    },
    'hist_ylabel': {
        'ru': 'Нормализованная Частота',
        'en': 'Normalized Frequency',
    },

    # --- Диагностические JPEG отдельных слоёв-предикторов (plot_geotiff_with_osm) ---
    'axis_longitude': {
        'ru': 'Долгота (°)',
        'en': 'Longitude (°)',
    },
    'axis_latitude': {
        'ru': 'Широта (°)',
        'en': 'Latitude (°)',
    },

    # --- Блок статистик на гистограмме (add_stats_to_plot) ---
    'stats_na': {
        'ru': 'н/д',
        'en': 'n/a',
    },
    'stats_text_categorical': {
        'ru': (
            '  Доля наблюдений: {share}\n'
            '  Доля региона: {share_region}\n'
            '  D Шёнера: {d}'
        ),
        'en': (
            '  Observed share: {share}\n'
            '  Region share: {share_region}\n'
            "  Schoener's D: {d}"
        ),
    },
    'stats_text_continuous': {
        'ru': (
            '  Наблюдения:\n'
            '  Мин: {obs_min}\n'
            '  P5:  {obs_p5}\n'
            '  Медиана: {obs_median}\n'
            '  Среднее: {obs_mean}\n'
            '  P95: {obs_p95}\n'
            '  Макс: {obs_max}\n'
            '  Ст. откл.: {obs_std}\n'
            '  Асимметрия: {obs_skew}\n'
            '  Эксцесс: {obs_kurt}\n\n'
            '  Весь слой:\n'
            '  Мин: {region_min}\n'
            '  P5:  {region_p5}\n'
            '  Медиана: {region_median}\n'
            '  Среднее: {region_mean}\n'
            '  P95: {region_p95}\n'
            '  Макс: {region_max}\n'
            '  Ст. откл.: {region_std}\n'
            '  Асимметрия: {region_skew}\n'
            '  Эксцесс: {region_kurt}\n\n'
            '  Сравнение:\n'
            '  Отн. ширина (P10-P90): {rel_width}\n'
            '  D Шёнера: {d}'
        ),
        'en': (
            '  Observations:\n'
            '  Min: {obs_min}\n'
            '  P5:  {obs_p5}\n'
            '  Median: {obs_median}\n'
            '  Mean: {obs_mean}\n'
            '  P95: {obs_p95}\n'
            '  Max: {obs_max}\n'
            '  Std. dev.: {obs_std}\n'
            '  Skewness: {obs_skew}\n'
            '  Kurtosis: {obs_kurt}\n\n'
            '  Whole layer:\n'
            '  Min: {region_min}\n'
            '  P5:  {region_p5}\n'
            '  Median: {region_median}\n'
            '  Mean: {region_mean}\n'
            '  P95: {region_p95}\n'
            '  Max: {region_max}\n'
            '  Std. dev.: {region_std}\n'
            '  Skewness: {region_skew}\n'
            '  Kurtosis: {region_kurt}\n\n'
            '  Comparison:\n'
            '  Rel. width (P10-P90): {rel_width}\n'
            "  Schoener's D: {d}"
        ),
    },

    # --- Консольный/лог-вывод (print) - sdm/sdm.py ---
    'log_input_params_header': {'ru': '-- Входящие параметры:', 'en': '-- Input parameters:'},
    'log_old_query_error_prefix': {'ru': 'Ошибка:', 'en': 'Error:'},
    'log_region_header': {'ru': '-- Регион для моделирования ({id}): ', 'en': '-- Modeling region ({id}): '},
    'log_step1_header': {'ru': '\n-- 1. Подготовка предикторов ({id})', 'en': '\n-- 1. Preparing predictors ({id})'},
    'log_step2_header': {'ru': '\n-- 2. Загрузка наблюдений ({id})', 'en': '\n-- 2. Loading occurrences ({id})'},
    'log_step3_header': {'ru': '\n-- 3. Загрузка предикторов ({id})', 'en': '\n-- 3. Loading predictors ({id})'},
    'log_scales_file_not_found': {'ru': 'Файл масштабов не найден', 'en': 'Scales file not found'},
    'log_predictors_loaded': {
        'ru': '\n-- Загружено предикторов: {bands} | Размер: {h} x {w} | CRS: {crs}',
        'en': '\n-- Predictors loaded: {bands} | Size: {h} x {w} | CRS: {crs}',
    },
    'log_layers_label': {'ru': 'Слои:', 'en': 'Layers:'},
    'log_step4_header': {
        'ru': '\n-- 4. Привязка присутствий к пикселям растра и фильтрация по маске валидности ({id})',
        'en': '\n-- 4. Binding presences to raster pixels and filtering by validity mask ({id})',
    },
    'log_presence_in_valid_area': {'ru': 'Присутствий внутри валидной области: {n}', 'en': 'Presences inside the valid area: {n}'},
    'log_step4_1_header': {
        'ru': '-- 4.1. Создаём полные растры для всего спектра слоёв-предикторов ({id})',
        'en': '-- 4.1. Building full rasters across the whole predictor stack ({id})',
    },
    'log_step5_header': {
        'ru': '\n-- 5. Дедупликация по пикселю — оставляем по одному наблюдению на клетку ({id})',
        'en': '\n-- 5. Per-pixel deduplication — keeping one observation per cell ({id})',
    },
    'log_warn_few_unique_presence': {
        'ru': 'Внимание: очень мало уникальных присутствий в пределах растра.',
        'en': 'Warning: very few unique presences within the raster.',
    },
    'log_unique_presence_count': {'ru': 'Уникальных присутствий по точкам растра: {n}', 'en': 'Unique presences by raster cell: {n}'},
    'log_step6_header': {
        'ru': '\n-- 6. Генерация фоновых точек и точек псевдоотсутствия ({id})',
        'en': '\n-- 6. Generating background and pseudo-absence points ({id})',
    },
    'log_auto_buffer_params': {
        'ru': 'Нужно генерировать точки псевдоприсутствия, и параметры огибающих заданы на авто. Определяем их.',
        'en': 'Pseudo-absence points are needed and the buffer parameters are set to auto. Determining them.',
    },
    'log_m_factor_values': {
        'ru': 'Текущие факторы мобильности: {cur}, {y2040}, {y2070}, {y2100}',
        'en': 'Current mobility factors: {cur}, {y2040}, {y2070}, {y2100}',
    },
    'log_bias_mode_enabled': {
        'ru': 'Активирован режим учета предвзятости наблюдений (human bias).',
        'en': 'Observation bias correction (human bias) mode is enabled.',
    },
    'log_bias_layer_loading': {'ru': 'Загрузка слоя предвзятости: {path}', 'en': 'Loading bias layer: {path}'},
    'log_bias_scale_params_found': {
        'ru': "Найдены параметры масштабирования для слоя '{name}'.",
        'en': "Scaling parameters found for layer '{name}'.",
    },
    'log_bias_layer_load_error': {'ru': 'Ошибка загрузки слоя предвзятости:', 'en': 'Error loading bias layer:'},
    'log_bg_generation_header': {
        'ru': '\n-- Генерация фоновых точек и точек псевдоотсутствия ({id})',
        'en': '\n-- Generating background and pseudo-absence points ({id})',
    },
    'log_bg_computed_params': {
        'ru': 'Вычисленные параметры точек: BG_PC={bg_pc},BG_DISTANCE_MIN={bg_min}, BG_DISTANCE_MAX={bg_max}',
        'en': 'Computed point parameters: BG_PC={bg_pc},BG_DISTANCE_MIN={bg_min}, BG_DISTANCE_MAX={bg_max}',
    },
    'log_bg_sampled_count': {
        'ru': 'Сэмплировано фоновых точек + точек псевдоотсутствия: {n}',
        'en': 'Background + pseudo-absence points sampled: {n}',
    },
    'log_warn_nan_dropped': {
        'ru': 'ВНИМАНИЕ: Найдено и удалено {n} точек с NaN-значениями в отдельных предикторах!',
        'en': 'WARNING: Found and removed {n} points with NaN values in individual predictors!',
    },
    'log_feature_matrix': {'ru': 'Матрица признаков: {shape}, классы: {classes}', 'en': 'Feature matrix: {shape}, classes: {classes}'},
    'log_step8_header': {'ru': '\n-- 8. Постройка гистограмм ({id})', 'en': '\n-- 8. Building histograms ({id})'},
    'log_no_predictors_for_histo': {
        'ru': 'Нет предикторов для построения гистограмм.',
        'en': 'No predictors available to build histograms.',
    },
    'log_histo_build_error': {'ru': 'Ошибка построения гистограммы:', 'en': 'Error building histogram:'},
    'log_scale_params_not_found': {
        'ru': "Предупреждение: Параметры масштабирования не найдены для '{band}'. Отображаются масштабированные значения.",
        'en': "Warning: scaling parameters not found for '{band}'. Showing scaled values.",
    },
    'log_histo_saved_progress': {'ru': 'Сохранена гистограмма: {i}/{total}', 'en': 'Histogram saved: {i}/{total}'},
    'log_histos_all_saved': {'ru': "Все гистограммы сохранены в папку: '{dir}'", 'en': "All histograms saved to folder: '{dir}'"},
    'log_no_files_to_zip': {'ru': 'В папке {dir} нет файлов для упаковки.', 'en': 'No files to archive in folder {dir}.'},
    'log_histos_zipped': {
        'ru': "Все файлы из '{dir}' успешно упакованы в '{archive}'.",
        'en': "All files from '{dir}' were successfully archived into '{archive}'.",
    },
    'log_step9_header': {'ru': '\n-- 9. Разделение на train/test ({id})', 'en': '\n-- 9. Splitting into train/test ({id})'},
    'log_split_done': {
        'ru': 'Успешно применено случайное стратифицированное разбиение (80/20).',
        'en': 'Random stratified split (80/20) applied successfully.',
    },
    'log_step10_header': {
        'ru': '\n-- 10. Обучение валидационной модели (на {n} точках) ({id})',
        'en': '\n-- 10. Training the validation model (on {n} points) ({id})',
    },
    'log_eval_model_done': {'ru': 'Обучение валидационной модели завершено', 'en': 'Validation model training complete'},
    'log_step10a_header': {
        'ru': '\n-- 10А. Обучение финальной боевой модели на 100% данных ({id})',
        'en': '\n-- 10a. Training the final production model on 100% of the data ({id})',
    },
    'log_step11_header': {
        'ru': '\n-- 11. Прогноз на всю область и сохранение карты пригодности ({id})',
        'en': '\n-- 11. Predicting across the whole area and saving the suitability map ({id})',
    },
    'log_applying_m_factor': {
        'ru': 'Применяем фактор мобильности (M-фактор): {km} км...',
        'en': 'Applying the mobility factor (M-factor): {km} km...',
    },
    'log_slope_layer_found': {
        'ru': "  Слой 'slope_deg' найден! Алгоритм будет учитывать рельеф при расчете доступности.",
        'en': "  Layer 'slope_deg' found! The algorithm will account for terrain when computing accessibility.",
    },
    'log_slope_layer_missing': {
        'ru': "  Слой 'slope_deg' не найден. Будет использовано стандартное евклидово расстояние.",
        'en': "  Layer 'slope_deg' not found. Standard Euclidean distance will be used.",
    },
    'log_elev_layer_found': {
        'ru': '  Слой высоты найден! Горы будут физическим барьером.',
        'en': '  Elevation layer found! Mountains will act as a physical barrier.',
    },
    'log_water_layer_found': {
        'ru': '  Слой открытой воды найден! Моря станут преградой.',
        'en': '  Open water layer found! Seas will act as a barrier.',
    },
    'log_m_factor_error': {'ru': 'Ошибка применения фактора мобильности: {e}', 'en': 'Error applying the mobility factor: {e}'},
    'log_m_factor_disabled': {
        'ru': 'Фактор мобильности (M-фактор) отключен пользователем (-1).',
        'en': 'The mobility factor (M-factor) is disabled by the user (-1).',
    },
    'log_suitability_map_saved': {'ru': 'Карта пригодности сохранена: {path}', 'en': 'Suitability map saved: {path}'},
    'log_m_factor_map_saved': {'ru': 'Карта расширения M-фактора сохранена: {path}', 'en': 'M-factor expansion map saved: {path}'},
    'log_m_factor_map_save_error': {'ru': 'Ошибка сохранения карты M-фактора: {e}', 'en': 'Error saving the M-factor map: {e}'},
    'log_presences_saved': {'ru': 'Сохранены использованные присутствия: {filename}', 'en': 'Used presences saved: {filename}'},
    'log_randoms_saved': {
        'ru': 'Сохранены фоновые точки (случайные): {filename}',
        'en': 'Background points saved (random): {filename}',
    },
    'log_buffer_saved': {
        'ru': 'Сохранены фоновые точки (буферные): {filename}',
        'en': 'Background points saved (buffer): {filename}',
    },
    'log_step12_header': {
        'ru': '\n-- 12. Вычисление пространственных метрик и метрик качества ({id})',
        'en': '\n-- 12. Computing spatial and quality metrics ({id})',
    },
    'log_cbi': {'ru': 'Continuous Boyce Index (CBI): {value:.3f}', 'en': 'Continuous Boyce Index (CBI): {value:.3f}'},
    'log_roc_auc_holdout': {'ru': 'ROC AUC (holdout): {value:.3f}', 'en': 'ROC AUC (holdout): {value:.3f}'},
    'log_tss': {'ru': 'TSS: {value:.4f}', 'en': 'TSS: {value:.4f}'},
    'log_cohens_kappa': {'ru': "Cohen's Kappa: {value:.4f}", 'en': "Cohen's Kappa: {value:.4f}"},
    'log_quality_eval_error': {'ru': 'Ошибка оценки качества модели:', 'en': 'Error evaluating model quality:'},
    'log_pca_niche_breadth': {
        'ru': 'Многомерная экологическая пластичность (PCA Niche Breadth, top-5): {value:.4f}',
        'en': 'Multivariate niche breadth (PCA Niche Breadth, top-5): {value:.4f}',
    },
    'log_feature_importance_header': {'ru': 'Важность предикторов:', 'en': 'Predictor importance:'},
    'log_step13_header': {'ru': '\n-- 13. Рисуем карту ({id})', 'en': '\n-- 13. Drawing the map ({id})'},
    'log_map_draw_error': {'ru': 'Ошибка рисования карты', 'en': 'Error drawing the map'},
    'log_map_jpeg_saved': {'ru': 'Карта присутствия сохранена в формате JPEG: {path}', 'en': 'Presence map saved as JPEG: {path}'},
    'log_step14_header': {'ru': '\n-- 14. Приступаю к прогнозу будущего ({id})', 'en': '\n-- 14. Starting the future forecast ({id})'},
    'log_future_forecast_for': {'ru': '\nПрогноз: {period} / {scenario}', 'en': '\nForecast: {period} / {scenario}'},
    'log_future_predict_error': {'ru': 'Ошибка прогноза будущего', 'en': 'Error forecasting the future'},
    'log_m_factor_future_error': {
        'ru': 'Ошибка применения фактора мобильности для будущего ({period}): {e}',
        'en': 'Error applying the mobility factor for the future ({period}): {e}',
    },
    'log_m_factor_future_disabled': {
        'ru': 'Фактор мобильности для будущего ({period}) отключен (-1).',
        'en': 'The future mobility factor ({period}) is disabled (-1).',
    },
    'log_saved_path': {'ru': 'Сохранено: {path}', 'en': 'Saved: {path}'},
    'log_animation_error': {'ru': 'Ошибка создания анимации: {e}', 'en': 'Error creating animation: {e}'},
    'log_creating_animation_future': {'ru': '\nСоздаём анимацию:', 'en': '\nCreating animation:'},
    'log_all_future_saved': {
        'ru': "\nВсе прогнозы сохранены в папку: '{dir}'",
        'en': "\nAll forecasts saved to folder: '{dir}'",
    },
    'log_step15_past_header': {'ru': '\n-- 15. Приступаю к прогнозу прошлого ({id})', 'en': '\n-- 15. Starting the past forecast ({id})'},
    'log_past_folder_missing': {'ru': 'Папка {dir} не найдена. Пропуск.', 'en': 'Folder {dir} not found. Skipping.'},
    'log_past_base_model_training': {
        'ru': '\nОбучение базовой модели для прошлого (на предикторах learning)...',
        'en': "\nTraining the baseline model for the past (on 'learning' predictors)...",
    },
    'log_feature_matrix_past': {
        'ru': 'Матрица признаков для прошлого: {shape}, классы: {classes}',
        'en': 'Feature matrix for the past: {shape}, classes: {classes}',
    },
    'log_past_forecast_for': {'ru': '\nПрогноз для прошлого: {period}', 'en': '\nForecast for the past: {period}'},
    'log_predictor_order_differs': {
        'ru': '  Порядок предикторов в {period} отличается. Выполняется автоматическая пересортировка...',
        'en': '  Predictor order in {period} differs. Automatic reordering in progress...',
    },
    'log_m_factor_past_error': {
        'ru': 'Ошибка применения фактора мобильности для прошлого ({period}): {e}',
        'en': 'Error applying the mobility factor for the past ({period}): {e}',
    },
    'log_past_forecast_error': {
        'ru': 'Ошибка прогноза прошлого для {period}: {e}',
        'en': 'Error forecasting the past for {period}: {e}',
    },
    'log_creating_animation_past': {'ru': '\nСоздаём анимацию прошлого:', 'en': '\nCreating the past animation:'},
    'log_all_past_saved': {
        'ru': "\nВсе прогнозы прошлого сохранены в папку: '{dir}'",
        'en': "\nAll past forecasts saved to folder: '{dir}'",
    },
    'log_step15_monthly_header': {'ru': '\n-- 15. Приступаю к помесячному моделированию', 'en': '\n-- 15. Starting monthly modeling'},
    'log_month_forecast': {'ru': '\n---- Прогноз для месяца {month}', 'en': '\n---- Forecast for month {month}'},
    'log_creating_seasonality_animation': {'ru': '\nСоздаём сезонности:', 'en': '\nCreating seasonal animation:'},
    'log_seasons_packing_header': {'ru': '\n-- Упаковка сезонных прогнозов', 'en': '\n-- Packing seasonal forecasts'},
    'log_monthly_done': {'ru': '-- Конец помесячного моделирования', 'en': '-- End of monthly modeling'},

    # Контекстные сообщения, передаваемые в handle_model_error (sdm/sdm.py)
    'log_err_prepare_predictors': {'ru': 'Ошибка подготовки предикторов:', 'en': 'Error preparing predictors:'},
    'log_err_occurrence_processing': {'ru': 'Ошибка обработки наблюдений:', 'en': 'Error processing occurrences:'},
    'log_err_loading_predictors': {'ru': 'Ошибка загрузки предикторов:', 'en': 'Error loading predictors:'},
    'log_err_occurrence_data': {'ru': 'Ошибка данных наблюдений', 'en': 'Occurrence data error'},
    'log_err_not_enough_points_region': {'ru': 'Недостаточно точек в регионе', 'en': 'Not enough points in region'},
    'log_err_not_enough_unique_points_region': {
        'ru': 'Недостаточно уникальных точек в регионе',
        'en': 'Not enough unique points in region',
    },
    'log_err_not_enough_unique_points': {'ru': 'Недостаточно уникальных точек', 'en': 'Not enough unique points'},
    'log_err_bg_generation': {'ru': 'Ошибка генерации фоновых точек:', 'en': 'Error generating background points:'},
    'log_err_eval_model_training': {'ru': 'Ошибка обучения валидационной модели', 'en': 'Error training the validation model'},
    'log_err_final_model_training': {'ru': 'Ошибка обучения финальной модели', 'en': 'Error training the final model'},
    'log_err_past_model_training': {'ru': 'Ошибка обучения модели прошлого:', 'en': 'Error training the past model:'},
    'log_err_filename_normalization': {'ru': 'Ошибка нормализации имён файлов:', 'en': 'Error normalizing file names:'},
    'log_err_monthly_modeling': {'ru': 'Ошибка помесячного моделирования:', 'en': 'Error in monthly modeling:'},

    # Сами тексты ошибок (err_msg), которые попадают в error_obj/handle_model_error
    'log_err_no_observations_in_input': {
        'ru': 'Во входных данных нет наблюдений. Проверьте источник.',
        'en': 'No observations found in the input data. Check the data source.',
    },
    'log_err_no_observations_after_filter': {
        'ru': 'После фильтрации по координатам и дате не осталось ни одного наблюдения, подходящего для моделирования.',
        'en': 'No observations suitable for modeling remained after filtering by coordinates and date.',
    },
    'log_err_not_enough_points_after_filter': {
        'ru': 'Недостаточно точек после фильтрации. Должно быть не менее 10, сейчас: {count}.',
        'en': 'Not enough points after filtering. At least 10 required, currently: {count}.',
    },
    'log_err_occ_not_initialized': {
        'ru': 'DataFrame с точками наблюдений (self.occ) не был инициализирован. Вероятно, произошла ошибка на этапе load_occurrences.',
        'en': 'The occurrence points DataFrame (self.occ) was not initialized. An error likely occurred during load_occurrences.',
    },
    'log_err_not_enough_points_in_region': {
        'ru': 'Внутри области моделирования недостаточно точек. Должно быть не менее 10, сейчас: {count}.',
        'en': 'Not enough points inside the modeling area. At least 10 required, currently: {count}.',
    },
    'log_err_not_enough_unique_presence_in_region': {
        'ru': 'Внутри области моделирования очень мало уникальных присутствий. Должно быть не менее 5, сейчас: {count}.',
        'en': 'Too few unique presence points inside the modeling area. At least 5 required, currently: {count}.',
    },
    'log_err_not_enough_unique_points_for_modeling': {
        'ru': 'Недостаточно уникальных точек для моделирования. Должно быть не менее 5, сейчас: {count}.',
        'en': 'Not enough unique points for modeling. At least 5 required, currently: {count}.',
    },
    'log_err_predictor_mismatch': {
        'ru': 'Несовпадение предикторов в папке {period_past}.',
        'en': 'Predictor mismatch in folder {period_past}.',
    },
    'log_err_predictor_mismatch_missing': {
        'ru': ' Отсутствуют слои: {layers}.',
        'en': ' Missing layers: {layers}.',
    },
    'log_err_predictor_mismatch_extra': {
        'ru': ' Лишние слои: {layers}.',
        'en': ' Extra layers: {layers}.',
    },
    'log_err_model_not_fitted': {
        'ru': 'Модель не была обучена. Вызовите метод .fit() сначала.',
        'en': 'The model has not been trained. Call .fit() first.',
    },
    'log_err_unexpected_feature_count': {
        'ru': 'Ожидалось {expected} исходных признаков, получено {actual}.',
        'en': 'Expected {expected} original features, got {actual}.',
    },
    'log_err_no_observations_in_data': {
        'ru': 'Входной файл пустой.',
        'en': 'The input file is empty.',
    },
    'log_err_input_file_processing': {
        'ru': 'Ошибка обработки входного файла: {error}',
        'en': 'Error processing the input file: {error}',
    },
    'log_err_csv_parse': {
        'ru': 'Ошибка обработки csv. Проверьте, что у входных данных корректный формат. Колонки с координатами должны называться lat, lon. Ячейки должны разделяться символом табуляции (при экспорте из Excel используйте формат текстовый файл с табуляцией).',
        'en': 'Error parsing the CSV. Check that the input data is in the correct format. Coordinate columns must be named lat, lon. Cells must be separated by a tab character (when exporting from Excel, use the tab-delimited text format).',
    },
    'log_err_missing_coord_columns': {
        'ru': 'В CSV нет столбцов {lon_col}/{lat_col}',
        'en': 'The CSV has no {lon_col}/{lat_col} columns',
    },
    'log_err_no_matching_predictor_files': {
        'ru': 'В папке {raster_dir} не найдены файлы, соответствующие предикторам. ',
        'en': 'No files matching the predictors were found in folder {raster_dir}. ',
    },
    'log_err_found_only': {
        'ru': 'Найдены только: {found}',
        'en': 'Found only: {found}',
    },
    'log_none_found': {'ru': 'ни одного', 'en': 'none'},
    'log_err_image_not_found': {
        'ru': 'Файл не найден: {img_path}',
        'en': 'File not found: {img_path}',
    },
    'log_err_image_read_failed': {
        'ru': 'Не удалось прочитать изображение: {img_path}. Возможно, файл поврежден или имеет неверный формат.',
        'en': 'Could not read the image: {img_path}. The file may be corrupted or in an unsupported format.',
    },
    'log_err_image_size_mismatch': {
        'ru': 'Изображения должны иметь одинаковый размер. Изображение {path0} имеет размер {w0}x{h0}, а изображение {path_i} имеет размер {wi}x{hi}.',
        'en': 'Images must be the same size. Image {path0} is {w0}x{h0}, but image {path_i} is {wi}x{hi}.',
    },
    'log_err_raster_geometry_mismatch': {
        'ru': 'Растр {fp} не согласован по геометрии с первым растром',
        'en': 'Raster {fp} is not geometrically aligned with the first raster',
    },
    'log_err_raster_crs_mismatch': {
        'ru': 'Растр {fp} имеет другой CRS: {crs} vs {ref_crs}',
        'en': 'Raster {fp} has a different CRS: {crs} vs {ref_crs}',
    },
    'log_err_no_predictors_after_glacier_exclusion': {
        'ru': 'Не осталось предикторов после исключения слоёв ледников.',
        'en': 'No predictors remained after excluding glacier layers.',
    },
    'log_err_nan_layers': {
        'ru': 'Невозможно продолжить: следующие слои не имеют данных (полностью NaN) в выбранном регионе:\n{layers}',
        'en': 'Cannot continue: the following layers have no data (entirely NaN) in the selected region:\n{layers}',
    },
    'log_err_no_valid_pixels': {
        'ru': 'Нет валидных пикселей для моделирования. Области данных на слоях не пересекаются.',
        'en': 'No valid pixels for modeling. The data extents of the layers do not overlap.',
    },
    'log_err_unsupported_predictor_period': {
        'ru': 'Неподдерживаемый период загрузки предикторов: {period}',
        'en': 'Unsupported predictor loading period: {period}',
    },
    'log_err_unknown_model_type': {
        'ru': "Неизвестный или неподдерживаемый тип алгоритма: '{model}'",
        'en': "Unknown or unsupported algorithm type: '{model}'",
    },
    'log_err_unknown_model_type_short': {
        'ru': "Неизвестный тип алгоритма: '{model}'",
        'en': "Unknown algorithm type: '{model}'",
    },
    'log_err_empty_training_set': {
        'ru': "Обучающая выборка пуста. Точки присутствия не попали в валидные области предикторов 'learning'. Возможно, несовпадение масок суши или отсутствие точек в этом периоде.",
        'en': "The training set is empty. Presence points did not fall within the valid areas of the 'learning' predictors. The land masks may not match, or there are no points in this period.",
    },

    # --- Консольный/лог-вывод (print) - sdm/utils/data_loader.py ---
    'log_species_detected': {'ru': 'Определён вид: {species}', 'en': 'Species determined: {species}'},
    'log_kingdom_class': {
        'ru': 'Вычислено царство {kingdom} и класс {dclass}',
        'en': 'Kingdom {kingdom} and class {dclass} determined',
    },
    'log_total_loaded': {'ru': 'Всего загружено записей: {n}', 'en': 'Total records loaded: {n}'},
    'log_reload_count': {
        'ru': 'Записей после дозагрузки: {n}, из них дозагружено: {n2}',
        'en': 'Records after merging additional data: {n}, of which newly added: {n2}',
    },
    'log_filter_gbif_header': {
        'ru': '-- 2.1. Фильтрация мусорных данных из GBIF ({id})',
        'en': '-- 2.1. Filtering junk GBIF data ({id})',
    },
    'log_filter_quality': {
        'ru': 'Осталось записей после фильтрации по качеству: {n}',
        'en': 'Records remaining after quality filtering: {n}',
    },
    'log_filter_bbox': {
        'ru': 'Осталось записей после фильтрации по региону (BBox): {n} (из {total})',
        'en': 'Records remaining after region (BBox) filtering: {n} (out of {total})',
    },
    'log_group_by_month_header': {'ru': '-- 2.2. Группировка по месяцам ({id})', 'en': '-- 2.2. Grouping by month ({id})'},
    'log_filter_year': {
        'ru': 'Осталось записей после фильтрации по году (>= {year}): {n} (из {total})',
        'en': 'Records remaining after year filtering (>= {year}): {n} (out of {total})',
    },
    'log_final_presences_header': {'ru': '-- 2.3. Финальные присутствия ({id})', 'en': '-- 2.3. Final presences ({id})'},
    'log_occurrence_processing_header': {'ru': '\n-- Обработка наблюдений', 'en': '\n-- Processing occurrences'},
    'log_final_csv_count': {'ru': 'Осталось записей финально CSV: {n}', 'en': 'Final CSV record count: {n}'},
    'log_separator_detected': {'ru': "Определен разделитель: '{sep}'", 'en': "Detected separator: '{sep}'"},
    'log_predictors_input_path': {'ru': 'Входной путь для предикторов: {path}', 'en': 'Predictor input path: {path}'},
    'log_predictors_static_path': {'ru': 'Статические предикторы: {path}', 'en': 'Static predictors: {path}'},
    'log_predictors_dynamic_path': {'ru': 'Динамические предикторы: {path}', 'en': 'Dynamic predictors: {path}'},
    'log_available_predictors_label': {'ru': 'Доступные предикторы:', 'en': 'Available predictors:'},
    'log_wanted_predictors_label': {'ru': 'Нужные предикторы:', 'en': 'Wanted predictors:'},
    'log_loaded_predictors_label': {'ru': 'Загруженные предикторы:', 'en': 'Loaded predictors:'},
    'log_missing_predictors_label': {'ru': 'Не найденные предикторы:', 'en': 'Predictors not found:'},
    'log_extrapolation_running': {
        'ru': 'Экстраполяция значений предикторов на акватории (Nearest Neighbor) в параллельном режиме...',
        'en': 'Extrapolating predictor values onto water bodies (Nearest Neighbor) in parallel...',
    },

    # --- Консольный/лог-вывод (print) - sdm/utils/preprocessing.py ---
    'log_processing_file': {'ru': '\nОбработка файла: {name}', 'en': '\nProcessing file: {name}'},
    'log_source_crs': {'ru': '  Исходный CRS: {crs}', 'en': '  Source CRS: {crs}'},
    'log_source_resolution': {
        'ru': '  Исходное разрешение: {x:.6f} x {y:.6f} градусов',
        'en': '  Source resolution: {x:.6f} x {y:.6f} degrees',
    },
    'log_source_bounds': {'ru': '  Исходные границы: {bounds}', 'en': '  Source bounds: {bounds}'},
    'log_file_saved': {'ru': '  Сохранен файл: {path}', 'en': '  File saved: {path}'},
    'log_saved_file_params': {
        'ru': '    CRS: {crs}, Разрешение: {res:.6f}, Границы: {bounds}',
        'en': '    CRS: {crs}, Resolution: {res:.6f}, Bounds: {bounds}',
    },
    'log_rasterio_io_error': {
        'ru': '  Ошибка чтения/записи файла {name}: {e}',
        'en': '  Error reading/writing file {name}: {e}',
    },
    'log_unexpected_processing_error': {
        'ru': '  Непредвиденная ошибка при обработке {name}: {e}',
        'en': '  Unexpected error while processing {name}: {e}',
    },
    'log_predictors_processing_header': {'ru': '\n-- Обработка предикторов', 'en': '\n-- Processing predictors'},
    'log_unsupported_resolution': {'ru': 'Неподдерживаемое разрешение растра: {res}', 'en': 'Unsupported raster resolution: {res}'},
    'log_predictors_processing_done': {'ru': '-- Обработка предикторов завершена', 'en': '-- Predictor processing complete'},
    'log_files_processed': {'ru': 'Обработано файлов: {n}', 'en': 'Files processed: {n}'},
    'log_results_saved_folder': {'ru': "Результаты сохранены в папку: '{dir}'", 'en': "Results saved to folder: '{dir}'"},

    # --- Консольный/лог-вывод (print) - sdm/utils/utils.py ---
    'log_scikit_fmm_missing': {
        'ru': "\nВНИМАНИЕ: Для учета рельефа/барьеров требуется библиотека 'scikit-fmm'. Выполните 'pip install scikit-fmm'.",
        'en': "\nWARNING: Accounting for terrain/barriers requires the 'scikit-fmm' library. Run 'pip install scikit-fmm'.",
    },
    'log_fallback_euclidean': {
        'ru': 'Используется обычное евклидово расстояние.',
        'en': 'Falling back to standard Euclidean distance.',
    },
    'log_pixel_area_error': {
        'ru': 'Ошибка при расчете площади пикселя для широты {lat}: {e}',
        'en': 'Error computing pixel area for latitude {lat}: {e}',
    },
    'log_bg_no_valid_pixels': {
        'ru': 'ВНИМАНИЕ: Нет доступных валидных пикселей для генерации случайного фона.',
        'en': 'WARNING: No valid pixels available to generate random background.',
    },
    'log_bg_hybrid_sampling': {
        'ru': 'Применяется гибридное взвешенное сэмплирование (сила = {pct}%)...',
        'en': 'Applying hybrid weighted sampling (strength = {pct}%)...',
    },
    'log_bg_inverse_scaling': {
        'ru': 'Выполняется обратное масштабирование весов предвзятости...',
        'en': 'Inverse-scaling bias weights...',
    },
    'log_bg_weighted_sampling_warning': {
        'ru': 'Предупреждение при взвешенном сэмплировании: {e}. Повторная нормализация.',
        'en': 'Warning during weighted sampling: {e}. Re-normalizing.',
    },
    'log_bg_uniform_sampling': {
        'ru': 'Применяется стандартное равномерное сэмплирование фона (bias слой не используется или сила=0)...',
        'en': 'Applying standard uniform background sampling (bias layer unused or strength=0)...',
    },
    'log_bg_generated_count': {'ru': 'Сгенерировано фоновых точек: {n}', 'en': 'Background points generated: {n}'},
    'log_bg_pseudo_absence_needed': {
        'ru': 'Нужно сгенерировать точек псевдоприсутствия: {n}',
        'en': 'Pseudo-absence points still needed: {n}',
    },
    'log_bg_no_presence_for_buffer': {
        'ru': 'ВНИМАНИЕ: Отсутствуют точки присутствия для генерации фона в огибающей.',
        'en': 'WARNING: No presence points available to generate buffer background.',
    },
    'log_bg_no_valid_pixels_in_radius': {
        'ru': 'ВНИМАНИЕ: Нет доступных валидных пикселей в радиусе ({dmin} - {dmax} пикселей) вокруг точек присутствия для генерации фона.',
        'en': 'WARNING: No valid pixels available within the radius ({dmin} - {dmax} pixels) around presence points to generate background.',
    },
    'log_bg_generated_less_than_requested': {
        'ru': 'ВНИМАНИЕ: Сгенерировано {current} фоновых точек, вместо желаемых {n}.',
        'en': 'WARNING: Generated {current} background points instead of the desired {n}.',
    },

    # --- Консольный/лог-вывод (print) - sdm/utils/models.py ---
    'log_glm_training': {
        'ru': 'Обучение GLM (логистическая регрессия, подбор C по {cv}-фолдовой кросс-валидации)...',
        'en': 'Training GLM (logistic regression, selecting C via {cv}-fold cross-validation)...',
    },
    'log_glm_training_done': {
        'ru': 'Обучение GLM завершено. Выбранная сила регуляризации C = {c:.5g}',
        'en': 'GLM training complete. Selected regularization strength C = {c:.5g}',
    },
    'log_maxent_training': {
        'ru': 'Обучение MaxEnt (elapid, {features}-признаки, beta_multiplier={beta})...',
        'en': 'Training MaxEnt (elapid, {features} features, beta_multiplier={beta})...',
    },
    'log_maxent_training_done': {'ru': 'Обучение MaxEnt завершено.', 'en': 'MaxEnt training complete.'},

    # --- Консольный/лог-вывод (print) - sdm_server.py / examples/example_minimal.py ---
    'log_modeling_done': {'ru': '\n-- Моделирование завершено', 'en': '\n-- Modeling complete'},
    'log_modeling_done_simple': {'ru': 'Моделирование завершено', 'en': 'Modeling complete'},
}


def t(key: str, lang: str = DEFAULT_LANG, **kwargs) -> str:
    """
    Возвращает переведённую строку по ключу из TRANSLATIONS.

    Args:
        key (str): Ключ в TRANSLATIONS.
        lang (str): Код языка (любое значение, принимаемое normalize_lang).
        **kwargs: Параметры для подстановки в шаблон через str.format (в том числе
                  со спецификаторами формата, например {auc:.3f}).

    Returns:
        str: Готовая строка на запрошенном языке (с откатом на DEFAULT_LANG, если
             перевод для этого языка/ключа отсутствует).
    """
    lang = normalize_lang(lang)
    entry = TRANSLATIONS.get(key)
    if entry is None:
        raise KeyError(f"Неизвестный ключ перевода: {key}")
    template = entry.get(lang, entry[DEFAULT_LANG])
    return template.format(**kwargs) if kwargs else template


def localized_predictor_title(band_info: dict, band_name: str, lang: str = DEFAULT_LANG) -> str:
    """
    Возвращает название предиктора (для заголовков гистограмм/карт слоёв) на запрошенном
    языке. Названия берутся из sdm/utils/predictors_info.py ('title'/'title_en'); если
    перевод для записи ещё не добавлен, используется русское название как откат.

    Args:
        band_info (dict): Запись справочника предикторов для конкретного слоя (одно
                           значение из словаря, который возвращает get_predictors_info()).
        band_name (str): Имя слоя (используется как запасной вариант, если band_info пуст).
        lang (str): Код языка.
    """
    if normalize_lang(lang) == 'en':
        return band_info.get('title_en') or band_info.get('title', band_name)
    return band_info.get('title', band_name)


def localized_predictor_unit(band_info: dict, lang: str = DEFAULT_LANG) -> str:
    """
    Возвращает единицу измерения предиктора на запрошенном языке, с откатом на русскую
    запись ('unit'), если 'unit_en' для слоя ещё не добавлен.
    """
    if normalize_lang(lang) == 'en':
        return band_info.get('unit_en') or band_info.get('unit', '')
    return band_info.get('unit', '')
