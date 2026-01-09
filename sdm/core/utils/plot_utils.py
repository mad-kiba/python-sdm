import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import array_bounds
import contextily as ctx
from pyproj import CRS, Transformer
from matplotlib.ticker import FuncFormatter, MaxNLocator
from shapely.geometry import Point
from PIL import Image

from .helpers import get_predictor_stats, format_float
from .gis_utils import read_and_to_3857


def create_animated_gif(image_paths, output_path="animation.gif", duration=500):
    """
    Создает анимированный GIF из списка путей к изображениям.

    Args:
        image_paths (list): Список строк, где каждая строка — путь к файлу изображения.
        output_path (str): Путь для сохранения выходного GIF файла.
        duration (int): Время отображения каждого кадра в миллисекундах.
                        (Например, 500ms = 0.5 секунды на кадр).
    """
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            # Важно: Для GIF лучше, чтобы все кадры имели одинаковый режим (например, 'RGB' или 'RGBA').
            # Если режим отличается, Pillow может работать некорректно.
            # Если ваши изображения имеют разные режимы, можно добавить конвертацию:
            # img = img.convert('RGB')
            images.append(img)
        except FileNotFoundError:
            print(f"Ошибка: Файл не найден по пути: {path}")
            return
        except Exception as e:
            print(f"Ошибка при открытии файла {path}: {e}")
            return

    if not images:
        print("Не удалось загрузить ни одного изображения.")
        return

    # Сохраняем GIF
    # append_images: список остальных кадров, кроме первого
    # save_all=True: указывает Pillow сохранить все кадры
    # duration: время отображения каждого кадра
    # loop: 0 означает бесконечную зацикленность GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"Анимированный GIF сохранен как: {output_path}")
    

def draw_map(OUTPUT_SUITABILITY_TIF, OUTPUT_SUITABILITY_JPG, title = '', rows=[], cols=[]):
    data, transform, width, height = read_and_to_3857(OUTPUT_SUITABILITY_TIF) 
    
    # Границы растра в координатах EPSG:3857
    xmin, ymin, xmax, ymax = array_bounds(height, width, transform)
    
    # Поля: одинаковая ширина в метрах, но не менее 5% от каждой стороны
    pad_x_req = (xmax - xmin) * 0.05
    pad_y_req = (ymax - ymin) * 0.05
    pad_m = max(pad_x_req, pad_y_req)
    
    xmin_v, xmax_v = xmin - pad_m, xmax + pad_m
    ymin_v, ymax_v = ymin - pad_m, ymax + pad_m
    
    # Колормэп с прозрачностью по NaN (нет данных)
    cmap = plt.cm.magma.copy()
    cmap.set_bad(alpha=0.0)
    
    plt.style.use('classic')
    
    # Подбор ориентации и размеров фигуры под соотношение сторон области
    width_v = xmax_v - xmin_v
    height_v = ymax_v - ymin_v
    ratio = width_v / height_v  # >1 — горизонтальная, <1 — вертикальная
    
    # Длинная сторона в дюймах и целевой размер по длинной стороне (минимум 2000 пикселей)
    long_inches = 10.0
    desired_long_px = 2000
    dpi = int(desired_long_px / long_inches)
    
    if ratio >= 1.0:
        # горизонтальная область
        fig_w = long_inches
        fig_h = max(long_inches / ratio, 1e-3)
    else:
        # вертикальная область
        fig_h = long_inches
        fig_w = max(long_inches * ratio, 1e-3)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor('white')  # Фон фигуры
    
    # Половинные (или чуть меньше) размеры шрифтов относительно текущих настроек
    base_fs = plt.rcParams.get('font.size', 10)
    tick_fs = max(6, int(base_fs * 0.5))   # подписи делений осей
    label_fs = max(7, int(base_fs * 0.6))  # подписи осей
    title_fs = max(9, int(base_fs * 0.7))  # заголовок
    
    # Видимая область и равный масштаб по X и Y — чтобы поля выглядели одинаково
    ax.set_xlim(xmin_v, xmax_v)
    ax.set_ylim(ymin_v, ymax_v)
    ax.set_aspect('equal', adjustable='box')
    
    # Подложка OSM
    ctx.add_basemap(
        ax,
        source=ctx.providers.OpenStreetMap.Mapnik
    )
    
    # Растровая подложка (ваш GeoTIFF) поверх OSM
    im = ax.imshow(
        data,
        extent=(xmin, xmax, ymin, ymax),  # (left, right, bottom, top)
        origin="upper",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",  # чуть более плавно и приятно глазу
        alpha=0.8
    )
    
    # Цветовая шкала
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Вероятность присутствия")
    
    #print(rows)
    #print(cols)
    
    #print(xmin, xmax)
    #print(ymin, ymax)
    
    transformer = Transformer.from_crs(CRS("EPSG:4326"), CRS("EPSG:3857"), always_xy=True)
    x_3857, y_3857 = transformer.transform(rows, cols)
    
    # наносим точки встреч на карту
    if len(rows)>0:
        ax.scatter(x_3857, y_3857, marker='o', s=5, color='red', alpha=0.7, zorder=100)
        ax.scatter(x_3857, y_3857, marker='o', s=4, color='yellow', alpha=0.7, zorder=100)
        #print(f"Added {len(x_3857)} observation points to the map in EPSG:3857 using manual coordinate calculation.")
    
    # Оси в градусах (слева и снизу)
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    cx = (xmin_v + xmax_v) / 2.0
    cy = (ymin_v + ymax_v) / 2.0
    
    def x_deg_formatter(x, pos):
        lon, _ = transformer.transform(x, cy)
        return f"{lon:.2f}°"
    
    def y_deg_formatter(y, pos):
        _, lat = transformer.transform(cx, y)
        return f"{lat:.2f}°"
    
    ax.xaxis.set_major_formatter(FuncFormatter(x_deg_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(y_deg_formatter))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    
    # Тик-марки только слева и снизу, внешний вид
    #ax.tick_params(axis='both', direction='out', top=False, right=False, labeltop=False, labelright=False)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs, direction='out', top=False, right=False)
    ax.tick_params(axis='both', which='minor', labelsize=max(6, tick_fs - 1))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor('#666666')
    
    #ax.set_xlabel("Долгота (°)")
    #ax.set_ylabel("Широта (°)")
    if (title==''):
        ax.set_title("Карта вероятности присутствия вида", pad=8, fontsize=title_fs)
    else:
        ax.set_title(title, pad=8, fontsize=title_fs)
    
    plt.tight_layout()
    
    # Сохранение с высоким разрешением (длина ≥ 2000 px)
    plt.savefig(
        OUTPUT_SUITABILITY_JPG,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    
    plt.close(fig)
    
    
def create_beautiful_histogram(ax: plt.Axes, data: np.ndarray, band_name: str, bins_num: int, data_full: np.ndarray, title = ''):
    """
    Рисует красивую и информативную гистограмму на заданных осях.
    """
    
    bio_info = {
        'roughness_std3x3': "Пересечённость рельефа (стандартное отклонение на сетке 3х3 км)",
        'slope_deg': "Уклон рельефа на заданном шаге модели",
        'wc2.1_30s_elev': "WC ELEV: Высота над уровнем моря",
        'wc2.1_30s_bio_1': "WC BIO1: Среднегодовая температура (Mean annual temperature)",
        'wc2.1_30s_bio_2': "WC BIO2: Среднегодовая суточная температурная амплитуда (Mean diurnal range)",
        'wc2.1_30s_bio_3': "WC BIO3: Изотермичность/сезонность температуры (Temperature seasonality, 100 * BIO2 / BIO7)",
        'wc2.1_30s_bio_4': "WC BIO4: Сезонность температуры (Temperature seasonality, стандартное отклонение * 100)", # BIO4 тоже про сезонность, но другой расчет
        'wc2.1_30s_bio_5': "WC BIO5: Средняя максимальная температура самого теплого месяца (Max temperature of warmest month)",
        'wc2.1_30s_bio_6': "WC BIO6: Средняя минимальная температура самого холодного месяца (Min temperature of coldest month)",
        'wc2.1_30s_bio_7': "WC BIO7: Годовая температурная амплитуда (Annual temperature range, BIO5 - BIO6)",
        'wc2.1_30s_bio_8': "WC BIO8: Средняя температура самого влажного квартала (Mean temperature of wettest quarter)",
        'wc2.1_30s_bio_9': "WC BIO9: Средняя температура самого сухого квартала (Mean temperature of driest quarter)",
        'wc2.1_30s_bio_10': "WC BIO10: Средняя температура самого теплого квартала (Mean temperature of warmest quarter)",
        'wc2.1_30s_bio_11': "WC BIO11: Средняя температура самого холодного квартала (Mean temperature of coldest quarter)",
        'wc2.1_30s_bio_12': "WC BIO12: Годовое количество осадков (Annual precipitation)",
        'wc2.1_30s_bio_13': "WC BIO13: Количество осадков самого влажного месяца (Precipitation of wettest month)",
        'wc2.1_30s_bio_14': "WC BIO14: Количество осадков самого сухого месяца (Precipitation of driest month)",
        'wc2.1_30s_bio_15': "WC BIO15: Сезонность осадков (Precipitation seasonality)",
        'wc2.1_30s_bio_16': "WC BIO16: Количество осадков самого влажного квартала (Precipitation of wettest quarter)",
        'wc2.1_30s_bio_17': "WC BIO17: Количество осадков самого сухого квартала (Precipitation of driest quarter)",
        'wc2.1_30s_bio_18': "WC BIO18: Количество осадков самого теплого квартала (Precipitation of warmest quarter)",
        'wc2.1_30s_bio_19': "WC BIO19: Количество осадков самого холодного квартала (Precipitation of coldest quarter)",
        'Consensus_reduced_class_1': "EE CLC1: Вечнозелёные/Листопадные хвойные деревья (Evergreen/Deciduous Needleleaf Trees)",
        'Consensus_reduced_class_2': "EE CLC2: Вечнозелёные широколистные деревья (Evergreen Broadleaf Trees)",
        'Consensus_reduced_class_3': "EE CLC3: Листопадные широколистные деревья (Deciduous Broadleaf Trees)",
        'Consensus_reduced_class_4': "EE CLC4: Смешанные/Другие деревья (Mixed/Other Trees)",
        'Consensus_reduced_class_5': "EE CLC5: Кустарники (Shrubs)",
        'Consensus_reduced_class_6': "EE CLC6: Травянистая растительность (Herbaceous Vegetation)",
        'Consensus_reduced_class_7': "EE CLC7: Культивируемая и управляемая растительность (Cultivated and Managed Vegetation)",
        'Consensus_reduced_class_8': "EE CLC8: Регулярно затопляемая растительность (Regularly Flooded Vegetation)",
        'Consensus_reduced_class_9': "EE CLC9: Городская/Застроенная территория (Urban/Built-up)",
        'Consensus_reduced_class_10': "EE CLC10: Снег/Лёд (Snow/Ice)",
        'Consensus_reduced_class_11': "EE CLC11: Голые земли / Бесплодные земли (Barren)",
        'Consensus_reduced_class_12': "EE CLC12: Открытые водные пространства (Open Water)",
        'ENVIREM_thermicityIndex': "ENVIREM: Компенсированный индекс термичности",
        'ENVIREM_monthCountByTemp10': "ENVIREM: Количество месяцев со средней температурой выше 10 гр.",
        'ENVIREM_minTempWarmest': "ENVIREM: Минимальная температура самого теплого месяца, x10",
        'ENVIREM_maxTempColdest': "ENVIREM: Максимальная температура самого холодного месяца, x10",
        'ENVIREM_growingDegDays5': "ENVIREM: Сумма средних месячных температур для месяцев со средней температурой выше 5℃, умноженная на количество дней",
        'ENVIREM_growingDegDays0': "ENVIREM: Сумма средних месячных температур для месяцев со средней температурой выше 0℃, умноженная на количество дней",
        'ENVIREM_embergerQ': "ENVIREM: Эмбержеровский плювиотермический коэффициент",
        'ENVIREM_continentality': "ENVIREM: Континентальность (средняя температура самого теплого месяца минус средняя температура самого холодного месяца)",
        'ENVIREM_climaticMoistureIndex': "ENVIREM: Климатический индекс влажности (показатель относительной влажности и сухости)",
        'ENVIREM_aridityIndexThornthwaite': "ENVIREM: Индекс аридности Торнтвейта (индекс степени водного дефицита)",
        'ENVIREM_PETWettestQuarter': "ENVIREM: Среднемесячная потенциальная эвапотранспирация самого влажного квартала",
        'ENVIREM_PETWarmestQuarter': "ENVIREM: Среднемесячная потенциальная эвапотранспирация самого теплого квартала",
        'ENVIREM_PETseasonality': "ENVIREM: Сезонность потенциальной эвапотранспирации",
        'ENVIREM_annualPET': "ENVIREM: Годовая потенциальная эвапотранспирация",
        'ENVIREM_PETDriestQuarter': "ENVIREM: Среднемесячная потенциальная эвапотранспирация самого сухого квартала",
        'ENVIREM_PETColdestQuarter': "ENVIREM: Среднемесячная потенциальная эвапотранспирация холодного квартала",
        
        'SG_ocs_0-30cm_mean_1000': "SoilGrid: Запас органического углерода в почве на глубине 0-30 см, х10 кг/кв. м",

		'SG_bdod_0-5cm_mean_1000': "SoilGrid: Плотность грунта на глубине 0-5 см, х100 кг/л",
		'SG_bdod_5-15cm_mean_1000': "SoilGrid: Плотность грунта на глубине 5-15 см, х100 кг/л",
		'SG_bdod_15-30cm_mean_1000': "SoilGrid: Плотность грунта на глубине 15-30 см, х100 кг/л",
		'SG_bdod_30-60cm_mean_1000': "SoilGrid: Плотность грунта на глубине 30-60 см, х100 кг/л",
		'SG_bdod_60-100cm_mean_1000': "SoilGrid: Плотность грунта на глубине 60-100 см, х100 кг/л",
		'SG_bdod_100-200cm_mean_1000': "SoilGrid: Плотность грунта на глубине 100-200 см, х100 кг/л",
		
		'SG_cec_0-5cm_mean_1000': "SoilGrid: Ёмкость катионного обмена почвы (ph=7) на глубине 0-5 см, х10 смоль/кг",
		'SG_cec_5-15cm_mean_1000': "SoilGrid: Ёмкость катионного обмена почвы (ph=7) на глубине 5-15 см, х10 смоль/кг",
		'SG_cec_15-30cm_mean_1000': "SoilGrid: Ёмкость катионного обмена почвы (ph=7) на глубине 15-30 см, х10 смоль/кг",
		'SG_cec_30-60cm_mean_1000': "SoilGrid: Ёмкость катионного обмена почвы (ph=7) на глубине 30-60 см, х10 смоль/кг",
		'SG_cec_60-100cm_mean_1000': "SoilGrid: Ёмкость катионного обмена почвы (ph=7) на глубине 60-100 см, х10 смоль/кг",
		'SG_cec_100-200cm_mean_1000': "SoilGrid: Ёмкость катионного обмена почвы (ph=7) на глубине 100-200 см, х10 смоль/кг",
		
		'SG_cfvo_0-5cm_mean_1000': "SoilGrid: Объёмная доля крупных частиц на глубине 0-5 см, х100 %",
		'SG_cfvo_5-15cm_mean_1000': "SoilGrid: Объёмная доля крупных частиц на глубине 5-15 см, х100 %",
		'SG_cfvo_15-30cm_mean_1000': "SoilGrid: Объёмная доля крупных частиц на глубине 15-30 см, х100 %",
		'SG_cfvo_30-60cm_mean_1000': "SoilGrid: Объёмная доля крупных частиц на глубине 30-60 см, х100 %",
		'SG_cfvo_60-100cm_mean_1000': "SoilGrid: Объёмная доля крупных частиц на глубине 60-100 см, х100 %",
		'SG_cfvo_100-200cm_mean_1000': "SoilGrid: Объёмная доля крупных частиц на глубине 100-200 см, х100 %",
		
		'SG_clay_0-5cm_mean_1000': "SoilGrid: Содержание глины в почве на глубине 0-5 см, х10 г/100 г",
		'SG_clay_5-15cm_mean_1000': "SoilGrid: Содержание глины в почве на глубине 5-15 см, х10 г/100 г",
		'SG_clay_15-30cm_mean_1000': "SoilGrid: Содержание глины в почве на глубине 15-30 см, х10 г/100 г",
		'SG_clay_30-60cm_mean_1000': "SoilGrid: Содержание глины в почве на глубине 30-60 см, х10 г/100 г",
		'SG_clay_60-100cm_mean_1000': "SoilGrid: Содержание глины в почве на глубине 60-100 см, х10 г/100 г",
		'SG_clay_100-200cm_mean_1000': "SoilGrid: Содержание глины в почве на глубине 100-200 см, х10 г/100 г",
		
		'SG_nitrogen_0-5cm_mean_1000': "SoilGrid: Содержание азота в почве на глубине 0-5 см, х100 г/кг",
		'SG_nitrogen_5-15cm_mean_1000': "SoilGrid: Содержание азота в почве на глубине 5-15 см, х100 г/кг",
		'SG_nitrogen_15-30cm_mean_1000': "SoilGrid: Содержание азота в почве на глубине 15-30 см, х100 г/кг",
		'SG_nitrogen_30-60cm_mean_1000': "SoilGrid: Содержание азота в почве на глубине 30-60 см, х100 г/кг",
		'SG_nitrogen_60-100cm_mean_1000': "SoilGrid: Содержание азота в почве на глубине 60-100 см, х100 г/кг",
		'SG_nitrogen_100-200cm_mean_1000': "SoilGrid: Содержание азота в почве на глубине 100-200 см, х100 г/кг",
		
		'SG_ocd_0-5cm_mean_1000': "SoilGrid: Плотность органического углерода в почве на глубине 0-5 см, х10 кг/куб.м",
		'SG_ocd_5-15cm_mean_1000': "SoilGrid: Плотность органического углерода в почве на глубине 5-15 см, х10 кг/куб.м",
		'SG_ocd_15-30cm_mean_1000': "SoilGrid: Плотность органического углерода в почве на глубине 15-30 см, х10 кг/куб.м",
		'SG_ocd_30-60cm_mean_1000': "SoilGrid: Плотность органического углерода в почве на глубине 30-60 см, х10 кг/куб.м",
		'SG_ocd_60-100cm_mean_1000': "SoilGrid: Плотность органического углерода в почве на глубине 60-100 см, х10 кг/куб.м",
		'SG_ocd_100-200cm_mean_1000': "SoilGrid: Плотность органического углерода в почве на глубине 100-200 см, х10 кг/куб.м",
		
		'SG_soc_0-5cm_mean_1000': "SoilGrid: Почвенный органический углерод на глубине 0-5 см, х10 г/кг",
		'SG_soc_5-15cm_mean_1000': "SoilGrid: Почвенный органический углерод на глубине 5-15 см, х10 г/кг",
		'SG_soc_15-30cm_mean_1000': "SoilGrid: Почвенный органический углерод на глубине 15-30 см, х10 г/кг",
		'SG_soc_30-60cm_mean_1000': "SoilGrid: Почвенный органический углерод на глубине 30-60 см, х10 г/кг",
		'SG_soc_60-100cm_mean_1000': "SoilGrid: Почвенный органический углерод на глубине 60-100 см, х10 г/кг",
		'SG_soc_100-200cm_mean_1000': "SoilGrid: Почвенный органический углерод на глубине 100-200 см, х10 г/кг",
		
		'SG_phh2o_0-5cm_mean_1000': "SoilGrid: pH воды на глубине 0-5 см, х10",
		'SG_phh2o_5-15cm_mean_1000': "SoilGrid: pH воды на глубине 5-15 см, х10",
		'SG_phh2o_15-30cm_mean_1000': "SoilGrid: pH воды на глубине 15-30 см, х10",
		'SG_phh2o_30-60cm_mean_1000': "SoilGrid: pH воды на глубине 30-60 см, х10",
		'SG_phh2o_60-100cm_mean_1000': "SoilGrid: pH воды на глубине 60-100 см, х10",
		'SG_phh2o_100-200cm_mean_1000': "SoilGrid: pH воды на глубине 100-200 см, х10",
		
		'SG_sand_0-5cm_mean_1000': "SoilGrid: Содержание песка в почве на глубине 0-5 см, х10 г/100 г",
		'SG_sand_5-15cm_mean_1000': "SoilGrid: Содержание песка в почве на глубине 5-15 см, х10 г/100 г",
		'SG_sand_15-30cm_mean_1000': "SoilGrid: Содержание песка в почве на глубине 15-30 см, х10 г/100 г",
		'SG_sand_30-60cm_mean_1000': "SoilGrid: Содержание песка в почве на глубине 30-60 см, х10 г/100 г",
		'SG_sand_60-100cm_mean_1000': "SoilGrid: Содержание песка в почве на глубине 60-100 см, х10 г/100 г",
		'SG_sand_100-200cm_mean_1000': "SoilGrid: Содержание песка в почве на глубине 100-200 см, х10 г/100 г",
		
		'SG_silt_0-5cm_mean_1000': "SoilGrid: Содержание илистых пород в почве на глубине 0-5 см, х10 г/100 г",
		'SG_silt_5-15cm_mean_1000': "SoilGrid: Содержание илистых пород в почве на глубине 5-15 см, х10 г/100 г",
		'SG_silt_15-30cm_mean_1000': "SoilGrid: Содержание илистых пород в почве на глубине 15-30 см, х10 г/100 г",
		'SG_silt_30-60cm_mean_1000': "SoilGrid: Содержание илистых пород в почве на глубине 30-60 см, х10 г/100 г",
		'SG_silt_60-100cm_mean_1000': "SoilGrid: Содержание илистых пород в почве на глубине 60-100 см, х10 г/100 г",
		'SG_silt_100-200cm_mean_1000': "SoilGrid: Содержание илистых пород в почве на глубине 100-200 см, х10 г/100 г",
		
		'SG_wv0010_0-5cm_mean_1000': "SoilGrid: Объемная влажность при 10 кПа на глубине 0-5 см, х10 %",
		'SG_wv0010_5-15cm_mean_1000': "SoilGrid: Объемная влажность при 10 кПа на глубине 5-15 см, х10 %",
		'SG_wv0010_15-30cm_mean_1000': "SoilGrid: Объемная влажность при 10 кПа на глубине 15-30 см, х10 %",
		'SG_wv0010_30-60cm_mean_1000': "SoilGrid: Объемная влажность при 10 кПа на глубине 30-60 см, х10 %",
		'SG_wv0010_60-100cm_mean_1000': "SoilGrid: Объемная влажность при 10 кПа на глубине 60-100 см, х10 %",
		'SG_wv0010_100-200cm_mean_1000': "SoilGrid: Объемная влажность при 10 кПа на глубине 100-200 см, х10 %",

        'SG_wv0033_0-5cm_mean_1000': "SoilGrid: Объемная влажность при 33 кПа на глубине 0-5 см, х10 %",
        'SG_wv0033_5-15cm_mean_1000': "SoilGrid: Объемная влажность при 33 кПа на глубине 5-15 см, х10 %",
        'SG_wv0033_15-30cm_mean_1000': "SoilGrid: Объемная влажность при 33 кПа на глубине 15-30 см, х10 %",
        'SG_wv0033_30-60cm_mean_1000': "SoilGrid: Объемная влажность при 33 кПа на глубине 30-60 см, х10 %",
        'SG_wv0033_60-100cm_mean_1000': "SoilGrid: Объемная влажность при 33 кПа на глубине 60-100 см, х10 %",
        'SG_wv0033_100-200cm_mean_1000': "SoilGrid: Объемная влажность при 33 кПа на глубине 100-200 см, х10 %",
        
        'SG_wv1500_0-5cm_mean_1000': "SoilGrid: Объемная влажность при 1500 кПа на глубине 0-5 см, х10 %",
        'SG_wv1500_5-15cm_mean_1000': "SoilGrid: Объемная влажность при 1500 кПа на глубине 5-15 см, х10 %",
        'SG_wv1500_15-30cm_mean_1000': "SoilGrid: Объемная влажность при 1500 кПа на глубине 15-30 см, х10 %",
        'SG_wv1500_30-60cm_mean_1000': "SoilGrid: Объемная влажность при 1500 кПа на глубине 30-60 см, х10 %",
        'SG_wv1500_60-100cm_mean_1000': "SoilGrid: Объемная влажность при 1500 кПа на глубине 60-100 см, х10 %",
        'SG_wv1500_100-200cm_mean_1000': "SoilGrid: Объемная влажность при 1500 кПа на глубине 100-200 см, х10 %",
    }
    
    
    # --- Статистика и гистограмма для данных наблюдений (data) ---
    stats_data = get_predictor_stats(data)
    
    # Определяем диапазон бинов.
    # Сначала убедимся, что data_full имеет хотя бы данные для расчета диапазона,
    # если data пустая.
    if data.size > 0:
        # Если data есть, используем ее диапазон для всех гистограмм
        counts_data, bin_edges_data = np.histogram(data, bins=bins_num)
        bins_range = (bin_edges_data.min(), bin_edges_data.max())
    elif data_full.size > 0:
        # Если data пустая, но data_full есть, используем диапазон data_full
        counts_full_temp, bin_edges_full_temp = np.histogram(data_full, bins=bins_num)
        bins_range = (bin_edges_full_temp.min(), bin_edges_full_temp.max())
    else:
        # Если оба массива пусты, не можем построить гистограмму.
        # Можно вывести сообщение или просто выйти.
        ax.set_title("Нет данных для отображения")
        return
    
    # --- Расчет counts и нормализация для data ---
    if data.size > 0:
        counts_data, _ = np.histogram(data, bins=bins_num, range=bins_range) # Используем общий bins_range
        max_count_data = counts_data.max() if counts_data.size > 0 else 0
        normalized_counts_data = counts_data / max_count_data if max_count_data > 0 else np.zeros_like(counts_data)
    else:
        normalized_counts_data = np.zeros(bins_num) # Если data пустая, все нормализованные counts = 0
        bin_edges_data = np.array([]) # Пустые границы, если нет данных

    # --- Расчет counts и нормализация для data_full ---
    counts_full, bin_edges_full = np.histogram(data_full, bins=bins_num, range=bins_range) # Используем общий bins_range
    max_count_full = counts_full.max() if counts_full.size > 0 else 0
    normalized_counts_full = counts_full / max_count_full if max_count_full > 0 else np.zeros_like(counts_full)
    
    # --- Определение общей максимальной высоты для графика ---
    total_y_max_normalized = 1.0 # Максимальное значение после нормализации

    plt.style.use('seaborn-v0_8-whitegrid') # Используйте подходящий стиль

    # --- Рисуем гистограмму для всего слоя (data_full) с помощью ax.bar ---
    # Мы рисуем столбец для каждого бина. x - левая граница бина, height - нормализованная частота.
    # Ширина бина = bin_edges_full[1] - bin_edges_full[0]
    # Важно, чтобы bin_edges_full содержали хотя бы 2 элемента, чтобы вычислить ширину.
    if len(bin_edges_full) > 1:
        bin_width_full = bin_edges_full[1] - bin_edges_full[0]
    else:
        # Если бинов очень мало или нет (из-за очень узкого диапазона или пустых данных)
        bin_width_full = 1 # Или другое значение по умолчанию, может потребовать настройки
    
    # Используем bin_edges_full[:-1] как x-координаты (левые границы бинов)
    ax.bar(bin_edges_full[:-1], normalized_counts_full, width=bin_width_full, align='edge',
           color='grey', edgecolor='black', alpha=0.3, label='Распределение всего слоя')

    # --- Рисуем гистограмму для данных наблюдений (data) с помощью ax.bar ---
    if len(bin_edges_data) > 1:
        bin_width_data = bin_edges_data[1] - bin_edges_data[0]
    else:
        bin_width_data = bin_width_full # Используем ту же ширину, что и для data_full
    
    ax.bar(bin_edges_data[:-1], normalized_counts_data, width=bin_width_data, align='edge',
           color='skyblue', edgecolor='black', alpha=0.7, label='Частота наблюдений')

    # --- Добавляем линии для основных статистик (для data) ---
    # Линии рисуем по исходным данным (data), а не по нормализованным counts.
    ax.axvline(stats_data['mean'], color='red', linestyle='dashed', linewidth=1.5, label=f'Среднее ({format_float(stats_data["mean"])})')
    ax.axvline(stats_data['median'], color='green', linestyle='dashed', linewidth=1.5, label=f'Медиана ({format_float(stats_data["median"])})')
    ax.axvline(stats_data['p5'], color='orange', linestyle='dotted', linewidth=1)
    ax.axvline(stats_data['p95'], color='orange', linestyle='dotted', linewidth=1, label='5%/95%')
    
    # --- Заголовок и подписи осей ---
    axtitle = f'Значения для предиктора: {band_name}'
    if (title != ''):
        axtitle = axtitle + "\n" + title
    ax.set_title(axtitle, fontsize=12, fontweight='bold')
    ax.set_ylabel('Нормализованная Частота', fontsize=10)
    
    xlabel_text = bio_info.get(band_name, f'Значения {band_name}')
    ax.set_xlabel(xlabel_text, fontsize=7)
    
    # --- Улучшаем внешний вид осей ---
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(axis='y', alpha=0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.2)
    
    # --- Добавляем информацию о статистике (для data) ---
    y_pos_for_stats_data_adjusted = 0.95
    add_stats_to_plot(ax, stats_data, y_pos=y_pos_for_stats_data_adjusted, y_max_overall=total_y_max_normalized)
    
    # --- Легенда ---
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    # --- Устанавливаем пределы оси Y ---
    ax.set_ylim(0, total_y_max_normalized * 1.4)


def add_stats_to_plot(ax: plt.Axes, stats: dict, y_pos: float, y_max_overall: float = None):
    """
    Добавляет статистическую информацию на оси гистограммы.
    Учитывает y_max_overall для корректного позиционирования текста,
    когда используется transform=ax.transAxes.

    Args:
        ax (plt.Axes): Объект осей Matplotlib.
        stats (dict): Словарь со статистиками ('min', 'p5', 'median', 'mean', 'p95', 'max').
        y_pos (float): Относительная позиция текста по оси Y (доля от 0 до 1)
                       при использовании transform=ax.transAxes.
                       Это значение теперь используется напрямую для позиционирования.
        y_max_overall (float, optional): Общее максимальное значение частоты для обеих гистограмм.
                                          Не используется напрямую для расчета позиции,
                                          но может быть полезен для понимания контекста.
                                          Оставляем для совместимости с сигнатурой вызова.
    """

    # Проверяем, что словарь stats содержит все ожидаемые ключи
    required_keys = ['min', 'p5', 'median', 'mean', 'p95', 'max']
    for key in required_keys:
        if key not in stats:
            stats[key] = np.nan # Устанавливаем NaN, если ключ отсутствует

    stats_text = (
        f"  Min: {format_float(stats.get('min'))}\n" # Используем .get() для безопасности
        f"  P5:  {format_float(stats.get('p5'))}\n"
        f"  Med: {format_float(stats.get('median'))}\n"
        f"  Mean:{format_float(stats.get('mean'))}\n"
        f"  P95: {format_float(stats.get('p95'))}\n"
        f"  Max: {format_float(stats.get('max'))}"
    )

    # Помещаем текст справа от гистограммы.
    # transform=ax.transAxes означает, что координаты (1.02, y_pos)
    # интерпретируются как доли от ширины и высоты самих осей.
    # x=1.02: 1.0 означает правую границу оси X, 0.02 - небольшой отступ вправо.
    # y_pos: Это значение (от 0 до 1) определяет вертикальное положение текста.
    #        0 - нижний край осей, 1 - верхний край осей.
    #        Текущий вызов `add_stats_to_plot(ax, stats_data, y_pos_for_stats_data_adjusted, y_max_overall=total_y_max)`
    #        где y_pos_for_stats_data_adjusted, похоже, рассчитано как доля от высоты гистограммы `data`.
    #        Если y_pos_for_stats_data_adjusted уже рассчитан как подходящая доля (например, 0.8 или 0.9),
    #        то его можно использовать напрямую.
    #        Если `y_pos` в вашей исходной функции `add_stats_to_plot` была, например, 0.8,
    #        и вы хотите, чтобы текст был в верхней части графика, то `y_pos` должно быть близко к 1.
    #        Следовательно, `y_pos_for_stats_data_adjusted` должно быть установлено соответственно.

    # Важно: Если `y_pos` в вызове `add_stats_to_plot` (это `y_pos_for_stats_data_adjusted`)
    # было рассчитано как относительное положение ДО использования y_max_overall,
    # и оно было, например, 0.8 (80% от высоты гистограммы data),
    # то при использовании `transform=ax.transAxes`, это значение будет применено
    # к общей высоте осей.
    # Если вы хотите, чтобы текст всегда был вверху, `y_pos` должна быть близка к 1.
    # Если `y_pos_for_stats_data_adjusted` уже соответствует желаемому положению (например, 0.95),
    # то оно должно работать.
    # Если `y_max_overall` был нужен для расчета `y_pos_for_stats_data_adjusted` в
    # `create_beautiful_histogram`, то его там следует использовать.
    # В данной функции `add_stats_to_plot`, `y_pos` напрямую используется для Y-координаты.

    ax.text(1.02, y_pos, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))