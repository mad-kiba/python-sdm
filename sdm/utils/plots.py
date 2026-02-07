import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling, transform as rasterio_transform
from rasterio.transform import array_bounds
import contextily as ctx
from pyproj import CRS, Transformer
from matplotlib.ticker import FuncFormatter, MaxNLocator
from shapely.geometry import Point
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon
import warnings
import osmnx as ox

from .utils import get_predictor_stats, format_float, calculate_histogram_similarity
from .utils import read_and_to_3857, round_to_significant_figures, wrap_long_lines




def create_avi_from_images(image_paths, output_mp4_path='output.mp4', fps=1):
    """
    Создает видеофайл AVI из массива путей к четырем изображениям.

    Args:
        image_paths (list): Список из четырех строк, где каждая строка -
                            путь к одному изображению.
        output_mp4_path (str): Путь, по которому будет сохранен выходной AVI файл.
                               По умолчанию 'output.avi'.
        fps (int): Количество кадров в секунду для выходного видео.
                   По умолчанию 1 кадр в секунду.
    """
    # 1. Чтение изображений
    images = []
    for img_path in image_paths:
        # Проверяем существование файла
        #print(img_path)
        if not os.path.exists(img_path):
            print(f"Файл не найден: {img_path}")
            raise FileNotFoundError(f"Файл не найден: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Не удалось прочитать изображение: {img_path}. Возможно, файл поврежден или имеет неверный формат.")
            raise IOError(f"Не удалось прочитать изображение: {img_path}. Возможно, файл поврежден или имеет неверный формат.")
        images.append(img)
    
    # 2. Определение размеров видео (все изображения должны иметь одинаковый размер)
    height, width, layers = images[0].shape
    
    for i, img in enumerate(images[1:]):
        h, w, _ = img.shape
        if h != height or w != width:
            raise ValueError(f"Изображения должны иметь одинаковый размер. "
                             f"Изображение {image_paths[0]} имеет размер {width}x{height}, "
                             f"а изображение {image_paths[i+1]} имеет размер {w}x{h}.")
    
    # 3. Определение кодека и создание объекта VideoWriter
    #    CV_FOURCC используется для указания кодека.
    #    'XVID' - один из популярных и хорошо поддерживаемых кодеков для AVI.
    #    Другие варианты: 'MJPG', 'DIVX', 'H264' (может требовать установки кодеков).
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Или 'avc1'
    video = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
    
    # 4. Запись каждого изображения в видео
    for img in images:
        video.write(img)
        
    video.release() 

    print(f"Видеофайл успешно создан: {output_mp4_path}")
    

def create_animated_gif(image_paths, output_path="animation.gif", duration=500):
    """
    Создает анимированный GIF из списка путей к изображениям.

    Args:
        image_paths (list): Список строк, где каждая строка — путь к файлу изображения.
        output_path (str): Путь для сохранения выходного GIF файла.
        duration (int): Время отображения каждого кадра в миллисекундах.
                        (Например, 500ms = 0.5 секунды на кадр).
    """
    #print('Файлы в GIF:')
    #print(image_paths)
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


def plot_geotiff_with_osm(geotiff_path: str, output_path: str, mean: float, scale: float, band: str, bio_info):
    """
    Строит график значений GeoTIFF, накладывая поверх него контекстную карту OSM.
    График отрисовывается в проекции Web Mercator (EPSG:3857) для корректного
    отображения OSM без искажений. Размер фигуры будет динамически подстроен
    под соотношение сторон области данных, а оси будут показывать
    географические координаты (градусы).

    Args:
        geotiff_path (str): Путь к входному файлу GeoTIFF.
        output_path (str): Путь к выходному файлу изображения (например, 'output.png').
        mean (float): Значение среднего для обратного преобразования шкалы.
        scale (float): Значение масштаба (стандартного отклонения) для обратного преобразования шкалы.
                       (Оригинальное значение = масштабированное значение * scale + mean)
    """

    print(f"Загрузка GeoTIFF: {geotiff_path}")
    with rasterio.open(geotiff_path) as src:
        # 1. Чтение и репроекция GeoTIFF данных в EPSG:3857 (Web Mercator)
        # Это позволяет базовой карте OSM отображаться без искажений.
        data_src = src.read(1)
        data_src = data_src.astype(np.float64)
        data_src[data_src == src.nodata] = np.nan
        target_crs = 'EPSG:3857'
        
        if src.crs == target_crs:
            reprojected_data = data_src
            dst_transform = src.transform
            width = src.width
            height = src.height
        else:
            # Вычисляем параметры для репроекции в Web Mercator
            out_transform, out_width, out_height = rasterio.warp.calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            reprojected_data = np.empty((out_height, out_width), dtype=src.dtypes[0])
            reproject(
                source=data_src,
                destination=reprojected_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest # Можно использовать bilinear, cubic и т.д.
            )
            dst_transform = out_transform
            width = out_width
            height = out_height
        # Вычисляем границы для imshow после репроекции (они теперь будут в метрах EPSG:3857)
        left_m, bottom_m, right_m, top_m = rasterio.transform.array_bounds(height, width, dst_transform)
        extent_m = [left_m, right_m, bottom_m, top_m]

        # 2. Динамический расчет figsize и dpi на основе границ в метрах Web Mercator
        # Эта логика взята из вашего примера.
        width_v_meters = right_m - left_m
        height_v_meters = top_m - bottom_m
        
        # Защита от деления на ноль, если область данных очень узкая или плоская
        if height_v_meters == 0: 
            height_v_meters = 1.0 # Присваиваем небольшое значение
        if width_v_meters == 0:
            width_v_meters = 1.0

        # Соотношение сторон области данных в метрах Web Mercator
        ratio = width_v_meters / height_v_meters  
        
        # Длинная сторона в дюймах и целевой размер по длинной стороне (минимум 2000 пикселей)
        long_inches = 10.0 # Базовый размер для длинной стороны фигуры
        desired_long_px = 2000 # Целевое разрешение по длинной стороне в пикселях
        dpi = int(desired_long_px / long_inches) # Рассчитываем DPI
        
        if ratio >= 1.0:
            # Горизонтальная область данных: фиксируем ширину фигуры
            fig_w = long_inches
            fig_h = max(long_inches / ratio, 1e-3) # Не допускаем нулевой высоты
        else:
            # Вертикальная область данных: фиксируем высоту фигуры
            fig_h = long_inches
            fig_w = max(long_inches * ratio, 1e-3) # Не допускаем нулевой ширины
        
        print(f"Динамически рассчитанные figsize: ({fig_w:.2f}, {fig_h:.2f}), dpi: {dpi}")
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        # Устанавливаем лимиты осей в метрах (Web Mercator)
        ax.set_xlim(left_m, right_m)
        ax.set_ylim(bottom_m, top_m)

        # 3. Настройка пропорциональности карты для Web Mercator (EPSG:3857)
        # В проекции Web Mercator, для неискаженного отображения, достаточно 'equal'.
        # Поскольку figsize уже соответствует аспекту данных, 'adjustable' можно и не указывать,
        # но для большей надёжности оставим.
        print(f"Установка аспекта 'equal' для EPSG:3857.")
        ax.set_aspect('equal', adjustable='box')

        # 4. Наложение базовой карты OSM с помощью contextily
        print("Загрузка контекстной карты OSM (contextily)...")
        try:
            ctx.add_basemap(ax, crs=target_crs, source=ctx.providers.OpenStreetMap.Mapnik, attribution=False)
            print("Контекстная карта OSM добавлена без искажений.")
        except Exception as e:
            print(f"Не удалось добавить контекстную карту OSM: {e}. Продолжаем без карты.")

        # Отображение GeoTIFF данных
        im = ax.imshow(reprojected_data, cmap='bone', extent=extent_m, origin='upper', aspect='auto', alpha=0.7, zorder=1)
        
        # 5. Настройка шкалы значений
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
        
        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks) 
        original_values_ticks = ticks * scale + mean
        cbar.set_ticklabels([f'{val:.2f}' for val in original_values_ticks])
        
        cbar.set_label(f'Значение слоя (оригинальные единицы, mean={mean:.2f}, scale={scale:.2f})')

        # 6. Преобразование меток осей из EPSG:3857 (метры) в EPSG:4326 (градусы)
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True) 

        # X-ось (долгота)
        xticks = ax.get_xticks()
        valid_xticks = [t for t in xticks if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
        
        # Для преобразования долготы, широта не влияет, но pyproj требует оба аргумента.
        # Используем центральную широту области в метрах Web Mercator.
        center_y_m = (top_m + bottom_m) / 2
        lon_labels, _ = transformer.transform(valid_xticks, [center_y_m] * len(valid_xticks))
        ax.set_xticks(valid_xticks) 
        ax.set_xticklabels([f'{lon:.2f}°' for lon in lon_labels])

        # Y-ось (широта)
        yticks = ax.get_yticks()
        valid_yticks = [t for t in yticks if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]

        # Для преобразования широты, долгота не влияет.
        center_x_m = (left_m + right_m) / 2
        _, lat_labels = transformer.transform([center_x_m] * len(valid_yticks), valid_yticks)
        ax.set_yticks(valid_yticks) 
        ax.set_yticklabels([f'{lat:.2f}°' for lat in lat_labels])
        
        # Заголовки осей
        ax.set_xlabel('Долгота (°)')
        ax.set_ylabel('Широта (°)')
        pred_title = band
        if bio_info:
            pred_title = bio_info.get(band)['title']
        ax.set_title(pred_title, fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200) # quality=90 - не найден такой параметр
        plt.close(fig)
        print(f"График успешно сохранен в {output_path}")


def draw_map(OUTPUT_SUITABILITY_TIF, OUTPUT_SUITABILITY_JPG, title = '', rows=[], cols=[], map_only=0):
    data, transform, width, height = read_and_to_3857(OUTPUT_SUITABILITY_TIF)
    
    # если наблюдений меньше пяти и моделирования не было, рисуем только точки
    if map_only==1:
        data = data * 0
        
    #print('---')
    #print(data)
    #print('---')
    
    # Границы растра в координатах EPSG:3857
    xmin, ymin, xmax, ymax = array_bounds(height, width, transform)
    
    # Поля: одинаковая ширина в метрах, но не менее 5% от каждой стороны
    pad_x_req = (xmax - xmin) * 0.05
    pad_y_req = (ymax - ymin) * 0.05
    pad_m = max(pad_x_req, pad_y_req)
    pad_m = 0 # попробуем без паддингов?
    
    xmin_v, xmax_v = xmin - pad_m, xmax + pad_m
    ymin_v, ymax_v = ymin - pad_m, ymax + pad_m
    
    # Колормэп с прозрачностью по NaN (нет данных)
    #cmap = plt.cm.Blues.copy()
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
    # print('Сохранение карты: '+OUTPUT_SUITABILITY_JPG)
    # Сохранение с высоким разрешением (длина ≥ 2000 px)
    plt.savefig(
        OUTPUT_SUITABILITY_JPG,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    
    plt.close(fig)
    
    
def create_beautiful_histogram(ax: plt.Axes, data: np.ndarray, band_name: str, bins_num: int, data_full: np.ndarray, bio_info, title = ''):
    """
    Рисует красивую и информативную гистограмму на заданных осях.
    """
    
    data = data[~np.isnan(data)]
    data_full = data_full[~np.isnan(data_full)]
    
    try:
        # --- Статистика и гистограмма для данных наблюдений (data) ---
        stats_data = get_predictor_stats(data)
        stats_predictor = get_predictor_stats(data_full)
        stats_data['similarity'] = calculate_histogram_similarity(data, data_full, bins_num)
    except Exception as e:
        print('Ошибка вычисления стат показателей диаграм:')
        print(e)
        
    #print(band_name)
    
    try:
        # Определяем диапазон бинов.
        # Сначала убедимся, что data_full имеет хотя бы данные для расчета диапазона,
        # если data пустая.
        if data.size > 0:
            # Если data есть, используем ее диапазон для всех гистограмм
            #counts_data, bin_edges_data = np.histogram(data, bins=bins_num)
            #bins_range = (bin_edges_data.min(), bin_edges_data.max())
            counts_full_temp, bin_edges_full_temp = np.histogram(data_full, bins=bins_num)
            counts_data, bin_edges_data = np.histogram(data_full, bins=bins_num)
            bins_range = (bin_edges_full_temp.min(), bin_edges_full_temp.max())
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
    except Exception as e:
        print('Ошибка вычисления bins_range')
        print(e)
    
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
    #print(10)
    ax.axvline(stats_data['mean'], color='red', linestyle='dashed', linewidth=1.5, label=f'Среднее ({format_float(stats_data["mean"])})')
    ax.axvline(stats_data['median'], color='green', linestyle='dashed', linewidth=1.5, label=f'Медиана ({format_float(stats_data["median"])})')
    ax.axvline(stats_data['p5'], color='orange', linestyle='dotted', linewidth=1)
    ax.axvline(stats_data['p95'], color='orange', linestyle='dotted', linewidth=1, label='5%/95%')
    #print(20)
    
    # --- Заголовок и подписи осей ---
    pred_title = bio_info.get(band_name)['title']
    axtitle = "Гистограмма отклика для предиктора:\n"+wrap_long_lines(pred_title, 100)
    if (title != ''):
        axtitle = axtitle + "\n" + title
    #ax.set_title(axtitle, fontsize=12, fontweight='bold')
    ax.set_title(axtitle, fontsize=9)
    ax.set_ylabel('Нормализованная Частота', fontsize=10)
    
    xlabel_text = pred_title
    ax.set_xlabel(xlabel_text, fontsize=7)
    
    # --- Улучшаем внешний вид осей ---
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(axis='y', alpha=0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.2)
    
    # --- Добавляем информацию о статистике (для data) ---
    y_pos_for_stats_data_adjusted = 0.95
    try:
        add_stats_to_plot(ax, stats_data, stats_predictor, y_pos=y_pos_for_stats_data_adjusted, y_max_overall=total_y_max_normalized)
    except Exception as e:
        print("Ошибка нанесения статистических признаков: " + str(e))
    
    # --- Легенда ---
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    # --- Устанавливаем пределы оси Y ---
    ax.set_ylim(0, total_y_max_normalized * 1.4)
    
    return stats_data, stats_predictor


def add_stats_to_plot(ax: plt.Axes, stats: dict, stats_pred: dict, y_pos: float, y_max_overall: float = None):
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
    required_keys = ['min', 'p5', 'median', 'mean', 'p95', 'max', 'std_dev', 'skewness', 'kurtosis', 'similarity']
    for key in required_keys:
        if key not in stats:
            stats[key] = np.nan # Устанавливаем NaN, если ключ отсутствует
            
    try:
        broad = round_to_significant_figures(stats.get('width_obs')/stats_pred.get('width_obs'), 2)
    except Exception as e:
        print("Ошибка вычисления относительной ширины: " + str(e))
        broad = 0
    
    broad = round(broad, 2)
    simil = round(stats.get('similarity'), 2)

    stats_text = (
        f"  Наблюдения:\n"
        f"  Min: {format_float(stats.get('min'))}\n" # Используем .get() для безопасности
        f"  P5:  {format_float(stats.get('p5'))}\n"
        f"  Med: {format_float(stats.get('median'))}\n"
        f"  Mean:{format_float(stats.get('mean'))}\n"
        f"  P95: {format_float(stats.get('p95'))}\n"
        f"  Max: {format_float(stats.get('max'))}\n"
        f"  StDev: {format_float(stats.get('std_dev'))}\n"
        f"  Ass: {format_float(stats.get('skewness'))}\n"
        f"  Exc: {format_float(stats.get('kurtosis'))}\n\n"
        
        f"  Весь слой:\n"
        f"  Min: {format_float(stats_pred.get('min'))}\n" # Используем .get() для безопасности
        f"  P5:  {format_float(stats_pred.get('p5'))}\n"
        f"  Med: {format_float(stats_pred.get('median'))}\n"
        f"  Mean:{format_float(stats_pred.get('mean'))}\n"
        f"  P95: {format_float(stats_pred.get('p95'))}\n"
        f"  Max: {format_float(stats_pred.get('max'))}\n"
        f"  StDev: {format_float(stats_pred.get('std_dev'))}\n"
        f"  Ass: {format_float(stats_pred.get('skewness'))}\n"
        f"  Exc: {format_float(stats_pred.get('kurtosis'))}\n\n"
        
        f"  Сравнение:\n"
        f"  Broad: {broad}\n"
        f"  Simil: {simil}"
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