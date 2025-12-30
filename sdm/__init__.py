# sdm/__init__.py

from sdm.core.modeling import run_sdm
from sdm.core.data_loading import load_occurrences
from sdm.core.preprocessing import clip_rasters
from sdm.core.utils.plot_utils import draw_map, create_beautiful_histogram

__version__ = "0.5.5" # Указываем версию библиотеки