import inspect
import os
from types import FrameType
from typing import Optional


def get_base_dir() -> str:
    """

    Returns:
        str: Корневая директория проекта.
    """
    frame: Optional[FrameType] = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Failed to get the current frame.")
    current_script_path = inspect.getfile(frame)
    scripts_dir = os.path.dirname(os.path.abspath(current_script_path))
    return os.path.dirname(scripts_dir)


def get_base_results_dir() -> str:
    """

    Returns:
        str: Путь к директории для сохранения результатов.
    """
    return os.path.join(get_base_dir(), "results")


def get_psd_data_dir(base_dir: Optional[str] = None) -> str:
    """

    Args:
        base_dir (Optional[str], optional): Корневая директория проекта. Defaults to None.

    Returns:
        str: Путь к директории PSD данных результатов.
    """
    if base_dir is None:
        base_dir = get_base_results_dir()
    return os.path.join(base_dir, "psd_data_2")


# ОПРЕДЕЛЕНИЕ ПАРАМЕТРОВ

# БАЗОВЫЕ ПАРАМЕТРЫ

CONDITIONS = ["pre", "post", "follow"]  # pre, MI-SES, MI-IES, post, follow
SUBJECT_DIR = [""]  # [""] для использования всех, (sub-01, ..., sub-27)
PRELOAD = True

DATA_ROOT = os.path.join(get_base_results_dir(), "PEEG")

# ПАРАМЕТРЫ СОЗДАНИЕ ЭПОХ

EPOCH_EVENT_ID = {"2": 2}
EPOCH_T_MIN = -3
EPOCH_T_MAX = 7
EPOCH_BASELINE = (None, 0)


# ПАРАМЕТРЫ ВЫЧИСЛЕНИЯ PSD

PSD_METHOD = "multitaper"
PSD_FMIN = 3
PSD_FMAX = 35
PSD_VERBOSE=False

# ПАРАМЕТРЫ UMAP

FREQ_BANDS = {
    "ALL": (3, 35),
    "THETA": (4, 7),
    "ALPHA": (9, 13),
    "BETA": (14, 35),
}
DR_FREQ_BAND = "ALL"  # "ALL", "THETA", "ALPHA", "BETA"

UMAP_N_COMPONENTS = 15
UMAP_N_NEIGHBORS = 50
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine'

# ПАРАМЕТРЫ PCA

PCA_N_COMPONENTS = 3

# ПАРАМЕТРЫ SPECTRALGROUPMODEL (SGM)

SGM_PEAK_WIDTH_LIMITS=[2, 12]
SGM_MAX_N_PEAKS=6
SGM_MIN_PEAK_HEIGHT=1
SGM_PEAK_THRESHOLD=0.1
SGM_APERIODIC_MODE='fixed'
SGM_PERIODIC_MODE='gaussian'
SGM_VERBOSE=False