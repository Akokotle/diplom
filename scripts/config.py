import inspect
import os
from types import FrameType
from typing import Optional


def get_base_dir() -> str:
    """

    Returns:
        str: root project directory (directory containing the "scripts" folder).
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
        str: path to the main folder for saving results (in the project root).
    """
    return os.path.join(get_base_dir(), "PSD_ANALYSIS_RESULTS")


def get_psd_data_dir(base_dir: Optional[str] = None) -> str:
    """

    Args:
        base_dir (Optional[str], optional): root project directory. Defaults to None.

    Returns:
        str: path to the folder for PSD data (.npz).
    """
    if base_dir is None:
        base_dir = get_base_results_dir()
    return os.path.join(base_dir, "PSD_DATA")


def get_psd_plots_dir(base_dir: Optional[str] = None) -> str:
    """

    Args:
        base_dir (Optional[str], optional): root project directory. Defaults to None.

    Returns:
        str: path to folder for 2D PSD graphics (.png).
    """
    if base_dir is None:
        base_dir = get_base_results_dir()
    return os.path.join(base_dir, "PSD_PLOTS")


def get_dr_plots_dir(base_dir: Optional[str] = None) -> str:
    """

    Args:
        base_dir (Optional[str], optional): root project directory. Defaults to None.

    Returns:
        str: path to the folder for dimension reduction graphs (UMAP/PCA, .html)
    """
    if base_dir is None:
        base_dir = get_base_results_dir()
    return os.path.join(base_dir, "DR_PLOTS")


# --- BASE CONSTANTS ---

CONDITIONS = ["pre", "post", "follow"]  # pre, MI-SES, MI-IES, post, follow
SUBJECT_DIR = [""]  # [""] to process all, (sub-01, ..., sub-27)
PRELOAD = True

# DIRECTORIES

DATA_ROOT = "/home/ilya/diplom/PEEG"

# EPOCH CREATING

EVENT_ID = {"2": 2}
T_MIN = -3
T_MAX = 7
BASELINE = (-3, 0)

# PSD PARAMETERS

PSD_METHOD = "multitaper"
FMIN_PSD = 3
FMAX_PSD = 35

# UMAP PARAMETERS

FREQ_BANDS = {
    "ALL": (3, 35),
    "THETA": (4, 7),
    "ALPHA": (9, 13),
    "BETA": (14, 35),
}
DR_FREQ_BAND = "ALL"  # "ALL", "THETA", "ALPHA", "BETA"

UMAP_N_COMPONENTS = 100
UMAP_N_NEIGHBORS = 20

# PCA PARAMETERS
PCA_N_COMPONENTS = 3

# PLOTS COLORS
colors_runs = {
    cond: c for cond, c in zip(CONDITIONS, ["#6A5ACD", "#3CB371", "#FF8C00"])
}
colors_mean = {
    cond: c for cond, c in zip(CONDITIONS, ["#483D8B", "#2E8B57", "#CC5500"])
}
