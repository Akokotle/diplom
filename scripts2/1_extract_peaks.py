import argparse
import glob
import os
import re

# import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from config import (
    CONDITIONS,
    DATA_ROOT,
    EPOCH_BASELINE,
    EPOCH_EVENT_ID,
    EPOCH_T_MAX,
    EPOCH_T_MIN,
    PSD_FMAX,
    PSD_FMIN,
    PSD_METHOD,
    PSD_VERBOSE,
    SGM_APERIODIC_MODE,
    SGM_MAX_N_PEAKS,
    SGM_MIN_PEAK_HEIGHT,
    SGM_PEAK_THRESHOLD,
    SGM_PEAK_WIDTH_LIMITS,
    SGM_PERIODIC_MODE,
    SGM_VERBOSE,
    SUBJECT_DIR,
    get_base_results_dir,
    get_psd_data_dir,
)
from specparam import SpectralGroupModel

# warnings.filterwarnings("ignore", message=".*expanding outside the data range.*")
# warnings.filterwarnings("ignore", message=".*'boundary' events.*")
# mne.set_log_level('WARNING')

matplotlib.use('Agg')

def fit_model_with_metrics(
    psds: np.ndarray, 
    freqs: np.ndarray, 
    calc_metrics: bool = False
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:

    """Параметризует спектр (разделяет на периодический и апериодический компоненты) и извлекает пики. Считает метрики.

    Args:
        psds (np.ndarray): Массив спектральной плотности мощности (3D: эпохи x каналы x частоты).
        freqs (np.ndarray): Одномерный массив частот.
        calc_metrics (bool, optional): Флаг для включения подсчета метрик качества (R-squared, RMSE).
            По умолчанию False. Не работает при использовании многопоточности (n_jobs > 1) в SpectralGroupModel.

    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]: Кортеж, содержащий:
            - 3D массив выделенных периодических компонентов (пиков) размерности (эпохи x каналы x частоты).
            - Список словарей с метриками качества подгонки (пустой список, если calc_metrics=False).
    """

    n_epochs, n_channels, n_freqs = psds.shape
    psds_2d = psds.reshape(-1, n_freqs)

    fg = SpectralGroupModel(
        peak_width_limits=SGM_PEAK_WIDTH_LIMITS, 
        max_n_peaks=SGM_MAX_N_PEAKS, 
        min_peak_height=SGM_MIN_PEAK_HEIGHT,
        peak_threshold=SGM_PEAK_THRESHOLD, 
        aperiodic_mode=SGM_APERIODIC_MODE, 
        periodic_mode=SGM_PERIODIC_MODE,
        verbose=SGM_VERBOSE
    )

    n_jobs_val = 1 if calc_metrics else -1
    fg.fit(freqs, psds_2d, n_jobs=n_jobs_val)

    peaks_list = []

    for i in range(fg.data.n_spectra):
        model_obj = fg.get_model(i)
        peaks_list.append(model_obj.results.model.get_component('peak'))

    all_peaks = np.array(peaks_list).reshape(n_epochs, n_channels, n_freqs)

    if calc_metrics:
        rmse = fg.get_metrics('error')
        rsquared = fg.get_metrics('gof')
        
        epoch_indices = np.repeat(np.arange(n_epochs), n_channels)
        channel_indices = np.tile(np.arange(n_channels), n_epochs)

        df_metrics_batch = pd.DataFrame({
            'epoch_idx': epoch_indices,
            'channel_idx': channel_indices,
            'rsquared': rsquared,
            'rmse': rmse
        })
        metrics_dict = df_metrics_batch.to_dict(orient='records')
    else:
        metrics_dict = []

    return all_peaks, metrics_dict


def psds_from_file(file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[mne.Info]]:

    """Загружает данные ЭЭГ из файла, нарезает их на эпохи и вычисляет спектральную мощность (PSD).

    Args:
        file_path (str): Путь к файлу данных.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[mne.Info]]: Кортеж, содержащий:
            - 3D массив спектров (эпохи x каналы x частоты) (или None при ошибке/отсутствии эпох).
            - 1D массив частот (или None при ошибке).
            - Объект mne.Info с метаданными записи (или None при ошибке).
    """

    try:
        if file_path.endswith('.set'):
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            events, _ = mne.events_from_annotations(raw)

            epochs = mne.Epochs(
                raw,
                events=events,
                event_id=EPOCH_EVENT_ID,
                tmin=EPOCH_T_MIN,
                tmax=EPOCH_T_MAX,
                preload=True,
                baseline=EPOCH_BASELINE,
            )
        elif file_path.endswith('.fif'):
            epochs = mne.read_epochs(file_path, preload=True)
        else:
            print(f"🟥 Неподдерживаемый формат файла: {file_path}")
            return None, None, None

        if len(epochs) == 0:
            return None, None, None

        epo_spectrum = epochs.compute_psd(
            method=PSD_METHOD,
            fmin=PSD_FMIN,
            fmax=PSD_FMAX,
            verbose=PSD_VERBOSE,
        )
        psds, freqs = epo_spectrum.get_data(return_freqs=True)

        return psds, freqs, epochs.info

    except Exception as e:
        print(f"🟥 Ошибка процессинга файла {os.path.basename(file_path)}: {e}.")
        return None, None, None


def plot_metrics_heatmaps(df_metrics: pd.DataFrame, output_dir: str) -> None:

    """Группирует метрики по каналам и строит хитмапы.

    Args:
        df_metrics (pd.DataFrame): Массив метрик.
        output_dir (str): Директория для сохранения хитмапов.
    """

    plots_dir = os.path.join(output_dir, "metrics_heatmaps")
    os.makedirs(plots_dir, exist_ok=True)
    
    metrics_to_plot = ['rsquared', 'rmse']
    conditions = df_metrics['condition'].unique()
    
    print("\n⏳ Построение хитмапов метрик...")
    
    for metric in metrics_to_plot:
        for cond in conditions:
            df_cond = df_metrics[df_metrics['condition'] == cond]
            df_agg = df_cond.groupby(['channel', 'subject'])[metric].mean().reset_index()
            df_pivot = df_agg.pivot(index='subject', columns='channel', values=metric)
                        
            fig, ax = plt.subplots(figsize=(10, 6))
            
            cmap_choice = 'viridis' if metric == 'rsquared' else 'plasma'
            im = ax.imshow(df_pivot.values, aspect='auto', cmap=cmap_choice)
            
            ax.set_xticks(np.arange(len(df_pivot.columns)))
            ax.set_xticklabels(df_pivot.columns, rotation=45, ha='right')
            ax.set_yticks(np.arange(len(df_pivot.index)))
            ax.set_yticklabels(df_pivot.index)
            
            ax.set_title(f"{metric.upper()}\n{cond}", fontsize=16, pad=15)
            ax.set_xlabel('Каналы', fontsize=12)
            ax.set_ylabel('Субъекты', fontsize=12)
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(metric.upper(), rotation=270, labelpad=15)
            
            fig.tight_layout()
            
            safe_cond = str(cond).replace('/', '_').replace(' ', '_')
            filename = os.path.join(plots_dir, f"heatmap_{metric}_{safe_cond}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f" 🟩 Хитмап для {metric.upper()} ({cond}) сохранен.")


def process_and_fit_all_subjects(
    data_root: str, 
    base_output_dir: str, 
    calc_metrics: bool = False
) -> pd.DataFrame:

    """Обрабатывает файлы всех субъектов и извлекает спектральные пики.

    Сканирует директорию с данными, проверяет наличие необходимых условий для каждого субъекта,
    нарезает данные на эпохи, проводит спектральную параметризацию и сохраняет результаты в сжатые архивы .npz.

    Args:
        data_root (str): Путь к корневой директории, содержащей папки субъектов.
        base_output_dir (str): путь к базовой директории для сохранения итоговых файлов (.npz) и метрик.
        calc_metrics (bool, optional): Флаг для включения подсчета метрик качества (R-squared, RMSE). 
            По умолчанию False.

    Returns:
        pd.DataFrame: Таблица pandas с метриками качества для всех эпох и каналов. 
            Если calc_metrics=False, возвращается пустой DataFrame.
    """

    subject_dir = SUBJECT_DIR
    conditions = CONDITIONS

    psd_output_dir = get_psd_data_dir(base_output_dir)
    os.makedirs(psd_output_dir, exist_ok=True)

    if len(subject_dir) == 1 and subject_dir[0] == "":
        subject_dir = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith("sub-")]

    global_metrics_list = []

    for subj_dir_name in sorted(subject_dir):
        subj_match = re.search(r"(sub-\d+)", subj_dir_name)
        if not subj_match:
            continue
        
        subj_id = subj_match.group(0)
        subj_path = os.path.join(data_root, subj_dir_name)
        
        missing_conditions = []
        files_by_condition = {}

        for condition in conditions:
            search_patterns = [
                os.path.join(subj_path, "ses-*", f"{subj_id}_{condition}_*_eeg.set"),
                os.path.join(subj_path, "ses-*", f"{subj_id}_{condition}_*_epo.fif"),
                os.path.join(subj_path, "ses-*", "eeg", f"{subj_id}_{condition}_*_eeg.set"),
                os.path.join(subj_path, "ses-*", "eeg", f"{subj_id}_{condition}_*_epo.fif")
            ]
            
            files = []
            for p in search_patterns:
                files.extend(glob.glob(p))
            
            files = sorted(list(set(files)))
            
            if not files:
                missing_conditions.append(condition)
            else:
                files_by_condition[condition] = files
        
        if missing_conditions:
            print(f"⚠️ Пропуск {subj_id}, нет данных для {missing_conditions}.")
            continue

        print(f"\n⏳ Работа с {subj_id}.")

        subject_data_storage = []
        subject_info = None

        for condition, files in files_by_condition.items():
            for file_path in files:
                run_match = re.search(r"_run-(\d+)", file_path)
                run_id = f"run-{run_match.group(1)}" if run_match else "run-NA"

                raw_psds, freqs, info = psds_from_file(file_path)
                
                if info is not None and subject_info is None:
                    subject_info = info

                if raw_psds is not None:
                    peaks, metrics = fit_model_with_metrics(raw_psds, freqs, calc_metrics=calc_metrics)
                    
                    if calc_metrics and metrics:
                        ch_names = info.ch_names if info is not None else []
                        for m in metrics:
                            ch_idx = m['channel_idx']
                            ch_name = ch_names[ch_idx] if ch_idx < len(ch_names) else f"Ch_{ch_idx}"
                            
                            m.update({
                                'subject': subj_id, 
                                'condition': condition, 
                                'run': run_id,
                                'channel': ch_name
                            })
                        global_metrics_list.extend(metrics)

                    subject_data_storage.append({
                        'peaks': peaks,
                        'freqs': freqs,
                        'condition': condition,
                        'run': run_id,
                        'channels': info.ch_names if info is not None else [] 
                    })

        if subject_data_storage:
            save_path = os.path.join(psd_output_dir, f"{subj_id}_spectral_results.npz")
            np.savez_compressed(save_path, data=subject_data_storage)

            if subject_info is not None:
                info_save_path = os.path.join(psd_output_dir, f"{subj_id}_info.fif")
                mne.io.write_info(info_save_path, subject_info)
                
            print(f"🟩 {subj_id} Готово.")

    if calc_metrics and global_metrics_list:
        df_all_metrics = pd.DataFrame(global_metrics_list)
        df_all_metrics.to_csv(os.path.join(psd_output_dir, "fit_quality_report.csv"), index=False)
        print(f"\n🟩 Вычисления завершены. CSV таблица метрик сохранена в {psd_output_dir}.")
        plot_metrics_heatmaps(df_all_metrics, base_output_dir)
        return df_all_metrics
    elif not calc_metrics:
        print(f"\n🟩 Завершено (подсчет метрик отключен). Данные сохранены в {psd_output_dir}.")
        return pd.DataFrame()
    else:
        print("\n🟥 Нет данных для условий.")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Выделяет периодику из данных ЭЭГ.")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--base_output_dir", type=str, default=get_base_results_dir())

    args = parser.parse_args()

    process_and_fit_all_subjects(
        args.data_root, 
        args.base_output_dir,
        calc_metrics=True
    )