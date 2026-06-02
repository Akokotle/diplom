import argparse
import glob
import os
import re
from typing import Optional, Tuple

import mne
import numpy as np
from config import (
    CONDITIONS,
    SUBJECT_DIR,
    get_base_results_dir,
)
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


def extract_riemann_features(
    file_path: str,
    fmin: float,
    fmax: float
) -> Tuple[Optional[np.ndarray], Optional[mne.Info]]:
    """Загружает данные ЭЭГ, фильтрует и переводит в касательное пространство Римана.

    Args:
        file_path (str): Путь к файлу данных.
        fmin (float): Нижняя граница полосового фильтра.
        fmax (float): Верхняя граница полосового фильтра.

    Returns:
        Tuple[Optional[np.ndarray], Optional[mne.Info]]: Кортеж, содержащий:
            - 2D массив признаков (эпохи x римановы признаки) (или None при ошибке).
            - Объект mne.Info с метаданными записи (или None при ошибке).
    """
    try:
        if file_path.endswith('.fif'):
            epochs = mne.read_epochs(file_path, preload=True)
        else:
            print(f"🟥 Неподдерживаемый формат файла: {file_path}")
            return None, None

        if len(epochs) == 0:
            return None, None

        epochs = epochs.filter(l_freq=fmin, h_freq=fmax, verbose=False)
        X = epochs.get_data(copy=False)

        cov = Covariances(estimator='oas').fit_transform(X)

        ts = TangentSpace(metric='riemann').fit(cov)
        feats = ts.transform(cov)

        return feats, epochs.info

    except Exception as e:
        print(f"🟥 Ошибка процессинга файла {os.path.basename(file_path)}: {e}.")
        return None, None


def process_all_subjects(
    data_root: str, 
    base_output_dir: str, 
    fmin: float,
    fmax: float
) -> None:
    """Обрабатывает файлы всех субъектов и сохраняет римановы признаки в .npz.

    Args:
        data_root (str): Путь к корневой директории, содержащей папки субъектов.
        base_output_dir (str): путь к базовой директории для сохранения итоговых файлов (.npz) и метрик.
        fmin (float): Нижняя граница полосового фильтра.
        fmax (float): Верхняя граница полосового фильтра.
    """
    subject_dir = SUBJECT_DIR
    conditions = CONDITIONS

    riemann_output_dir = os.path.join(base_output_dir, "riemann_features")
    os.makedirs(riemann_output_dir, exist_ok=True)

    if len(subject_dir) == 1 and subject_dir[0] == "":
        subject_dir = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith("sub-")]

    for subj_dir_name in sorted(subject_dir):
        subj_match = re.search(r"(sub-\d+)", subj_dir_name)
        if not subj_match:
            continue
        
        subj_id = subj_match.group(0)
        subj_path = os.path.join(data_root, subj_dir_name)
        
        missing_conditions = []
        files_by_condition = {}

        for condition in conditions:
            search_patterns = os.path.join(subj_path, "ses-*", f"{subj_id}_{condition}_*_epo.fif")
            files = sorted(list(set(glob.glob(search_patterns))))
            
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

                feats, info = extract_riemann_features(file_path, fmin=fmin, fmax=fmax)
                
                if info is not None and subject_info is None:
                    subject_info = info

                if feats is not None:
                    subject_data_storage.append({
                        'features': feats,
                        'condition': condition,
                        'run': run_id,
                        'channels': info.ch_names if info is not None else [] 
                    })

        if subject_data_storage:
            save_path = os.path.join(riemann_output_dir, f"{subj_id}_riemann_results.npz")
            np.savez_compressed(save_path, data=subject_data_storage)

            if subject_info is not None:
                info_save_path = os.path.join(riemann_output_dir, f"{subj_id}_info.fif")
                mne.io.write_info(info_save_path, subject_info)
                
            print(f"🟩 {subj_id} Готово.")

    print(f"\n🟩 Вычисления завершены. Данные сохранены в {riemann_output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Извлекает признаки в касательном пространстве Римана.")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--base_output_dir", type=str, default=get_base_results_dir())
    parser.add_argument("--fmin", type=float, default=8.0, help="Нижняя граница фильтра")
    parser.add_argument("--fmax", type=float, default=13.0, help="Верхняя граница фильтра")

    args = parser.parse_args()

    process_all_subjects(
        args.data_root, 
        args.base_output_dir,
        fmin=args.fmin,
        fmax=args.fmax
    )