import argparse
import glob
import os

import mne
import numpy as np
from config import (
    DATA_ROOT,
    get_base_results_dir,
)
from mne.preprocessing import ICA
from mne_icalabel import label_components


def process_all_raw_subjects(data_root: str, base_output_dir: str):
    """Препроцессирует сырые данные ЭЭГ.

    Args:
        data_root (str): Путь к корневой директории с сырыми данными.
        base_output_dir (str): Путь к базовой директории для сохранения результатов.
    """
    output_dir = os.path.join(base_output_dir, 'PEEG2')
    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(data_root, 'sub-*', 'ses-*', '*_ori.set')
    file_list = glob.glob(search_pattern)

    print(f"Найдено файлов для обработки: {len(file_list)}")

    for file_path in file_list:
        print(f"\n{'='*50}\nНачало обработки: {file_path}")

        rel_dir = os.path.relpath(os.path.dirname(file_path), data_root)
        save_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(save_dir, exist_ok=True)

        basename = os.path.basename(file_path)
        parts = basename.replace('_ori.set', '').split('_')

        if len(parts) >= 3:
            sub = parts[0]
            run = parts[-1]
            cond = '_'.join(parts[1:-1])

            new_basename = f"{sub}_{cond}_{run}_epo.fif"
            save_path = os.path.join(save_dir, new_basename)

            try:

                raw = mne.io.read_raw_eeglab(file_path, preload=True)

                channels_to_drop = [
                    'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL',
                    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF8', 'F7', 'F8', 'FT7', 'FT8', 'M1',
                    'T7', 'T8', 'M2', 'TP7', 'TP8', 'P7', 'P8', 'PO7', 'PO8', 'CB1', 
                    'O1', 'Oz', 'O2'
                ] # M1, M2, CB1 нет в данных
                existing_drop_chs = [ch for ch in channels_to_drop if ch in raw.ch_names]
                if existing_drop_chs:
                    raw.drop_channels(existing_drop_chs)

                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, match_case=False, on_missing='ignore')

                raw.set_eeg_reference('average')

                spectrum = raw.compute_psd(method='welch', fmin=1.0, fmax=40.0)
                psds, freqs = spectrum.get_data(return_freqs=True)

                psds_uv = psds * 1e12
                psds_log = 10 * np.log10(psds_uv)
                avg_log_power = np.mean(psds_log, axis=1)

                mean_power_all_channels = np.mean(avg_log_power)
                std_power_all_channels = np.std(avg_log_power)

                z_scores = (avg_log_power - mean_power_all_channels) / std_power_all_channels

                bad_channel_indices = np.where(np.abs(z_scores) > 50)[0]
                bad_channels = [raw.ch_names[i] for i in bad_channel_indices]

                print(f"Плохие каналы для интерполяции (выявлены по спектру): {bad_channels}")

                raw.info['bads'].extend(bad_channels)
                raw.interpolate_bads(reset_bads=True, method='spline')

                raw.resample(sfreq=250.0)

                raw.filter(l_freq=3.0, h_freq=35.0, fir_design='firwin')
                raw.notch_filter(freqs=50.0, fir_design='firwin')

                events, event_dict = mne.events_from_annotations(raw)
                target_events = {k: v for k, v in event_dict.items() if k in ['2', '8']}

                if target_events:
                    epochs = mne.Epochs(raw, events, event_id=target_events, 
                                        tmin=7.0, tmax=9.0, #tmin=-3.0, tmax=9.0,
                                        baseline=None, preload=True)
                    
                    reject_criteria = dict(eeg=150e-6) 
                    epochs.drop_bad(reject=reject_criteria)

                    ica = ICA(n_components=0.95, method='infomax', fit_params=dict(extended=True), random_state=42)
                    ica.fit(epochs)

                    ic_labels = label_components(epochs, ica, method='iclabel')
                    labels = ic_labels['labels']
                    probabilities = ic_labels['y_pred_proba']

                    exclude_idx = []
                    for idx, (label, prob) in enumerate(zip(labels, probabilities)):
                        if label in ['eye blink', 'muscle artifact'] and prob >= 0.90:
                            exclude_idx.append(idx)
                            print(f"Компонента {idx} удалена: {label} (вероятность {prob:.2f})")
                    
                    ica.exclude = exclude_idx
                    epochs_clean = ica.apply(epochs.copy())

                    epochs_clean.save(save_path, overwrite=True)
                    print(f"Предобработка успешно завершена! Файл сохранен: {save_path}")

                else:
                    print(f"Внимание: Маркеры '2' или '8' не найдены в файле {os.path.basename(file_path)}. Эпохи не созданы, пайплайн прерван.")

            except Exception as e:
                print(f"КРИТИЧЕСКАЯ ОШИБКА при обработке {basename}: {e}")
        else:
            print(f"Нестандартное имя файла, пропускаем: {basename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Препроцессинг сырых данных ЭЭГ (ICA, CAR, фильтрация).")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Корневая папка с сырыми данными")
    parser.add_argument("--base_output_dir", type=str, default=get_base_results_dir(), help="Базовая папка для сохранения результатов")

    args = parser.parse_args()

    process_all_raw_subjects(
        data_root=args.data_root,
        base_output_dir=args.base_output_dir
    )
