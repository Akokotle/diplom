import argparse
import glob
import os
import re
from typing import List, Optional, Tuple

import mne
import numpy as np
from config import (
    BASELINE,
    CONDITIONS,
    DATA_ROOT,
    EVENT_ID,
    FMAX_PSD,
    FMIN_PSD,
    PSD_METHOD,
    SUBJECT_DIR,
    T_MAX,
    T_MIN,
    get_base_results_dir,
    get_psd_data_dir,
)


def load_preprocess_and_get_all_epoch_psds(
    file_path: str, info: Optional[mne.Info] = None
) -> Tuple[np.ndarray | None, np.ndarray | None, mne.Info | None]:
    """Uploads, creates epochs, and calculates PSD for a single file.

    Args:
        file_path (str): Path to data file.
        info (Optional[mne.Info], optional): An instance of the mne.Info class,
            used to pass channel information between files. Defaults to None.

    Returns:
        Tuple[np.ndarray | None, np.ndarray | None, mne.Info | None]: A tuple containing:
            1. psds: np.ndarray | None
                PSD for each epoch: (n_epochs, n_channels, n_frequencies).
            2. freqs: np.ndarray | None
                The frequency values for the requested range.
            3. info: mne.Info | None
                An instance of the mne.Info class (returned only on the first successful call).
    """
    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        events, _ = mne.events_from_annotations(raw)

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=EVENT_ID,
            tmin=T_MIN,
            tmax=T_MAX,
            preload=True,
            baseline=BASELINE,
        )

        if len(epochs) == 0:
            return None, None, None

        epo_spectrum = epochs.compute_psd(
            method=PSD_METHOD,
            fmin=FMIN_PSD,
            fmax=FMAX_PSD,
            verbose=False,
        )
        # psds shape (n_epochs, n_channels, n_frequencies)
        psds, freqs = epo_spectrum.get_data(return_freqs=True)

        if info is None:
            info = raw.info.copy()
            return psds, freqs, info

        return psds, freqs, None

    except Exception as e:
        print(f" ❌ Error processing file {os.path.basename(file_path)}: {e}")
        return None, None, None


def process_subjects_and_save_psd(
    data_root: str,
    conditions: List[str],
    base_output_dir: str,
    subject_dir: List[str],
) -> None:
    """Processes subjects, calculates epoch-level PSD,
        and saves data for DR/Plotting in the format (N_epochs, N_features).

    Args:
        data_root (str): Root data directory.
        conditions (List[str]): List of conditions to process (e.g., ["pre", "post"]).
        base_output_dir (str): Base directory for saving results.
        subject_dir (List[str]): List of directory name(s) for a subject(s) (e.g., "sub-01").
        If [""] - processing all subjects.
    """

    psd_output_dir = get_psd_data_dir(base_output_dir)

    os.makedirs(psd_output_dir, exist_ok=True)
    print(f"📂 The PSD epoch data will be stored in: {psd_output_dir}")

    if len(subject_dir) == 1 and subject_dir[0] == "":
        subject_dir = [
            d
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and d.startswith("sub-")
        ]

        print(f"{len(subject_dir)} potential subjects found in {data_root}.")

    processed_count = 0

    for subj_dir_name in sorted(subject_dir):
        subject_id_match = re.search(r"(sub-\d+)", subj_dir_name)
        if not subject_id_match:
            continue

        current_subject_id = subject_id_match.group(0)
        subj_dir = os.path.join(data_root, subj_dir_name)

        print(f"\n===== SUBJECT PROCESSING: {current_subject_id} =====")

        all_epochs_psds = []  # List of arrays (n_epochs_in_run, n_channels, n_freqs)
        all_epochs_labels = []  # Condition labels
        all_epochs_run_labels = []  # Run labels
        info = None
        freqs = None
        has_all_conditions = True

        for condition in conditions:
            search_pattern = os.path.join(
                subj_dir, "ses-*", "eeg", f"{current_subject_id}_{condition}_*_eeg.set"
            )
            all_set_files_for_condition = sorted(glob.glob(search_pattern))

            if not all_set_files_for_condition:
                print(
                    f"  ❌ No files found for condition {condition}. Skipping subject."
                )
                has_all_conditions = False
                break

            for file_path in all_set_files_for_condition:
                run_match = re.search(r"_run-(\d+)", file_path)
                run_id = f"run-{run_match.group(1)}" if run_match else "run-NA"

                epoch_psds, current_freqs, current_info = (
                    load_preprocess_and_get_all_epoch_psds(file_path, info)
                )

                if epoch_psds is not None:
                    all_epochs_psds.append(epoch_psds)

                    # Create labels for all epochs in this run
                    n_epochs = epoch_psds.shape[0]
                    labels = np.full(n_epochs, condition)
                    all_epochs_labels.append(labels)

                    run_labels = np.full(n_epochs, run_id)
                    all_epochs_run_labels.append(run_labels)

                    if info is None and current_info is not None:
                        info = current_info
                        freqs = current_freqs

        if not has_all_conditions or not all_epochs_psds:
            continue

        # Combine PSD of all runs and conditions into one array
        final_psd_data_epoch = np.concatenate(
            all_epochs_psds, axis=0
        )  # (N_total_epochs, n_channels, n_freqs)
        final_labels = np.concatenate(all_epochs_labels)
        final_run_labels = np.concatenate(all_epochs_run_labels)

        # Reshape for DR: (N_total_epochs, N_channels * N_freqs)
        N_epochs = final_psd_data_epoch.shape[0]
        data_for_dr = final_psd_data_epoch.reshape(N_epochs, -1)

        if info is None or freqs is None:
            print(
                f" ❌ Unable to obtain information about channels/frequencies for {current_subject_id}. Skipping file storage."
            )
            continue

        psd_data_to_save = {
            "data_for_dr": data_for_dr,  # (N_epochs, N_features) - for DR
            "epoch_psds": final_psd_data_epoch,  # (N_epochs, N_channels, N_freqs) - for Plotting
            "labels": final_labels,  # (N_epochs,) - Condition
            "run_labels": final_run_labels,  # (N_epochs,) - Run
            "freqs": freqs,
            "channels": info.ch_names,
            "conditions": conditions,
        }

        save_path = os.path.join(
            psd_output_dir, f"{current_subject_id}_epoch_psd_data.npz"
        )
        np.savez_compressed(save_path, **psd_data_to_save)

        processed_count += 1
        print(
            f"  ✅ The epoch PSD data is stored in {save_path}. Data shape for DR: {data_for_dr.shape}"
        )

    print("\n==========================================")
    print(f"Processing complete. Total subjects processed: {processed_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for calculating epoch-level PSD for all subjects and saving the results."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DATA_ROOT,
        help=f"Root data directory (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default=get_base_results_dir(),
        help="Base directory for saving results. (default: PSD_ANALYSIS_RESULTS in the project directory)",
    )

    args = parser.parse_args()

    process_subjects_and_save_psd(
        args.data_root, CONDITIONS, args.base_output_dir, SUBJECT_DIR
    )
