import argparse
import glob
import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from config import (
    colors_mean,
    colors_runs,
    get_base_results_dir,
    get_psd_data_dir,
    get_psd_plots_dir,
)


def plot_psd_graphs(
    psd_df: pd.DataFrame,
    freqs: np.ndarray,
    subject_id: str,
    conditions: List[str],
    plot_output_dir: str,
) -> None:
    """Plots and saves a single PSD graph combining individual runs (light solid lines)
    and averaged condition lines (dark dashed lines).

    Args:
        psd_df (pd.DataFrame): DataFrame with columns 'PSD_Value', 'Run', 'Condition', 'Freq_Index'.
        freqs (np.ndarray): The array of frequency values.
        subject_id (str): The ID of the current subject.
        conditions (List[str]): List of conditions to plot.
        plot_output_dir (str): Directory where the plot will be saved.
    """

    plt.figure(figsize=(12, 6))
    plt.title(f"[{subject_id}] PSD (Channel Averaged)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (Power/Hz)")

    legend_handles = []
    legend_labels = []

    for cond_name in conditions:
        cond_data = psd_df[psd_df["Condition"] == cond_name]

        # Individual runs
        # Group by run and frequency, averaging PSD values over epochs
        run_means = (
            cond_data.groupby(["Run", "Freq_Index"])["PSD_Value"].mean().reset_index()
        )
        color_run = colors_runs[cond_name]

        for run_id in sorted(run_means["Run"].unique()):
            run_data = run_means[run_means["Run"] == run_id]
            # Use run_data['Freq_Index'] to map to the freqs array
            plt.plot(
                freqs[run_data["Freq_Index"]],
                run_data["PSD_Value"],
                color=color_run,
                alpha=0.5,
                linewidth=1,
            )

        # Averaged condition
        # Averaging across all runs and epochs in this condition
        psd_final_mean = cond_data.groupby("Freq_Index")["PSD_Value"].mean()
        color_mean = colors_mean[cond_name]

        (mean_line,) = plt.plot(
            freqs[psd_final_mean.index],
            psd_final_mean.to_numpy(),
            color=color_mean,
            linewidth=2,
            linestyle="--",
        )

        (cond_handle,) = plt.plot([], [], color=color_mean, linewidth=2, linestyle="-")

        legend_handles.append(cond_handle)
        legend_labels.append(cond_name)

    (mean_style_handle,) = plt.plot([], [], color="k", linewidth=2, linestyle="--")

    legend_handles.append(mean_style_handle)
    legend_labels.append("Session Average")

    plt.legend(legend_handles, legend_labels, loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(freqs.min(), freqs.max())

    # Save to PSD_PLOTS
    plt.savefig(os.path.join(plot_output_dir, f"{subject_id}_psd_combined.png"))
    plt.close()


def load_and_plot_subjects(base_input_dir: str) -> None:
    """Loads PSD data from .npz files and generates plots.

    Args:
        base_input_dir (str): Base directory containing the PSD_DATA and PSD_PLOTS folders.
    """

    data_input_dir = get_psd_data_dir(base_input_dir)
    plot_output_dir = get_psd_plots_dir(base_input_dir)

    os.makedirs(plot_output_dir, exist_ok=True)

    print(f"📂 PSD data will be loaded from: {data_input_dir}")
    print(f"📈 Plots will be saved to: {plot_output_dir}")

    # Search for all .npz files in the data folder
    npz_files = glob.glob(os.path.join(data_input_dir, "*_epoch_psd_data.npz"))

    if not npz_files:
        print(
            f"❌ No *_epoch_psd_data.npz files found in '{data_input_dir}'. Please run 1_calculate_psd_for_umap.py first."
        )
        return

    for file_path in sorted(npz_files):
        file_name = os.path.basename(file_path)
        subject_id = file_name.replace("_epoch_psd_data.npz", "")

        print(f"\n===== PLOTTING FOR SUBJECT: {subject_id} =====")

        try:
            data = np.load(file_path, allow_pickle=True)
            freqs = data["freqs"]
            conditions = data["conditions"].tolist()

            # Load epoch-level data
            epoch_psds = data["epoch_psds"]  # (N_epochs, N_channels, N_freqs)
            labels = data["labels"]
            run_labels = data["run_labels"]

            # Preliminary averaging: across channels for all epochs (aggregation step)
            avg_psds_per_epoch = epoch_psds.mean(axis=1)  # (N_epochs, N_freqs)

            # Create "long" DataFrame for easy grouping
            n_epochs, n_freqs = avg_psds_per_epoch.shape

            freq_indices = np.tile(np.arange(n_freqs), n_epochs)
            psd_values = avg_psds_per_epoch.flatten()
            cond_labels_repeated = np.repeat(labels, n_freqs)
            run_labels_repeated = np.repeat(run_labels, n_freqs)

            psd_df = pd.DataFrame(
                {
                    "PSD_Value": psd_values,
                    "Condition": cond_labels_repeated,
                    "Run": run_labels_repeated,
                    "Freq_Index": freq_indices,
                }
            )

            plot_psd_graphs(psd_df, freqs, subject_id, conditions, plot_output_dir)
            print(f"  ✅ Plot saved to {plot_output_dir}")

        except Exception as e:
            print(f" ❌ Error loading or plotting for {subject_id}: {e}")
            continue

    print("\n==========================================")
    print("Plotting complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for plotting PSD graphs based on saved .npz files."
    )
    parser.add_argument(
        "--base_input_dir",
        type=str,
        default=get_base_results_dir(),
        help="Base directory containing PSD_DATA and DR_PLOTS folders. (default: PSD_ANALYSIS_RESULTS in the project directory)",
    )

    args = parser.parse_args()
    load_and_plot_subjects(args.base_input_dir)
