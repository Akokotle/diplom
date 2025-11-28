import argparse
import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
from config import (
    CONDITIONS,
    DR_FREQ_BAND,
    FREQ_BANDS,
    PCA_N_COMPONENTS,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    get_base_results_dir,
    get_dr_plots_dir,
    get_psd_data_dir,
)
from sklearn.decomposition import PCA
from umap import UMAP


def analyze_and_plot_dr_interactive(
    base_input_dir: str, umap_n_comp: int, umap_n_neigh: int
) -> None:
    """Loads data, applies UMAP and PCA, generates an interactive 3D plot,
    and saves the HTML file to the DR_PLOTS folder.

    Args:
        base_input_dir (str): Base directory containing the PSD_DATA folder.
        umap_n_comp (int): Intermediate dimensionality for UMAP.
        umap_n_neigh (int): Number of neighbors for UMAP.
    """

    data_input_dir = get_psd_data_dir(base_input_dir)
    plot_output_dir = get_dr_plots_dir(base_input_dir)
    os.makedirs(plot_output_dir, exist_ok=True)

    npz_files = glob.glob(os.path.join(data_input_dir, "*_epoch_psd_data.npz"))

    if not npz_files:
        print(
            f"❌ No *_epoch_psd_data.npz files found in '{data_input_dir}'. Please run 1_calculate_psd.py first."
        )
        return

    for file_path in sorted(npz_files):
        file_name = os.path.basename(file_path)
        subject_id = file_name.replace("_epoch_psd_data.npz", "")

        print(f"\n===== DR ANALYSIS FOR SUBJECT: {subject_id} =====")

        try:
            data = np.load(file_path, allow_pickle=True)
            X = data["data_for_dr"]
            labels = data["labels"]
            run_labels = data["run_labels"]
            freqs = data["freqs"]

            # Filter data by conditions from config
            mask = np.isin(labels, CONDITIONS)
            X_filtered = X[mask]
            labels_filtered = labels[mask]
            run_labels_filtered = run_labels[mask]

            if DR_FREQ_BAND != "ALL":
                f_min, f_max = FREQ_BANDS[DR_FREQ_BAND]

                freq_indices = np.where((freqs >= f_min) & (freqs <= f_max))[0]

                if freq_indices.size == 0:
                    print(
                        f"⚠️ There are no frequencies in the selected range '{DR_FREQ_BAND}' ({f_min}-{f_max} Hz). Skip."
                    )
                    continue

                n_freqs_all = len(freqs)
                # n_channels * n_freqs_all = X_filtered.shape[1]
                n_channels = X_filtered.shape[1] // n_freqs_all

                # (N_epochs, N_channels, N_freqs)
                X_reshaped = X_filtered.reshape(
                    X_filtered.shape[0], n_channels, n_freqs_all
                )

                X_band_selected = X_reshaped[:, :, freq_indices]
                X_filtered = X_band_selected.reshape(X_band_selected.shape[0], -1)

                print(
                    f"✅ Data filtered by range: {DR_FREQ_BAND} ({f_min}-{f_max} Hz). New shape: {X_filtered.shape[1]}"
                )

            if X_filtered.shape[0] < 2 * umap_n_neigh:
                print("⚠️ Not enough epochs for UMAP. Skipping.")
                continue

            print(f"Step 1/3: UMAP (N={umap_n_neigh}, D={umap_n_comp})")

            reducer = UMAP(
                n_neighbors=umap_n_neigh,
                n_components=umap_n_comp,
                metric="euclidean",
                random_state=42,
                verbose=False,
            )
            X_umap = reducer.fit_transform(X_filtered)

            # Use PCA_N_COMPONENTS from config.py
            print(f"Step 2/3: PCA (D={PCA_N_COMPONENTS})")
            pca = PCA(n_components=PCA_N_COMPONENTS)
            X_pca_3d = pca.fit_transform(X_umap)

            # Prepare data for Plotly
            df = pd.DataFrame(X_pca_3d, columns=["PC 1", "PC 2", "PC 3"])
            df["Condition"] = labels_filtered
            df["Run"] = run_labels_filtered

            # Create a combined label for coloring
            df["Condition_Run"] = (
                df["Condition"].astype(str) + "_" + df["Run"].astype(str)
            )

            # 3. Interactive 3D visualization with Plotly
            print("Step 3/3: Interactive Plotly visualization...")

            plot_title = (
                f"[{subject_id}] PSD DR: UMAP -> PCA (Band: {DR_FREQ_BAND})<br>"
                f"UMAP:<br>"
                f"Number of neighbors = {UMAP_N_NEIGHBORS}<br>"
                f"Number of components = {UMAP_N_COMPONENTS}"
            )

            fig = px.scatter_3d(
                df,
                x="PC 1",
                y="PC 2",
                z="PC 3",
                color="Condition",
                symbol="Condition_Run",
                hover_data=["Condition", "Run"],
                title=plot_title,
                opacity=0.7,
                height=700,
            )

            fig.update_traces(marker=dict(size=4))

            # Save the plot to an interactive HTML file in DR_PLOTS
            save_path = os.path.join(
                plot_output_dir, f"{subject_id}_dr_umap_pca_3d_interactive.html"
            )
            fig.write_html(save_path)

            print(f"✅ Interactive 3D plot saved to {save_path}")

        except Exception as e:
            print(f"❌ Error during DR or plotting for {subject_id}: {e}")
            continue

    print("\n==========================================")
    print("Interactive dimensionality analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for interactive epoch-level PSD analysis using UMAP and PCA (Plotly)."
    )
    parser.add_argument(
        "--base_input_dir",
        type=str,
        default=get_base_results_dir(),
        help="Base directory containing the PSD_DATA folder.",
    )
    parser.add_argument(
        "--umap_dim",
        type=int,
        default=UMAP_N_COMPONENTS,
        help="Intermediate dimensionality for UMAP (default: 50).",
    )
    parser.add_argument(
        "--umap_neighbors",
        type=int,
        default=UMAP_N_NEIGHBORS,
        help="Number of neighbors for UMAP (default: 20).",
    )

    args = parser.parse_args()

    analyze_and_plot_dr_interactive(
        args.base_input_dir, args.umap_dim, args.umap_neighbors
    )
