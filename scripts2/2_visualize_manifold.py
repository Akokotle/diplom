import argparse
import glob
import itertools
import os
from typing import Any, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from config import (
    CONDITIONS,
    PCA_N_COMPONENTS,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    get_base_results_dir,
    get_psd_data_dir,
)
from sklearn.decomposition import PCA

matplotlib.use('Agg')

def compute_hybrid_manifold_projections(peaks: np.ndarray) -> np.ndarray:

    """Уменьшает размерность входящих данных до трех с помощью UMAP и PCA.

    Args:
        peaks (np.ndarray): Спектр (2D массив, (эпохи x каналы) x частоты).

    Returns:
        np.ndarray: Трехмерный массив координат (n_samples, 3) для построения 3D-графика.
    """

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST, 
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC
    ) #, random_state=42
    umap_embeddings = reducer.fit_transform(peaks)
    
    pca = PCA(n_components=PCA_N_COMPONENTS) #, random_state=42
    embeddings_3d = pca.fit_transform(umap_embeddings)
    
    return embeddings_3d

def plot_latent_space_3d(embeddings_3d: np.ndarray, metadata: pd.DataFrame, output_path: str):

    """Строит и сохраняет интерактивный 3D-график проекции данных.

    Args:
        embeddings (np.ndarray): Трехмерные координаты данных, полученные после понижения размерности.
        metadata (pd.DataFrame): Таблица с метаданными (условие, номер прогона, субъект) для каждой точки.
        output_path (str): Полный путь для сохранения итогового интерактивного HTML-файла.
    """

    plot_df = pd.DataFrame({
    'PC1': embeddings_3d[:, 0],
    'PC2': embeddings_3d[:, 1],
    'PC3': embeddings_3d[:, 2],
    'condition': metadata['condition'].values,
    'subject': metadata['subject'].values,
    'run': metadata['run'].values
    })

    subj_name = metadata['subject'].iloc[0]
    
    fig = px.scatter_3d(
        plot_df, x='PC1', y='PC2', z='PC3',
        color='condition', symbol='run',
        title=f'UMAP-PCA 3D проекция {subj_name}',
        opacity=0.7, template='plotly_white'
    )
    fig.update_traces(marker=dict(size=3))
    fig.write_html(output_path)

def plot_latent_displacement_topography(
    embeddings_3d: np.ndarray, metadata: pd.DataFrame, info: mne.Info, 
    condition_col: str, cond_1: str, cond_2: str, ax: Any 
) -> np.ndarray:
    
    """Вычисляет и визуализирует величину спектрального сдвига в 3D пространстве для каждого электрода.

    Args:
        embeddings_3d (np.ndarray): 3D координаты (после UMAP+PCA).
        metadata (pd.DataFrame): Таблица метаданных, сопоставленная с признаками.
        info (mne.Info): Объект MNE с информацией о пространственном расположении каналов (монтаже).
        condition_col (str): Название колонки с условиями в таблице метаданных.
        cond_1 (str): Название первого (базового) условия.
        cond_2 (str): Название второго (целевого) условия.
        ax (Any): Ось графика.

    Returns:
        np.ndarray: Одномерный массив сдвигов координат (n_chanels).
    """

    metadata['emb_x'] = embeddings_3d[:, 0]
    metadata['emb_y'] = embeddings_3d[:, 1]
    metadata['emb_z'] = embeddings_3d[:, 2] 
    
    displacements = []
    for ch_idx, _ in enumerate(info.ch_names):
        ch_data = metadata[metadata['channel_idx'] == ch_idx]
        
        pre_data = ch_data[ch_data[condition_col] == cond_1]
        post_data = ch_data[ch_data[condition_col] == cond_2]
            
        centroid_pre = pre_data[['emb_x', 'emb_y', 'emb_z']].mean().values
        centroid_post = post_data[['emb_x', 'emb_y', 'emb_z']].mean().values
        
        shift = np.linalg.norm(centroid_post - centroid_pre)
        displacements.append(shift)
        
    displacement_array = np.array(displacements)
    
    mne.viz.plot_topomap(
        displacement_array, info, axes=ax, cmap='plasma', 
        show=False, contours=6, extrapolate='head', names=info.ch_names
    )
    
    ax.set_title(f'{cond_1} - {cond_2}', fontsize=14)

    return displacement_array


def visualize_per_subject(base_output_dir: str) -> None:
    
    """Загружает данные и вызывает функции генерации 3D проекций и топомапы сдвигов для каждого субъекта.

    Args:
        base_output_dir (str): Корневая директория, содержащая обработанные данные и папку для сохранения графиков.
    """

    conditions = CONDITIONS
    psd_output_dir = get_psd_data_dir(base_output_dir)
    plots_base_dir = os.path.join(base_output_dir, "subject_plots")
    os.makedirs(plots_base_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(psd_output_dir, "*_spectral_results.npz")))

    condition_pairs = list(itertools.combinations(conditions, 2))
    group_shifts: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {pair: {} for pair in condition_pairs}
    global_ch_names: list[str] = []

    for npz_path in npz_files:
        filename = os.path.basename(npz_path)
        subj_id = filename.split('_')[0]
        
        print(f"\n⏳ Работа с {subj_id}")
        
        subject_data = np.load(npz_path, allow_pickle=True)['data']
        
        all_peaks = []
        meta_records = []
                
        for block in subject_data:
            peaks_3d = block['peaks'] 
            cond = block['condition']
            run_id = block['run']
            
            n_epochs, n_channels, n_freqs = peaks_3d.shape
            
            peaks_2d = peaks_3d.reshape(-1, n_freqs)
            all_peaks.append(peaks_2d)
            
            epoch_idx = np.repeat(np.arange(n_epochs), n_channels)
            chan_idx = np.tile(np.arange(n_channels), n_epochs)
            
            block_meta = pd.DataFrame({
                'epoch_idx': epoch_idx,
                'channel_idx': chan_idx,
                'condition': cond,
                'run': run_id,
                'subject': subj_id
            })
            meta_records.append(block_meta)

        subj_features = np.vstack(all_peaks)
        subj_metadata = pd.concat(meta_records, ignore_index=True)

        info_path = os.path.join(psd_output_dir, f"{subj_id}_info.fif")
        if os.path.exists(info_path):
            info = mne.io.read_info(info_path)
            if not global_ch_names:
                global_ch_names = info.ch_names
        else:
            print(f"⚠️ Не найден файл {info_path}.")
            continue

        subj_out_dir = os.path.join(plots_base_dir, subj_id)
        os.makedirs(subj_out_dir, exist_ok=True)

        print(f"Снижение размерности с UMAP и PCA для {subj_id}...")
        embeddings_3d = compute_hybrid_manifold_projections(subj_features)

        print(f"3D проекция сохранена в {subj_id}_umap_pca_3d.html")
        plot_latent_space_3d(
            embeddings_3d, subj_metadata,
            output_path=os.path.join(subj_out_dir, f"{subj_id}_umap_pca_3d.html")
        )

        if len(conditions) >= 2:
            print(f"Топография сохранена в {subj_id}_topomap_shifts.png")
            
            n_pairs = len(condition_pairs)
            
            fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 6))
            
            if n_pairs == 1:
                axes = [axes]
                
            fig.suptitle(f'Топографическая карта {subj_id}', fontsize=18, y=1.05)

            for (cond_1, cond_2), ax in zip(condition_pairs, axes):
                shift_array = plot_latent_displacement_topography(
                    embeddings_3d, subj_metadata, info, 
                    condition_col='condition', cond_1=cond_1, cond_2=cond_2,
                    ax=ax
                )
                group_shifts[(cond_1, cond_2)][subj_id] = shift_array
            
            fig.tight_layout()
            combined_filename = f"{subj_id}_topomap_shifts.png"
            plt.savefig(os.path.join(subj_out_dir, combined_filename), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
    print(f"\n🟩 Графики сохранены в {plots_base_dir}")


    if group_shifts and global_ch_names:
        print("\n⏳ Построение групповых тепловых карт (хитмапов)...")
        
        for (cond_1, cond_2), subj_data_dict in group_shifts.items():
            df_heat = pd.DataFrame(subj_data_dict, index=global_ch_names).T   

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(df_heat.values, aspect='auto', cmap='plasma')
            
            ax.set_xticks(np.arange(len(df_heat.columns)))
            ax.set_xticklabels(df_heat.columns, rotation=90, ha='center')
            ax.set_yticks(np.arange(len(df_heat.index)))
            ax.set_yticklabels(df_heat.index)
            
            ax.set_title(f'Хитмап сдвигов {cond_1} - {cond_2}', fontsize=16, pad=15)
            
            ax.set_xlabel('Каналы', fontsize=12)
            ax.set_ylabel('Субъекты', fontsize=12)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Величина сдвига', rotation=270, labelpad=15)
            
            fig.tight_layout()
            
            safe_pre = str(cond_1).replace('/', '_').replace(' ', '_')
            safe_post = str(cond_2).replace('/', '_').replace(' ', '_')
            heatmap_filename = os.path.join(plots_base_dir, f"group_heatmap_{safe_pre}_to_{safe_post}.png")
            plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f" 🟩 Хитмап сохранен: {os.path.basename(heatmap_filename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_output_dir", type=str, default=get_base_results_dir())
    args = parser.parse_args()
    
    visualize_per_subject(args.base_output_dir)