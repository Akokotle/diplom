import argparse
import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import umap
from config import (
    PCA_N_COMPONENTS,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    UMAP_TARGET_N_NEIGHBORS,
    UMAP_TARGET_WEIGHT,
    get_base_results_dir,
)
from sklearn.decomposition import PCA


def compute_hybrid_manifold_projections(features: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    
    reducer_params = {
        'n_neighbors': UMAP_N_NEIGHBORS,
        'min_dist': UMAP_MIN_DIST, 
        'n_components': UMAP_N_COMPONENTS,
        'metric': UMAP_METRIC,
    }
    
    if mode == 'supervised':
        reducer_params['target_weight'] = UMAP_TARGET_WEIGHT
        reducer_params['target_n_neighbors'] = UMAP_TARGET_N_NEIGHBORS
        reducer = umap.UMAP(**reducer_params)
        umap_embeddings = reducer.fit_transform(features, y=y)
    else:
        reducer = umap.UMAP(**reducer_params)
        umap_embeddings = reducer.fit_transform(features)
    
    pca = PCA(n_components=PCA_N_COMPONENTS) 
    embeddings_3d = pca.fit_transform(umap_embeddings)
    
    return embeddings_3d


def plot_latent_space_3d(embeddings_3d: np.ndarray, metadata: pd.DataFrame, output_path: str, title: str):
    
    plot_df = pd.DataFrame({
        'PC1': embeddings_3d[:, 0],
        'PC2': embeddings_3d[:, 1],
        'PC3': embeddings_3d[:, 2],
        'condition': metadata['condition'].values,
        'subject': metadata['subject'].values,
        'run': metadata['run'].values
    })

    fig = px.scatter_3d(
        plot_df, x='PC1', y='PC2', z='PC3',
        color='condition', symbol='run',
        title=title, 
        opacity=0.7, template='plotly_white'
    )
    fig.update_traces(marker=dict(size=3))
    fig.write_html(output_path)


def visualize_per_subject(base_output_dir: str, mode: str) -> None:
    
    riemann_output_dir = os.path.join(base_output_dir, "riemann_features")
    plots_base_dir = os.path.join(base_output_dir, "subject_plots")
    os.makedirs(plots_base_dir, exist_ok=True)

    npz_files = sorted(glob.glob(os.path.join(riemann_output_dir, "*_riemann_results.npz")))

    if not npz_files:
        print(f"🟥 Не найдено файлов с признаками в {riemann_output_dir}")
        return

    prefix = "sup" if mode == 'supervised' else "unsup"
    title_type = "Supervised" if mode == 'supervised' else "Unsupervised"

    for npz_path in npz_files:
        filename = os.path.basename(npz_path)
        subj_id = filename.split('_')[0]
        
        print(f"\n⏳ Работа с {subj_id} ({mode})")
        
        subject_data = np.load(npz_path, allow_pickle=True)['data']
        
        all_features = []
        meta_records = []
                
        for block in subject_data:
            feats_2d = block['features'] 
            cond = block['condition']
            run_id = block['run']
            
            n_epochs = feats_2d.shape[0]
            
            all_features.append(feats_2d)
            
            block_meta = pd.DataFrame({
                'epoch_idx': np.arange(n_epochs),
                'condition': cond,
                'run': run_id,
                'subject': subj_id
            })
            meta_records.append(block_meta)

        subj_features = np.vstack(all_features)
        subj_metadata = pd.concat(meta_records, ignore_index=True)

        subj_out_dir = os.path.join(plots_base_dir, subj_id)
        os.makedirs(subj_out_dir, exist_ok=True)

        label_mapping = {'pre': 0, 'post': 1, 'follow': 2}
        y_labels = subj_metadata['condition'].map(label_mapping).fillna(-1).astype(int).values

        print(f"Снижение размерности UMAP и PCA для {subj_id}...")
        
        embeddings_3d = compute_hybrid_manifold_projections(subj_features, y=y_labels, mode=mode)

        plot_name = f"{subj_id}_{prefix}_riemann_umap_pca_3d.html"
        out_path = os.path.join(subj_out_dir, plot_name)
        
        print(f"3D проекция сохранена в {plot_name}")
        plot_latent_space_3d(
            embeddings_3d, 
            subj_metadata, 
            output_path=out_path,
            title=f'{title_type} UMAP-PCA 3D проекция {subj_id} (PyRiemann)'
        )
            
    print(f"\n🟩 Графики сохранены в {plots_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуализация римановых признаков через UMAP.")
    parser.add_argument("--base_output_dir", type=str, default=get_base_results_dir())
    parser.add_argument("--mode", type=str, choices=['supervised', 'unsupervised'], default='supervised')

    args = parser.parse_args()
    
    visualize_per_subject(args.base_output_dir, args.mode)