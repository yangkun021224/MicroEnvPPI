import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import json
import sys
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

# --- Add project root to sys.path ---
try:
    from models import CodeBook
    from dataloader import ProteinDatasetDGL
except ImportError:
    print("Warning: Could not import from 'src'. Assuming the script is run from a directory where 'src' is accessible.")
    if 'src' not in sys.path:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from models import CodeBook
    from dataloader import ProteinDatasetDGL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Caching Mechanism ---
def save_cache(data, cache_path):
    """Saves data to a pickle cache file."""
    try:
        with open(cache_path, 'wb') as f: pickle.dump(data, f)
        # print(f"Cached data to: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

def load_cache(cache_path):
    """Loads data from a pickle cache file if it exists."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f: data = pickle.load(f)
            # print(f"Loaded data from cache: {cache_path}")
            return data
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}. Error: {e}")
    return None

# --- Core Functions ---
def load_config_params(config_path, dataset_name, split_mode):
    """Loads model and training parameters from the main JSON config file."""
    try:
        with open(config_path, 'r') as f: full_config = json.load(f)
        params = full_config.get(dataset_name, {}).get(split_mode, {})
        if not params: raise ValueError(f"Params for {dataset_name}/{split_mode} not found in {config_path}")
        print(f"Successfully loaded configuration for {dataset_name}/{split_mode}.")
        return params
    except Exception as e:
        print(f"ERROR: Failed to load configuration from {config_path}. {e}"); sys.exit(1)

def load_vae_model(params, ckpt_path):
    """Loads the pre-trained VAE (CodeBook) model."""
    try:
        model = CodeBook(params, protein_data=None).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"VAE model loaded from {ckpt_path}.")
        return model
    except Exception as e:
        print(f"ERROR: Failed to initialize or load VAE model. {e}"); sys.exit(1)
        
def load_protein_dataset(data_dir, dataset_name):
    """Loads the DGL dataset for protein structures."""
    try:
        prot_r_edge_path = os.path.join(data_dir, f'protein.rball.edges.{dataset_name}.npy')
        prot_k_edge_path = os.path.join(data_dir, f'protein.knn.edges.{dataset_name}.npy')
        prot_node_path = os.path.join(data_dir, f'protein.nodes.esm2_650M.{dataset_name}.pt')
        
        # We need to know the number of proteins to initialize the dataset correctly.
        # This is often derived from the sequence dictionary file.
        seq_file_path = os.path.join(data_dir, f"protein.{dataset_name}.sequences.dictionary.csv")
        num_proteins = sum(1 for line in open(seq_file_path))
        
        protein_dataset = ProteinDatasetDGL(
            prot_r_edge_path, prot_k_edge_path, prot_node_path,
            dataset=dataset_name, expected_num_proteins=num_proteins
        )
        if len(protein_dataset) == 0: raise ValueError("ProteinDatasetDGL is empty.")
        print(f"Loaded ProteinDatasetDGL with {len(protein_dataset)} proteins.")
        return protein_dataset
    except Exception as e:
        print(f"ERROR: Failed to initialize ProteinDatasetDGL. {e}"); sys.exit(1)

def get_codebook_usage_stats(vae_model, protein_dataset, cache_path, max_proteins=5000):
    """Calculates or loads from cache the usage statistics of codebook vectors."""
    cached_stats = load_cache(cache_path)
    if cached_stats: return cached_stats

    print("Calculating codebook usage statistics...")
    vae_model.eval()
    num_embeddings = vae_model.vq_layer.num_embeddings
    usage_counts = np.zeros(num_embeddings)
    protein_sets = [set() for _ in range(num_embeddings)]
    
    process_size = min(max_proteins, len(protein_dataset))
    
    with torch.no_grad():
        for idx in tqdm(range(process_size), desc="Analyzing code usage"):
            graph = protein_dataset[idx]
            if graph is None or graph.num_nodes('amino_acid') == 0: continue
            
            h = vae_model.Protein_Encoder.encoding(graph.to(device))
            if h.shape[0] == 0: continue
            
            _, _, _, indices = vae_model.vq_layer(h)
            unique_codes, counts = torch.unique(indices, return_counts=True)
            
            for code, count in zip(unique_codes.cpu().numpy(), counts.cpu().numpy()):
                usage_counts[code] += count
                protein_sets[code].add(idx)

    total_usage = np.sum(usage_counts)
    stats = {
        'usage_counts': usage_counts,
        'relative_frequencies': usage_counts / total_usage if total_usage > 0 else np.zeros_like(usage_counts),
        'protein_breadth': np.array([len(s) for s in protein_sets]),
    }
    save_cache(stats, cache_path)
    return stats

def create_statistical_summary_plot(stats, cluster_labels, codebook_vectors, filename):
    """Creates the multi-panel statistical summary plot (Figure 5)."""
    print(f"Creating statistical summary plot: {filename}")
    plt.style.use('default')
    FONT_CONFIG = {
        'suptitle_size': 24, 'title_size': 20, 'label_size': 18, 
        'tick_size': 16, 'legend_size': 16
    }
    plt.rcParams.update({
        'font.size': FONT_CONFIG['tick_size'], 'axes.titlesize': FONT_CONFIG['title_size'],
        'axes.labelsize': FONT_CONFIG['label_size'], 'xtick.labelsize': FONT_CONFIG['tick_size'],
        'ytick.labelsize': FONT_CONFIG['tick_size'], 'legend.fontsize': FONT_CONFIG['legend_size'],
        'font.weight': 'bold', 'axes.titleweight': 'bold', 'axes.labelweight': 'bold'
    })

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Dict analysis using ESM features', fontsize=FONT_CONFIG['suptitle_size'], y=0.98)
    
    # Panel A: Usage Frequency Distribution
    axes[0, 0].hist(stats['usage_counts'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Usage Frequency Distribution')
    axes[0, 0].set_xlabel('Usage Count')
    axes[0, 0].set_ylabel('Number of Codes')

    # Panel B: Protein Distribution Breadth
    axes[0, 1].hist(stats['protein_breadth'], bins=50, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Protein Distribution Breadth')
    axes[0, 1].set_xlabel('Number of Proteins')
    axes[0, 1].set_ylabel('Number of Codes')

    # Panel C: Cluster Size Distribution
    cluster_sizes = np.bincount(cluster_labels)
    axes[0, 2].bar(range(len(cluster_sizes)), cluster_sizes, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title('Cluster Size Distribution')
    axes[0, 2].set_xlabel('Cluster ID')
    axes[0, 2].set_ylabel('Number of Codes')

    # Panel D: Usage vs Protein Breadth
    axes[1, 0].scatter(stats['usage_counts'], stats['protein_breadth'], alpha=0.6, c=cluster_labels, cmap='tab10', edgecolors='black')
    axes[1, 0].set_title('Usage vs Protein Distribution')
    axes[1, 0].set_xlabel('Usage Count')
    axes[1, 0].set_ylabel('Protein Breadth')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')

    # Panel E: Cumulative Usage
    sorted_usage = np.sort(stats['usage_counts'])[::-1]
    cumulative_usage = np.cumsum(sorted_usage) / np.sum(sorted_usage) * 100
    axes[1, 1].plot(cumulative_usage, 'b-', linewidth=3)
    axes[1, 1].axhline(y=80, color='r', linestyle='--', label='80% Threshold')
    axes[1, 1].set_title('Cumulative Usage Distribution')
    axes[1, 1].set_xlabel('Code Vector Rank (by usage)')
    axes[1, 1].set_ylabel('Cumulative Usage (%)')
    axes[1, 1].legend()

    # Panel F: PCA Explained Variance
    pca = PCA()
    pca.fit(codebook_vectors)
    explained_var = pca.explained_variance_ratio_[:20]
    axes[1, 2].bar(range(len(explained_var)), explained_var, color='orange', edgecolor='black')
    axes[1, 2].set_title('Codebook Dimensionality Analysis')
    axes[1, 2].set_xlabel('Principal Component')
    axes[1, 2].set_ylabel('Explained Variance Ratio')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to {filename}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="MicroEnvPPI Landscape and Statistical Analysis Script")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the param_configs.json file.")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to the trained VAE model checkpoint (.ckpt).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the processed_data directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots and cache.")
    parser.add_argument("--dataset_name", type=str, default="STRING", help="Name of the dataset (e.g., STRING).")
    parser.add_argument("--split_mode", type=str, default="dfs", help="Data split mode (e.g., dfs).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache_microenvppi")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Device: {device}")

    # Load data and model
    params = load_config_params(args.config_path, args.dataset_name, args.split_mode)
    protein_dataset = load_protein_dataset(args.data_dir, args.dataset_name)
    vae_model = load_vae_model(params, args.vae_ckpt_path)

    # Get codebook vectors
    codebook_vectors = vae_model.vq_layer.embeddings.weight.data.cpu().numpy()

    # Get usage statistics
    stats_cache_path = os.path.join(cache_dir, f"usage_stats_{args.dataset_name}.pkl")
    usage_stats = get_codebook_usage_stats(vae_model, protein_dataset, stats_cache_path)

    # Perform clustering
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(codebook_vectors)

    # Create and save the plot
    output_filename = os.path.join(args.output_dir, f"statistical_summary_microenvppi_{args.dataset_name}.png")
    create_statistical_summary_plot(usage_stats, cluster_labels, codebook_vectors, output_filename)

    print("\n--- Landscape analysis for MicroEnvPPI completed! ---")

if __name__ == "__main__":
    main()
