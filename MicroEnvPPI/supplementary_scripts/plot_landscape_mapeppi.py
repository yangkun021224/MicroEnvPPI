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
import dgl
import warnings
warnings.filterwarnings('ignore')

# --- Add project root to sys.path ---
try:
    from models import CodeBook
except ImportError:
    print("Warning: Could not import from 'src'.")
    if 'src' not in sys.path:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from models import CodeBook

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Custom Dataset for MAPE-PPI (handles numpy array features) ---
class ProteinDatasetDGL_MAPEPPI(torch.utils.data.Dataset):
    def __init__(self, r_edge_path, k_edge_path, node_path, num_proteins):
        self.r_edge = np.load(r_edge_path, allow_pickle=True) if os.path.exists(r_edge_path) else [[]]*num_proteins
        self.k_edge = np.load(k_edge_path, allow_pickle=True) if os.path.exists(k_edge_path) else [[]]*num_proteins
        raw_features = torch.load(node_path, map_location='cpu')
        self.node_features = [torch.from_numpy(f).float() if isinstance(f, np.ndarray) else f.float() for f in raw_features]
        self.num_proteins = min(len(self.r_edge), len(self.node_features), num_proteins)

    def __len__(self): return self.num_proteins
    def __getitem__(self, idx):
        node_feats = self.node_features[idx]
        num_nodes = node_feats.shape[0]
        if num_nodes == 0: return None
        def filter_edges(edges, max_idx): return [(u, v) for u, v in edges if u < max_idx and v < max_idx]
        graph_data = {
            ('amino_acid', 'SEQ', 'amino_acid'): [(i, i+1) for i in range(num_nodes-1)] + [(i+1, i) for i in range(num_nodes-1)],
            ('amino_acid', 'STR_KNN', 'amino_acid'): filter_edges(self.k_edge[idx], num_nodes),
            ('amino_acid', 'STR_DIS', 'amino_acid'): filter_edges(self.r_edge[idx], num_nodes)
        }
        g = dgl.heterograph(graph_data, num_nodes_dict={'amino_acid': num_nodes})
        g.nodes['amino_acid'].data['x'] = node_feats
        return g

# --- Caching and Utility Functions ---
def save_cache(data, cache_path):
    try:
        with open(cache_path, 'wb') as f: pickle.dump(data, f)
    except Exception as e: print(f"Warning: Cache save failed for {cache_path}. {e}")

def load_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f: return pickle.load(f)
        except Exception as e: print(f"Warning: Cache load failed for {cache_path}. {e}")
    return None

def load_mapeppi_params():
    """Returns a hardcoded parameter dictionary for the baseline MAPE-PPI model."""
    return {
        "input_dim": 7, "prot_hidden_dim": 128, "prot_num_layers": 4,
        "num_embeddings": 256, "commitment_cost": 0.25, "dropout_ratio": 0.0,
        "protein_embedding_mode": "concat"
    }

def load_protein_dataset_mapeppi(data_dir, dataset_name):
    """Loads the DGL dataset using the custom loader for MAPE-PPI."""
    try:
        r_edge_path = os.path.join(data_dir, f'protein.rball.edges.{dataset_name}.npy')
        k_edge_path = os.path.join(data_dir, f'protein.knn.edges.{dataset_name}.npy')
        node_path = os.path.join(data_dir, f'protein.nodes.{dataset_name}.pt') # 7D features
        seq_file_path = os.path.join(data_dir, f"protein.{dataset_name}.sequences.dictionary.csv")
        num_proteins = sum(1 for line in open(seq_file_path))

        dataset = ProteinDatasetDGL_MAPEPPI(r_edge_path, k_edge_path, node_path, num_proteins)
        if len(dataset) == 0: raise ValueError("Dataset is empty.")
        print(f"Loaded MAPE-PPI dataset with {len(dataset)} proteins.")
        return dataset
    except Exception as e:
        print(f"ERROR: Failed to initialize MAPE-PPI dataset. {e}"); sys.exit(1)

def get_codebook_usage_stats_mapeppi(vae_model, protein_dataset, cache_path, max_proteins=5000):
    """Calculates usage stats for MAPE-PPI codebook vectors."""
    cached_stats = load_cache(cache_path)
    if cached_stats: return cached_stats

    print("Calculating codebook usage statistics for MAPE-PPI...")
    vae_model.eval()
    num_embeddings = vae_model.vq_layer.num_embeddings
    usage_counts = np.zeros(num_embeddings)
    protein_sets = [set() for _ in range(num_embeddings)]
    
    process_size = min(max_proteins, len(protein_dataset))
    
    with torch.no_grad():
        for idx in tqdm(range(process_size), desc="Analyzing code usage (MAPE-PPI)"):
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

def create_statistical_summary_plot_mapeppi(stats, cluster_labels, codebook_vectors, filename):
    """Creates the multi-panel statistical summary plot for MAPE-PPI (Figure 5 comparison)."""
    print(f"Creating statistical summary plot for MAPE-PPI: {filename}")
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
    fig.suptitle('Dict analysis using seven-dimensional physicochemical properties', fontsize=FONT_CONFIG['suptitle_size'], y=0.98)
    
    # Panel A: Usage Frequency Distribution
    axes[0, 0].hist(stats['usage_counts'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Original Usage Frequency Distribution')
    axes[0, 0].set_xlabel('Usage Count')
    axes[0, 0].set_ylabel('Number of Codes')

    # Panel B: Protein Distribution Breadth
    axes[0, 1].hist(stats['protein_breadth'], bins=30, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Original Protein Distribution Breadth')
    axes[0, 1].set_xlabel('Number of Proteins')
    axes[0, 1].set_ylabel('Number of Codes')

    # Panel C: Cluster Size Distribution
    cluster_sizes = np.bincount(cluster_labels)
    axes[0, 2].bar(range(len(cluster_sizes)), cluster_sizes, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title('Original Cluster Size Distribution')
    axes[0, 2].set_xlabel('Cluster ID')
    axes[0, 2].set_ylabel('Number of Codes')

    # Panel D: Usage vs Protein Breadth
    axes[1, 0].scatter(stats['usage_counts'], stats['protein_breadth'], alpha=0.6, c=cluster_labels, cmap='tab10', edgecolors='black')
    axes[1, 0].set_title('Original Usage vs Protein Distribution')
    axes[1, 0].set_xlabel('Usage Count')
    axes[1, 0].set_ylabel('Protein Breadth')
    
    # Panel E: Cumulative Usage
    sorted_usage = np.sort(stats['usage_counts'])[::-1]
    cumulative_usage = np.cumsum(sorted_usage) / np.sum(sorted_usage) * 100
    axes[1, 1].plot(cumulative_usage, 'b-', linewidth=3)
    axes[1, 1].axhline(y=80, color='r', linestyle='--', label='80% Threshold')
    axes[1, 1].set_title('Original Cumulative Usage Distribution')
    axes[1, 1].set_xlabel('Code Vector Rank (by usage)')
    axes[1, 1].set_ylabel('Cumulative Usage (%)')
    axes[1, 1].legend()

    # Panel F: PCA Explained Variance
    pca = PCA()
    pca.fit(codebook_vectors)
    explained_var = pca.explained_variance_ratio_[:20]
    axes[1, 2].bar(range(len(explained_var)), explained_var, color='orange', edgecolor='black')
    axes[1, 2].set_title('Original Codebook Dimensionality Analysis')
    axes[1, 2].set_xlabel('Principal Component')
    axes[1, 2].set_ylabel('Explained Variance Ratio')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MAPE-PPI summary plot saved to {filename}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="MAPE-PPI (Baseline) Landscape and Statistical Analysis Script")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to the trained MAPE-PPI VAE model checkpoint (.ckpt).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the processed_data directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots and cache.")
    parser.add_argument("--dataset_name", type=str, default="STRING", help="Name of the dataset (e.g., STRING).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache_mapeppi")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Device: {device}")

    # Load data and model
    params = load_mapeppi_params()
    protein_dataset = load_protein_dataset_mapeppi(args.data_dir, args.dataset_name)
    try:
        vae_model = CodeBook(params, protein_data=None).to(device)
        vae_model.load_state_dict(torch.load(args.vae_ckpt_path, map_location=device), strict=False)
        vae_model.eval()
    except Exception as e:
        print(f"ERROR: Failed to load MAPE-PPI VAE model. {e}"); sys.exit(1)

    # Get codebook vectors
    codebook_vectors = vae_model.vq_layer.embeddings.weight.data.cpu().numpy()

    # Get usage statistics
    stats_cache_path = os.path.join(cache_dir, f"usage_stats_{args.dataset_name}_mapeppi.pkl")
    usage_stats = get_codebook_usage_stats_mapeppi(vae_model, protein_dataset, stats_cache_path)

    # Perform clustering
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10) # MAPE-PPI has a smaller codebook
    cluster_labels = kmeans.fit_predict(codebook_vectors)

    # Create and save the plot
    output_filename = os.path.join(args.output_dir, f"statistical_summary_mapeppi_{args.dataset_name}.png")
    create_statistical_summary_plot_mapeppi(usage_stats, cluster_labels, codebook_vectors, output_filename)

    print("\n--- Landscape analysis for MAPE-PPI completed! ---")

if __name__ == "__main__":
    main()
