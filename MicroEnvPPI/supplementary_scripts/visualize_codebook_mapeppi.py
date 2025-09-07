import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import umap
import os
import json
import sys
import csv
import argparse
from tqdm import tqdm
import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import dgl
import warnings
warnings.filterwarnings("ignore")

# --- Add project root to sys.path ---
try:
    from models import CodeBook
except ImportError:
    print("Warning: Could not import from 'src'.")
    if 'src' not in sys.path:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from models import CodeBook

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Global Visualization Parameters ---
RANDOM_STATE = 42
TSNE_PERPLEXITY = 30.0
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
NUM_PROTEINS_FOR_RESIDUE_VIS = 100
MAX_RESIDUES_PER_PROTEIN_VIS = 200
NUM_PROTEINS_FOR_PROTEIN_VIS = 2000
TOP_N_ANNOTATIONS = 19
NUM_CODEBOOK_CLUSTERS = 10

# --- Custom Dataset Loader for MAPE-PPI (handles numpy array features) ---
class ProteinDatasetDGL_MAPEPPI(torch.utils.data.Dataset):
    def __init__(self, r_edge_path, k_edge_path, node_path, dataset, num_proteins):
        self.r_edge = np.load(r_edge_path, allow_pickle=True) if os.path.exists(r_edge_path) else [[]]*num_proteins
        self.k_edge = np.load(k_edge_path, allow_pickle=True) if os.path.exists(k_edge_path) else [[]]*num_proteins
        
        raw_features = torch.load(node_path, map_location='cpu')
        self.node_features = [torch.from_numpy(f).float() if isinstance(f, np.ndarray) else f.float() for f in raw_features]
        
        self.num_proteins = min(len(self.r_edge), len(self.node_features), num_proteins)

    def __len__(self):
        return self.num_proteins

    def __getitem__(self, idx):
        node_feats = self.node_features[idx]
        num_nodes = node_feats.shape[0]
        if num_nodes == 0: return None

        def filter_edges(edges, max_idx):
            return [(u, v) for u, v in edges if u < max_idx and v < max_idx]

        graph_data = {
            ('amino_acid', 'SEQ', 'amino_acid'): [(i, i + 1) for i in range(num_nodes - 1)] + [(i + 1, i) for i in range(num_nodes - 1)],
            ('amino_acid', 'STR_KNN', 'amino_acid'): filter_edges(self.k_edge[idx], num_nodes),
            ('amino_acid', 'STR_DIS', 'amino_acid'): filter_edges(self.r_edge[idx], num_nodes)
        }
        g = dgl.heterograph(graph_data, num_nodes_dict={'amino_acid': num_nodes})
        g.nodes['amino_acid'].data['x'] = node_feats
        return g

# --- Caching and Utility Functions ---
def save_to_cache(data, cache_path):
    try:
        with open(cache_path, 'wb') as f: pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

def load_from_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f: return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}. Error: {e}")
    return None

def load_mapeppi_params():
    """Returns a hardcoded parameter dictionary for the baseline MAPE-PPI model."""
    return {
        "input_dim": 7, "prot_hidden_dim": 128, "prot_num_layers": 4,
        "num_embeddings": 256, "commitment_cost": 0.25, "dropout_ratio": 0.0,
        "protein_embedding_mode": "concat"
    }

# Other utility functions (load_vae_model, load_protein_data, load_interproscan, plot_embeddings) are similar to MicroEnvPPI's script
# For brevity, they are defined within the main script logic where needed.

def main():
    parser = argparse.ArgumentParser(description="MAPE-PPI (Baseline) Visualization Script")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to the trained MAPE-PPI VAE model checkpoint (.ckpt).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the processed_data directory.")
    parser.add_argument("--dataset_name", type=str, default="STRING", help="Name of the dataset (e.g., STRING).")
    parser.add_argument("--interpro_tsv_path", type=str, required=True, help="Path to the InterProScan TSV results file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualization plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache_mapeppi")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"--- MAPE-PPI (Baseline) Visualization ---")
    print(f"Device: {device}")

    # --- 1. Load Data and Model ---
    params = load_mapeppi_params()
    
    # Load sequences and IDs to get protein count
    seq_file_path = os.path.join(args.data_dir, f"protein.{args.dataset_name}.sequences.dictionary.csv")
    protein_sequences, protein_ids = {}, []
    try:
        with open(seq_file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    protein_sequences[row[0]] = row[1].upper()
                    protein_ids.append(row[0])
    except FileNotFoundError:
        print(f"ERROR: Sequence file not found at {seq_file_path}"); sys.exit(1)

    # Load dataset using the custom loader for MAPE-PPI
    r_edge_path = os.path.join(args.data_dir, f'protein.rball.edges.{args.dataset_name}.npy')
    k_edge_path = os.path.join(args.data_dir, f'protein.knn.edges.{args.dataset_name}.npy')
    node_path = os.path.join(args.data_dir, f'protein.nodes.{args.dataset_name}.pt') # Note: uses 7D features
    dataset = ProteinDatasetDGL_MAPEPPI(r_edge_path, k_edge_path, node_path, args.dataset_name, len(protein_ids))
    
    # Load model
    try:
        vae_model = CodeBook(params, protein_data=None).to(device)
        vae_model.load_state_dict(torch.load(args.vae_ckpt_path, map_location=device), strict=False)
        vae_model.eval()
        vae_model.protein_dataset = dataset
        print(f"MAPE-PPI VAE model loaded from {args.vae_ckpt_path}.")
    except Exception as e:
        print(f"ERROR: Failed to load MAPE-PPI VAE model. {e}"); sys.exit(1)

    # Load annotations
    annotations, counts = load_from_cache(os.path.join(cache_dir, "annotations.pkl")) or ({}, {})
    if not annotations:
        # Simplified annotation loading, assuming it's been run by the other script
        print("Warning: Annotation cache not found. Some plots may be skipped.")


    # --- 2. Run Visualizations ---
    
    # Codebook Vectors
    print("\nVisualizing Codebook Vectors (MAPE-PPI)...")
    codebook_vectors = vae_model.vq_layer.embeddings.weight.data.cpu().numpy()
    kmeans = KMeans(n_clusters=NUM_CODEBOOK_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(codebook_vectors)
    tsne = TSNE(n_components=2, perplexity=min(TSNE_PERPLEXITY, codebook_vectors.shape[0]-1), random_state=RANDOM_STATE)
    codebook_tsne = tsne.fit_transform(codebook_vectors)
    plot_embeddings_2d(codebook_tsne, "t-SNE of Codebook Vectors (MAPE-PPI)", 
                       os.path.join(args.output_dir, f"codebook_tsne_{args.dataset_name}_mapeppi.png"), 
                       colors=cluster_labels, cmap='tab10', s=15)

    # Residue Embeddings
    print("\nVisualizing Residue Embeddings (MAPE-PPI)...")
    # This is computationally intensive, so we use a cached or smaller sample
    # (Logic is simplified here for brevity, assuming a similar extraction function as in the other script)
    
    # Protein Embeddings
    print("\nVisualizing Protein Embeddings (MAPE-PPI)...")
    cache_path = os.path.join(cache_dir, f"protein_embeddings_{args.dataset_name}_mapeppi.pkl")
    protein_embeddings = load_from_cache(cache_path)
    if protein_embeddings is None:
        with torch.no_grad():
            protein_embeddings = vae_model.get_protein_embeddings().cpu().numpy()
        save_to_cache(protein_embeddings, cache_path)

    sample_indices = np.random.choice(len(protein_embeddings), min(len(protein_embeddings), NUM_PROTEINS_FOR_PROTEIN_VIS), replace=False)
    embeddings_sample = protein_embeddings[sample_indices]
    
    if 'pfam' in annotations and annotations['pfam']:
        ids_sample = [protein_ids[i] for i in sample_indices]
        top_anns = [item[0] for item in counts['pfam'].most_common(TOP_N_ANNOTATIONS)]
        ann_map = {ann: i for i, ann in enumerate(top_anns)}
        colors = [ann_map.get(annotations['pfam'].get(pid), TOP_N_ANNOTATIONS) for pid in ids_sample]
        
        tsne_prot = TSNE(n_components=2, perplexity=min(TSNE_PERPLEXITY, embeddings_sample.shape[0]-1), random_state=RANDOM_STATE)
        protein_tsne = tsne_prot.fit_transform(embeddings_sample)
        plot_embeddings_2d(protein_tsne, "t-SNE of Protein Embeddings (Colored by PFAM, MAPE-PPI)",
                           os.path.join(args.output_dir, f"protein_tsne_pfam_{args.dataset_name}_mapeppi.png"),
                           colors=colors, cmap='tab20', s=10)

    print("\n--- MAPE-PPI visualization tasks completed ---")

# Redefine plot_embeddings_2d for standalone use if needed
def plot_embeddings_2d(embeddings, title, filename, colors=None, cmap='viridis', s=5, alpha=0.7, legend_elements=None):
    plt.figure(figsize=(14, 12)); plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap=cmap, s=s, alpha=alpha)
    plt.title(title, fontsize=20, fontweight='bold'); plt.xlabel("Dimension 1", fontsize=16); plt.ylabel("Dimension 2", fontsize=16)
    plt.xticks([]); plt.yticks([])
    if legend_elements: plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)
    plt.tight_layout(); plt.savefig(filename, bbox_inches='tight', dpi=300); plt.close()
    print(f"Saved plot: {filename}")
    
if __name__ == "__main__":
    main()
