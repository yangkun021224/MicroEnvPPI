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
from sklearn.metrics import f1_score, average_precision_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

# --- Add project root to sys.path ---
# This allows importing modules from the 'src' directory
try:
    from models import CodeBook
    from dataloader import ProteinDatasetDGL
except ImportError:
    # If the script is not run from the 'visualizations' directory, this might fail.
    # A more robust solution is adding the project's root directory to PYTHONPATH.
    print("Warning: Could not import from 'src'. Assuming the script is run from a directory where 'src' is accessible.")
    # A simple fallback for running from the root of the project
    if 'src' not in sys.path:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from models import CodeBook
    from dataloader import ProteinDatasetDGL


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

# --- Caching Mechanism ---
def save_to_cache(data, cache_path):
    """Saves data to a pickle cache file."""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        # print(f"Data successfully cached to: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache file {cache_path}. Error: {e}")

def load_from_cache(cache_path):
    """Loads data from a pickle cache file if it exists."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            # print(f"Successfully loaded data from cache: {cache_path}")
            return data
        except Exception as e:
            print(f"Warning: Failed to load cache file {cache_path}. Error: {e}")
    return None

# --- Data Loading and Model Initialization ---
def load_config_params(config_path, dataset_name, split_mode):
    """Loads model and training parameters from the main JSON config file."""
    try:
        with open(config_path, 'r') as f:
            full_config = json.load(f)
        params = full_config.get(dataset_name, {}).get(split_mode, {})
        if not params:
            raise ValueError(f"Parameters for dataset '{dataset_name}' with split '{split_mode}' not found in {config_path}")
        print(f"Successfully loaded configuration for {dataset_name}/{split_mode}.")
        return params
    except Exception as e:
        print(f"ERROR: Failed to load configuration from {config_path}. {e}")
        sys.exit(1)

def load_vae_model(params, ckpt_path):
    """Loads the pre-trained VAE (CodeBook) model."""
    try:
        model = CodeBook(params, protein_data=None).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        load_result = model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"VAE model loaded from {ckpt_path}.")
        print(f"  > Missing keys: {load_result.missing_keys}")
        print(f"  > Unexpected keys: {load_result.unexpected_keys}")
        return model
    except Exception as e:
        print(f"ERROR: Failed to initialize or load VAE model. {e}")
        sys.exit(1)

def load_protein_data(data_dir, dataset_name):
    """Loads protein sequences, IDs, and the DGL dataset."""
    seq_file_path = os.path.join(data_dir, f"protein.{dataset_name}.sequences.dictionary.csv")
    prot_r_edge_path = os.path.join(data_dir, f'protein.rball.edges.{dataset_name}.npy')
    prot_k_edge_path = os.path.join(data_dir, f'protein.knn.edges.{dataset_name}.npy')
    prot_node_path_esm = os.path.join(data_dir, f'protein.nodes.esm2_650M.{dataset_name}.pt')

    # Load sequences and IDs
    protein_sequences, protein_ids = {}, []
    try:
        with open(seq_file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    protein_sequences[row[0]] = row[1].upper()
                    protein_ids.append(row[0])
        print(f"Loaded {len(protein_ids)} protein sequences from {dataset_name}.")
    except FileNotFoundError:
        print(f"ERROR: Sequence dictionary not found at {seq_file_path}")
        sys.exit(1)

    # Load DGL dataset
    try:
        dataset = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_node_path_esm, dataset_name, expected_num_proteins=len(protein_ids))
        if len(dataset) == 0:
            raise ValueError("Initialized ProteinDatasetDGL is empty.")
        print(f"Loaded ProteinDatasetDGL with {len(dataset)} proteins.")
    except Exception as e:
        print(f"ERROR: Failed to initialize ProteinDatasetDGL. {e}")
        sys.exit(1)
        
    return protein_sequences, protein_ids, dataset

def load_interproscan_annotations(tsv_path, cache_dir, dataset_name):
    """Loads protein functional annotations from InterProScan TSV output."""
    cache_path = os.path.join(cache_dir, f"interproscan_annotations_{dataset_name}.pkl")
    cached_data = load_from_cache(cache_path)
    if cached_data:
        return cached_data['annotations'], cached_data['counts']

    annotations = defaultdict(dict)
    counts = defaultdict(collections.Counter)
    
    try:
        with open(tsv_path, 'r') as f:
            for line in tqdm(f, desc="Parsing InterProScan TSV"):
                fields = line.strip().split('\t')
                if len(fields) < 14: continue
                
                protein_id, db, signature_id, interpro_id, go_terms = fields[0], fields[3], fields[4], fields[11], fields[13]
                
                if db.upper() == "PFAM" and protein_id not in annotations['pfam']:
                    annotations['pfam'][protein_id] = signature_id
                    counts['pfam'][signature_id] += 1
                if interpro_id != '-' and protein_id not in annotations['interpro']:
                    annotations['interpro'][protein_id] = interpro_id
                    counts['interpro'][interpro_id] += 1
                if go_terms != '-' and protein_id not in annotations['go']:
                    first_go = go_terms.split('|')[0]
                    annotations['go'][protein_id] = first_go
                    counts['go'][first_go] += 1
        print(f"Loaded annotations for {len(annotations['pfam'])} proteins from PFAM.")
    except FileNotFoundError:
        print(f"Warning: InterProScan results not found at {tsv_path}. Annotation plots will be skipped.")
    
    save_to_cache({'annotations': annotations, 'counts': counts}, cache_path)
    return annotations, counts

# --- Visualization and Analysis Functions ---
def plot_embeddings_2d(embeddings, title, filename, colors=None, cmap='viridis', s=5, alpha=0.7, legend_elements=None):
    """Generic 2D scatter plot for embeddings."""
    plt.figure(figsize=(14, 12))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap=cmap, s=s, alpha=alpha)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel("Dimension 1", fontsize=16)
    plt.ylabel("Dimension 2", fontsize=16)
    plt.xticks([])
    plt.yticks([])
    if legend_elements:
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10, markerscale=2)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved plot to: {filename}")
    plt.close()

def visualize_codebook_vectors(vae_model, output_dir, dataset_name):
    """Visualizes the codebook vectors using t-SNE and UMAP."""
    print("\n--- Visualizing Codebook Vectors ---")
    codebook_vectors = vae_model.vq_layer.embeddings.weight.data.cpu().numpy()
    num_embeddings = codebook_vectors.shape[0]

    # K-Means Clustering for coloring
    kmeans = KMeans(n_clusters=NUM_CODEBOOK_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(codebook_vectors)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(TSNE_PERPLEXITY, num_embeddings - 1), random_state=RANDOM_STATE, n_iter=1000)
    codebook_tsne = tsne.fit_transform(codebook_vectors)
    plot_embeddings_2d(codebook_tsne, f"t-SNE of {num_embeddings} Codebook Vectors (MicroEnvPPI)", 
                       os.path.join(output_dir, f"codebook_tsne_{dataset_name}.png"), colors=cluster_labels, cmap='tab10', s=15)

    # UMAP
    reducer = umap.UMAP(n_neighbors=min(UMAP_N_NEIGHBORS, num_embeddings - 1), min_dist=UMAP_MIN_DIST, random_state=RANDOM_STATE)
    codebook_umap = reducer.fit_transform(codebook_vectors)
    plot_embeddings_2d(codebook_umap, f"UMAP of {num_embeddings} Codebook Vectors (MicroEnvPPI)", 
                       os.path.join(output_dir, f"codebook_umap_{dataset_name}.png"), colors=cluster_labels, cmap='tab10', s=15)

def visualize_residue_embeddings(vae_model, dataset, protein_sequences, output_dir, dataset_name):
    """Visualizes residue embeddings from a sample of proteins."""
    print("\n--- Visualizing Residue Embeddings ---")
    all_embeddings, all_aa_labels, all_codebook_indices = [], [], []
    sample_indices = np.random.choice(len(dataset), NUM_PROTEINS_FOR_RESIDUE_VIS, replace=False)
    
    aa_map = {'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19}

    for idx in tqdm(sample_indices, desc="Extracting Residue Embeddings"):
        graph = dataset[idx]
        if graph is None or graph.num_nodes('amino_acid') == 0: continue
        
        protein_id = list(protein_sequences.keys())[idx]
        sequence = protein_sequences[protein_id]
        
        with torch.no_grad():
            h = vae_model.Protein_Encoder.encoding(graph.to(device))
            _, _, _, indices = vae_model.vq_layer(h)
        
        num_residues = h.shape[0]
        res_indices = np.random.choice(num_residues, min(num_residues, MAX_RESIDUES_PER_PROTEIN_VIS), replace=False)
        
        all_embeddings.append(h[res_indices].cpu())
        all_codebook_indices.append(indices[res_indices].cpu())
        all_aa_labels.extend([aa_map.get(sequence[i], 20) for i in res_indices])
        
    if not all_embeddings:
        print("Warning: No residue embeddings were extracted. Skipping visualization.")
        return
        
    embeddings_np = torch.cat(all_embeddings).numpy()
    indices_np = torch.cat(all_codebook_indices).numpy()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(TSNE_PERPLEXITY, embeddings_np.shape[0] - 1), random_state=RANDOM_STATE)
    residue_tsne = tsne.fit_transform(embeddings_np)
    plot_embeddings_2d(residue_tsne, "t-SNE of Residue Embeddings (Colored by AA Type)", 
                       os.path.join(output_dir, f"residue_tsne_aa_type_{dataset_name}.png"), colors=all_aa_labels, cmap='tab20', s=3)
    plot_embeddings_2d(residue_tsne, "t-SNE of Residue Embeddings (Colored by Codebook Index)", 
                       os.path.join(output_dir, f"residue_tsne_codebook_idx_{dataset_name}.png"), colors=indices_np, cmap='viridis', s=3)

def visualize_protein_embeddings(vae_model, dataset, protein_ids, annotations, counts, output_dir, dataset_name):
    """Visualizes full protein embeddings, colored by functional annotations."""
    print("\n--- Visualizing Protein Embeddings ---")
    
    cache_path = os.path.join(output_dir, f"protein_embeddings_cache_{dataset_name}.pkl")
    cached_data = load_from_cache(cache_path)
    
    if cached_data:
        protein_embeddings = cached_data['embeddings']
    else:
        with torch.no_grad():
            protein_embeddings = vae_model.get_protein_embeddings().cpu().numpy()
        save_to_cache({'embeddings': protein_embeddings}, cache_path)

    sample_indices = np.random.choice(len(protein_embeddings), min(len(protein_embeddings), NUM_PROTEINS_FOR_PROTEIN_VIS), replace=False)
    embeddings_sample = protein_embeddings[sample_indices]
    ids_sample = [protein_ids[i] for i in sample_indices]

    # Plot for each annotation type
    for ann_type in ['pfam', 'go']:
        if not annotations[ann_type]: continue

        top_anns = [item[0] for item in counts[ann_type].most_common(TOP_N_ANNOTATIONS)]
        ann_map = {ann: i for i, ann in enumerate(top_anns)}
        
        colors = [ann_map.get(annotations[ann_type].get(pid), TOP_N_ANNOTATIONS) for pid in ids_sample]
        
        tsne = TSNE(n_components=2, perplexity=min(TSNE_PERPLEXITY, embeddings_sample.shape[0] - 1), random_state=RANDOM_STATE)
        protein_tsne = tsne.fit_transform(embeddings_sample)

        legend_elements = [Line2D([0], [0], marker='o', color='w', label=ann, markerfacecolor=plt.cm.tab20(i/TOP_N_ANNOTATIONS)) for i, ann in enumerate(top_anns)]
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='grey'))

        plot_embeddings_2d(protein_tsne, f"t-SNE of Protein Embeddings (Colored by {ann_type.upper()})",
                           os.path.join(output_dir, f"protein_tsne_{ann_type}_{dataset_name}.png"),
                           colors=colors, cmap='tab20', s=10, legend_elements=legend_elements)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="MicroEnvPPI Visualization Script")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the param_configs.json file.")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="Path to the trained VAE model checkpoint (.ckpt).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the processed_data directory.")
    parser.add_argument("--dataset_name", type=str, default="STRING", help="Name of the dataset (e.g., STRING, SHS27k).")
    parser.add_argument("--split_mode", type=str, default="dfs", help="Data split mode (e.g., dfs, bfs, random).")
    parser.add_argument("--interpro_tsv_path", type=str, required=True, help="Path to the InterProScan TSV results file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualization plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Device: {device}")
    
    # Load everything
    params = load_config_params(args.config_path, args.dataset_name, args.split_mode)
    protein_sequences, protein_ids, dataset = load_protein_data(args.data_dir, args.dataset_name)
    vae_model = load_vae_model(params, args.vae_ckpt_path)
    vae_model.protein_dataset = dataset # Link dataset to model for embedding generation
    annotations, counts = load_interproscan_annotations(args.interpro_tsv_path, cache_dir, args.dataset_name)

    # Run visualizations
    visualize_codebook_vectors(vae_model, args.output_dir, args.dataset_name)
    visualize_residue_embeddings(vae_model, dataset, protein_sequences, args.output_dir, args.dataset_name)
    visualize_protein_embeddings(vae_model, dataset, protein_ids, annotations, counts, args.output_dir, args.dataset_name)

    print("\n--- All visualization tasks completed ---")

if __name__ == "__main__":
    main()
