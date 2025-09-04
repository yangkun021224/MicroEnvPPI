import os
import csv
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
import sys
import traceback

import dgl
import torch
from torch.utils.data import Dataset

try:
    from utils import get_bfs_sub_graph, get_dfs_sub_graph
except ImportError:
    # Fallback implementation in case utils is not available
    def get_bfs_sub_graph(*args): return []
    def get_dfs_sub_graph(*args): return []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(param):
    """
    Loads data based on the provided parameters.
    This function is now driven by the param dictionary to support different feature types (ESM or 7D).
    """
    dataset = param['dataset']
    split_mode = param['split_mode']
    seed = param['seed']
    input_dim = param['input_dim']
    skip_head=True
    
    protein_name_to_id = {}
    ppi_pair_to_edge_idx = {}
    ppi_list = []
    ppi_label_list = []

    class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}
    num_classes = len(class_map)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir, '..', 'data', 'processed_data')

    if not os.path.isdir(base_data_path):
        print(f"Data directory {base_data_path} not found!", file=sys.stderr)
        exit(1)

    # --- MODIFICATION START: Dynamically determine feature filename ---
    embedding_model_name = param.get("embedding_model_name", "esm2_650M")
    print(f"Loading node features of type: {embedding_model_name}")
    # --- MODIFICATION END ---

    ppi_path = os.path.join(base_data_path, f'protein.actions.{dataset}.txt')
    prot_seq_path = os.path.join(base_data_path, f'protein.{dataset}.sequences.dictionary.csv')
    prot_r_edge_path = os.path.join(base_data_path, f'protein.rball.edges.{dataset}.npy')
    prot_k_edge_path = os.path.join(base_data_path, f'protein.knn.edges.{dataset}.npy')
    prot_node_path = os.path.join(base_data_path, f'protein.nodes.{embedding_model_name}.{dataset}.pt')


    if not os.path.exists(prot_node_path):
        print(f"Node feature file {prot_node_path} not found!", file=sys.stderr)
        print("Please ensure you have generated the 7D features for the ablation study (by running generate_7d_features.py).", file=sys.stderr)
        exit(1)

    ppi_pkl_path = os.path.join(base_data_path, f'{dataset}_ppi.pkl')
    ppi_label_pkl_path = os.path.join(base_data_path, f'{dataset}_ppi_label.pkl')

    current_id = 0
    protein_id_to_name = {}
    if os.path.exists(prot_seq_path):
        with open(prot_seq_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 1 and row[0]:
                    protein_name = row[0]
                    if protein_name not in protein_name_to_id:
                        protein_name_to_id[protein_name] = current_id
                        protein_id_to_name[current_id] = protein_name
                        current_id += 1
    else:
        print(f"Protein sequence dictionary {prot_seq_path} not found!", file=sys.stderr)
        exit(1)
    
    num_proteins = len(protein_name_to_id)

    if os.path.exists(ppi_pkl_path) and os.path.exists(ppi_label_pkl_path):
        with open(ppi_pkl_path, "rb") as tf: 
            ppi_list = pickle.load(tf)
        with open(ppi_label_pkl_path, "rb") as tf: 
            ppi_label_list = pickle.load(tf)
    else:
        ppi_pair_to_edge_idx = {}
        edge_idx_counter = 0
        ppi_list = []
        ppi_label_list = []
        
        with open(ppi_path, 'r') as f_ppi:
            if skip_head:
                header = next(f_ppi, None)
            for line in tqdm(f_ppi, desc="Processing PPI actions", unit=" lines"):
                parts = line.strip().split('\t')
                if len(parts) < 3: continue

                p1_name, p2_name, interaction_type = parts[0], parts[1], parts[2]

                if p1_name not in protein_name_to_id or p2_name not in protein_name_to_id: continue
                if interaction_type not in class_map: continue

                p1_id, p2_id = protein_name_to_id[p1_name], protein_name_to_id[p2_name]

                if p1_id == p2_id: continue

                pair_key = tuple(sorted((p1_id, p2_id)))
                type_idx = class_map[interaction_type]

                if pair_key not in ppi_pair_to_edge_idx:
                    ppi_pair_to_edge_idx[pair_key] = edge_idx_counter
                    ppi_list.append(list(pair_key))
                    new_label = [0] * num_classes
                    new_label[type_idx] = 1
                    ppi_label_list.append(new_label)
                    edge_idx_counter += 1
                else:
                    existing_edge_idx = ppi_pair_to_edge_idx[pair_key]
                    ppi_label_list[existing_edge_idx][type_idx] = 1

        if ppi_list and ppi_label_list:
            with open(ppi_pkl_path, "wb") as tf: pickle.dump(ppi_list, tf)
            with open(ppi_label_pkl_path, "wb") as tf: pickle.dump(ppi_label_list, tf)

    if not ppi_list: 
        print("No valid PPI data!", file=sys.stderr)
        exit(1)
    
    try:
        ppi_g = dgl.graph(ppi_list, num_nodes=num_proteins)
        ppi_g = dgl.to_bidirected(ppi_g, copy_ndata=False).to(device)
    except Exception as e: 
        print(f"Error building PPI graph: {e}", file=sys.stderr)
        exit(1)

    protein_data = None
    try:
        # --- MODIFICATION START: Pass input_dim ---
        protein_data = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset, expected_num_proteins=num_proteins, expected_node_dim=input_dim)
        # --- MODIFICATION END ---
    except Exception as e: 
        print(f"Error initializing ProteinDatasetDGL: {e}", file=sys.stderr)
        exit(1)

    ppi_split_dict = split_dataset(ppi_list, dataset, split_mode, seed)
    if ppi_split_dict is None: 
        print("Dataset split failed!", file=sys.stderr)
        exit(1)

    try:
        labels_tensor = torch.from_numpy(np.array(ppi_label_list, dtype=np.float32)).to(device)
    except Exception as e: 
        print(f"Error creating label tensor: {e}", file=sys.stderr)
        exit(1)

    return protein_data, ppi_g, ppi_list, labels_tensor, ppi_split_dict


class ProteinDatasetDGL(Dataset):
    # --- MODIFICATION START: Add expected_node_dim parameter ---
    def __init__(self, prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset, expected_num_proteins, expected_node_dim):
    # --- MODIFICATION END ---
        self.dataset = dataset
        self.prot_node_path = prot_node_path
        self.expected_num_proteins = expected_num_proteins

        self.prot_r_edge, self.prot_k_edge = None, None
        try:
            if os.path.exists(prot_r_edge_path): self.prot_r_edge = np.load(prot_r_edge_path, allow_pickle=True)
            if os.path.exists(prot_k_edge_path): self.prot_k_edge = np.load(prot_k_edge_path, allow_pickle=True)
            len_r = len(self.prot_r_edge) if self.prot_r_edge is not None else 0
            len_k = len(self.prot_k_edge) if self.prot_k_edge is not None else 0
            self.num_edge_entries = min(len_r, len_k) if len_r > 0 and len_k > 0 else max(len_r, len_k)
        except Exception as e: 
            print(f"Error loading edge files: {e}", file=sys.stderr)
            exit(1)

        self.prot_node_features_list = None
        try:
            self.prot_node_features_list = torch.load(prot_node_path, map_location='cpu')
            if not isinstance(self.prot_node_features_list, list):
                print(f"Node feature file content is not a list!", file=sys.stderr)
                exit(1)
            self.num_feature_entries = len(self.prot_node_features_list)
        except Exception as e: 
            print(f"Error loading node features: {e}", file=sys.stderr)
            exit(1)

        self.num_proteins = min(self.num_edge_entries, self.num_feature_entries) if self.num_edge_entries > 0 else self.num_feature_entries
        if expected_num_proteins != -1: self.num_proteins = expected_num_proteins

        if self.num_proteins == 0:
            print("No available protein data!", file=sys.stderr)
            exit(1)

        # --- MODIFICATION START: Validate feature dimension ---
        self.expected_node_dim = expected_node_dim
        # --- MODIFICATION END ---

    def __len__(self):
        return self.num_proteins

    def __getitem__(self, idx):
        if not (0 <= idx < self.num_proteins):
            raise IndexError(f"Index {idx} out of range")

        try:
            if idx >= len(self.prot_node_features_list): return None
            node_features = self.prot_node_features_list[idx]
            if not isinstance(node_features, torch.Tensor) or node_features.numel() == 0: return None
            
            # --- MODIFICATION START: Validate feature dimension ---
            if node_features.shape[1] != self.expected_node_dim:
                # print(f"Warning: Protein {idx} has feature dimension {node_features.shape[1]}, expected {self.expected_node_dim}. Skipping.", file=sys.stderr)
                return None
            # --- MODIFICATION END ---

            num_nodes = node_features.shape[0]
            
            r_edges = self.prot_r_edge[idx] if self.prot_r_edge is not None and idx < len(self.prot_r_edge) else []
            k_edges = self.prot_k_edge[idx] if self.prot_k_edge is not None and idx < len(self.prot_k_edge) else []
            prot_seq = [(j, j+1) for j in range(num_nodes-1)] + [(j+1, j) for j in range(num_nodes-1)] if num_nodes > 1 else []

            def filter_edges(edges, max_idx):
                return [(u, v) for u, v in edges if u < max_idx and v < max_idx] if hasattr(edges, '__iter__') else []

            graph_data = {
                ('amino_acid', 'SEQ', 'amino_acid'): prot_seq,
                ('amino_acid', 'STR_KNN', 'amino_acid'): filter_edges(k_edges, num_nodes),
                ('amino_acid', 'STR_DIS', 'amino_acid'): filter_edges(r_edges, num_nodes)
            }
            prot_g = dgl.heterograph(graph_data, num_nodes_dict={'amino_acid': num_nodes})
            prot_g.nodes['amino_acid'].data['x'] = node_features.float()
            return prot_g
        except Exception:
            return None

def collate(samples):
    valid_samples = [s for s in samples if isinstance(s, dgl.DGLGraph)]
    return dgl.batch(valid_samples) if valid_samples else None

def split_dataset(ppi_list, dataset, split_mode, seed):
    split_file_path = os.path.join('..', 'data', 'processed_data', f'{dataset}_{split_mode}.json')

    if not os.path.exists(split_file_path):
        random.seed(seed)
        ppi_num = len(ppi_list)
        indices = list(range(ppi_num))
        
        ppi_split_dict = {}
        if split_mode == 'random':
            random.shuffle(indices)
            train_end, val_end = int(ppi_num * 0.6), int(ppi_num * 0.8)
            ppi_split_dict = {'train_index': indices[:train_end], 'val_index': indices[train_end:val_end], 'test_index': indices[val_end:]}
        elif split_mode in ['bfs', 'dfs']:
            node_to_edge_index = {i: [] for i in range(np.max(ppi_list) + 1)}
            for i, edge in enumerate(ppi_list): node_to_edge_index[edge[0]].append(i); node_to_edge_index[edge[1]].append(i)
            
            sub_graph_size = int(ppi_num * 0.4)
            func = get_bfs_sub_graph if split_mode == 'bfs' else get_dfs_sub_graph
            selected_edge_index = func(ppi_list, len(node_to_edge_index), node_to_edge_index, sub_graph_size)
            
            train_indices = list(set(indices) - set(selected_edge_index))
            val_test_indices = list(set(selected_edge_index))
            random.shuffle(val_test_indices)
            val_num = int(ppi_num * 0.2)
            ppi_split_dict = {'train_index': train_indices, 'val_index': val_test_indices[:val_num], 'test_index': val_test_indices[val_num:]}
        
        with open(split_file_path, 'w') as f: json.dump(ppi_split_dict, f, indent=4)
    else:
        with open(split_file_path, 'r') as f: ppi_split_dict = json.load(f)

    return ppi_split_dict

def load_pretrain_data(dataset):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir, '..', 'data', 'processed_data')
    embedding_model_name = "esm2_650M" # Pre-training always uses ESM

    prot_r_edge_path = os.path.join(base_data_path, f'protein.rball.edges.{dataset}.npy')
    prot_k_edge_path = os.path.join(base_data_path, f'protein.knn.edges.{dataset}.npy')
    prot_node_path = os.path.join(base_data_path, f'protein.nodes.{embedding_model_name}.{dataset}.pt')

    if not os.path.exists(prot_node_path): 
        print(f"Pretrain node file {prot_node_path} not found!", file=sys.stderr)
        exit(1)

    try:
        protein_data = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset, expected_num_proteins=-1, expected_node_dim=1280)
        return protein_data
    except Exception as e: 
        print(f"Error loading pretrain data: {e}", file=sys.stderr)
        exit(1)

