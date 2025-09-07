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
    def get_bfs_sub_graph(*args): return []
    def get_dfs_sub_graph(*args): return []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
expected_node_dim = 1280
embedding_model_name = "esm2_650M"


def load_data(dataset, split_mode, seed, skip_head=True):
    protein_name_to_id = {}
    ppi_pair_to_edge_idx = {}
    ppi_list = []
    ppi_label_list = []

    class_map = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3, 'inhibition':4, 'catalysis':5, 'expression':6}
    num_classes = len(class_map)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir, '..', 'data', 'processed_data')

    if not os.path.isdir(base_data_path):
        print(f"Data directory {base_data_path} not found!")
        exit(1)

    ppi_path = os.path.join(base_data_path, f'protein.actions.{dataset}.txt')
    prot_seq_path = os.path.join(base_data_path, f'protein.{dataset}.sequences.dictionary.csv')
    prot_r_edge_path = os.path.join(base_data_path, f'protein.rball.edges.{dataset}.npy')
    prot_k_edge_path = os.path.join(base_data_path, f'protein.knn.edges.{dataset}.npy')
    prot_node_path = os.path.join(base_data_path, f'protein.nodes.{embedding_model_name}.{dataset}.pt')

    if not os.path.exists(prot_node_path):
        print(f"Node feature file {prot_node_path} not found!")
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
        print(f"Protein sequence dictionary {prot_seq_path} not found!")
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

        line_count = 0
        processed_count = 0
        skipped_missing_protein = 0
        skipped_unknown_type = 0
        skipped_self_loop = 0
        
        with open(ppi_path, 'r') as f_ppi:
            if skip_head:
                header = next(f_ppi, None)
            for line in tqdm(f_ppi, desc="Processing PPI actions", unit=" lines"):
                line_count += 1
                parts = line.strip().split('\t')
                if len(parts) < 3: 
                    continue

                p1_name, p2_name, interaction_type = parts[0], parts[1], parts[2]

                if p1_name not in protein_name_to_id or p2_name not in protein_name_to_id:
                    skipped_missing_protein += 1
                    continue
                if interaction_type not in class_map:
                    skipped_unknown_type += 1
                    continue

                p1_id = protein_name_to_id[p1_name]
                p2_id = protein_name_to_id[p2_name]

                if p1_id == p2_id: 
                    skipped_self_loop += 1
                    continue

                pair_key = tuple(sorted((p1_id, p2_id)))
                pair_list_entry = list(pair_key)
                type_idx = class_map[interaction_type]

                if pair_key not in ppi_pair_to_edge_idx:
                    current_edge_idx = edge_idx_counter
                    ppi_pair_to_edge_idx[pair_key] = current_edge_idx
                    ppi_list.append(pair_list_entry)
                    new_label = [0] * num_classes
                    new_label[type_idx] = 1
                    ppi_label_list.append(new_label)
                    edge_idx_counter += 1
                else:
                    existing_edge_idx = ppi_pair_to_edge_idx[pair_key]
                    ppi_label_list[existing_edge_idx][type_idx] = 1

                processed_count += 1

        if ppi_list and ppi_label_list:
            with open(ppi_pkl_path, "wb") as tf: 
                pickle.dump(ppi_list, tf)
            with open(ppi_label_pkl_path, "wb") as tf: 
                pickle.dump(ppi_label_list, tf)

    if not ppi_list: 
        print("No valid PPI data!")
        exit(1)

    ppi_g = None
    try:
        valid_ppi_list_for_graph = []
        invalid_count = 0
        max_id_in_ppi = -1
        for p in ppi_list:
            if isinstance(p, list) and len(p) == 2 and all(isinstance(i, int) and 0 <= i < num_proteins for i in p):
                valid_ppi_list_for_graph.append(p)
                max_id_in_ppi = max(max_id_in_ppi, p[0], p[1])
            else:
                invalid_count += 1

        if not valid_ppi_list_for_graph: 
            raise ValueError("No valid edges to build graph.")

        ppi_g = dgl.graph(valid_ppi_list_for_graph, num_nodes=num_proteins)
        ppi_g = dgl.to_bidirected(ppi_g, copy_ndata=False)
        ppi_g = ppi_g.to(device)
    except Exception as e: 
        print(f"Error building PPI graph: {e}")
        exit(1)

    protein_data = None
    try:
        protein_data = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset, expected_num_proteins=num_proteins)
    except Exception as e: 
        print(f"Error initializing ProteinDatasetDGL: {e}")
        exit(1)

    ppi_split_dict = split_dataset(ppi_list, dataset, split_mode, seed)
    if ppi_split_dict is None: 
        print("Dataset split failed!")
        exit(1)

    labels_tensor = None
    try:
        if not isinstance(ppi_label_list, (np.ndarray, list)): 
            raise TypeError("ppi_label_list format error")
        labels_array = np.array(ppi_label_list, dtype=np.float32)
        if labels_array.shape[0] != len(ppi_list): 
            raise ValueError("Label count mismatch with PPI list count")
        if labels_array.shape[1] != num_classes: 
            raise ValueError(f"Label dimension ({labels_array.shape[1]}) mismatch with class count ({num_classes})")
        labels_tensor = torch.from_numpy(labels_array).to(device)
    except Exception as e: 
        print(f"Error creating label tensor: {e}")
        exit(1)

    return protein_data, ppi_g, ppi_list, labels_tensor, ppi_split_dict


class ProteinDatasetDGL(Dataset):
    def __init__(self, prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset, expected_num_proteins):
        self.dataset = dataset
        self.prot_node_path = prot_node_path
        self.expected_num_proteins = expected_num_proteins

        self.prot_r_edge = None
        self.prot_k_edge = None
        self.num_edge_entries = 0
        try:
            if os.path.exists(prot_r_edge_path):
                self.prot_r_edge = np.load(prot_r_edge_path, allow_pickle=True)
            if os.path.exists(prot_k_edge_path):
                self.prot_k_edge = np.load(prot_k_edge_path, allow_pickle=True)

            len_r = len(self.prot_r_edge) if self.prot_r_edge is not None else 0
            len_k = len(self.prot_k_edge) if self.prot_k_edge is not None else 0

            if len_r > 0 and len_k > 0 and len_r != len_k:
                self.num_edge_entries = min(len_r, len_k)
            elif len_r > 0: 
                self.num_edge_entries = len_r
            elif len_k > 0: 
                self.num_edge_entries = len_k
            else: 
                self.num_edge_entries = 0

        except Exception as e: 
            print(f"Error loading edge files: {e}")
            exit(1)

        self.prot_node_features_list = None
        self.num_feature_entries = 0
        try:
            self.prot_node_features_list = torch.load(prot_node_path, map_location='cpu')
            if not isinstance(self.prot_node_features_list, list):
                print(f"Node feature file content is not a list!")
                exit(1)
            self.num_feature_entries = len(self.prot_node_features_list)

            if self.num_feature_entries > 0:
                first_feat = self.prot_node_features_list[0]
                if isinstance(first_feat, torch.Tensor):
                    feat_shape = first_feat.shape
                    if len(feat_shape) != 2 or feat_shape[1] != expected_node_dim:
                        pass
            elif self.num_feature_entries == 0: 
                print("Loaded node feature list is empty!")
                exit(1)

        except FileNotFoundError as e: 
            print(f"Error loading node feature file: {e}")
            exit(1)
        except Exception as e: 
            print(f"Error loading node features: {e}")
            exit(1)

        self.num_proteins = min(self.num_edge_entries, self.num_feature_entries)
        if expected_num_proteins != -1 and self.num_proteins != expected_num_proteins:
            if expected_num_proteins > 0:
                self.num_proteins = expected_num_proteins

        if self.num_proteins == 0:
            print("No available protein data!")
            exit(1)

    def __len__(self):
        return self.num_proteins

    def __getitem__(self, idx):
        if not (0 <= idx < self.num_proteins):
            raise IndexError(f"Index {idx} out of range")

        try:
            if self.prot_node_features_list is None or idx >= len(self.prot_node_features_list):
                return None
            node_features = self.prot_node_features_list[idx]
            if not isinstance(node_features, torch.Tensor):
                return None
            num_nodes = node_features.shape[0]
            if num_nodes == 0:
                return None
            if len(node_features.shape) != 2 or node_features.shape[1] != expected_node_dim:
                return None

            r_edges_np = []
            k_edges_np = []
            if self.prot_r_edge is not None and idx < len(self.prot_r_edge):
                r_edges_np = self.prot_r_edge[idx]
            if self.prot_k_edge is not None and idx < len(self.prot_k_edge):
                k_edges_np = self.prot_k_edge[idx]

            prot_seq = []
            if num_nodes > 1:
                nodes_arange = torch.arange(num_nodes - 1)
                edges_forward = torch.stack([nodes_arange, nodes_arange + 1], dim=1)
                edges_backward = torch.stack([nodes_arange + 1, nodes_arange], dim=1)
                prot_seq = torch.cat([edges_forward, edges_backward]).tolist()

            def filter_edges(edges, max_node_idx):
                if edges is None or not hasattr(edges, '__iter__'): 
                    return []
                valid_edges = []
                for edge in edges:
                    try:
                        if isinstance(edge, (list, tuple)) and len(edge) == 2:
                            u, v = int(edge[0]), int(edge[1])
                            if 0 <= u < max_node_idx and 0 <= v < max_node_idx:
                                valid_edges.append((u, v))
                    except (ValueError, TypeError): 
                        continue
                return valid_edges

            current_r_edges = filter_edges(r_edges_np, num_nodes)
            current_k_edges = filter_edges(k_edges_np, num_nodes)

            graph_data = {
                ('amino_acid', 'SEQ', 'amino_acid'): prot_seq,
                ('amino_acid', 'STR_KNN', 'amino_acid'): current_k_edges,
                ('amino_acid', 'STR_DIS', 'amino_acid'): current_r_edges
            }
            prot_g = dgl.heterograph(graph_data, num_nodes_dict={'amino_acid': num_nodes})
            prot_g.nodes['amino_acid'].data['x'] = node_features.float()

            return prot_g

        except Exception as e:
            return None


def collate(samples):
    valid_samples = [s for s in samples if isinstance(s, dgl.DGLGraph)]
    if not valid_samples: 
        return None
    try:
        batched_graph = dgl.batch(valid_samples)
        return batched_graph
    except Exception as e:
        return None


def split_dataset(ppi_list, dataset, split_mode, seed):
    split_file_path = os.path.join('..', 'data', 'processed_data', f'{dataset}_{split_mode}.json')

    if not os.path.exists(split_file_path):
        random.seed(seed)
        if not isinstance(ppi_list, list): 
            return None
        ppi_num = len(ppi_list)
        if ppi_num == 0: 
            return None

        valid_edge_indices_for_split = [i for i, edge in enumerate(ppi_list) if isinstance(edge, list) and len(edge) == 2 and all(isinstance(x, int) for x in edge)]
        if not valid_edge_indices_for_split: 
            return None

        ppi_split_dict = {}
        if split_mode == 'random':
            indices = valid_edge_indices_for_split[:]
            random.shuffle(indices)
            train_end = int(len(indices) * 0.6)
            val_end = int(len(indices) * 0.8)
            ppi_split_dict = {'train_index': indices[:train_end], 'val_index': indices[train_end:val_end], 'test_index': indices[val_end:]}

        elif split_mode in ['bfs', 'dfs']:
            node_to_edge_index = {}
            max_node_id = -1
            for i in valid_edge_indices_for_split:
                edge = ppi_list[i]
                u, v = edge[0], edge[1]
                max_node_id = max(max_node_id, u, v)
                if u not in node_to_edge_index: 
                    node_to_edge_index[u] = []
                node_to_edge_index[u].append(i)
                if v not in node_to_edge_index: 
                    node_to_edge_index[v] = []
                node_to_edge_index[v].append(i)
            node_num = max_node_id + 1

            sub_graph_size = int(len(valid_edge_indices_for_split) * 0.4)
            sub_graph_size = max(1, sub_graph_size)

            if not node_to_edge_index: 
                return None

            selected_edge_index = []
            if split_mode == 'bfs':
                selected_edge_index = get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size)
            else:
                selected_edge_index = get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size)

            min_required_val_test = max(2, int(len(valid_edge_indices_for_split) * 0.4 * 0.1))
            if len(selected_edge_index) < min_required_val_test:
                indices = valid_edge_indices_for_split[:]
                random.shuffle(indices)
                train_end = int(len(indices) * 0.6)
                val_end = int(len(indices) * 0.8)
                ppi_split_dict = {'train_index': indices[:train_end], 'val_index': indices[train_end:val_end], 'test_index': indices[val_end:]}
            else:
                valid_selected_set = set(selected_edge_index).intersection(set(valid_edge_indices_for_split))
                all_valid_set = set(valid_edge_indices_for_split)
                train_indices = list(all_valid_set - valid_selected_set)
                val_test_indices = list(valid_selected_set)
                random.shuffle(val_test_indices)

                val_num = int(len(valid_edge_indices_for_split) * 0.2)
                val_num = max(1, val_num)
                val_num = min(val_num, len(val_test_indices) -1)

                val_indices = val_test_indices[:val_num]
                test_indices = val_test_indices[val_num:]

                if not train_indices or not val_indices or not test_indices:
                    indices = valid_edge_indices_for_split[:]
                    random.shuffle(indices)
                    train_end = int(len(indices) * 0.6)
                    val_end = int(len(indices) * 0.8)
                    ppi_split_dict = {'train_index': indices[:train_end], 'val_index': indices[train_end:val_end], 'test_index': indices[val_end:]}
                else:
                    ppi_split_dict = {'train_index': train_indices, 'val_index': val_indices, 'test_index': test_indices}
        else: 
            return None

        try:
            with open(split_file_path, 'w') as f: 
                json.dump(ppi_split_dict, f, indent=4)
        except Exception as e: 
            pass
    else:
        try:
            with open(split_file_path, 'r') as f: 
                ppi_split_dict = json.load(f)
            if not all(k in ppi_split_dict for k in ['train_index', 'val_index', 'test_index']):
                os.remove(split_file_path)
                return split_dataset(ppi_list, dataset, split_mode, seed)
            if not all(isinstance(ppi_split_dict[k], list) for k in ['train_index', 'val_index', 'test_index']):
                os.remove(split_file_path)
                return split_dataset(ppi_list, dataset, split_mode, seed)
        except Exception as e: 
            return None

    return ppi_split_dict


def load_pretrain_data(dataset):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir, '..', 'data', 'processed_data')

    prot_r_edge_path = os.path.join(base_data_path, f'protein.rball.edges.{dataset}.npy')
    prot_k_edge_path = os.path.join(base_data_path, f'protein.knn.edges.{dataset}.npy')
    prot_node_path = os.path.join(base_data_path, f'protein.nodes.{embedding_model_name}.{dataset}.pt')

    if not os.path.exists(prot_node_path): 
        print(f"Pretrain node file {prot_node_path} not found!")
        exit(1)

    try:
        protein_data = ProteinDatasetDGL(prot_r_edge_path, prot_k_edge_path, prot_node_path, dataset, expected_num_proteins=-1)
        return protein_data
    except Exception as e: 
        print(f"Error loading pretrain data: {e}")
        exit(1)