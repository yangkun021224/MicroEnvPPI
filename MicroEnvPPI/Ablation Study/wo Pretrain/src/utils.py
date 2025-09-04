import os
import dgl
import torch
import torch.nn.functional as F
import shutil
import random
import numpy as np
import warnings

from sklearn.metrics import f1_score, average_precision_score


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)


def evaluat_metrics(output, label):
    if output is None or label is None or output.numel() == 0 or label.numel() == 0:
        return 0.0, 0.0

    if not isinstance(output, torch.Tensor): 
        output = torch.tensor(output)
    if not isinstance(label, torch.Tensor): 
        label = torch.tensor(label)

    if output.device != torch.device('cpu'): 
        output = output.detach().cpu()
    if label.device != torch.device('cpu'): 
        label = label.detach().cpu()

    try: 
        y_prob = torch.sigmoid(output).numpy()
    except Exception as e: 
        return 0.0, 0.0

    y_pred = (y_prob > 0.5).astype(int)
    y_true = label.numpy().astype(int)

    if y_pred.shape != y_true.shape or y_prob.shape != y_true.shape:
        return 0.0, 0.0

    try: 
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    except Exception as e: 
        micro_f1 = 0.0

    micro_aupr = 0.0
    try:
        if np.sum(y_true) > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*is ill-defined and being set to 0.0 due to no true samples.*")
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
                micro_aupr = average_precision_score(y_true, y_prob, average='micro')
                if np.isnan(micro_aupr): 
                    micro_aupr = 0.0
    except ValueError as ve: 
        micro_aupr = 0.0
    except Exception as e: 
        micro_aupr = 0.0

    return micro_f1, micro_aupr


def node_feature_masking(graph, mask_rate, mask_value=0.0):
    if 'x' not in graph.ndata:
        warnings.warn("Node feature masking: Cannot find 'x' in graph.ndata.", RuntimeWarning)
        return graph

    feat = graph.ndata['x']
    num_nodes, feat_dim = feat.shape
    if num_nodes == 0: 
        return graph

    mask = torch.bernoulli(torch.full((num_nodes, feat_dim), 1.0 - mask_rate, device=feat.device)).bool()
    feat[~mask] = mask_value
    return graph

def edge_dropping(graph, unified_drop_rate):
    if not isinstance(graph, dgl.DGLGraph): 
        return graph
    if unified_drop_rate <= 0: 
        return graph

    etypes_to_process = [
        ('amino_acid', 'SEQ', 'amino_acid'),
        ('amino_acid', 'STR_KNN', 'amino_acid'),
        ('amino_acid', 'STR_DIS', 'amino_acid')
    ]

    for etype in graph.canonical_etypes:
        if etype in etypes_to_process:
            num_edges = graph.num_edges(etype)
            if num_edges > 0:
                num_edges_to_drop = int(num_edges * unified_drop_rate)
                if num_edges_to_drop > 0:
                    eids_to_drop = torch.randperm(num_edges, device=graph.device)[:num_edges_to_drop]
                    if hasattr(graph, 'remove_edges'):
                        graph.remove_edges(eids_to_drop, etype=etype)
                    else:
                        warnings.warn(f"DGL version might not support edge removal easily for etype {etype}. Skipping.", RuntimeWarning)
    return graph

def combined_augmentation(graph, node_mask_rate=0.1, edge_drop_rate=0.1):
    if not isinstance(graph, dgl.DGLGraph):
        warnings.warn("combined_augmentation received non-DGLGraph input.", RuntimeWarning)
        return graph
    
    if node_mask_rate > 0:
        graph = node_feature_masking(graph, node_mask_rate)
    
    if edge_drop_rate > 0:
        graph = edge_dropping(graph, edge_drop_rate)
    return graph


def info_nce_loss(features1, features2, temperature=0.1, batch_info=None):
    if features1 is None or features2 is None or features1.numel() == 0 or features2.numel() == 0:
        return torch.tensor(0.0, device=features1.device if features1 is not None else 'cpu', requires_grad=True)
    if features1.shape != features2.shape:
        warnings.warn(f"InfoNCE Loss: Feature shapes mismatch! f1={features1.shape}, f2={features2.shape}. Cannot compute loss.", RuntimeWarning)
        return torch.tensor(0.0, device=features1.device, requires_grad=True)

    device = features1.device
    features1 = F.normalize(features1, p=2, dim=-1)
    features2 = F.normalize(features2, p=2, dim=-1)

    if batch_info is None or not isinstance(batch_info, dgl.DGLGraph) or not hasattr(batch_info, 'batch_num_nodes'):
        warnings.warn("InfoNCE Loss: Missing or invalid batch_info. Cannot compute loss reliably.", RuntimeWarning)
        return torch.tensor(0.0, device=device, requires_grad=True)

    try:
        node_counts = batch_info.batch_num_nodes('amino_acid')
        if not isinstance(node_counts, torch.Tensor): 
            node_counts = torch.tensor(node_counts, device=device)
        if node_counts.sum() != features1.shape[0]:
            warnings.warn(f"InfoNCE Loss: Node count mismatch between batch_info ({node_counts.sum()}) and features ({features1.shape[0]}). Cannot compute loss.", RuntimeWarning)
            return torch.tensor(0.0, device=device, requires_grad=True)
        graph_ids = torch.repeat_interleave(torch.arange(batch_info.batch_size, device=device), node_counts).long()
    except Exception as e:
        warnings.warn(f"InfoNCE Loss: Error getting graph IDs: {e}. Cannot compute loss.", RuntimeWarning)
        return torch.tensor(0.0, device=device, requires_grad=True)

    similarity_matrix = torch.matmul(features1, features2.T) / temperature

    N_total = features1.shape[0]
    labels = torch.arange(N_total, device=device).long()

    mask = (graph_ids.unsqueeze(1) != graph_ids.unsqueeze(0)).float()
    mask.fill_diagonal_(0)

    logits = similarity_matrix
    exp_logits = torch.exp(logits) * mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + torch.exp(logits.diag().unsqueeze(1)))

    loss = -log_prob.diag().mean()

    if torch.isnan(loss):
        warnings.warn("InfoNCE loss resulted in NaN.", RuntimeWarning)
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss


def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    visited_nodes = set()
    processed_edges = set()
    if not node_to_edge_index or node_num <= 0: 
        return []
    valid_nodes = list(node_to_edge_index.keys())
    start_node = -1
    if not valid_nodes: 
        return []
    potential_starts = [n for n in valid_nodes if node_to_edge_index.get(n) and len(node_to_edge_index[n]) <= 20]
    if potential_starts: 
        start_node = random.choice(potential_starts)
    else: 
        start_node = random.choice(valid_nodes)
    candiate_node.append(start_node)
    visited_nodes.add(start_node)
    while candiate_node and len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        if cur_node not in node_to_edge_index: 
            continue
        edges_to_process = list(node_to_edge_index[cur_node])
        random.shuffle(edges_to_process)
        for edge_index in edges_to_process:
            if edge_index not in processed_edges and len(selected_edge_index) < sub_graph_size:
                if not (isinstance(edge_index, int) and 0 <= edge_index < len(ppi_list)): 
                    continue
                selected_edge_index.append(edge_index)
                processed_edges.add(edge_index)
                try:
                    edge = ppi_list[edge_index]
                    if not (isinstance(edge, (list, tuple)) and len(edge) == 2): 
                        continue
                    neighbor_node = edge[1] if edge[0] == cur_node else edge[0] if edge[1] == cur_node else None
                    if neighbor_node is not None and neighbor_node not in visited_nodes:
                        visited_nodes.add(neighbor_node)
                        candiate_node.append(neighbor_node)
                except IndexError: 
                    continue
    return selected_edge_index

def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    visited_nodes = set()
    visited_edges = set()
    if not node_to_edge_index or node_num <= 0: 
        return []
    valid_nodes = list(node_to_edge_index.keys())
    start_node = -1
    if not valid_nodes: 
        return []
    potential_starts = [n for n in valid_nodes if node_to_edge_index.get(n) and len(node_to_edge_index[n]) <= 20]
    if potential_starts: 
        start_node = random.choice(potential_starts)
    else: 
        start_node = random.choice(valid_nodes)
    stack.append(start_node)
    while stack and len(selected_edge_index) < sub_graph_size:
        cur_node = stack[-1]
        if cur_node not in visited_nodes: 
            visited_nodes.add(cur_node)
        found_neighbor = False
        if cur_node in node_to_edge_index:
            edges_to_explore = list(node_to_edge_index[cur_node])
            random.shuffle(edges_to_explore)
            for edge_index in edges_to_explore:
                if len(selected_edge_index) >= sub_graph_size: 
                    break
                if not (isinstance(edge_index, int) and 0 <= edge_index < len(ppi_list)): 
                    continue
                try:
                    edge = ppi_list[edge_index]
                    if not (isinstance(edge, (list, tuple)) and len(edge) == 2): 
                        continue
                    neighbor_node = edge[1] if edge[0] == cur_node else edge[0] if edge[1] == cur_node else None
                    if edge_index not in visited_edges and len(selected_edge_index) < sub_graph_size:
                        selected_edge_index.append(edge_index)
                        visited_edges.add(edge_index)
                        if len(selected_edge_index) >= sub_graph_size: 
                            break
                    if neighbor_node is not None and neighbor_node not in visited_nodes:
                        stack.append(neighbor_node)
                        found_neighbor = True
                        break
                except IndexError: 
                    continue
        if not found_neighbor:
            if stack: 
                stack.pop()
    return selected_edge_index