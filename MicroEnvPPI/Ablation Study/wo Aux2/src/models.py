import dgl
import numpy as np
import gc
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
import traceback

from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv

try:
    from dataloader import collate
except ImportError:
    def collate(samples):
        valid_samples = [s for s in samples if isinstance(s, dgl.DGLGraph)]
        if not valid_samples: return None
        try:
            batched_graph = dgl.batch(valid_samples)
            return batched_graph
        except Exception as e:
            raise e

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_PROJECTION_DIM = 128


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=DEFAULT_PROJECTION_DIM, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, param):
        super(GIN, self).__init__()
        self.num_layers = param['ppi_num_layers']
        self.dropout_p = param.get('dropout_ratio', 0.0)
        ppi_hidden_dim = param['ppi_hidden_dim']
        self.output_dim = param['output_dim']
        self.interaction_mode = param.get('ppi_interaction_mode', 'mlp')
        gin_input_dim = param.get('protein_embedding_dim', param['prot_hidden_dim'] * 2)

        self.dropout = nn.Dropout(self.dropout_p)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        input_mlp = nn.Sequential(
            nn.Linear(gin_input_dim, ppi_hidden_dim), nn.ReLU(),
            nn.Linear(ppi_hidden_dim, ppi_hidden_dim), nn.ReLU()
        )
        self.layers.append(GINConv(apply_func=input_mlp, aggregator_type='sum', learn_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(ppi_hidden_dim))

        for i in range(self.num_layers - 1):
            hidden_mlp = nn.Sequential(
                nn.Linear(ppi_hidden_dim, ppi_hidden_dim), nn.ReLU(),
                nn.Linear(ppi_hidden_dim, ppi_hidden_dim), nn.ReLU()
            )
            self.layers.append(GINConv(apply_func=hidden_mlp, aggregator_type='sum', learn_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(ppi_hidden_dim))

        self.linear = nn.Linear(ppi_hidden_dim, ppi_hidden_dim)
        gin_final_dropout = param.get('gin_final_dropout', 0.5)
        self.gin_final_dropout_layer = nn.Dropout(gin_final_dropout)

        if self.interaction_mode == 'dot':
            self.interaction_layer = nn.Linear(ppi_hidden_dim, self.output_dim)
        elif self.interaction_mode == 'mlp':
            self.interaction_layer = nn.Sequential(
                nn.Linear(ppi_hidden_dim * 2, ppi_hidden_dim), nn.ReLU(),
                nn.Dropout(p=0.5), nn.Linear(ppi_hidden_dim, self.output_dim)
            )
        elif self.interaction_mode == 'bilinear':
            self.interaction_layer = nn.Bilinear(ppi_hidden_dim, ppi_hidden_dim, self.output_dim)
        else:
            raise ValueError(f"Unsupported PPI interaction mode: {self.interaction_mode}")

    def forward(self, g, ppi_list, idx):
        if 'feat' not in g.ndata: 
            raise KeyError("Input graph must contain node features 'feat'")
        h = g.ndata['feat']
        if h.device != device: 
            h = h.to(device)

        for i, layer in enumerate(self.layers):
            if g.device != device: 
                g = g.to(device)
            h_new = layer(g, h)
            h = h_new
            if h.shape[0] > 0:
                if h.shape[0] > 1 or not self.training:
                    try: 
                        h = self.batch_norms[i](h)
                    except ValueError as e: 
                        continue
            h = F.relu(h)
            h = self.dropout(h)

        h = F.relu(self.linear(h))
        h = self.gin_final_dropout_layer(h)

        try:
            if not isinstance(idx, (list, torch.Tensor, np.ndarray)): 
                raise TypeError("idx must be list, Tensor, or ndarray")
            if isinstance(idx, torch.Tensor): 
                idx = idx.cpu().numpy()
            if isinstance(idx, list): 
                idx = np.array(idx, dtype=np.int64)

            if len(idx) == 0: 
                return torch.empty(0, self.output_dim, device=h.device)

            if not hasattr(self, 'ppi_array_cache') or self.ppi_array_cache is None or len(self.ppi_array_cache) != len(ppi_list):
                if not ppi_list or not isinstance(ppi_list[0], (list, tuple)): 
                    raise TypeError("ppi_list format incorrect")
                self.ppi_array_cache = np.array(ppi_list, dtype=np.int64)

            if idx.max() >= len(self.ppi_array_cache) or idx.min() < 0:
                raise IndexError(f"idx out of ppi_list range. Max idx: {idx.max()}, ppi_list len: {len(self.ppi_array_cache)}")

            node_id_pairs = self.ppi_array_cache[idx]
        except Exception as e: 
            raise e

        max_node_idx_needed = node_id_pairs.max() if node_id_pairs.size > 0 else -1
        if h.shape[0] <= max_node_idx_needed:
            raise IndexError(f"Node index {max_node_idx_needed} out of range (feature shape: {h.shape})")

        h1 = h[node_id_pairs[:, 0]]
        h2 = h[node_id_pairs[:, 1]]

        if self.interaction_mode == 'dot': 
            output = self.interaction_layer(torch.mul(h1, h2))
        elif self.interaction_mode == 'mlp': 
            output = self.interaction_layer(torch.cat((h1, h2), dim=1))
        elif self.interaction_mode == 'bilinear': 
            output = self.interaction_layer(h1, h2)
        else: 
            raise ValueError(f"Unsupported PPI interaction mode: {self.interaction_mode}")
        return output


class GCN_Encoder(nn.Module):
    def __init__(self, param):
        super(GCN_Encoder, self).__init__()
        self.num_layers = param['prot_num_layers']
        self.dropout_p = param.get('dropout_ratio', 0.0)
        self.hidden_dim = param['prot_hidden_dim']
        self.input_dim = param['input_dim']
        if self.input_dim != 1280: 
            pass

        self.dropout = nn.Dropout(self.dropout_p)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        self.layers.append(HeteroGraphConv({
            'SEQ' : GraphConv(self.input_dim, self.hidden_dim, allow_zero_in_degree=True),
            'STR_KNN' : GraphConv(self.input_dim, self.hidden_dim, allow_zero_in_degree=True),
            'STR_DIS' : GraphConv(self.input_dim, self.hidden_dim, allow_zero_in_degree=True)
        }, aggregate='sum'))
        self.fc_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.norms.append(nn.BatchNorm1d(self.hidden_dim))

        for i in range(self.num_layers - 1):
            self.layers.append(HeteroGraphConv({
                'SEQ' : GraphConv(self.hidden_dim, self.hidden_dim, allow_zero_in_degree=True),
                'STR_KNN' : GraphConv(self.hidden_dim, self.hidden_dim, allow_zero_in_degree=True),
                'STR_DIS' : GraphConv(self.hidden_dim, self.hidden_dim, allow_zero_in_degree=True)
            }, aggregate='sum'))
            self.fc_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))

    def encoding(self, batch_graph):
        if batch_graph.device != device: 
            batch_graph = batch_graph.to(device)
        if 'amino_acid' not in batch_graph.ntypes or 'x' not in batch_graph.nodes['amino_acid'].data:
            raise KeyError("Input graph missing 'amino_acid' nodes or 'x' features.")

        x = batch_graph.nodes['amino_acid'].data['x']
        if x.device != device: 
            x = x.to(device)

        if x.shape[0] == 0: 
            return torch.empty(0, self.hidden_dim, device=device)
        if x.shape[1] != self.input_dim: 
            raise ValueError(f"GCN Encoder input dim error! Expected {self.input_dim}, got {x.shape[1]}")

        for l, (layer, fc_layer, norm_layer) in enumerate(zip(self.layers, self.fc_layers, self.norms)):
            x_dict = {'amino_acid': x}
            try:
                h_dict = layer(batch_graph, x_dict)
                x_new = h_dict.get('amino_acid', torch.tensor([], device=device))
                if x_new.shape[0] > 0:
                    x_new = fc_layer(x_new)
                    x_new = F.relu(x_new)
                    if x_new.shape[0] > 1 or not self.training:
                        try: 
                            x_new = norm_layer(x_new)
                        except ValueError as e: 
                            pass
                    if l < self.num_layers - 1: 
                        x_new = self.dropout(x_new)
                    x = x_new
                else: 
                    x = torch.empty(0, self.hidden_dim, device=device)
                    break
            except Exception as e: 
                raise e
        return x


class GCN_Decoder(nn.Module):
    def __init__(self, param):
        super(GCN_Decoder, self).__init__()
        self.num_layers = param['prot_num_layers']
        self.dropout_p = param.get('dropout_ratio', 0.0)
        self.hidden_dim = param['prot_hidden_dim']
        self.output_dim = param['input_dim']
        self.dropout = nn.Dropout(self.dropout_p)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        self.layers.append(HeteroGraphConv({'SEQ':GraphConv(self.hidden_dim,self.hidden_dim,allow_zero_in_degree=True),'STR_KNN':GraphConv(self.hidden_dim,self.hidden_dim,allow_zero_in_degree=True),'STR_DIS':GraphConv(self.hidden_dim,self.hidden_dim,allow_zero_in_degree=True)}, aggregate='sum'))
        self.fc_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        if self.num_layers > 1: 
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))

        for _ in range(self.num_layers - 2):
            self.layers.append(HeteroGraphConv({'SEQ':GraphConv(self.hidden_dim,self.hidden_dim,allow_zero_in_degree=True),'STR_KNN':GraphConv(self.hidden_dim,self.hidden_dim,allow_zero_in_degree=True),'STR_DIS':GraphConv(self.hidden_dim,self.hidden_dim,allow_zero_in_degree=True)}, aggregate='sum'))
            self.fc_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.layers.append(HeteroGraphConv({'SEQ':GraphConv(self.hidden_dim,self.hidden_dim,allow_zero_in_degree=True),'STR_KNN':GraphConv(self.hidden_dim,self.hidden_dim,allow_zero_in_degree=True),'STR_DIS':GraphConv(self.hidden_dim,self.hidden_dim,allow_zero_in_degree=True)}, aggregate='sum'))
        self.fc_layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def decoding(self, batch_graph, x):
        if batch_graph.device != device: 
            batch_graph = batch_graph.to(device)
        if x.device != device: 
            x = x.to(device)
        if x.shape[0] == 0: 
            return torch.empty(0, self.output_dim, device=device)

        for l, (layer, fc_layer) in enumerate(zip(self.layers, self.fc_layers)):
            h_dict = layer(batch_graph, {'amino_acid': x})
            x_new = h_dict.get('amino_acid', torch.tensor([], device=device))
            if x_new.shape[0] > 0:
                x_new = fc_layer(x_new)
                if l < self.num_layers - 1:
                    if l < len(self.norms):
                        if x_new.shape[0] > 1 or not self.training:
                            try: 
                                x_new = self.norms[l](x_new)
                            except ValueError as e: 
                                pass
                    x_new = F.relu(x_new)
                    x_new = self.dropout(x_new)
                x = x_new
            else: 
                x = torch.empty(0, self.output_dim if l == self.num_layers - 1 else self.hidden_dim, device=device)
                break
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, x):
        if x.shape[0] == 0:
            zero_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            empty_tensor = torch.empty(0, self.embedding_dim, device=x.device)
            empty_indices = torch.empty(0, dtype=torch.long, device=x.device)
            return empty_tensor, zero_loss, zero_loss, empty_indices

        if x.device != self.embeddings.weight.device: 
            x = x.to(self.embeddings.weight.device)

        x_normalized = F.normalize(x, p=2, dim=-1)
        codebook_normalized = F.normalize(self.embeddings.weight, p=2, dim=-1)
        distances = torch.sum(x_normalized.pow(2), dim=1, keepdim=True) + \
                     torch.sum(codebook_normalized.pow(2), dim=1) - \
                     2 * torch.matmul(x_normalized, codebook_normalized.t())
        encoding_indices = torch.argmin(distances, dim=1)
        quantized_unnormalized = self.embeddings(encoding_indices)

        codebook_loss = F.mse_loss(quantized_unnormalized, x.detach())
        commitment_loss = F.mse_loss(x, quantized_unnormalized.detach())
        quantized_ste = x + (quantized_unnormalized - x).detach()

        return quantized_ste, codebook_loss, commitment_loss, encoding_indices.detach()

    def quantize(self, encoding_indices):
        return self.embeddings(encoding_indices)


class CodeBook(nn.Module):
    def __init__(self, param, protein_data):
        super(CodeBook, self).__init__()
        self.param = param
        self.original_input_dim = param['input_dim']
        self.prot_hidden_dim = param['prot_hidden_dim']
        self.protein_embedding_mode = 'concat'

        self.Protein_Encoder = GCN_Encoder(param)
        self.Protein_Decoder = GCN_Decoder(param)
        self.vq_layer = VectorQuantizer(self.prot_hidden_dim, param['num_embeddings'], param['commitment_cost'])

        self.input_recon_head = nn.Linear(self.prot_hidden_dim, self.original_input_dim)

        self.protein_dataset = protein_data
        if self.protein_dataset is None: 
            pass
        elif not hasattr(self.protein_dataset, '__len__') or not hasattr(self.protein_dataset, '__getitem__'):
            self.protein_dataset = None

    def forward(self, batch_graph, random_mcm_mask, return_encoder_output=False):
        if batch_graph.device != device: 
            batch_graph = batch_graph.to(device)

        h_m_L = self.Protein_Encoder.encoding(batch_graph)

        if h_m_L.shape[0] == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            empty_h = torch.empty(0, self.prot_hidden_dim, device=device)
            if return_encoder_output: 
                return zero_loss, zero_loss, zero_loss, zero_loss, empty_h
            else: 
                return zero_loss, zero_loss, zero_loss, zero_loss

        e_quantized_ste, codebook_loss_term, commitment_loss_term, encoding_indices = self.vq_layer(h_m_L)

        x_recon_vq = self.Protein_Decoder.decoding(batch_graph, e_quantized_ste)

        recon_loss = torch.tensor(0.0, device=device)
        try:
            pass
        except Exception as e: 
            raise e

        mask = random_mcm_mask
        if mask.device != encoding_indices.device: 
            mask = mask.to(encoding_indices.device)
        node_mask = mask[encoding_indices]

        mask_loss_sce = torch.tensor(0.0, device=device)
        if node_mask.sum() > 0:
            e_masked = e_quantized_ste.clone()
            e_masked[node_mask] = 0.0
            x_mask_recon = self.Protein_Decoder.decoding(batch_graph, e_masked)

            try:
                pass
            except Exception as e: 
                mask_loss_sce = torch.tensor(0.0, device=device)

        if return_encoder_output:
            return torch.tensor(0.0), codebook_loss_term, commitment_loss_term, torch.tensor(0.0), h_m_L
        else:
            return torch.tensor(0.0), codebook_loss_term, commitment_loss_term, torch.tensor(0.0)

    def get_protein_embeddings(self):
        if self.protein_dataset is None: 
            return None
        if not hasattr(self, 'Protein_Encoder') or not hasattr(self, 'vq_layer'): 
            return None

        self.eval()
        dataset_len = len(self.protein_dataset)
        batch_size = self.param.get('vae_batch_size', 64)
        num_batches = math.ceil(dataset_len / batch_size)

        out_dim = self.prot_hidden_dim * 2

        all_embeds_collected = []
        with torch.no_grad():
            pbar = tqdm(range(num_batches), desc="Encoding proteins", mininterval=1.0)
            processed_count = 0
            for i in pbar:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, dataset_len)
                indices = list(range(start_idx, end_idx))
                if not indices: 
                    continue

                batch_samples = []
                valid_indices_in_batch = []
                for idx in indices:
                    try:
                        sample = self.protein_dataset[idx]
                        if sample is not None and isinstance(sample, dgl.DGLGraph) and sample.num_nodes('amino_acid') > 0:
                            batch_samples.append(sample)
                            valid_indices_in_batch.append(idx)
                    except Exception as fetch_e: 
                        pass

                num_expected_in_batch = len(indices)
                num_valid_in_batch = len(batch_samples)
                num_failed_in_batch = num_expected_in_batch - num_valid_in_batch

                if not batch_samples:
                    zero_embeds = torch.zeros(num_expected_in_batch, out_dim, device='cpu')
                    all_embeds_collected.append(zero_embeds)
                    continue

                try: 
                    batch_graph = collate(batch_samples)
                except Exception as collate_e:
                    zero_embeds = torch.zeros(num_expected_in_batch, out_dim, device='cpu')
                    all_embeds_collected.append(zero_embeds)
                    continue

                if batch_graph is None: 
                    zero_embeds = torch.zeros(num_expected_in_batch, out_dim, device='cpu')
                    all_embeds_collected.append(zero_embeds)
                    continue

                batch_graph = batch_graph.to(device)
                batch_prot_embeds = torch.zeros(num_valid_in_batch, out_dim, device='cpu')

                try:
                    num_graphs_in_batch_actual = batch_graph.batch_size if hasattr(batch_graph, 'batch_size') else num_valid_in_batch
                    if num_graphs_in_batch_actual != num_valid_in_batch:
                        num_graphs_in_batch_actual = num_valid_in_batch

                    if batch_graph.num_nodes('amino_acid') > 0:
                        h = self.Protein_Encoder.encoding(batch_graph)
                        if h.shape[0] > 0 :
                            z_quantized_ste, _, _, _ = self.vq_layer(h)
                            residue_embeddings_combined = torch.cat([h, z_quantized_ste], dim=-1)
                            if residue_embeddings_combined.shape[0] > 0:
                                batch_graph.nodes['amino_acid'].data['h_combined'] = residue_embeddings_combined
                                if batch_graph.num_nodes('amino_acid') > 0:
                                    prot_embed = dgl.mean_nodes(batch_graph, 'h_combined', ntype='amino_acid')
                                    if prot_embed.shape[0] == num_graphs_in_batch_actual:
                                        batch_prot_embeds = prot_embed.cpu()
                                        processed_count += prot_embed.shape[0]
                                    else:
                                        padded_embed = torch.zeros(num_graphs_in_batch_actual, out_dim, device='cpu')
                                        copy_len = min(prot_embed.shape[0], num_graphs_in_batch_actual)
                                        padded_embed[:copy_len, :] = prot_embed[:copy_len, :].cpu()
                                        batch_prot_embeds = padded_embed
                                        processed_count += copy_len
                except Exception as e:
                    pass

                if num_failed_in_batch > 0:
                    zero_padding_failed = torch.zeros(num_failed_in_batch, out_dim, device='cpu')
                    final_batch_embeds = torch.cat((batch_prot_embeds, zero_padding_failed), dim=0)
                    if final_batch_embeds.shape[0] != num_expected_in_batch:
                        corrected_batch = torch.zeros(num_expected_in_batch, out_dim, device='cpu')
                        copy_len = min(num_expected_in_batch, final_batch_embeds.shape[0])
                        corrected_batch[:copy_len, :] = final_batch_embeds[:copy_len, :]
                        final_batch_embeds = corrected_batch
                    all_embeds_collected.append(final_batch_embeds)
                else:
                    all_embeds_collected.append(batch_prot_embeds)

        if not all_embeds_collected: 
            return torch.tensor([])

        try: 
            final_embeddings = torch.cat(all_embeds_collected, dim=0)
        except Exception as cat_e:
            valid_embeds = [t for t in all_embeds_collected if t.numel() > 0 and len(t.shape)==2]
            if not valid_embeds: 
                return torch.tensor([])
            try: 
                final_embeddings = torch.cat(valid_embeds, dim=0)
            except: 
                return torch.tensor([])

        if final_embeddings.shape[0] != dataset_len:
            corrected_embeddings = torch.zeros(dataset_len, out_dim, device='cpu')
            copy_len = min(dataset_len, final_embeddings.shape[0])
            corrected_embeddings[:copy_len, :] = final_embeddings[:copy_len, :]
            final_embeddings = corrected_embeddings

        if final_embeddings.shape[0] != dataset_len: 
            return torch.tensor([])

        return final_embeddings