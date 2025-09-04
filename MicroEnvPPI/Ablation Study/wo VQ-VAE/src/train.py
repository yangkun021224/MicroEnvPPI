import dgl
import csv
import time
import json
import math
import copy
import argparse
import warnings
import numpy as np
import os
import gc
from tqdm import tqdm
import sys
import traceback
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

try:
    from utils import set_seed, check_writable, evaluat_metrics, combined_augmentation, info_nce_loss
    from models import CodeBook, GIN, ProjectionHead
    from dataloader import load_data, collate, ProteinDatasetDGL
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_ESM_DIM = 1280
DEFAULT_NUM_WORKERS = 4
DEFAULT_VAE_BATCH_SIZE = 64
DEFAULT_PROJECTION_DIM = 128
DEFAULT_CONTRASTIVE_TEMP = 0.1
DEFAULT_CONTRASTIVE_WEIGHT = 0.1
DEFAULT_NODE_MASK_RATE = 0.1
DEFAULT_EDGE_DROP_RATE = 0.1
DEFAULT_VAE_LR = 0.0005
DEFAULT_VAE_SCHEDULER = "cosine"
DEFAULT_VAE_ETA_MIN = 1e-7
DEFAULT_PRETRAIN_PATIENCE = 50
DEFAULT_INPUT_MASK_RATIO = 0.15
DEFAULT_MASKED_FEAT_LOSS_WEIGHT = 0.5

VAE_CHECKPOINT_FILENAME = "vae_cl_aux_randmcm_checkpoint.pth"
GIN_CHECKPOINT_FILENAME = "gin_cl_aux_randmcm_checkpoint.pth"
GIN_BEST_MODEL_FILENAME = "model_cl_aux_randmcm_best_state.pth"
FINAL_BEST_VAE_MODEL_FILENAME = "vae_model.ckpt"


def pretrain_vae(param, timestamp, protein_data, resume_path=None):
    if protein_data is None:
        print("Error: pretrain_vae received invalid protein_data!")
        return None

    required_params = [
        'dataset', 'weight_decay', 'pre_epoch', 'log_num',
        'commitment_cost', 'mask_loss', 'prot_hidden_dim', 'num_embeddings',
        'mask_ratio', 'sce_scale', 'input_dim', 'vae_batch_size', 'num_workers',
        'save_ckpt_every',
        'contrastive_loss_weight', 'contrastive_temperature',
        'node_mask_rate', 'edge_drop_rate',
        'vae_learning_rate', 'vae_scheduler', 'vae_lr_eta_min',
        'pretrain_patience',
        'input_mask_ratio', 'masked_feat_loss_weight'
    ]
    for p in required_params:
        if p not in param:
            default_values = { 
                'contrastive_loss_weight': DEFAULT_CONTRASTIVE_WEIGHT, 
                'contrastive_temperature': DEFAULT_CONTRASTIVE_TEMP, 
                'node_mask_rate': DEFAULT_NODE_MASK_RATE, 
                'edge_drop_rate': DEFAULT_EDGE_DROP_RATE, 
                'num_workers': DEFAULT_NUM_WORKERS, 
                'vae_batch_size': DEFAULT_VAE_BATCH_SIZE, 
                'save_ckpt_every': 10, 
                'vae_learning_rate': DEFAULT_VAE_LR, 
                'vae_scheduler': DEFAULT_VAE_SCHEDULER, 
                'vae_lr_eta_min': DEFAULT_VAE_ETA_MIN, 
                'pretrain_patience': DEFAULT_PRETRAIN_PATIENCE, 
                'input_mask_ratio': DEFAULT_INPUT_MASK_RATIO, 
                'masked_feat_loss_weight': DEFAULT_MASKED_FEAT_LOSS_WEIGHT 
            }
            if p in default_values: 
                param[p] = default_values[p]
            elif p in ['input_dim', 'prot_hidden_dim', 'num_embeddings']: 
                return None

    output_dir = f"../results/{param['dataset']}/{timestamp}/VAE_CL_Aux_RandMCM/"
    check_writable(output_dir, overwrite=False)
    log_file_path = os.path.join(output_dir, "pretrain_log.txt")
    log_file = None
    try:
        log_mode = 'a+' if resume_path and os.path.exists(resume_path) else 'w+'
        log_file = open(log_file_path, log_mode)
        if log_mode == 'a+': 
            log_file.write(f"\n--- Resuming VAE_CL_Aux_RandMCM Pre-training at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    except IOError as e: 
        pass
    
    config_path = os.path.join(output_dir, "config.json")
    try:
        params_to_save = {k: v for k, v in param.items() if k in required_params or k in ['seed']}
        with open(config_path, 'w') as tf: 
            json.dump(params_to_save, tf, indent=2)
    except IOError as e: 
        pass
    
    pin_memory_flag = torch.cuda.is_available()
    num_workers = param.get('num_workers', DEFAULT_NUM_WORKERS)
    vae_batch_size = param.get('vae_batch_size', DEFAULT_VAE_BATCH_SIZE)
    try:
        vae_dataloader = DataLoader( protein_data, batch_size=vae_batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=True )
        if len(vae_dataloader) == 0: 
            return None
    except Exception as e: 
        return None
    
    vae_model = None
    try: 
        vae_model = CodeBook(param, protein_data).to(device)
    except Exception as e: 
        return None
    
    projection_head = None
    try:
        projection_head = ProjectionHead( input_dim=param['prot_hidden_dim'], hidden_dim=param['prot_hidden_dim'], output_dim=DEFAULT_PROJECTION_DIM, dropout=param.get('dropout_ratio', 0.1) ).to(device)
    except Exception as e: 
        return None
    
    try:
        vae_optimizer = torch.optim.Adam( list(vae_model.parameters()) + list(projection_head.parameters()), lr=float(param['vae_learning_rate']), weight_decay=float(param['weight_decay']) )
    except Exception as e: 
        return None

    start_epoch = 1
    vae_scheduler = None
    best_pretrain_loss = float('inf')
    pretrain_epochs_no_improve = 0
    best_pretrain_vae_state_no_heads = None
    vae_checkpoint_file = os.path.join(output_dir, VAE_CHECKPOINT_FILENAME)
    vae_checkpoint_file_to_load = resume_path if resume_path and os.path.exists(resume_path) else vae_checkpoint_file
    
    if os.path.exists(vae_checkpoint_file_to_load):
        try:
            checkpoint = torch.load(vae_checkpoint_file_to_load, map_location=device)
            model_load_result = vae_model.load_state_dict(checkpoint['vae_model_state_dict'], strict=False)
            if 'projection_head_state_dict' in checkpoint: 
                proj_load_result = projection_head.load_state_dict(checkpoint['projection_head_state_dict'], strict=False)
            vae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_pretrain_loss = checkpoint.get('best_pretrain_loss', float('inf'))
            pretrain_epochs_no_improve = checkpoint.get('pretrain_epochs_no_improve', 0)
            vae_scheduler_type = param.get('vae_scheduler')
            if vae_scheduler_type == 'cosine':
                t_max_epochs_total = param['pre_epoch']
                eta_min_val = float(param.get('vae_lr_eta_min', DEFAULT_VAE_ETA_MIN))
                if t_max_epochs_total < start_epoch: 
                    t_max_epochs_total = start_epoch
                t_max_for_resume = max(1, t_max_epochs_total - start_epoch + 1)
                try:
                    vae_scheduler = CosineAnnealingLR(vae_optimizer, T_max=t_max_for_resume, eta_min=eta_min_val, verbose=False)
                    if 'vae_scheduler_state_dict' in checkpoint and checkpoint['vae_scheduler_state_dict'] is not None: 
                        vae_scheduler.load_state_dict(checkpoint['vae_scheduler_state_dict'])
                    else: 
                        vae_scheduler.last_epoch = -1
                except Exception as e: 
                    vae_scheduler = None
            else: 
                vae_scheduler = None
        except Exception as e:
            start_epoch = 1
            vae_scheduler = None
            best_pretrain_loss = float('inf')
            pretrain_epochs_no_improve = 0
            best_pretrain_vae_state_no_heads = None
    else:
        if start_epoch == 1:
            vae_scheduler_type = param.get('vae_scheduler')
            if vae_scheduler_type == 'cosine':
                t_max_epochs = param['pre_epoch']
                eta_min_val = float(param.get('vae_lr_eta_min', DEFAULT_VAE_ETA_MIN))
                if t_max_epochs <= 0: 
                    t_max_epochs = 1
                try: 
                    vae_scheduler = CosineAnnealingLR(vae_optimizer, T_max=t_max_epochs, eta_min=eta_min_val, last_epoch=-1, verbose=True)
                except Exception as e: 
                    vae_scheduler = None
            else: 
                vae_scheduler = None

    pretrain_patience = param.get('pretrain_patience', DEFAULT_PRETRAIN_PATIENCE)
    sys.stdout.flush()

    contrastive_loss_weight = float(param.get('contrastive_loss_weight', DEFAULT_CONTRASTIVE_WEIGHT))
    contrastive_temperature = float(param.get('contrastive_temperature', DEFAULT_CONTRASTIVE_TEMP))
    node_mask_rate = float(param.get('node_mask_rate', DEFAULT_NODE_MASK_RATE))
    edge_drop_rate = float(param.get('edge_drop_rate', DEFAULT_EDGE_DROP_RATE))
    input_mask_ratio = float(param.get('input_mask_ratio', DEFAULT_INPUT_MASK_RATIO))
    masked_feat_loss_weight = float(param.get('masked_feat_loss_weight', DEFAULT_MASKED_FEAT_LOSS_WEIGHT))

    for epoch in range(start_epoch, param["pre_epoch"] + 1):
        vae_model.train()
        projection_head.train()

        epoch_loss_total_sum=0.0
        epoch_recon_loss_sum=0.0
        epoch_codebook_loss_sum=0.0
        epoch_commitment_loss_sum=0.0
        epoch_mcm_sce_loss_sum=0.0
        epoch_aux_loss_sum=0.0
        epoch_cl_loss_sum = 0.0
        num_batches_processed = 0
        
        pbar = tqdm(enumerate(vae_dataloader), total=len(vae_dataloader), desc=f"Pretrain Epoch {epoch}/{param['pre_epoch']}", leave=False, mininterval=1.0)

        for iter_num, batch_graph in pbar:
            if batch_graph is None or not isinstance(batch_graph, dgl.DGLGraph) or batch_graph.num_nodes('amino_acid') == 0: 
                continue
            try:
                if 'x' not in batch_graph.nodes['amino_acid'].data: 
                    continue
                original_features = batch_graph.nodes['amino_acid'].data['x'].clone().to(device)
                graph_for_vq = copy.deepcopy(batch_graph).to(device)
                graph_for_aug1 = copy.deepcopy(batch_graph)
                graph_for_aug2 = copy.deepcopy(batch_graph)
            except Exception as e: 
                continue

            try:
                feat_shape = graph_for_vq.nodes['amino_acid'].data['x'].shape
                input_mask = torch.rand(feat_shape, device=device) < input_mask_ratio
                masked_input_features = graph_for_vq.nodes['amino_acid'].data['x'].clone()
                masked_input_features[input_mask] = 0.0
                graph_for_vq.nodes['amino_acid'].data['x'] = masked_input_features
            except Exception as e: 
                continue

            try:
                view1 = combined_augmentation(graph_for_aug1, node_mask_rate, edge_drop_rate).to(device)
                view2 = combined_augmentation(graph_for_aug2, node_mask_rate, edge_drop_rate).to(device)
            except Exception as e: 
                continue

            mcm_mask_ratio = param.get('mask_ratio', 0.15)
            num_embeddings = param['num_embeddings']
            random_mcm_mask = torch.bernoulli(torch.full(size=(num_embeddings,), fill_value=mcm_mask_ratio, device=device)).bool()

            try:
                _, codebook_loss, commitment_loss, _, h_orig_vq = vae_model(graph_for_vq, random_mcm_mask, return_encoder_output=True)

                e_quantized_ste_vq, _, _, _ = vae_model.vq_layer(h_orig_vq)
                x_recon_vq = vae_model.Protein_Decoder.decoding(graph_for_vq, e_quantized_ste_vq)
                recon_loss = torch.tensor(0.0, device=device)
                if x_recon_vq.shape == original_features.shape: 
                    recon_loss = F.mse_loss(x_recon_vq, original_features)
                else: 
                    warnings.warn("Shape mismatch for VQ Recon Loss", RuntimeWarning)

                mask_loss_sce = torch.tensor(0.0, device=device)
                _, _, _, encoding_indices_vq = vae_model.vq_layer(h_orig_vq)
                node_mask_mcm = random_mcm_mask[encoding_indices_vq]
                if node_mask_mcm.sum() > 0:
                    e_masked_mcm = e_quantized_ste_vq.clone()
                    e_masked_mcm[node_mask_mcm] = 0.0
                    x_mask_recon_mcm = vae_model.Protein_Decoder.decoding(graph_for_vq, e_masked_mcm)
                    orig_masked_mcm = original_features[node_mask_mcm]
                    recon_masked_mcm = x_mask_recon_mcm[node_mask_mcm]
                    if orig_masked_mcm.shape == recon_masked_mcm.shape and orig_masked_mcm.numel() > 0:
                        cos_sim_mcm = F.cosine_similarity(orig_masked_mcm, recon_masked_mcm, dim=-1)
                        loss_per_node_mcm = (1.0 - torch.clamp(cos_sim_mcm, -1.0, 1.0))
                        sce_scale = param.get('sce_scale', 1.0)
                        loss_per_node_scaled_mcm = torch.pow(loss_per_node_mcm, sce_scale)
                        mask_loss_sce = torch.mean(loss_per_node_scaled_mcm)
                    else: 
                        warnings.warn("Shape mismatch for MCM Loss", RuntimeWarning)

                loss_components = [recon_loss, codebook_loss, commitment_loss, mask_loss_sce]
                if not all(isinstance(l, torch.Tensor) and l.numel() >= 0 and not torch.isnan(l).any() and not torch.isinf(l).any() for l in loss_components) or not isinstance(h_orig_vq, torch.Tensor):
                    warnings.warn(f"Invalid VQ/MCM loss or h_orig_vq returned. Skipping batch.", UserWarning)
                    continue
            except Exception as e: 
                continue

            masked_feat_loss = torch.tensor(0.0, device=device)
            try:
                if hasattr(vae_model, 'input_recon_head') and h_orig_vq.shape[0] > 0 and input_mask.sum() > 0:
                    predicted_input_features = vae_model.input_recon_head(h_orig_vq)
                    loss_func_aux = nn.MSELoss(reduction='none')
                    if predicted_input_features.shape == original_features.shape:
                        per_node_loss = loss_func_aux(predicted_input_features, original_features)
                        if input_mask.shape == per_node_loss.shape:
                            masked_loss = per_node_loss[input_mask]
                            if masked_loss.numel() > 0: 
                                masked_feat_loss = masked_loss.mean()
                            if torch.isnan(masked_feat_loss) or torch.isinf(masked_feat_loss): 
                                warnings.warn(f"NaN/Inf Aux loss. Resetting to 0.", RuntimeWarning)
                                masked_feat_loss = torch.tensor(0.0, device=device)
                        else: 
                            warnings.warn(f"Shape mismatch: input_mask vs per_node_loss. Skipping Aux loss.", RuntimeWarning)
                    else: 
                        warnings.warn(f"Shape mismatch: predicted_input vs original_features. Skipping Aux loss.", RuntimeWarning)
            except Exception as e: 
                masked_feat_loss = torch.tensor(0.0, device=device)

            loss_cl = torch.tensor(0.0, device=device)
            try:
                h1 = vae_model.Protein_Encoder.encoding(view1)
                h2 = vae_model.Protein_Encoder.encoding(view2)
                if h1.shape[0] > 0 and h2.shape[0] > 0 and h1.shape == h2.shape:
                    p1 = projection_head(h1)
                    p2 = projection_head(h2)
                    loss_cl = info_nce_loss(p1, p2, temperature=contrastive_temperature, batch_info=graph_for_vq)
                    if torch.isnan(loss_cl) or torch.isinf(loss_cl): 
                        warnings.warn(f"NaN/Inf CL loss. Resetting to 0.", RuntimeWarning)
                        loss_cl = torch.tensor(0.0, device=device)
            except Exception as e: 
                loss_cl = torch.tensor(0.0, device=device)

            try:
                beta = float(param['commitment_cost'])
                eta = float(param['mask_loss'])
                lambda_aux = masked_feat_loss_weight
                lambda_cl = contrastive_loss_weight
                loss_vq_vae = recon_loss + codebook_loss + beta * commitment_loss
                total_loss = loss_vq_vae + eta * mask_loss_sce + lambda_aux * masked_feat_loss + lambda_cl * loss_cl
                if torch.isnan(total_loss) or torch.isinf(total_loss): 
                    warnings.warn(f"NaN/Inf Total loss. Skipping step.", RuntimeWarning)
                    continue
            except Exception as e: 
                continue

            try:
                vae_optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                vae_optimizer.step()
            except Exception as e: 
                continue

            epoch_loss_total_sum += total_loss.item()
            epoch_recon_loss_sum += recon_loss.item()
            epoch_codebook_loss_sum += codebook_loss.item()
            epoch_commitment_loss_sum += commitment_loss.item()
            epoch_mcm_sce_loss_sum += mask_loss_sce.item()
            epoch_aux_loss_sum += masked_feat_loss.item()
            epoch_cl_loss_sum += loss_cl.item()
            num_batches_processed += 1
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}", 'Rec': f"{recon_loss.item():.4f}", 'MCM': f"{mask_loss_sce.item():.5f}", 'Aux': f"{masked_feat_loss.item():.4f}", 'CL': f"{loss_cl.item():.4f}"})

        current_vae_lr = vae_optimizer.param_groups[0]['lr']
        if vae_scheduler is not None:
            if isinstance(vae_scheduler, CosineAnnealingLR): 
                vae_scheduler.step()
        
        if num_batches_processed > 0:
            avg_loss = epoch_loss_total_sum / num_batches_processed
            avg_rec = epoch_recon_loss_sum / num_batches_processed
            avg_cb = epoch_codebook_loss_sum / num_batches_processed
            avg_cmt = epoch_commitment_loss_sum / num_batches_processed
            avg_mcm = epoch_mcm_sce_loss_sum / num_batches_processed
            avg_aux = epoch_aux_loss_sum / num_batches_processed
            avg_cl = epoch_cl_loss_sum / num_batches_processed
            epoch_summary = (f"Pretrain Epoch {epoch} LR: {current_vae_lr:.6g} Avg Loss: {avg_loss:.5f} | "
                             f"VQ(R:{avg_rec:.5f}, CB:{avg_cb:.5f}, CMT:{avg_cmt:.5f}), "
                             f"MCM: {avg_mcm:.5f}, Aux: {avg_aux:.5f}, CL: {avg_cl:.5f}")
            print(f"\n{epoch_summary}")
            if log_file: 
                log_file.write(f"{epoch_summary}\n")
                log_file.flush()
            current_loss_to_monitor = avg_loss
            if current_loss_to_monitor < best_pretrain_loss:
                best_pretrain_loss = current_loss_to_monitor
                pretrain_epochs_no_improve = 0
                try:
                    full_state = copy.deepcopy(vae_model.state_dict())
                    best_pretrain_vae_state_no_heads = {k: v for k, v in full_state.items() if not k.startswith('input_recon_head')}
                except Exception as copy_e: 
                    best_pretrain_vae_state_no_heads = None
            else: 
                pretrain_epochs_no_improve += 1
            if pretrain_epochs_no_improve >= pretrain_patience: 
                break

        save_ckpt_every = param.get('save_ckpt_every', 10)
        if epoch % save_ckpt_every == 0 or epoch == param["pre_epoch"]:
            vae_ckpt_save_path = os.path.join(output_dir, VAE_CHECKPOINT_FILENAME)
            try: 
                torch.save({'epoch': epoch, 'vae_model_state_dict': vae_model.state_dict(), 'projection_head_state_dict': projection_head.state_dict(), 'optimizer_state_dict': vae_optimizer.state_dict(), 'vae_scheduler_state_dict': vae_scheduler.state_dict() if vae_scheduler else None, 'best_pretrain_loss': best_pretrain_loss, 'pretrain_epochs_no_improve': pretrain_epochs_no_improve, 'param': param }, vae_ckpt_save_path)
            except Exception as e: 
                pass
        sys.stdout.flush()

    final_vae_path = os.path.join(output_dir, FINAL_BEST_VAE_MODEL_FILENAME)
    if best_pretrain_vae_state_no_heads is not None:
        try: 
            torch.save(best_pretrain_vae_state_no_heads, final_vae_path)
        except Exception as e: 
            pass
    else:
        try:
            last_epoch_state = vae_model.state_dict()
            vae_state_to_save = {k: v for k, v in last_epoch_state.items() if not k.startswith('input_recon_head')}
            torch.save(vae_state_to_save, final_vae_path)
        except Exception as e: 
            pass
    
    if log_file:
        try: 
            log_file.close()
        except Exception as e: 
            pass
    
    del vae_model, projection_head, vae_dataloader, vae_optimizer, vae_scheduler, pbar
    del best_pretrain_vae_state_no_heads
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return final_vae_path


def evaluator(model, ppi_g, ppi_list, labels, index, batch_size, mode='metric'):
    model.eval()
    eval_output_list, eval_labels_list = [], []
    num_eval_samples = len(index)
    batch_num = math.ceil(num_eval_samples / batch_size) if num_eval_samples > 0 else 0
    if batch_num == 0: 
        return (0.0, 0.0) if mode == 'metric' else (torch.tensor([], device='cpu'), torch.tensor([], device='cpu'))
    
    pbar_eval = tqdm(range(batch_num), desc=f"Evaluating ({mode})", leave=False, mininterval=1.0)
    with torch.no_grad():
        for batch in pbar_eval:
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, num_eval_samples)
            if start_idx >= end_idx: 
                continue
            batch_idx = index[start_idx:end_idx]
            try:
                output = model(ppi_g, ppi_list, batch_idx)
                if labels.device != device: 
                    labels = labels.to(device)
                batch_labels = labels[batch_idx]
                if output.numel() == 0 or batch_labels.numel() == 0: 
                    continue
                eval_output_list.append(output.cpu())
                eval_labels_list.append(batch_labels.cpu())
            except Exception as e: 
                continue
    
    if not eval_output_list or not eval_labels_list: 
        return (0.0, 0.0) if mode == 'metric' else (torch.tensor([], device='cpu'), torch.tensor([], device='cpu'))
    try: 
        all_outputs = torch.cat(eval_output_list, dim=0)
        all_labels = torch.cat(eval_labels_list, dim=0)
    except Exception as e: 
        return (0.0, 0.0) if mode == 'metric' else (torch.tensor([], device='cpu'), torch.tensor([], device='cpu'))
    
    if mode == 'metric': 
        micro_f1, micro_aupr = evaluat_metrics(all_outputs, all_labels)
        return micro_f1, micro_aupr
    elif mode == 'output': 
        return all_outputs, all_labels
    else: 
        raise ValueError(f"Unknown evaluation mode: {mode}")


def train(model, ppi_g, ppi_list, labels, train_index, batch_size, optimizer, loss_fn, epoch, scheduler):
    model.train()
    f1_sum = 0.0
    loss_sum = 0.0
    num_train_samples = len(train_index)
    batch_num = math.ceil(num_train_samples / batch_size) if num_train_samples > 0 else 0
    if batch_num == 0: 
        return 0.0, 0.0
    
    shuffled_train_index = random.sample(train_index, len(train_index))
    pbar_train = tqdm(range(batch_num), desc=f"Train Epoch {epoch}", leave=False, mininterval=1.0)
    current_lr = optimizer.param_groups[0]['lr']
    num_batches_processed_gin = 0
    
    for batch in pbar_train:
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_train_samples)
        if start_idx >= end_idx: 
            continue
        batch_idx = shuffled_train_index[start_idx:end_idx]
        try:
            output = model(ppi_g, ppi_list, batch_idx)
            if labels.device != device: 
                labels = labels.to(device)
            batch_labels = labels[batch_idx]
            if output.shape[0] != batch_labels.shape[0] or output.numel() == 0 or batch_labels.numel() == 0: 
                continue
            loss = F.binary_cross_entropy_with_logits(output, batch_labels)
            if torch.isnan(loss) or torch.isinf(loss): 
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            loss_sum += loss_item
            with torch.no_grad(): 
                f1_score_batch, _ = evaluat_metrics(output.detach(), batch_labels.detach())
            f1_sum += f1_score_batch
            num_batches_processed_gin += 1
            pbar_train.set_postfix({'Loss': f"{loss_item:.4f}", 'F1': f"{f1_score_batch:.4f}", 'LR': f"{current_lr:.6f}"})
        except Exception as e: 
            continue
    
    avg_loss = loss_sum / num_batches_processed_gin if num_batches_processed_gin > 0 else 0
    avg_f1 = f1_sum / num_batches_processed_gin if num_batches_processed_gin > 0 else 0
    return avg_loss, avg_f1


def main(param, timestamp, loaded_protein_data, loaded_ppi_g, loaded_ppi_list, loaded_labels, loaded_ppi_split_dict, vae_ckpt_path, resume_path=None):
    protein_data = loaded_protein_data
    ppi_g = loaded_ppi_g
    ppi_list = loaded_ppi_list
    labels = loaded_labels
    ppi_split_dict = loaded_ppi_split_dict
    if protein_data is None or ppi_g is None or labels is None or ppi_split_dict is None: 
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    vae_model_for_embed = None
    prot_embed = None
    log_file = None
    try:
        vae_model_for_embed = CodeBook(param, protein_data).to(device)
        final_vae_path = vae_ckpt_path
        if not (final_vae_path and os.path.exists(final_vae_path)): 
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        vae_state_dict_no_heads = torch.load(final_vae_path, map_location=device)
        model_load_result = vae_model_for_embed.load_state_dict(vae_state_dict_no_heads, strict=False)
        if not any(k.startswith('input_recon_head') for k in model_load_result.unexpected_keys): 
            pass

        vae_model_for_embed.eval()
        with torch.no_grad():
            prot_embed_cpu = vae_model_for_embed.get_protein_embeddings()
            if prot_embed_cpu is None or prot_embed_cpu.numel() == 0: 
                raise RuntimeError("get_protein_embeddings failed")
            prot_embed = prot_embed_cpu.to(device)
        param['protein_embedding_dim'] = prot_embed.shape[1]
        if ppi_g.num_nodes() == prot_embed.shape[0]:
            if ppi_g.device != device: 
                ppi_g = ppi_g.to(device)
            ppi_g.ndata['feat'] = prot_embed
        else: 
            raise ValueError(f"PPI graph node count ({ppi_g.num_nodes()}) mismatch with embedding count ({prot_embed.shape[0]})!")
    except Exception as e: 
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    finally: 
        del vae_model_for_embed
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    gin_output_dir = f"../results/{param['dataset']}/{timestamp}/GIN_CL_Aux_RandMCM_{param['seed']}/"
    check_writable(gin_output_dir, overwrite=False)
    log_file_path = os.path.join(gin_output_dir, "train_log.txt")
    try:
        log_mode_gin = 'a+' if resume_path and os.path.exists(resume_path) else 'w+'
        log_file = open(log_file_path, log_mode_gin)
        if log_mode_gin == 'w+': 
            log_file.write("--- GIN Configuration ---")
            log_file.write(json.dumps(param, indent=2))
            log_file.write("\n--- End Config ---\n\n")
        if log_mode_gin == 'a+': 
            log_file.write(f"\n--- Resuming GIN Training at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    except IOError as e: 
        log_file = None

    model = None
    try: 
        model = GIN(param).to(device)
    except Exception as e: 
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(param['learning_rate']), weight_decay=float(param['weight_decay']))
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=param.get('scheduler_patience', 100), verbose=True)
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    start_epoch_gin = 1
    es = 0
    best_epoch = 0
    val_best_f1 = 0.0
    val_best_aupr = 0.0
    test_f1_at_best_val_f1 = 0.0
    test_aupr_at_best_val_f1 = 0.0
    test_best_f1 = 0.0
    test_best_aupr = 0.0
    best_model_state = None
    gin_checkpoint_file = os.path.join(gin_output_dir, GIN_CHECKPOINT_FILENAME)
    gin_checkpoint_file_to_load = resume_path if resume_path and os.path.exists(resume_path) else gin_checkpoint_file
    
    if os.path.exists(gin_checkpoint_file_to_load):
        try:
            checkpoint = torch.load(gin_checkpoint_file_to_load, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint: 
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                if isinstance(scheduler, ReduceLROnPlateau): 
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch_gin = checkpoint.get('epoch', 0) + 1
            val_best_f1 = checkpoint.get('val_best_f1', 0.0)
            val_best_aupr = checkpoint.get('val_best_aupr', 0.0)
            test_f1_at_best_val_f1 = checkpoint.get('test_f1_at_best_val_f1', 0.0)
            test_aupr_at_best_val_f1 = checkpoint.get('test_aupr_at_best_val_f1', 0.0)
            test_best_f1 = checkpoint.get('test_best_f1', 0.0)
            test_best_aupr = checkpoint.get('test_best_aupr', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            es = checkpoint.get('es', 0)
            if 'best_model_state_dict' in checkpoint and checkpoint['best_model_state_dict'] is not None: 
                best_model_state = checkpoint['best_model_state_dict']
            else: 
                best_model_path_alt = os.path.join(gin_output_dir, GIN_BEST_MODEL_FILENAME)
            if os.path.exists(best_model_path_alt):
                try: 
                    best_model_state = torch.load(best_model_path_alt, map_location=device)
                except: 
                    pass
            if log_file: 
                log_file.write(f"Resumed GIN training from epoch {start_epoch_gin - 1}.\n")
        except Exception as e:
            start_epoch_gin=1
            es=0
            best_epoch=0
            val_best_f1=0.0
            val_best_aupr=0.0
            test_f1_at_best_val_f1=0.0
            test_aupr_at_best_val_f1=0.0
            test_best_f1=0.0
            test_best_aupr=0.0
            best_model_state=None

    sys.stdout.flush()
    for epoch in range(start_epoch_gin, param["max_epoch"] + 1):
        train_loss, train_f1_score = train(model, ppi_g, ppi_list, labels, ppi_split_dict['train_index'], param['batch_size'], optimizer, loss_fn, epoch, scheduler)
        if (epoch % param.get('log_num', 100) == 0) or (epoch == param["max_epoch"]):
            val_f1_score, val_aupr_score = evaluator(model, ppi_g, ppi_list, labels, ppi_split_dict['val_index'], param['batch_size'], 'metric')
            test_f1_score, test_aupr_score = evaluator(model, ppi_g, ppi_list, labels, ppi_split_dict['test_index'], param['batch_size'], 'metric')
            if test_f1_score > test_best_f1: 
                test_best_f1 = test_f1_score
            if test_aupr_score > test_best_aupr: 
                test_best_aupr = test_aupr_score
            if val_f1_score >= val_best_f1:
                val_best_f1 = val_f1_score
                val_best_aupr = val_aupr_score
                test_f1_at_best_val_f1 = test_f1_score
                test_aupr_at_best_val_f1 = test_aupr_score
                best_epoch = epoch
                es = 0
                try: 
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_save_path = os.path.join(gin_output_dir, GIN_BEST_MODEL_FILENAME)
                    torch.save(best_model_state, best_save_path)
                except Exception as e: 
                    pass
            else: 
                es += param.get('log_num', 100)
            current_lr = optimizer.param_groups[0]['lr']
            log_line = (f" Epoch: {epoch}, LR: {current_lr:.6f} | TrLoss: {train_loss:.5f}, TrF1: {train_f1_score:.4f} | ValF1: {val_f1_score:.4f}, ValAUPR: {val_aupr_score:.4f}, TestF1: {test_f1_score:.4f}, TestAUPR: {test_aupr_score:.4f} | ValBestF1: {val_best_f1:.4f}, ValBestAUPR: {val_best_aupr:.4f}, TestF1@Best: {test_f1_at_best_val_f1:.4f}, TestAUPR@Best: {test_aupr_at_best_val_f1:.4f}, TestBestF1: {test_best_f1:.4f}, TestBestAUPR: {test_best_aupr:.4f} | BestEp: {best_epoch}, ES: {es}\n")
            print(log_line.strip())
            if log_file: 
                log_file.write(log_line)
                log_file.flush()
            scheduler.step(val_f1_score)
            patience_epochs = param.get('patience', 500)
            if es >= patience_epochs : 
                break
            if epoch % param.get('save_ckpt_every', 1000) == 0:
                gin_ckpt_save_path = os.path.join(gin_output_dir, GIN_CHECKPOINT_FILENAME)
                try: 
                    torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'val_best_f1': val_best_f1, 'val_best_aupr': val_best_aupr, 'test_f1_at_best_val_f1': test_f1_at_best_val_f1, 'test_aupr_at_best_val_f1': test_aupr_at_best_val_f1, 'test_best_f1': test_best_f1, 'test_best_aupr': test_best_aupr, 'best_epoch': best_epoch, 'es': es, 'param': param, 'best_model_state_dict': best_model_state }, gin_ckpt_save_path)
                except Exception as e: 
                    pass
        sys.stdout.flush()

    best_state_path = os.path.join(gin_output_dir, GIN_BEST_MODEL_FILENAME)
    loaded_best = False
    if best_model_state is not None: 
        model.load_state_dict(best_model_state)
        loaded_best = True
    elif os.path.exists(best_state_path):
        try: 
            model.load_state_dict(torch.load(best_state_path, map_location=device))
            loaded_best = True
        except Exception as e: 
            pass
    
    sys.stdout.flush()
    final_test_output, final_test_labels = evaluator(model, ppi_g, ppi_list, labels, ppi_split_dict['test_index'], param['batch_size'], 'output')
    final_test_f1, final_test_aupr = 0.0, 0.0
    if final_test_output.numel() > 0 and final_test_labels.numel() > 0: 
        final_test_f1, final_test_aupr = evaluat_metrics(final_test_output, final_test_labels)

    try: 
        np.save(os.path.join(gin_output_dir, "eval_output.npy"), final_test_output.numpy())
        np.save(os.path.join(gin_output_dir, "eval_labels.npy"), final_test_labels.numpy())
    except Exception as e: 
        pass
    
    if log_file:
        try: 
            log_file.write(f"\n--- Training Finished ---\n")
            log_file.write(f"Best Val F1: {val_best_f1:.4f}\n")
            log_file.write(f"Best Val AUPR (at best F1 epoch): {val_best_aupr:.4f}\n")
            log_file.write(f"Test F1 @ Best Val F1 Epoch: {test_f1_at_best_val_f1:.4f}\n")
            log_file.write(f"Test AUPR @ Best Val F1 Epoch: {test_aupr_at_best_val_f1:.4f}\n")
            log_file.write(f"Final Test F1: {final_test_f1:.4f}\n")
            log_file.write(f"Final Test AUPR: {final_test_aupr:.4f}\n")
            log_file.write(f"Best Test F1 during run: {test_best_f1:.4f}\n")
            log_file.write(f"Best Test AUPR during run: {test_best_aupr:.4f}\n")
            log_file.write(f"Best Epoch: {best_epoch}\n")
            log_file.close()
        except Exception as e: 
            pass
    
    return test_f1_at_best_val_f1, val_best_f1, test_best_f1, best_epoch, test_aupr_at_best_val_f1, val_best_aupr, test_best_aupr


if __name__ == "__main__":
    default_param = {}
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dataset", default="SHS27k")
    pre_parser.add_argument("--split_mode", default="random")
    pre_args, _ = pre_parser.parse_known_args()
    dataset_key = pre_args.dataset
    split_mode_key = pre_args.split_mode
    config_file_path = "../configs/param_configs.json"
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f: 
                config_from_file = json.load(f)
            if dataset_key in config_from_file and split_mode_key in config_from_file[dataset_key]: 
                default_param.update(config_from_file[dataset_key][split_mode_key])
        except Exception as e: 
            pass

    parser = argparse.ArgumentParser(description="MicroEnvPPI: PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default=default_param.get('dataset', "SHS27k"))
    parser.add_argument("--split_mode", type=str, default=default_param.get('split_mode', "random"), choices=['random', 'bfs', 'dfs'])
    parser.add_argument("--seed", type=int, default=default_param.get('seed', 0))
    parser.add_argument("--input_dim", type=int, default=DEFAULT_ESM_DIM, help="Input dimension (fixed)")
    parser.add_argument("--output_dim", type=int, default=default_param.get('output_dim', 7))
    parser.add_argument("--ppi_hidden_dim", type=int, default=default_param.get('ppi_hidden_dim', 1024))
    parser.add_argument("--prot_hidden_dim", type=int, default=default_param.get('prot_hidden_dim', 256))
    parser.add_argument("--ppi_num_layers", type=int, default=default_param.get('ppi_num_layers', 2))
    parser.add_argument("--prot_num_layers", type=int, default=default_param.get('prot_num_layers', 4))
    parser.add_argument("--dropout_ratio", type=float, default=default_param.get('dropout_ratio', 0.0))
    parser.add_argument("--gin_final_dropout", type=float, default=default_param.get('gin_final_dropout', 0.5))
    parser.add_argument("--learning_rate", type=float, default=default_param.get('learning_rate', 0.001), help="LR for GIN")
    parser.add_argument("--weight_decay", type=float, default=default_param.get('weight_decay', 0.0001))
    parser.add_argument("--max_epoch", type=int, default=default_param.get('max_epoch', 500))
    parser.add_argument("--batch_size", type=int, default=default_param.get('batch_size', 10000), help="GIN batch size")
    parser.add_argument("--scheduler_patience", type=int, default=default_param.get('scheduler_patience', 100))
    parser.add_argument("--patience", type=int, default=default_param.get('patience', 500), help="GIN early stopping")
    parser.add_argument("--log_num", type=int, default=default_param.get('log_num', 100))
    parser.add_argument("--num_workers", type=int, default=default_param.get('num_workers', DEFAULT_NUM_WORKERS))
    parser.add_argument("--pre_epoch", type=int, default=default_param.get('pre_epoch', 50))
    parser.add_argument("--vae_batch_size", type=int, default=default_param.get('vae_batch_size', DEFAULT_VAE_BATCH_SIZE))
    parser.add_argument("--commitment_cost", type=float, default=default_param.get('commitment_cost', 0.25))
    parser.add_argument("--num_embeddings", type=int, default=default_param.get('num_embeddings', 512))
    parser.add_argument("--mask_ratio", type=float, default=default_param.get('mask_ratio', 0.15))
    parser.add_argument("--sce_scale", type=float, default=default_param.get('sce_scale', 1.0))
    parser.add_argument("--mask_loss", type=float, default=default_param.get('mask_loss', 1.0))
    parser.add_argument("--contrastive_loss_weight", type=float, default=default_param.get('contrastive_loss_weight', DEFAULT_CONTRASTIVE_WEIGHT))
    parser.add_argument("--contrastive_temperature", type=float, default=default_param.get('contrastive_temperature', DEFAULT_CONTRASTIVE_TEMP))
    parser.add_argument("--node_mask_rate", type=float, default=default_param.get('node_mask_rate', DEFAULT_NODE_MASK_RATE))
    parser.add_argument("--edge_drop_rate", type=float, default=default_param.get('edge_drop_rate', DEFAULT_EDGE_DROP_RATE), help="Unified edge drop rate")
    parser.add_argument("--vae_learning_rate", type=float, default=default_param.get('vae_learning_rate', DEFAULT_VAE_LR))
    parser.add_argument("--vae_scheduler", type=str, default=default_param.get('vae_scheduler', DEFAULT_VAE_SCHEDULER), choices=["none", "cosine"])
    parser.add_argument("--vae_lr_eta_min", type=float, default=default_param.get('vae_lr_eta_min', DEFAULT_VAE_ETA_MIN))
    parser.add_argument("--pretrain_patience", type=int, default=default_param.get('pretrain_patience', DEFAULT_PRETRAIN_PATIENCE))
    parser.add_argument("--input_mask_ratio", type=float, default=default_param.get('input_mask_ratio', DEFAULT_INPUT_MASK_RATIO))
    parser.add_argument("--masked_feat_loss_weight", type=float, default=default_param.get('masked_feat_loss_weight', DEFAULT_MASKED_FEAT_LOSS_WEIGHT))
    parser.add_argument("--save_ckpt_every", type=int, default=default_param.get('save_ckpt_every', 100))
    parser.add_argument("--ppi_interaction_mode", type=str, default=default_param.get('ppi_interaction_mode', "mlp"), choices=['dot', 'mlp', 'bilinear'])
    parser.add_argument("--data_mode", type=int, default=default_param.get('data_mode', -1), help="Overrides config dataset")
    parser.add_argument("--data_split_mode", type=int, default=default_param.get('data_split_mode', -1), help="Overrides config split")
    parser.add_argument("--pre_train_data", type=str, default=default_param.get('pre_train_data', None))
    parser.add_argument("--ckpt_path", type=str, default=default_param.get('ckpt_path', None), help="Path to PRE-TRAINED final VAE ckpt (no heads) to skip pre-training")
    parser.add_argument("--resume", type=str, default=default_param.get('resume', None), help="Path to VAE or GIN checkpoint to resume training")

    args = parser.parse_args()
    param = args.__dict__
    param['protein_embedding_mode'] = 'concat'

    if param['data_mode'] == 0: 
        param['dataset'] = 'SHS27k'
    elif param['data_mode'] == 1: 
        param['dataset'] = 'SHS148k'
    elif param['data_mode'] == 2: 
        param['dataset'] = 'STRING'
    if param['data_split_mode'] == 0: 
        param['split_mode'] = 'random'
    elif param['data_split_mode'] == 1: 
        param['split_mode'] = 'bfs'
    elif param['data_split_mode'] == 2: 
        param['split_mode'] = 'dfs'
    param['input_dim'] = DEFAULT_ESM_DIM

    params_to_print = {k:v for k,v in param.items() if k not in ['projection_dim', 'edge_drop_seq', 'edge_drop_knn', 'edge_drop_dis', 'mcm_masking_strategy', 'mcm_num_clusters', 'mcm_analysis_freq']}
    sys.stdout.flush()

    set_seed(param['seed'])

    sys.stdout.flush()
    protein_data, ppi_g, ppi_list, labels, ppi_split_dict = None, None, None, None, None
    try:
        if param.get('pre_train_data'): 
            pass
        protein_data, ppi_g, ppi_list, labels, ppi_split_dict = load_data(param['dataset'], param['split_mode'], param['seed'])
        sys.stdout.flush()
    except Exception as e: 
        sys.exit(1)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{int((time.time() - int(time.time())) * 1000):03d}"

    vae_checkpoint_to_use = param.get('ckpt_path')
    resume_ckpt_path = param.get('resume')

    if vae_checkpoint_to_use is None:
        sys.stdout.flush()
        resume_path_for_vae = None
        if resume_ckpt_path and os.path.exists(resume_ckpt_path) and VAE_CHECKPOINT_FILENAME in os.path.basename(resume_ckpt_path): 
            resume_path_for_vae = resume_ckpt_path
        vae_checkpoint_to_use = pretrain_vae(param, timestamp, protein_data, resume_path=resume_path_for_vae)
        if not (vae_checkpoint_to_use and os.path.exists(vae_checkpoint_to_use)): 
            sys.exit(1)
        if resume_path_for_vae: 
            resume_ckpt_path = None
    else:
        sys.stdout.flush()
        if not os.path.exists(vae_checkpoint_to_use): 
            sys.exit(1)
        if resume_ckpt_path and VAE_CHECKPOINT_FILENAME in os.path.basename(resume_ckpt_path): 
            resume_ckpt_path = None

    sys.stdout.flush()
    resume_path_for_gin = None
    if resume_ckpt_path and os.path.exists(resume_ckpt_path) and GIN_CHECKPOINT_FILENAME in os.path.basename(resume_ckpt_path): 
        resume_path_for_gin = resume_ckpt_path

    results = main(param, timestamp, protein_data, ppi_g, ppi_list, labels, ppi_split_dict, vae_checkpoint_to_use, resume_path=resume_path_for_gin)
    final_test_f1, final_val_best_f1, final_test_best_f1, final_best_epoch, final_test_aupr, final_val_best_aupr, final_test_best_aupr = results

    sys.stdout.flush()

    csv_file_path = '../PerformMetrics_Metrics.csv'
    sys.stdout.flush()
    try:
        write_header = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0
        final_params_used = {k: param[k] for k in parser.parse_args().__dict__ if k in param and k not in ['projection_dim', 'edge_drop_seq', 'edge_drop_knn', 'edge_drop_dis', 'mcm_masking_strategy', 'mcm_num_clusters', 'mcm_analysis_freq']}
        param_keys = sorted([k for k in final_params_used.keys() if k not in ['device']])
        header = ["Timestamp"] + param_keys + [ "TestF1_AtBestValF1", "BestValF1", "BestTestF1", "BestEpoch", "TestAUPR_AtBestValF1", "BestValAUPR", "BestTestAUPR"]
        with open(csv_file_path,'a+', newline='') as outFile:
            writer = csv.writer(outFile, dialect='excel')
            if write_header: 
                writer.writerow(header)
            results_row = [timestamp] + [final_params_used.get(k, 'N/A') for k in param_keys] + [f"{final_test_f1:.4f}", f"{final_val_best_f1:.4f}", f"{final_test_best_f1:.4f}", final_best_epoch] + [f"{final_test_aupr:.4f}", f"{final_val_best_aupr:.4f}", f"{final_test_best_aupr:.4f}"]
            writer.writerow(results_row)
    except Exception as e: 
        pass