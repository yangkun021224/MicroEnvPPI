import os
import re
import argparse
import numpy as np
import torch
from tqdm import tqdm
import sys

# This function is adapted from the user-provided MicroEnvPPI/src/data_process.py
def match_feature(x, all_for_assign):
    """Maps a sequence of 3-letter amino acid codes to a 7D feature matrix."""
    # Mapping from 3-letter code to index in all_for_assign
    aa_map = {
        'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4, 'GLY': 5, 'HIS': 6,
        'ILE': 7, 'LYS': 8, 'LEU': 9, 'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13,
        'ARG': 14, 'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19
    }
    x_p = np.zeros((len(x), 7))
    for j, aa_code in enumerate(x):
        if aa_code in aa_map:
            x_p[j] = all_for_assign[aa_map[aa_code], :]
        # else, keep as zeros, a reasonable default for unknown/other residues
    return x_p

# This function is adapted from the user-provided MicroEnvPPI/src/data_process.py
def read_atoms(file_path):
    """Reads the amino acid sequence from a PDB file."""
    ajs = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    ajs_id = line[17:20].strip()
                    ajs.append(ajs_id)
    except Exception as e:
        print(f"Error reading PDB file {file_path}: {e}", file=sys.stderr)
    return ajs

def process_7d_features(dataset, pdb_dir, data_dir):
    """Generates and saves 7D features for a specified dataset."""
    print(f"--- Starting 7D feature generation for {dataset} ---")

    # Define paths
    processed_data_dir = os.path.join(data_dir, 'processed_data')
    assign_file = os.path.join(data_dir, 'all_assign.txt')
    seq_dict_file = os.path.join(processed_data_dir, f'protein.{dataset}.sequences.dictionary.csv')
    output_file = os.path.join(processed_data_dir, f'protein.nodes.7d.{dataset}.pt')
    
    # Ensure output directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    # Check for required files and directories
    if not os.path.exists(assign_file):
        print(f"Error: `all_assign.txt` not found in `{data_dir}`. Please add this file.", file=sys.stderr)
        return
    if not os.path.isdir(pdb_dir):
        print(f"Error: PDB directory `{pdb_dir}` not found.", file=sys.stderr)
        return
    if not os.path.exists(seq_dict_file):
        print(f"Error: Sequence dictionary file not found at `{seq_dict_file}`.", file=sys.stderr)
        return

    # Load the 7D feature mapping
    try:
        all_for_assign = np.loadtxt(assign_file)
        if all_for_assign.shape != (20, 7):
            print(f"Warning: `all_assign.txt` has shape {all_for_assign.shape}, expected (20, 7).", file=sys.stderr)
    except Exception as e:
        print(f"Error loading `all_assign.txt`: {e}", file=sys.stderr)
        return

    # Get protein ID list and order from sequence dictionary
    protein_ids = []
    try:
        with open(seq_dict_file, 'r') as f:
            next(f, None) # Skip potential header
            for row in f:
                parts = row.strip().split(',')
                if len(parts) > 0 and parts[0]:
                    protein_ids.append(parts[0])
    except FileNotFoundError:
        print(f"Error: Could not open sequence dictionary file {seq_dict_file}", file=sys.stderr)
        return

    print(f"Found {len(protein_ids)} proteins to process for dataset {dataset}.")

    node_feature_list = []
    # Use the original protein list to ensure consistent order
    for protein_id in tqdm(protein_ids, desc=f"Processing PDBs for {dataset}"):
        pdb_file_path = os.path.join(pdb_dir, protein_id + '.pdb')
        if not os.path.exists(pdb_file_path):
            print(f"Warning: PDB file for {protein_id} not found, skipping.", file=sys.stderr)
            # Add an empty Tensor placeholder to maintain index consistency
            node_feature_list.append(torch.empty(0, 7, dtype=torch.float32))
            continue

        amino_acid_sequence = read_atoms(pdb_file_path)
        if not amino_acid_sequence:
            print(f"Warning: No CA atoms found for {protein_id}, skipping.", file=sys.stderr)
            node_feature_list.append(torch.empty(0, 7, dtype=torch.float32))
            continue

        feature_matrix = match_feature(amino_acid_sequence, all_for_assign)
        node_feature_list.append(torch.tensor(feature_matrix, dtype=torch.float32))

    # Save the generated features
    try:
        torch.save(node_feature_list, output_file)
        print(f"Successfully saved 7D features for {len(node_feature_list)} proteins to:")
        print(output_file)
    except Exception as e:
        print(f"Error saving output file: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 7D physicochemical features for proteins.")
    parser.add_argument("--dataset", type=str, required=True, choices=["SHS27k", "SHS148k", "STRING"], help="Name of the dataset.")
    parser.add_argument("--pdb_dir", type=str, default="../STRING_AF2DB", help="Directory containing PDB files.")
    parser.add_argument("--data_dir", type=str, default="../data", help="Base directory containing `all_assign.txt` and the `processed_data` folder.")
    args = parser.parse_args()

    process_7d_features(args.dataset, args.pdb_dir, args.data_dir)

