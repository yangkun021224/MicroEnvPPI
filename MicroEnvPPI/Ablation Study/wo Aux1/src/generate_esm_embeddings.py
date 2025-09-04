import torch
from transformers import AutoTokenizer, EsmModel
import csv
import os
from tqdm import tqdm
import gc
import sys

dataset = "SHS148k"#"STRING"or"SHS148k"
local_model_path = "/root/autodl-tmp/esm2_t33_650M_UR50D/"
script_dir = os.path.dirname(os.path.abspath(__file__)) 
base_dir = os.path.dirname(script_dir) 
data_dir = os.path.join(base_dir, 'data', 'processed_data')
seq_file = os.path.join(data_dir, f'protein.{dataset}.sequences.dictionary.csv')
output_pt_file = os.path.join(data_dir, f'protein.nodes.esm2_650M.{dataset}.pt')
max_seq_len = 1022
batch_size = 64
embedding_dim = 1280

print("--- Starting ESM-2 650M embedding generation ---")
print(f"Dataset: {dataset}")
print(f"Local model path: {local_model_path}")
print(f"Sequence file: {seq_file}")
print(f"Output file: {output_pt_file}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Batch size: {batch_size}")
sys.stdout.flush()

output_dir_check = os.path.dirname(output_pt_file)
if not os.path.exists(output_dir_check):
    try:
        os.makedirs(output_dir_check)
        print(f"Created directory: {output_dir_check}")
    except OSError as e:
        print(f"Error: Cannot create output directory {output_dir_check}: {e}")
        exit(1)

print(f"Loading model and tokenizer from {local_model_path}...")
sys.stdout.flush()
try:
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = EsmModel.from_pretrained(local_model_path)
    model.eval()
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error: Error loading model or tokenizer: {e}")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sys.stdout.flush()
if device == torch.device("cuda"):
    try:
        model = model.to(device)
        print("Model moved to GPU.")
    except Exception as e:
        print(f"Warning: Error moving model to GPU: {e}")
        print("Will try to run on CPU.")
        device = torch.device("cpu")
sys.stdout.flush()

print(f"Loading sequences from {seq_file}...")
sys.stdout.flush()
protein_sequences = []
protein_order = []
try:
    with open(seq_file, 'r') as f:
        reader = csv.reader(f)
        line_num = 0
        for row in reader:
            line_num += 1
            if len(row) >= 2:
                protein_id = row[0]
                sequence = row[1]
                if not protein_id or not sequence:
                    print(f"Warning: Line {line_num} protein ID or sequence is empty, skipped.")
                    continue
                protein_sequences.append((protein_id, sequence))
                protein_order.append(protein_id)
            else:
                print(f"Warning: Line {line_num} format incorrect (insufficient columns), skipped. Content: {row}")

except FileNotFoundError:
    print(f"Error: Sequence file {seq_file} not found!")
    exit(1)
except Exception as e:
    print(f"Error: Error reading sequence file {seq_file}: {e}")
    exit(1)

num_sequences_loaded = len(protein_sequences)
print(f"Loaded {num_sequences_loaded} valid protein sequences.")
if num_sequences_loaded == 0:
    print("Error: No valid protein sequences loaded from file. Please check file content and format.")
    exit(1)
sys.stdout.flush()

all_embeddings_list = []
print(f"Starting embedding generation, batch size: {batch_size}...")
sys.stdout.flush()

batch_iterator = range(0, num_sequences_loaded, batch_size)
progress_bar = tqdm(batch_iterator, desc="Processing batches", mininterval=1.0)

error_count = 0
oom_count = 0

with torch.no_grad():
    for i in progress_bar:
        batch_sequences = protein_sequences[i:min(i + batch_size, num_sequences_loaded)]
        if not batch_sequences: 
            continue

        batch_ids = [item[0] for item in batch_sequences]
        batch_seq_data = [item[1] for item in batch_sequences]

        try:
            inputs = tokenizer(batch_seq_data, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            hidden_states = outputs.last_hidden_state

            for j in range(hidden_states.shape[0]):
                original_seq_len = len(batch_seq_data[j])
                valid_len = min(original_seq_len, max_seq_len)
                embeddings = hidden_states[j, 1:valid_len+1, :].cpu()

                if embeddings.shape[0] == 0:
                    print(f"Warning: Protein {batch_ids[j]} generated empty embedding (valid length 0?), skipped.")
                    continue

                if embeddings.shape[1] != embedding_dim:
                    print(f"Error: Protein {batch_ids[j]} embedding dimension is {embeddings.shape[1]}, expected {embedding_dim}. Please check model or code.")
                    error_count += 1
                    continue

                all_embeddings_list.append(embeddings)

            del inputs, outputs, hidden_states
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            error_message = str(e).lower()
            if "out of memory" in error_message:
                oom_count += 1
                progress_bar.write(f"\nError: GPU OOM when processing batch {i // batch_size} - {e}")
                progress_bar.write("Please stop script, reduce batch_size and try again.")
                break
            else:
                error_count += 1
                progress_bar.write(f"\nError: Runtime error when processing batch {i // batch_size}: {e}")
                continue
        except Exception as e:
            error_count += 1
            progress_bar.write(f"\nError: Unknown error when processing batch {i // batch_size}: {e}")
            continue

progress_bar.close()

print(f"\nEmbedding generation loop ended. OOM count: {oom_count}, Other error count: {error_count}")
sys.stdout.flush()

num_embeddings_generated = len(all_embeddings_list)
print(f"Successfully generated embeddings for {num_embeddings_generated} proteins.")
if num_embeddings_generated != num_sequences_loaded - error_count:
    print(f"Warning: Final embedding count ({num_embeddings_generated}) does not match expected ({num_sequences_loaded - error_count})! Possible data loss.")

if num_embeddings_generated == 0:
    print("Error: No protein embeddings generated successfully. Cannot save file.")
    exit(1)

print(f"Preparing to save {num_embeddings_generated} embeddings to {output_pt_file}...")
sys.stdout.flush()
try:
    if not os.path.exists(output_dir_check):
        os.makedirs(output_dir_check)
        print(f"Reconfirmed and created directory: {output_dir_check}")

    torch.save(all_embeddings_list, output_pt_file)
    print(f"Embedding file attempted to save to: {output_pt_file}")

    if os.path.exists(output_pt_file):
        print("File saved successfully! Verified file exists.")
        file_size = os.path.getsize(output_pt_file)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        if file_size < 1024:
            print("Warning: Saved file is very small, might be incomplete or data is empty.")
    else:
        print("Error: After saving, file check shows file does not exist! Please check write permissions and disk space.")

except Exception as e:
    print(f"Error: Error saving embedding file: {e}")
    print("Please check disk space, write permissions, and path correctness.")

print("--- ESM-2 650M embedding generation complete ---")