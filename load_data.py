import torch
import torch.utils.data as Data
import numpy as np
import os
import re

def read_pssm(pssm_path):
    data = []
    with open(pssm_path, 'r') as f:
        for line in f:
            m = re.match(r'\s*\d+\s+[A-Z]', line)
            if m:
                parts = line.strip().split()
                if len(parts) >= 22:  # ensure 20 columns of scores exist
                    scores = [float(x) for x in parts[2:22]]
                    data.append(scores)
    return np.array(data, dtype=np.float32)

def one_hot_encode(seq):
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = {aa: idx for idx, aa in enumerate(aa_list)}
    one_hot = np.zeros((len(seq), len(aa_list)), dtype=np.float32)
    for i, aa in enumerate(seq):
        idx = aa_dict.get(aa, 20)
        if idx is not None:
            one_hot[i, idx] = 1.0
    return one_hot

def feature_generator(fasta_file, pssm_dir, max_rows=None):
    fastas = []
    with open(fasta_file, 'r') as f:
        name = None
        seq = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if name:
                    fastas.append((name, "".join(seq)))
                name = line.lstrip('>').split('|')[0]
                seq = []
            else:
                seq.append(line)
        if name:
            fastas.append((name, "".join(seq)))

    datas = []
    for name, seq in fastas:
        pssm_file = os.path.join(pssm_dir, f"{name}.pssm")
        pssm_feat = read_pssm(pssm_file)  # (L,20)
        oh_feat = one_hot_encode(seq)     # (L,20)
        aa_feat = np.hstack([pssm_feat, oh_feat])  # (L,40)
        datas.append(aa_feat)

    # padding or truncating
    if max_rows is None:
        max_rows = max(d.shape[0] for d in datas)

    padded_datas = []
    for d in datas:
        if d.shape[0] < max_rows:
            pad = np.zeros((max_rows - d.shape[0], d.shape[1]), dtype=np.float32)
            padded_datas.append(np.vstack([d, pad]))
        elif d.shape[0] > max_rows:
            padded_datas.append(d[:max_rows, :])
        else:
            padded_datas.append(d)

    return np.stack(padded_datas).astype(np.float32)

def read_protein_sequences_from_txt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    sequences = []
    labels = []
    sequence = ''
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if sequence:
                        sequences.append(sequence)
                        sequence = ''
                    if 'pos' in line.lower():
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    sequence += line
            if sequence:
                sequences.append(sequence)
    except Exception as e:
        raise IOError(f"Error reading FASTA file: {e}")
        
    return sequences, labels

# load PSSM encoding
def load_encoding_from_txt(fasta_file, pssm_dir, max_rows=None):
    sequences, labels = read_protein_sequences_from_txt(fasta_file)
    encoded_sequences = feature_generator(fasta_file, pssm_dir, max_rows)
    return encoded_sequences, labels

# load esm features
def load_features_from_txt(feature_file_path):
    features = np.loadtxt(feature_file_path)
    return features

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, features, labels):
        self.input_ids = input_ids
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.float32),
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
