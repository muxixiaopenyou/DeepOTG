import os
import numpy as np
import re

# read PSSM file
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


# one-hot encoding
def one_hot_encode(seq):
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = {aa: idx for idx, aa in enumerate(aa_list)}
    one_hot = np.zeros((len(seq), len(aa_list)), dtype=np.float32)
    for i, aa in enumerate(seq):
        idx = aa_dict.get(aa, 20)
        if idx is not None:
            one_hot[i, idx] = 1.0
    return one_hot

# return PSSM+one-hot features
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
