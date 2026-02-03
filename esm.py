import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

def extract_esm2_features(fasta_file, output_file, model_name="facebook/esm2_t12_35M_UR50D"):
    print(f"  Loading ESM-2 model:  {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    model.to(device)
    model.eval()
    
    print(f"Reading sequences from {fasta_file}")
    sequences = []
    seq_ids = []
    current_seq = ''
    current_id = ''
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    seq_ids.append(current_id)
                    current_seq = ''
                current_id = line[1:]
            else:
                current_seq += line
        
        # add the last sequence
        if current_seq:
            sequences.append(current_seq)
            seq_ids.append(current_id)
    
    print(f"Found {len(sequences)} sequences")
    
    features_list = []
    with torch.no_grad():
        for idx, (seq_id, seq) in enumerate(zip(seq_ids, sequences)):
            if (idx + 1) % 10 == 0 or (idx + 1) == len(sequences):
                print(f"   Progress: {idx + 1}/{len(sequences)}")
            
            inputs = tokenizer(
                seq, 
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # [CLS]
            cls_embedding = outputs.last_hidden_state[0, 0, : ].cpu().numpy()
            features_list.append(cls_embedding)
    
    features_array = np.array(features_list)
    np.savetxt(output_file, features_array, fmt='%.6f')
    
    print(f"Saved features to {output_file}")
    print(f"Shape: {features_array.shape}")
    return features_array

if __name__ == "__main__": 
    input_file = "data/train.txt"
    output_file = "data/train_esm.txt"
    model_name = "facebook/esm2_t6_8M_UR50D" 
    extract_esm2_features(input_file, output_file, model_name)