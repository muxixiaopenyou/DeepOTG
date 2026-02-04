import torch
import torch.utils.data as Data
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
import numpy as np
import os
import glob
import random
from load_data import *
from model import *

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)

    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = mcc_num / (mcc_den + 1e-8)

    roc_auc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    bacc = (sensitivity + specificity) / 2
    
    return {
        'MCC': mcc,
        'SN': sensitivity,
        'SP': specificity,
        'ACC': accuracy,
        'BACC': bacc,
        'PRECISION': precision,
        'F1': f1,
        'AUC': roc_auc,
        'AUCPR': pr_auc
    }

def run_test(test_name, seq_path, pssm_path, feat_path, result_file, max_rows=None):
    print(f"\n{'='*10} Starting {test_name} {'='*10}")

    try:
        test_sequences, test_labels = load_encoding_from_txt(seq_path, pssm_path, max_rows=max_rows)
        test_features = load_features_from_txt(feat_path)
    except Exception as e:
        print(f"Error loading data for {test_name}: {e}")
        return

    test_sequences = np.array(test_sequences)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    print(f"Data loaded. PSSM features: {test_sequences.shape}, ESM Features: {test_features.shape}")

    test_dataset = MyDataSet(test_sequences, test_features, test_labels)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_model_paths = sorted(glob.glob('./saved_models/*.pth'))
    if not best_model_paths:
        print("No model files found in ./saved_models/ directory")
        return

    print(f"Found {len(best_model_paths)} models. Starting ensemble prediction")
    
    all_fold_probs = []

    for model_path in best_model_paths:
        model = FusionPepNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        fold_probs = []
        
        with torch.no_grad():
            for input_ids, sequence_features, _ in test_loader:
                input_ids = input_ids.to(device)
                sequence_features = sequence_features.to(device)
                
                outputs, _, _, _, _ = model(input_ids, sequence_features)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                fold_probs.extend(probs)
        
        all_fold_probs.append(fold_probs)

    if not all_fold_probs:
        print("No predictions made.")
        return

    avg_probs = np.mean(all_fold_probs, axis=0)
    
    y_pred = (avg_probs >= 0.5).astype(int)
    y_true = test_labels
    
    metrics = calculate_metrics(y_true, y_pred, avg_probs)

    print(f"\n====== {test_name} Results ======")
    for k, v in metrics.items():
        print(f"{k:10}: {v:.3f}")

    np.savetxt(result_file, np.column_stack((y_true, y_pred, avg_probs)), 
            fmt='%d\t%d\t%.6f', header='True_Label\tPredicted_Label\tProbability')
    print(f"\nResults saved to {result_file}")

set_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

run_test("Balanced Independent Test", './data/balanced.txt', './db/balanced/pssm', './data/balanced_esm.txt', 'balanced_results.txt')
run_test("Imbalanced Independent Test", './data/imbalanced.txt', './db/imbalanced/pssm', './data/imbalanced_esm.txt', 'imbalanced_results.txt')