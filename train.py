import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
import numpy as np
import os
import logging
from datetime import datetime
from load_data import *
from model import *

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.makedirs('models', exist_ok=True)
os.makedirs('log', exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'log/training_{current_time}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

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

set_seed(0)

# PSSM and one-hot features
sequence_file = "./data/train.txt"
pssm_dir = "./db/train/pssm"
sequences, labels = load_encoding_from_txt(sequence_file, pssm_dir)
# esm features
features = load_features_from_txt('./data/train_esm.txt')

sequences = np.array(sequences)
features = np.array(features)
labels = np.array(labels)

max_len = sequences.shape[1]
logger.info(f"Max sequence length: {max_len}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
lr = 3e-4
weight_decay = 5e-4
beta_vib = 5e-3
n_epochs = 200
k_folds = 5
patience = 10

logger.info("="*30)
logger.info("Hyperparameters:")
logger.info(f"Batch Size: {batch_size}")
logger.info(f"Learning Rate: {lr}")
logger.info(f"Weight Decay: {weight_decay}")
logger.info(f"Beta VIB: {beta_vib}")
logger.info(f"Epochs: {n_epochs}")
logger.info(f"K-Folds: {k_folds}")
logger.info(f"Patience: {patience}")
logger.info("="*30)

criterion = nn.CrossEntropyLoss()

kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)

fold_metrics = [] 

for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):

    train_seq, val_seq = sequences[train_idx], sequences[val_idx]
    train_feat, val_feat = features[train_idx], features[val_idx]
    train_lbl, val_lbl = labels[train_idx], labels[val_idx]

    dataset_train = MyDataSet(train_seq, train_feat, train_lbl)
    dataset_val = MyDataSet(val_seq, val_feat, val_lbl)

    train_loader = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model = FusionPepNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_one_epoch(model, loader, optimizer, beta=beta_vib):
        model.train()
        total_loss = 0.0
        total_kl = 0.0
        for input_ids, sequence_features, labels in loader:
            input_ids, sequence_features, labels = input_ids.to(device), sequence_features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, mu, std, _, _ = model(input_ids, sequence_features)
            loss = vib_loss(logits, labels, mu, std, beta=beta)
            loss.backward()
            optimizer.step()

            kl = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - torch.log(std.pow(2) + 1e-8) - 1)
            total_kl += kl.item()
            total_loss += loss.item()

        return total_loss / len(loader), total_kl / len(loader)

    def evaluate_model(model, loader):
        model.eval()
        total_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for input_ids, sequence_features, labels in loader:
                input_ids, sequence_features, labels = input_ids.to(device), sequence_features.to(device), labels.to(device)
                
                outputs, mu, std, _, _ = model(input_ids, sequence_features)
                loss = vib_loss(outputs, labels, mu, std, beta=beta_vib)
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        metrics['Val Loss'] = total_loss / len(loader)
        return metrics

    best_metrics = None
    min_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    kl_losses = []

    for epoch in range(n_epochs):
        train_loss, kl_loss = train_one_epoch(model, train_loader, optimizer, beta=beta_vib)
        val_metrics = evaluate_model(model, val_loader)
        
        train_losses.append(train_loss)
        kl_losses.append(kl_loss)
        val_losses.append(val_metrics['Val Loss'])

        current_val_loss = val_metrics['Val Loss']

        if current_val_loss < min_val_loss - 1e-6:
            min_val_loss = current_val_loss
            best_metrics = val_metrics
            torch.save(model.state_dict(), f"./models/fold_{fold+1}.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        
    fold_metrics.append(best_metrics)
    for k, v in best_metrics.items():
        if k != 'Val Loss':
            logger.info(f"{k:10}: {v:.4f}")

metric_names = ['MCC', 'SN', 'SP', 'ACC', 'BACC', 'PRECISION', 'F1', 'AUC', 'AUCPR']
for name in metric_names:
    values = [m[name] for m in fold_metrics]
    logger.info(f"{name:10}: {np.mean(values):.4f} ± {np.std(values):.4f}")
