import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 40 refer to PSSM(20) + One-hot(20)
class MultiCNN_BiGRU(nn.Module):
    def __init__(self):
        super(MultiCNN_BiGRU, self).__init__()
        self.filter_sizes = [1, 2, 3, 4, 5, 6]
        filter_num = 32 
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (fsz, 40)) for fsz in self.filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(input_size=len(self.filter_sizes) * filter_num,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        # 2D-CNN
        # Channel=1, Height=sequence length, Width=40
        x = x.unsqueeze(1)
        conv_outputs = [F.relu(conv(x)) for conv in self.convs]
        pooled_outputs = [F.max_pool2d(item, kernel_size=(item.size(2), item.size(3))) for item in conv_outputs]
        pooled_outputs = [item.view(item.size(0), -1) for item in pooled_outputs]
        cnn_feature = torch.cat(pooled_outputs, dim=1)  
        
        gru_in = cnn_feature.view(cnn_feature.size(0), 1, -1) 
        gru_out, _ = self.gru(gru_in)  
        gru_feature = gru_out[:, -1, :]  
        gru_feature = self.dropout(gru_feature)
        output = self.block1(gru_feature)  
        return output

# 320 refer to the esm-2 8m model
class MLPBranch(nn.Module):
    def __init__(self):
        super(MLPBranch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(320, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        return self.mlp(x) 

class MultiHeadFusion(nn.Module):
    def __init__(self, feature_dim=64, num_heads=4):
        super().__init__()
        assert feature_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)

        self.out = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        batch_size = features.size(0)

        Q = self.W_q(features)  
        K = self.W_k(features)
        V = self.W_v(features)

        Q = Q.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)  
        K = K.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) 
        attn_weights = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, V) 

        context = context.transpose(1, 2).contiguous().view(batch_size, 2, -1)  

        output = self.out(context.sum(dim=1))  
        return self.layer_norm(output)

class FusionPepNet(nn.Module):
    def __init__(self):
        super(FusionPepNet, self).__init__()
        self.branch1 = MultiCNN_BiGRU()  
        self.branch2 = MLPBranch()         
        self.fusion_attn = MultiHeadFusion(feature_dim=64, num_heads=8)
        self.dvib = DVIB(input_dim=64, latent_dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        
    def forward(self, input_ids, features):
        feat1 = self.branch1(input_ids)  
        feat2 = self.branch2(features)   
        combined = torch.stack([feat1, feat2], dim=1)
        fused = self.fusion_attn(combined)
        z, mu, std = self.dvib(fused)
        logits = self.classifier(z)
        return logits, mu, std, feat1, feat2

class DVIB(nn.Module):
    def __init__(self, input_dim=64, latent_dim=64):
        super(DVIB, self).__init__()
        self.enc_mean = nn.Linear(input_dim, latent_dim)
        self.enc_std  = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x):
        mu = self.enc_mean(x)
        std = F.softplus(self.enc_std(x) - 5)
        if self.training:
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        return z, mu, std

def vib_loss(logits, labels, mu, std, beta=1e-3):
    ce = F.cross_entropy(logits, labels)
    kl = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
    return ce + beta * kl / logits.size(0)
