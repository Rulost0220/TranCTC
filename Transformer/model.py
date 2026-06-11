import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader, Sampler
import os
from tqdm import tqdm
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional
import sys
import pdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from utils import _collect_csv_files

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    """
    English documentation retained from the original workflow.
    English documentation retained from the original workflow.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # English note retained from the original workflow.
        self.last_attn_weights = None  # English note retained from the original workflow.

    def _sa_block(self, x, attn_mask, key_padding_mask):
        # x: [B, T, E]
        # English note retained from the original workflow.
        attn_output, attn_output_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        # English note retained from the original workflow.
        self.last_attn_weights = attn_output_weights
        return self.dropout1(attn_output)

class MultiFieldCrossAttention(nn.Module):
    def __init__(
        self,
        d_feat: int,
        nhead: int,
        query_indices: list,          # e.g. [0,1,2,3,4]
        dropout: float = 0.1,
        agg: str = "mean"  # English note retained from the original workflow.
    ):
        super().__init__()
        self.query_indices = sorted(query_indices)
        self.agg = agg

        # English note retained from the original workflow.
        self.q_proj = nn.Linear(1, d_feat, bias=False)
        self.k_proj = nn.Linear(1, d_feat, bias=False)
        self.v_proj = nn.Linear(1, d_feat, bias=False)

        self.attn = nn.MultiheadAttention(d_feat, nhead, batch_first=True, dropout=dropout)
        self.out = nn.Linear(d_feat, d_feat)

        if self.agg == "attn":
            # English note retained from the original workflow.
            self.query_pool = nn.MultiheadAttention(d_feat, nhead, batch_first=True, dropout=dropout)
            self.pool_token = nn.Parameter(torch.randn(1, 1, d_feat))  # English note retained from the original workflow.

    def forward(self, x):  # English note retained from the original workflow.
        B, T, F = x.shape
        z = x.reshape(B * T, F, 1)                       # [B*T, F, 1]

        # English note retained from the original workflow.
        q_feats = z[:, self.query_indices, :]            # [B*T, Q, 1], Q=len(query_indices)
        Q = self.q_proj(q_feats)                         # [B*T, Q, d_feat]

        # English note retained from the original workflow.
        K = self.k_proj(z)                               # [B*T, F, d_feat]
        V = self.v_proj(z)                               # [B*T, F, d_feat]

        # English note retained from the original workflow.
        ctx_q, _ = self.attn(Q, K, V)                    # [B*T, Q, d_feat]
        ctx_q = self.out(ctx_q)                          # [B*T, Q, d_feat]

        # English note retained from the original workflow.
        if self.agg == "mean":
            ctx = ctx_q.mean(dim=1, keepdim=True)        # [B*T, 1, d_feat]
        else:  # English note retained from the original workflow.
            pool = self.pool_token.expand(ctx_q.size(0), 1, -1)  # [B*T,1,d_feat]
            ctx, _ = self.query_pool(pool, ctx_q, ctx_q)         # [B*T,1,d_feat]

        ctx = ctx.reshape(B, T, -1)                      # [B, T, d_feat]
        return ctx
    

class SingleFieldCrossAttention(nn.Module):
    def __init__(self, d_feat: int, nhead: int, query_idx: int, dropout: float = 0.1):
        super().__init__()
        self.query_idx = query_idx

        # English note retained from the original workflow.
        self.q_proj = nn.Linear(1, d_feat, bias=False)
        self.k_proj = nn.Linear(1, d_feat, bias=False)
        self.v_proj = nn.Linear(1, d_feat, bias=False)

        self.attn = nn.MultiheadAttention(d_feat, nhead, batch_first=True, dropout=dropout)
        self.out = nn.Linear(d_feat, d_feat)

    def forward(self, x):  # x: [B, T, F]
        B, T, F = x.shape
        z = x.reshape(B * T, F, 1)                  # [B*T, F, 1]

        # English note retained from the original workflow.
        q_feats = z[:, [self.query_idx], :]   # [B*T, 1, 1]
        Q = self.q_proj(q_feats)                             # [B*T, 1, d_feat]

        # English note retained from the original workflow.
        K = self.k_proj(z)                                   # [B*T, F, d_feat]
        V = self.v_proj(z)                                   # [B*T, F, d_feat]

        # English note retained from the original workflow.
        ctx, _ = self.attn(Q, K, V)                          # [B*T, 1, d_feat]
        _, attn_weights = self.attn(Q, K, V, need_weights=True, average_attn_weights=False)
        ctx = self.out(ctx)                                  # [B*T, 1, d_feat]

        ctx = ctx.reshape(B, T, -1)                          # [B, T, d_feat]
        return ctx, attn_weights


class AttentionPool(nn.Module):
    def __init__(self, d_model, device=None):
        super().__init__()
        # English note retained from the original workflow.
        self.query = nn.Parameter(torch.randn(1, 1, d_model, device=device))
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
        # English note retained from the original workflow.
        if device is not None:
            self.attn.to(device)

    def forward(self, x):  # x: [B, S=8, d_model]
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, d_model]
        # English note retained from the original workflow.
        if q.device != x.device:
            raise ValueError(f"q on {q.device}, x on {x.device}")
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)  # [B, d_model]
    
# English note retained from the original workflow.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


# English note retained from the original workflow.
# English note retained from the original workflow.

class Transformer_Feature_extract_and_Predict4(nn.Module):
    def __init__(
            self, 
            seq_len=9, 
            discretized_size=16,
            feature_dim=15, 
            d_model=128, 
            nhead=4, 
            num_layers=3, 
            dim_feedforward=256, 
            dropout=0.1, 
            device=None, 
            multi_mlp=False, 
            cross_attn_agg="attn",  # English note retained from the original workflow.
            use_cls_pool=True
        ):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.multi_mlp = multi_mlp  # English note retained from the original workflow.
        # English note retained from the original workflow.
        self.attn_pool = AttentionPool(d_model, device=device)
        self.use_cls_pool = use_cls_pool
        # English note retained from the original workflow.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),  # English note retained from the original workflow.
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4)
        )

        # English note retained from the original workflow.
        self.mlp_fv_bv = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # English note retained from the original workflow.
        )
        self.mlp_fs_bs = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # English note retained from the original workflow.
        )
        # English note retained from the original workflow.
        # English note retained from the original workflow.
        self.seq_len = seq_len
        if seq_len % 2 == 0:
            raise ValueError("seq_len must be odd to have a clear middle row")
        
        self.ipd_related_indices = list((0, -1, -2, -3, -4)) if feature_dim == 11 else list((-1, -2, -3, -4))
        self.multi_field_attn = MultiFieldCrossAttention(
            d_feat=d_model,
            nhead=nhead,
            query_indices=self.ipd_related_indices,
            dropout=dropout,
            agg=cross_attn_agg
        )
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))


    def feature_norm(self, x):
        B, S, F = x.shape
        x_out = x.clone()
        if F == 11:
            feature_max = torch.tensor([
                1.0,  # English note retained from the original workflow.
                255,        # tc
                65535,      # plen
                255,        # nh
                15,         # tcp_dataofs
                511,        # tcp_flags
                65535,      # tcp_window
                1,          # fv
                1,          # bv
                1,          # fs
                1           # bs
            ], dtype=torch.float32).to(x.device)
            ipd = x[:, :, 0:1]
            ipd_min = ipd.amin(dim=1, keepdim=True)
            ipd_max = ipd.amax(dim=1, keepdim=True)
            ipd = (ipd - ipd_min) / (ipd_max - ipd_min + 1e-6)
            x_out[:, :, 0:1] = ipd			
            # English note retained from the original workflow.
            x_out[:, :, 1:] = x[:, :, 1:] / feature_max[1:]

        if F == 10:
            feature_max = torch.tensor([
                255,        # tc
                65535,      # plen
                255,        # nh
                15,         # tcp_dataofs
                511,        # tcp_flags
                65535,      # tcp_window
                1,          # fv
                1,          # bv
                1,          # fs
                1           # bs
            ], dtype=torch.float32).to(x.device)

            x_out = x / feature_max
            return x_out
            
    def forward(self, src):
        # src: [B, 9, 15]
        src = src.float()  # English note retained from the original workflow.
        src = self.feature_norm(src)  # English note retained from the original workflow.
        if torch.isnan(src).any() or torch.isinf(src).any():
            raise ValueError("Input contains NaN or Inf")
        B = src.size(0)

        # English note retained from the original workflow.
        mid_idx = self.seq_len // 2

        src = self.embedding(src) + self.multi_field_attn(src)
        # English note retained from the original workflow.
        src = self.pos_encoder(src)  # [B, seq_len, d_model]
        src = self.ln(src)
        
        if self.use_cls_pool:
            cls = self.cls.expand(B, 1, -1)
            src = torch.cat([cls, src], dim=1)        # [B, 1+T, d_model]
            encoded = self.encoder(src)
            src_feature = encoded[:, 0]               # [B, d_model]
        else:
            encoded = self.encoder(src)
            src_feature = self.attn_pool(encoded)


        # English note retained from the original workflow.
        if self.multi_mlp is False:
            output = self.mlp(src_feature)
        else:
        # English note retained from the original workflow.
            pred_fv_bv = self.mlp_fv_bv(src_feature)  # [B, 2]
            pred_fs_bs = self.mlp_fs_bs(src_feature)  # [B, 2]
            output = torch.cat([pred_fv_bv, pred_fs_bs], dim=-1)  # [B, 4]

        # output = self.mlp(src_feature)
        # residual = self.mlp_residual(fused)
        # output = output + residual
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError("Output contains NaN or Inf")
        return output

class Transformer_packets_feature_model(nn.Module):
    def __init__(
            self, 
            discretized_size=16,
            feature_dim=15, 
            d_model=128, 
            nhead=4, 
            num_layers=3, 
            dim_feedforward=256, 
            dropout=0.1, 
            is_middle=False,
            cross_attn_agg="attn",  # English note retained from the original workflow.
            masked_idx=-1
        ):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # English note retained from the original workflow.
        self.attn_pool = AttentionPool(d_model)
        # English note retained from the original workflow.
        encoder_layer = TransformerEncoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, discretized_size)
        self.ipd_related_indices = -1
        self.single_field_attn = SingleFieldCrossAttention(
            d_feat=d_model,
            nhead=nhead,
            query_idx=self.ipd_related_indices,
            dropout=dropout
        )
        self.masked_idx = masked_idx
    def feature_norm(self, x):
        masked_idx = self.masked_idx
        # print(f"-----------masked_idx = {masked_idx}")
        B, S, F = x.shape
        x_out = x.clone()
        if F == 1: 
            feature_max = torch.tensor([
                255,        # tc
                65535,      # plen
                255,        # nh
                15,         # tcp_dataofs
                511,        # tcp_flags
                65535,      # tcp_window
                16
            ], dtype=torch.float32).to(x.device)
        elif F == 16: 
            feature_max = torch.tensor([
                255,        # tc
                1048575,    # fl
                65535,      # plen
                255,        # nh
                255,        # hl
                65535,      # tcp_sport
                65535,      # tcp_dport
                4294967295, # tcp_seq
                4294967295, # tcp_ack
                15,         # tcp_dataofs
                7,          # tcp_reserved
                511,        # tcp_flags
                65535,      # tcp_window
                65535,      # tcp_chksum
                65535,      # tcp_urgptr
                16
            ], dtype=torch.float32).to(x.device)
        elif F == 12: # ['tc','plen','nh','hlim','tcp_ack','tcp_dataofs','tcp_reserved','tcp_flags','tcp_window','tcp_chksum','tcp_urgptr']
            feature_max = torch.tensor([
                255,        # tc
                65535,      # plen
                255,        # nh
                255,        # hlim
                4294967295, # tcp_ack
                15,         # tcp_dataofs
                7,          # tcp_reserved
                511,        # tcp_flags
                65535,      # tcp_window
                65535,      # tcp_chksum
                65535,      # tcp_urgptr
                16
            ], dtype=torch.float32).to(x.device)
        elif F == 11:
            feature_max = torch.tensor([
                255,        # tc
                65535,      # plen
                255,        # nh
                255,        # hlim
                4294967295, # tcp_ack
                15,         # tcp_dataofs
                7,          # tcp_reserved
                511,        # tcp_flags
                65535,      # tcp_window
                65535,      # tcp_chksum
                65535,      # tcp_urgptr
                16
            ], dtype=torch.float32).to(x.device)
            if masked_idx != -1:  # English note retained from the original workflow.
                print(f"masked_idx = {masked_idx}")
                all_indices = torch.arange(feature_max.shape[0], device=feature_max.device)
                # English note retained from the original workflow.
                keep_indices = all_indices[all_indices != masked_idx]
                # English note retained from the original workflow.
                feature_max = feature_max[keep_indices]  # English note retained from the original workflow.
                print(feature_max.shape)
                
        elif F == 7:
            feature_max = torch.tensor([
                65535,      # plen
                255,        # nh
                255,        # hlim
                15,         # tcp_dataofs
                511,        # tcp_flags
                65535,      # tcp_window
                16
            ], dtype=torch.float32).to(x.device)
            if masked_idx != -1:  # English note retained from the original workflow.
                print(f"masked_idx = {masked_idx}")
                all_indices = torch.arange(feature_max.shape[0], device=feature_max.device)
                # English note retained from the original workflow.
                keep_indices = all_indices[all_indices != masked_idx]
                # English note retained from the original workflow.
                feature_max = feature_max[keep_indices]  # English note retained from the original workflow.
                print(feature_max.shape)

        x_out = x / feature_max
        return x_out
        
            
    def forward(self, src):
        # src: [B, 9, 15]
        src = src.float()  # English note retained from the original workflow.
        src = self.feature_norm(src)  # English note retained from the original workflow.


        if torch.isnan(src).any() or torch.isinf(src).any():
            raise ValueError("Input contains NaN or Inf")
        B = src.size(0)
        # English note retained from the original workflow.

        attn_output, attn_weights = self.single_field_attn(src)
        src = self.embedding(src)
        # English note retained from the original workflow.
        src = self.pos_encoder(src)  # [B, seq_len, d_model]
        src = self.ln(src)

        token_attn_maps = []                    # list of [B, nhead, T, T]
        h = src
        for layer in self.encoder.layers:
            h = layer(h)  # English note retained from the original workflow.
            token_attn_maps.append(layer.last_attn_weights)

        if self.encoder.norm is not None:
            h = self.encoder.norm(h)
            
        # encoded = self.encoder(src)
        encoded = h
        src_feature = self.attn_pool(encoded)
        # English note retained from the original workflow.
        # feat = src_feature[:, -1, :]
        logits = self.fc(src_feature)
        return logits, token_attn_maps, attn_weights


class Transformer_ipd_model(nn.Module):
    def __init__(self, discretized_size, feature_dim, d_model, nhead, num_layers,dim_feedforward, dropout, is_middle):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = discretized_size, embedding_dim = d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.ln = nn.LayerNorm(d_model)

        encoder_layer = TransformerEncoderLayerWithAttn(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, discretized_size)
        self.is_middle = is_middle
        # English note retained from the original workflow.
        self.attn_maps = []
    def forward(self, x, src_key_padding_mask=None, attn_mask=None):
            """
            English documentation retained from the original workflow.
            English documentation retained from the original workflow.
            """
            self.attn_maps = []
            seq_len = x.size(1)
            x = self.embedding(x)           # [B, T] -> [B, T, E]
            x = self.pos_encoder(x)  # English note retained from the original workflow.
            x = self.ln(x)  # English note retained from the original workflow.

            out = x
            for layer in self.encoder.layers:
                # English note retained from the original workflow.
                if hasattr(self.encoder, 'norm') and self.encoder.norm is not None and self.encoder.norm is not layer:
                    pass  # English note retained from the original workflow.

                out2 = layer(out, src_mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
                # English note retained from the original workflow.
                self.attn_maps.append(layer.last_attn_weights)  # [B, nhead, T, T]
                out = out2

            if self.encoder.norm is not None:
                out = self.encoder.norm(out)

            if not self.is_middle:
                feat = out[:, -1, :]  # English note retained from the original workflow.
            else:
                feat = out[:, (seq_len // 2) - 1: (seq_len // 2) + 1, :]
                feat = feat.mean(dim=1)  # English note retained from the original workflow.
            logits = self.fc(feat)
            return logits, self.attn_maps, None # [B, nhead, T, T]

            # out = self.encoder(x)          # out: [B, T, E]
            # if not self.is_middle:
            # English note retained from the original workflow.
            # else:
            # English note retained from the original workflow.
            # 	feat = feat.mean(dim=1)
            # logits = self.fc(feat)         # [B, C]
            # return logits, None


# English note retained from the original workflow.
class SeqClsDataset(Dataset):
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray):
            assert X_np.ndim == 2  # (N, seq_len)
            assert y_np.ndim == 1  # (N,)
            self.X = torch.as_tensor(X_np, dtype=torch.long)  # English note retained from the original workflow.
            self.y = torch.as_tensor(y_np, dtype=torch.long)  # English note retained from the original workflow.

    def __len__(self):
            return self.X.shape[0]

    def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

class IntervalDataset_Middle(Dataset):
    def __init__(self, data, is_middle):
        # data: [num_samples, seq_len, feature_dim]
        self.is_middle = is_middle
        self.data = data
        if len(data.shape) != 3 or data.shape[1] % 2 == 0:
            raise ValueError("Data must have shape [num_samples, seq_len, feature_dim] with odd seq_len")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.as_tensor(self.data[idx], dtype=torch.float32)  # [seq_len, feature_dim]
        mid_idx = int((sample.shape[0] - 1) // 2)  # English note retained from the original workflow.
        # print('sample.shape = ',sample.shape)
        # print('ori_sample = ',sample)
        if not self.is_middle:
            # print('ori_data = ,',sample)
            # print('ori_label = ', sample[-1, -1])
            label = sample[-1, -1].clone()  # English note retained from the original workflow.
            input_data = sample
            input_data[-1, -1] = 0
            input_data[-2, -1], _ = mask_data(input_data[-2, -1])
            # input_data = sample[:-1,:]
            # English note retained from the original workflow.
            # print('input_data = ', input_data)
            # print('label = ', label)
        else:
            label = sample[mid_idx, -1].clone()
            input_data = sample.clone()
            input_data[mid_idx, -1] = 0
            input_data[mid_idx - 1, -1], _ = mask_data(input_data[mid_idx-1, -1])
            _, input_data[mid_idx + 1, -1] = mask_data(input_data[mid_idx+1, -1])

            # input_f = sample[:mid_idx, :]
            # input_b = sample[mid_idx + 1:, :]
            # input_f[-1, -1], _ = mask_data(input_f[-1, -1])
            # _, input_b[0, -1] = mask_data(input_b[0, -1])
            # input_data = torch.cat([input_f, input_b], dim=0)

        return input_data, label

def train_val(model, train_loader, val_loader, criterion, optimizer, epochs, save_dir, device):
    best_val_loss = float('inf')
    # English note retained from the original workflow.
    val_bit_accuracies = [[] for _ in range(4)]  # English note retained from the original workflow.
    val_bit_f1_scores = [[] for _ in range(4)]  # English note retained from the original workflow.

    model = model.to(device)

    for epoch in range(epochs):
        # English note retained from the original workflow.
        all_outputs = []
        model.train()
        train_total_loss = 0
        train_total_correct = 0
        train_total_num = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
            # print(batch_x.shape)

            # if not torch.all((batch_y == 0) | (batch_y == 1)):
            #     raise ValueError("Labels contain non-binary values")
            optimizer.zero_grad()
            output, _, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item() * batch_y.size(0)
            train_total_num += batch_y.size(0)
            preds = output.argmax(dim=-1)
            train_total_correct  += (preds == batch_y).sum().item()

        # English note retained from the original workflow.
        model.eval()
        val_total_loss = 0
        val_total_correct = 0
        val_total_num = 0

        with torch.no_grad():
            fv_correct = 0
            bv_correct = 0
            fs_correct = 0
            bs_correct = 0
            # English note retained from the original workflow.
            for batch_x, batch_y in val_loader:

                batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
                output, _, _ = model(batch_x)
                loss = criterion(output, batch_y)
                val_total_loss += loss.item() * batch_y.size(0)

                val_total_num += batch_y.size(0)
                preds = output.argmax(dim=-1)
                val_total_correct  += (preds == batch_y).sum().item()

                fv_pre = (preds >> 3) & 1  # English note retained from the original workflow.
                bv_pre = (preds >> 2) & 1  # English note retained from the original workflow.
                fs_pre = (preds >> 1) & 1  # English note retained from the original workflow.
                bs_pre = preds & 1

                fv_label = (batch_y >> 3) & 1  # English note retained from the original workflow.
                bv_label = (batch_y >> 2) & 1  # English note retained from the original workflow.
                fs_label = (batch_y >> 1) & 1  # English note retained from the original workflow.
                bs_label = batch_y & 1

                fv_correct += (fv_pre == fv_label).sum().item()
                bv_correct += (bv_pre == bv_label).sum().item()
                fs_correct += (fs_pre == fs_label).sum().item()
                bs_correct += (bs_pre == bs_label).sum().item()
        # val_loss /= len(val_loader)

        print(f"Epoch_{epoch + 1}: loss = {(train_total_loss / train_total_num):.4f}, acc = {(train_total_correct / train_total_num):.4f}, fv_acc = {(fv_correct / val_total_num):.4f}, bv_acc = {(bv_correct / val_total_num):.4f}, fs_acc = {(fs_correct / val_total_num):.4f}, bs_acc = {(bs_correct / val_total_num):.4f},")

        # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
        # 	  f"Val Loss: {(val_loss / val_total):.4f}, Val Acc: {val_acc:.4f}")

        # English note retained from the original workflow.
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pth"))


def train_val_ipd_transformer(model, train_loader, val_loader, criterion,  optimizer, epochs, save_dir, device):
    # model = model.to(device)
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    best_loss = float("inf")
    for epoch in range(epochs):
        train_loss, train_correct, train_count = 0.0, 0, 0
        fv_correct = 0
        bv_correct = 0
        fs_correct = 0
        bs_correct = 0
        total_num = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_y.size(0)
            total_num += batch_y.size(0)
            preds = output.argmax(dim=-1)
            train_correct  += (preds == batch_y).sum().item()
            train_count += batch_y.size(0)

            fv_pre = (preds >> 3) & 1  # English note retained from the original workflow.
            bv_pre = (preds >> 2) & 1  # English note retained from the original workflow.
            fs_pre = (preds >> 1) & 1  # English note retained from the original workflow.
            bs_pre = preds & 1

            fv_label = (batch_y >> 3) & 1  # English note retained from the original workflow.
            bv_label = (batch_y >> 2) & 1  # English note retained from the original workflow.
            fs_label = (batch_y >> 1) & 1  # English note retained from the original workflow.
            bs_label = batch_y & 1

            fv_correct += (fv_pre == fv_label).sum().item()
            bv_correct += (bv_pre == bv_label).sum().item()
            fs_correct += (fs_pre == fs_label).sum().item()
            bs_correct += (bs_pre == bs_label).sum().item()
        print(f"Epoch_{epoch + 1}: loss = {(train_loss / total_num):.4f}, acc = {(train_correct / train_count):.4f}, fv_acc = {(fv_correct / train_count):.4f}, bv_acc = {(bv_correct / train_count):.4f}, fs_acc = {(fs_correct / train_count):.4f}, bs_acc = {(bs_correct / train_count):.4f},")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pth"))


class FlowCSVDataset(Dataset):
    def __init__(
        self,
        csv_dir,
        float_fields,
        seq_len,
        median,
        num_samples=None,
        is_middle=False
    ):
        self.csv_files = _collect_csv_files(csv_dir)
        self.float_fields = float_fields
        self.seq_len = seq_len + 1  # English note retained from the original workflow.
        self.median = median
        self.is_middle = is_middle

        self.feature_dim = len(float_fields) + 1

        if self.seq_len % 2 == 0:
            raise ValueError("seq_len must be odd (after +1), consistent with IntervalDataset_Middle")

        # index_map[k] = (csv_file, start_row_idx)
        self.index_map = []

        total = 0
        for csv_file in tqdm(self.csv_files):
            n_flows = flow_csv_count(
                csv_file,
                self.seq_len,
                self.float_fields,
                self.median
            )

            for start_idx in range(n_flows):
                self.index_map.append((csv_file, start_idx))
                total += 1
                if num_samples and total >= num_samples:
                    break

            if num_samples and total >= num_samples:
                break

        if len(self.index_map) == 0:
            raise RuntimeError(f"No valid samples found in {csv_dir}")

        print(
            f"[FlowCSVDataset] Indexed {len(self.index_map)} sliding-window samples "
            f"from {len(self.csv_files)} CSV files"
        )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        csv_file, start_idx = self.index_map[idx]

        # English note retained from the original workflow.
        data = flow_csv_to_np_data(
            csv_file,
            self.seq_len,
            self.float_fields,
            self.median,
            num_samples=start_idx + self.seq_len
        )

        sample = data[start_idx]                  # [seq_len, feature_dim]
        sample = torch.as_tensor(sample, dtype=torch.float32)

        # English note retained from the original workflow.
        mid_idx = (sample.shape[0] - 1) // 2

        if not self.is_middle:
            label = sample[-1, -1].clone()
            input_data = sample.clone()
            input_data[-1, -1] = 0
            input_data[-2, -1], _ = mask_data(input_data[-2, -1])
        else:
            label = sample[mid_idx, -1].clone()
            input_data = sample.clone()
            input_data[mid_idx, -1] = 0
            input_data[mid_idx - 1, -1], _ = mask_data(input_data[mid_idx - 1, -1])
            _, input_data[mid_idx + 1, -1] = mask_data(input_data[mid_idx + 1, -1])

        return input_data, label



def get_dataloader(mode, csv_dir, feature_dim, seq_len, median, num_samples, batch_size, is_middle, masked_idx=-1):
    if mode == 'Transformer_packets_feature_model':
#        print(f"feature_dim = {feature_dim}")
        if feature_dim == 1:
            float_fields = ['tc', 'plen', 'nh', 'tcp_dataofs', 'tcp_flags', 'tcp_window']
        elif feature_dim == 7:
            float_fields = ['plen', 'nh', 'hlim', 'tcp_dataofs', 'tcp_flags', 'tcp_window']
        elif feature_dim == 16:
            float_fields = ['tc','fl','plen','nh','hlim','tcp_sport','tcp_dport','tcp_seq','tcp_ack','tcp_dataofs','tcp_reserved','tcp_flags','tcp_window','tcp_chksum','tcp_urgptr']
        # elif feature_dim == 10:
        # 	float_fields = ['plen','nh','hlim','tcp_dataofs','tcp_reserved','tcp_flags','tcp_window','tcp_chksum','tcp_urgptr']
        elif feature_dim == 12:
            float_fields = ['tc','plen','nh','hlim','tcp_ack','tcp_dataofs','tcp_reserved','tcp_flags','tcp_window','tcp_chksum','tcp_urgptr']
        elif feature_dim == 11:
            float_fields = ['tc','plen','nh','hlim','tcp_ack','tcp_dataofs','tcp_reserved','tcp_flags','tcp_window','tcp_chksum','tcp_urgptr']
            if masked_idx != -1:
                masek_field = float_fields[masked_idx]
                float_fields.remove(masek_field)
        data_npy = load_multi_features_from_csv(csv_dir, float_fields, seq_len, median, num_samples)
        dataset = IntervalDataset_Middle(data_npy,is_middle)
        # dataset = FlowCSVDataset(csv_dir=csv_dir,float_fields=float_fields,seq_len=seq_len,median=median,num_samples=num_samples,is_middle=is_middle)


        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        csv_files = _collect_csv_files(csv_dir)
        dataset_lists = []
        
        for csv_file in csv_files:
            ipd_list = ipd_csv_to_ipd_list(csv_file)
            discretization_list = gas_ipd_discretization(ipd_list, median=median)
            dataset_X, dataset_Y = construct_single_ipd_prediction_dataset(discretization_list, raw_len=0, time_step_x=1, time_step_y=seq_len, if_unique=False, is_middle=is_middle)
            train_dataset = SeqClsDataset(dataset_X, dataset_Y)
            dataset_lists.append(train_dataset)
        dataset = ConcatDataset(dataset_lists)
        if num_samples != 0 and num_samples is not None:
            print(num_samples)
            dataset = Subset(dataset, range(num_samples))
        data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader
    