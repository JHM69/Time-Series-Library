import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Autoformer_EncDec import series_decomp
import numpy as np

class GroupPolicyLayer(nn.Module):
    """
    Group Relative Policy Optimization layer for enhanced time series modeling
    """
    def __init__(self, configs):
        super(GroupPolicyLayer, self).__init__()
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads if hasattr(configs, 'n_heads') else 8
        self.dropout = configs.dropout
        self.num_groups = configs.num_groups if hasattr(configs, 'num_groups') else 4
        self.group_size = configs.group_size if hasattr(configs, 'group_size') else configs.seq_len // self.num_groups
        self.policy_clip_range = configs.policy_clip_range if hasattr(configs, 'policy_clip_range') else 0.2
        
        # Multi-head attention for policy learning
        self.policy_attention = nn.MultiheadAttention(
            self.d_model, 
            self.n_heads,
            dropout=self.dropout
        )
        
        # Policy value estimation
        self.value_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, 1)
        )
        
        # Policy advantage estimation
        self.advantage_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        # Cross-group feature fusion
        self.group_fusion = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model)
        )

    def compute_policy_loss(self, new_policy, old_policy, advantages):
        ratio = torch.exp(new_policy - old_policy.detach())
        clipped_ratio = torch.clamp(ratio, 1 - self.policy_clip_range, 1 + self.policy_clip_range)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        return loss

    def forward(self, x, layer_past=None):
        B, L, D = x.shape
        
        # Split sequence into groups
        groups = x.view(B, self.num_groups, -1, D)
        
        # Initialize group policies
        group_policies = []
        group_values = []
        
        for i in range(self.num_groups):
            # Get group features
            group_x = groups[:, i]
            
            # Self-attention within group
            group_out = group_x.transpose(0, 1)
            group_attn_out, _ = self.policy_attention(
                group_out, group_out, group_out
            )
            group_out = group_attn_out.transpose(0, 1)
            
            # Compute value and advantage
            value = self.value_net(group_out)
            advantage = self.advantage_net(group_out)
            
            # Store group outputs
            group_policies.append(group_out)
            group_values.append(value)
        
        # Stack group outputs
        group_policies = torch.stack(group_policies, dim=1)
        group_values = torch.stack(group_values, dim=1)
        
        # Cross-group feature fusion
        fused_features = self.group_fusion(
            group_policies.view(B, self.num_groups, -1, D)
        )
        
        # Combine with original input using residual connection
        output = fused_features.view(B, L, D) + x
        
        return output, group_values

class TimeGRPO(nn.Module):
    def __init__(self, configs):
        super(TimeGRPO, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        
        # Data decomposition
        self.decomp = series_decomp(configs.moving_avg if hasattr(configs, 'moving_avg') else 25)
        
        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq, 
            configs.dropout
        )
        
        # GRPO layers
        self.grpo_layers = nn.ModuleList([
            GroupPolicyLayer(configs) for _ in range(configs.e_layers)
        ])
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, 
                    configs.n_heads
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ])
        
        # Projection layers
        self.trend_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.seasonal_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
        # Adjusted fusion layer to handle correct dimensions
        self.fusion = nn.Sequential(
            nn.Linear(configs.c_out * 2, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.c_out)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Decompose input into trend and seasonal components
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # Embed inputs
        trend_enc = self.enc_embedding(trend_init, x_mark_enc)
        seasonal_enc = self.enc_embedding(seasonal_init, x_mark_enc)
        
        # Initialize attention caches
        trend_attns = []
        seasonal_attns = []
        
        # Process through GRPO and transformer layers
        for grpo_layer, encoder_layer in zip(self.grpo_layers, self.encoder_layers):
            # GRPO processing
            trend_enc, trend_values = grpo_layer(trend_enc)
            seasonal_enc, seasonal_values = grpo_layer(seasonal_enc)
            
            # Transformer processing
            trend_enc, trend_attn = encoder_layer(trend_enc)
            seasonal_enc, seasonal_attn = encoder_layer(seasonal_enc)
            
            trend_attns.append(trend_attn)
            seasonal_attns.append(seasonal_attn)
        
        # Project components
        trend_out = self.trend_projection(trend_enc)
        seasonal_out = self.seasonal_projection(seasonal_enc)
        
        # Combine trend and seasonal projections with proper reshaping
        combined_features = torch.cat([trend_out, seasonal_out], dim=-1)
        dec_out = self.fusion(combined_features)
        
        return dec_out[:, -self.pred_len:, :]  # Return predictions for pred_len

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.model = TimeGRPO(configs)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        else:
            raise NotImplementedError("Currently only forecasting tasks are supported")