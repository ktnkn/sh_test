import torch
import torch.nn as nn
import torch.nn.functional as F
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_attention_head, n_kv_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_attention_head = n_attention_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_attention_head
        self.num_rep = n_attention_head // n_kv_head
        
        assert d_model % n_attention_head == 0
        assert n_attention_head % n_kv_head == 0

        self.q_proj = nn.Linear(d_model, n_attention_head * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout_p = dropout

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_attention_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Repeat k, v
        if self.num_rep > 1:
            k = k.repeat_interleave(self.num_rep, dim=1)
            v = v.repeat_interleave(self.num_rep, dim=1)
            
        # Flash Attention (Scaled Dot Product Attention)
        # PyTorch 2.0+ automatically uses Flash Attention if available and conditions are met
        # Ensure inputs are contiguous for optimal performance with Flash Attention kernels
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(output)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

class MoELayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_experts, num_experts_per_tok, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        # Router logits: (batch * seq_len, num_experts)
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=1)
        
        # Select top-k experts
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        # Normalize weights
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
        
        # Calculate auxiliary loss (load balancing)
        if self.training:
            # P_i: fraction of weight given to expert i (mean of routing_weights[:, i])
            density_1_proxy = routing_weights.mean(dim=0)
            
            # f_i: fraction of tokens routed to expert i
            mask = F.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=1) # (T, num_experts)
            density_1 = mask.float().mean(dim=0)
            
            aux_loss = (density_1_proxy * density_1).sum() * self.num_experts
        else:
            aux_loss = 0.0

        # Compute output
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            # Find which tokens selected expert i
            idx = (selected_experts == i).nonzero(as_tuple=True) # (row_indices, col_indices_in_topk)
            
            if len(idx[0]) == 0:
                continue
                
            batch_idx = idx[0]
            
            # Process tokens
            expert_input = x_flat[batch_idx]
            expert_output = self.experts[i](expert_input)
            
            # Weight by the routing weight
            weight = routing_weights_topk[idx] # (num_selected_tokens,)
            
            # Add to final output
            final_output.index_add_(0, batch_idx, weight.unsqueeze(1) * expert_output)
            
        return final_output.view(batch_size, seq_len, d_model), aux_loss

class TransformerEncoderLayerGQA(nn.Module):
    def __init__(self, d_model, n_attention_head, n_kv_head, dim_feedforward, dropout=0.1, num_experts=1, num_experts_per_tok=1):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, n_attention_head, n_kv_head, dropout)
        
        self.num_experts = num_experts
        if num_experts > 1:
            self.moe = MoELayer(d_model, dim_feedforward, num_experts, num_experts_per_tok, dropout)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.activation = nn.GELU()
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        aux_loss = 0.0
        if self.num_experts > 1:
            src2, aux_loss = self.moe(src)
        else:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, aux_loss

class TransformerPolicyValueNetwork(nn.Module):
    def __init__(self, d_model=256, n_attention_head=8, n_kv_head=2, num_layers=8, dim_feedforward=512, dropout=0.1, mtp_heads=0, num_experts=1, num_experts_per_tok=1):
        super(TransformerPolicyValueNetwork, self).__init__()
        
        self.d_model = d_model
        self.mtp_heads = mtp_heads
        self.num_experts = num_experts

        # Embedding for board features (x1)
        self.embedding1 = nn.Linear(FEATURES1_NUM, d_model)
        
        # Embedding for global/hand features (x2)
        self.embedding2 = nn.Linear(FEATURES2_NUM, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 81, d_model))
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayerGQA(d_model, n_attention_head, n_kv_head, dim_feedforward, dropout, num_experts, num_experts_per_tok)
            for _ in range(num_layers)
        ])
        
        # Policy head (Main)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, MAX_MOVE_LABEL_NUM),
            nn.Flatten(start_dim=1) 
        )
        
        # MTP Policy heads
        self.mtp_policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, MAX_MOVE_LABEL_NUM),
                nn.Flatten(start_dim=1) 
            ) for _ in range(mtp_heads)
        ])
        
        # MTP Value heads
        self.mtp_value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 81, 256),
                nn.GELU(),
                nn.Linear(256, 1)
            ) for _ in range(mtp_heads)
        ])
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 81, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, x1, x2):
        b, c1, h, w = x1.shape
        c2 = x2.shape[1]
        
        x1_flat = x1.view(b, c1, -1).permute(0, 2, 1) 
        x2_flat = x2.view(b, c2, -1).permute(0, 2, 1)
        
        x = self.embedding1(x1_flat) + self.embedding2(x2_flat)
        x = x + self.pos_encoder
        
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
        
        y1 = self.policy_head(x)
        y2 = self.value_head(x.view(b, -1))
        
        if self.mtp_heads > 0:
            mtp_policy_logits = [head(x) for head in self.mtp_policy_heads]
            mtp_value_logits = [head(x.view(b, -1)) for head in self.mtp_value_heads]
            return y1, y2, mtp_policy_logits, mtp_value_logits, total_aux_loss
        
        return y1, y2, total_aux_loss
