import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM



class LinearPolicyHead(nn.Module):
    def __init__(self, d_model, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model, bias=False)
        self.bn = nn.BatchNorm1d(d_model)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_model, out_channels)

    def forward(self, x):
        # x: (B, 81, d_model)
        x = self.linear1(x)
        
        # BatchNorm applied to (N, C)
        b, s, d = x.shape
        x = x.view(b * s, d)
        x = self.bn(x)
        x = x.view(b, s, d)
        
        x = self.gelu(x)
        x = self.linear2(x) # (B, 81, out_channels)
        
        # (B, 81, out) -> (B, out, 81) to match dlshogi label order (Direction-major)
        x = x.transpose(1, 2)
        
        return x.flatten(start_dim=1)


class BoardEmbedding(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

    def forward(self, x):
        # x: (B, C, 9, 9)
        out = self.conv(x)
        b, c, h, w = out.shape
        # (B, d_model, 9, 9) -> (B, 81, d_model)
        return out.view(b, c, -1).permute(0, 2, 1)










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
    def __init__(self, d_model, dim_feedforward, num_experts, num_experts_per_tok, dropout=0.1, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.capacity_factor = capacity_factor
        
        # Experts weights: (num_experts, in_features, out_features)
        # w1: d_model -> dim_feedforward
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, dim_feedforward))
        # w2: dim_feedforward -> d_model
        self.w2 = nn.Parameter(torch.empty(num_experts, dim_feedforward, d_model))
        
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Bias for load balancing (DeepSeek V3 style)
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))
        self.bias_update_rate = 0.0001 

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights similar to nn.Linear
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.reshape(-1, d_model)
        
        # Router logits: (batch * seq_len, num_experts)
        router_logits = self.gate(x_flat)
        
        # DeepSeek V3 uses Sigmoid
        routing_probs = router_logits.sigmoid()
        
        # Add bias for selection
        routing_probs_for_choice = routing_probs
        if self.training:
             routing_probs_for_choice = routing_probs + self.e_score_correction_bias
        
        # Select top-k experts
        _, selected_experts = torch.topk(routing_probs_for_choice, self.num_experts_per_tok, dim=-1)
        
        # Get weights for selected experts (using original probs)
        routing_weights_topk = routing_probs.gather(1, selected_experts)
        
        # Normalize weights
        routing_weights_topk = routing_weights_topk / (routing_weights_topk.sum(dim=-1, keepdim=True) + 1e-6)
        
        if self.training:
            # Update bias based on usage (DeepSeek style balancing)
            with torch.no_grad():
                mask = F.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=1) # (T, num_experts)
                usage = mask.float().mean(dim=0) # (num_experts,)
                target_usage = self.num_experts_per_tok / self.num_experts
                
                # If usage < target, increase bias. If usage > target, decrease bias.
                error = target_usage - usage
                self.e_score_correction_bias += self.bias_update_rate * error

        # --- Capacity limited routing (Fixed Capacity) ---
        # Solving the torch.compile error by making tensor shapes data-independent
        
        # Calculate capacity per expert
        # capacity = ceil(num_tokens * k * factor / E)
        capacity = int(self.capacity_factor * num_tokens * self.num_experts_per_tok / self.num_experts)
        
        # Flatten selected experts: (B*T, k) -> (B*T*k)
        flat_selected_experts = selected_experts.view(-1)
        
        # Calculate rank of each token within its expert
        expert_mask = F.one_hot(flat_selected_experts, num_classes=self.num_experts) # (Total, E)
        
        # Cumsum to get rank.
        # (Total, E)
        token_priority = torch.cumsum(expert_mask, dim=0) 
        
        # Get the rank for the assigned expert
        # token_priority is (Total, E), we pick the one matching flat_selected_experts
        ranks = token_priority.gather(1, flat_selected_experts.unsqueeze(1)).squeeze(1) - 1
        
        # Filter tokens exceeding capacity
        valid_mask = ranks < capacity
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        
        # Calculate scatter destination indices: expert_id * capacity + rank
        valid_experts = flat_selected_experts[valid_indices]
        valid_ranks = ranks[valid_indices]
        scatter_indices = valid_experts * capacity + valid_ranks
        
        # Repeat x for each k: (B*T, D) -> (B*T, k, D) -> (B*T*k, D)
        x_repeated = x_flat.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, d_model)
        
        # Create padded buffer: (E * C, D) using FIXED capacity
        # This shape depends only on batch size (num_tokens), not data content
        padded_x = torch.zeros(self.num_experts * capacity, d_model, device=x.device, dtype=x.dtype)
        
        # Scatter inputs
        padded_x.index_copy_(0, scatter_indices, x_repeated[valid_indices])
        
        # Reshape for computation: (E, C, D)
        padded_x = padded_x.view(self.num_experts, capacity, d_model)
        
        # BMM 1: (E, C, D) @ (E, D, H) -> (E, C, H)
        h = torch.bmm(padded_x, self.w1)
        h = F.gelu(h)
        h = self.dropout(h)
        
        # BMM 2: (E, C, H) @ (E, H, D) -> (E, C, D)
        out_padded = torch.bmm(h, self.w2)
        
        # Gather outputs back
        # Reshape back to (E*C, D)
        out_padded_flat = out_padded.view(-1, d_model)
        
        # Output container (same shape as x_repeated)
        out_repeated = torch.zeros_like(x_repeated)
        
        # Gather: we pull from padded buffer back to valid positions
        out_repeated.index_copy_(0, valid_indices, out_padded_flat[scatter_indices])
        
        # Weight outputs and sum over k
        # routing_weights_topk: (B*T, k) -> flatten -> (B*T*k)
        weights_flat = routing_weights_topk.flatten()
        out_weighted = out_repeated * weights_flat.unsqueeze(1)
        
        # Reshape to (B*T, k, D) and sum
        out_weighted = out_weighted.view(-1, self.num_experts_per_tok, d_model)
        final_output = out_weighted.sum(dim=1)
            
        return final_output.view(batch_size, seq_len, d_model)

class TransformerEncoderLayerGQA(nn.Module):
    def __init__(self, d_model, n_attention_head, n_kv_head, dim_feedforward, dropout=0.1, num_experts=1, num_experts_per_tok=1, capacity_factor=1.25):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, n_attention_head, n_kv_head, dropout)
        
        self.num_experts = num_experts
        if num_experts > 1:
            self.moe = MoELayer(d_model, dim_feedforward, num_experts, num_experts_per_tok, dropout, capacity_factor)
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
        
        if self.num_experts > 1:
            src2 = self.moe(src)
        else:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src




class TransformerPolicyValueNetwork(nn.Module):
    def __init__(self, d_model=256, n_attention_head=8, n_kv_head=2, num_layers=4, dim_feedforward=512, dropout=0.1, mtp_heads=0, num_experts=1, num_experts_per_tok=1, capacity_factor=1.25):
        super(TransformerPolicyValueNetwork, self).__init__()
        
        self.d_model = d_model
        self.mtp_heads = mtp_heads
        self.num_experts = num_experts

        # Embedding for board features (x1)
        #self.embedding1 = BoardEmbedding(FEATURES1_NUM, d_model)
        self.embedding1 = nn.Linear(FEATURES1_NUM, d_model) 
        
        # Embedding for global/hand features (x2)
        self.embedding2 = nn.Linear(FEATURES2_NUM, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 81, d_model))
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayerGQA(d_model, n_attention_head, n_kv_head, dim_feedforward, dropout, num_experts, num_experts_per_tok, capacity_factor)
            for _ in range(num_layers)
        ])
        
        # Policy head (Main)
        self.policy_head = LinearPolicyHead(d_model, MAX_MOVE_LABEL_NUM)
        
        # MTP Policy heads
        self.mtp_policy_heads = nn.ModuleList([
            LinearPolicyHead(d_model, MAX_MOVE_LABEL_NUM)
            for _ in range(mtp_heads)
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
            nn.Linear(d_model, 1024),
            nn.GELU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x1, x2):
        b, c1, h, w = x1.shape
        c2 = x2.shape[1]
        
        # x1: (B, C1, 9, 9) -> embedding1 -> (B, 81, d_model)
        x = self.embedding1(x1)
        
        x2_flat = x2.view(b, c2, -1).permute(0, 2, 1)
        
        x = x + self.embedding2(x2_flat)
        x = x + self.pos_encoder
        
        for layer in self.layers:
            x = layer(x)
        
        y1 = self.policy_head(x)
        
        # Value head: use mean pooling over sequence dimension
        # (B, 81, d_model) -> (B, d_model)
        y2 = self.value_head(x.mean(dim=1))
        
        if self.mtp_heads > 0:
            mtp_policy_logits = [head(x) for head in self.mtp_policy_heads]
            mtp_value_logits = [head(x.view(b, -1)) for head in self.mtp_value_heads]
            return y1, y2, mtp_policy_logits, mtp_value_logits
        
        return y1, y2
