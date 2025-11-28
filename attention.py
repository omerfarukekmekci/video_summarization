import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, encoding_for_q, encoding_for_k, encoding_for_v, mask=None):
        q = self.W_q(encoding_for_q)
        k = self.W_k(encoding_for_k)
        v = self.W_v(encoding_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / (k.size(-1) ** 0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask == 0, float("-inf"))

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=8, num_heads=2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # shared projections (one matrix produces all three of q, k, v)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)

        # final projection after concatenation of heads
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)

        qkv = self.W_qkv(x)  # (batch, seq_len, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape from (batch, seq_len, d_model) to (batch, num_heads, seq_len, d_head)
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.d_head).transpose(
            1, 2
        )
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.d_head).transpose(
            1, 2
        )
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.d_head).transpose(
            1, 2
        )

        # attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)

        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=scores.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (b, 1, seq_len, seq_len?) allow broadcast
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)

        # merge heads again
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.d_model)

        return self.out_proj(out)
