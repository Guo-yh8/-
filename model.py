import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 搭建transformer模型
class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // num_head
        self.fc_Q = nn.Linear(dim_model, dim_model)
        self.fc_K = nn.Linear(dim_model, dim_model)
        self.fc_V = nn.Linear(dim_model, dim_model)
        self.fc = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.fc_Q(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1, 2)
        K = self.fc_K(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1, 2)
        V = self.fc_V(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_head ** 0.5)  

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()  
        scores = scores.masked_fill(~mask, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)  
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        pe = torch.zeros(pad_size, embed)
        position = torch.arange(0, pad_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed, 2).float() * -(np.log(10000.0) / embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0).to(x.device)
        return self.dropout(x)


class LanguageTransformer(nn.Module):
    def __init__(self, embed, n_vocab, num_layer, dim_model, num_head, hidden, pad_size, dropout, device):
        super(LanguageTransformer, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed, padding_idx=n_vocab - 1)
        self.position_embedding = Positional_Encoding(embed, pad_size, dropout, device)
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(dim_model, num_head, hidden, dropout) for _ in range(num_layer)
        ])
        self.fc_out = nn.Linear(dim_model, n_vocab)

    def forward(self, x):
        out = self.embedding(x)  
        out = self.position_embedding(out)
        for layer in self.decoder_layers:
            out = layer(out)
        logits = self.fc_out(out) 
        return logits
