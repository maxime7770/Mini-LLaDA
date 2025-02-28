import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)  # apply dropout

class MiniLLaDA(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, max_len=500):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src):
        # src shape: (batch_size, seq_len)
        x = self.embedding(src)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        # Transformer expects shape: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        logits = self.fc_out(x)  # (batch_size, seq_len, vocab_size)
        return logits