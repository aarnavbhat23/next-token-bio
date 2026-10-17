"""Two parameter-matched neural next-token predictors: a GPT and an LSTM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int
    max_len: int = 1024
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 6
    lstm_hidden: int = 384
    lstm_layers: int = 2
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.mlp = nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(), nn.Linear(4 * c.d_model, c.d_model))
        self.n_head = c.n_head
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


class LSTMLM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, 128)
        self.rnn = nn.LSTM(128, c.lstm_hidden, num_layers=c.lstm_layers, batch_first=True, dropout=c.dropout)
        self.out = nn.Sequential(nn.Linear(c.lstm_hidden, 128), nn.Linear(128, c.vocab))

    def forward(self, idx):
        h, _ = self.rnn(self.tok(idx))
        return self.out(h)


def build(name: str, c: Config) -> nn.Module:
    return {"gpt": GPT, "lstm": LSTMLM}[name](c)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
