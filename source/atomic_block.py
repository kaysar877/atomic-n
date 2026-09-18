"""Atomic-N feedforward cell (challenger for the arena test).

Implements the strongest variant from the design notes:
    N = [feature | context]  ->  split  ->  A' = act(W_f x_f), B' = act(W_c x_c)
    I = A' (*) B'  (Hadamard interaction)
    N_out = act_out( W_o [A', B', I] ) + b      # collapse to one N

Drop-in replacement for the MLP inside CausalBlock. Inner dim defaults to
k = 2d so total params ~= the baseline d -> 4d -> d MLP (8d^2), keeping the
fight at equal budget. No training logic here - wiring happens later.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AtomicFFN(nn.Module):
    def __init__(self, d_model: int, inner: int | None = None,
                 act=torch.nn.GELU, out_act: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        k = inner or 2 * d_model
        assert d_model % 2 == 0, "d_model must split evenly into feature|context"
        h = d_model // 2
        self.w_f = nn.Linear(h, k)
        self.w_c = nn.Linear(h, k)
        self.w_o = nn.Linear(3 * k, d_model)
        self.act = act()
        self.out_act = act() if out_act else None
        # same output regularization as the baseline MLP (no params added)
        self.drop = nn.Dropout(dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        xf, xc = x.chunk(2, dim=-1)
        a = self.act(self.w_f(xf))
        b = self.act(self.w_c(xc))
        i = a * b
        y = self.w_o(torch.cat([a, b, i], dim=-1))
        if self.out_act is not None:
            y = self.out_act(y)
        return self.drop(y)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def baseline_mlp_params(d_model: int, mult: int = 4) -> int:
    return d_model * (mult * d_model) + (mult * d_model) + \
        (mult * d_model) * d_model + d_model
