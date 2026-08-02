"""Compact unidirectional LSTM and deterministic training utilities."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from xjtu_sy_tcc.rul.metrics import regression_metrics


class RULLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0,
        context_size: int = 0,
    ):
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("Positive model dimensions required")
        if dropout and num_layers == 1:
            raise ValueError("Recurrent dropout is inactive for one layer")
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + context_size, hidden_size // 2 or 1),
            nn.ReLU(),
            nn.Linear(hidden_size // 2 or 1, 1),
        )
        self.context_size = context_size

    def forward(self, x, context=None):
        _, (hidden, _) = self.lstm(x)
        representation = hidden[-1]
        if self.context_size:
            if context is None or context.shape[-1] != self.context_size:
                raise ValueError("Context shape mismatch")
            representation = torch.cat((representation, context), dim=1)
        return self.head(representation).squeeze(1)


@dataclass(slots=True)
class TrainingResult:
    model: RULLSTM
    history: pd.DataFrame
    best_epoch: int
    validation_mae: float
    validation_rmse: float
    training_seconds: float
    parameter_count: int


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def predict(model, x, context, batch_size, device):
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            a = torch.as_tensor(x[start : start + batch_size], device=device)
            c = (
                torch.as_tensor(context[start : start + batch_size], device=device)
                if model.context_size
                else None
            )
            results.append(model(a, c).cpu().numpy())
    return np.concatenate(results)


def train_model(
    model,
    train_x,
    train_context,
    train_y,
    weights,
    val_x,
    val_context,
    val_y,
    val_bearings,
    preprocessor,
    config,
    learning_rate,
    device,
):
    set_deterministic(config.random_seed)
    model.to(device)
    started = time.perf_counter()
    dataset = TensorDataset(
        *(torch.as_tensor(x) for x in (train_x, train_context, train_y, weights))
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.random_seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.SmoothL1Loss(beta=config.huber_beta, reduction="none")
    best = None
    best_mae = float("inf")
    stale = 0
    rows = []
    for epoch in range(1, config.max_epochs + 1):
        tick = time.perf_counter()
        model.train()
        total = 0.0
        weight_total = 0.0
        for bx, bc, by, bw in loader:
            bx, bc, by, bw = (v.to(device) for v in (bx, bc, by, bw))
            optimizer.zero_grad()
            output = model(bx, bc if model.context_size else None)
            loss = (loss_fn(output, by) * bw).sum() / bw.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += float((loss_fn(output.detach(), by) * bw).sum())
            weight_total += float(bw.sum())
        raw = preprocessor.inverse_target(
            predict(model, val_x, val_context, config.batch_size, device)
        )
        per = []
        for bearing in sorted(set(val_bearings)):
            mask = np.asarray(val_bearings) == bearing
            per.append(regression_metrics(val_y[mask], np.maximum(raw[mask], 0)))
        mae = float(np.mean([x["mae"] for x in per]))
        rmse = float(np.mean([x["rmse"] for x in per]))
        improved = mae < best_mae - config.minimum_delta
        if improved:
            best_mae = mae
            best = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_rmse = rmse
            stale = 0
        else:
            stale += 1
        rows.append(
            {
                "epoch": epoch,
                "weighted_training_loss": total / weight_total,
                "validation_macro_mae": mae,
                "validation_macro_rmse": rmse,
                "learning_rate": learning_rate,
                "epoch_seconds": time.perf_counter() - tick,
                "best_epoch_flag": improved,
            }
        )
        if stale >= config.patience:
            break
    model.load_state_dict(best)
    return TrainingResult(
        model,
        pd.DataFrame(rows),
        best_epoch,
        best_mae,
        best_rmse,
        time.perf_counter() - started,
        sum(p.numel() for p in model.parameters()),
    )
