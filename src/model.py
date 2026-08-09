from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(requested: str = "auto"):
    if requested != "auto":
        return torch.device(requested)
    # For this small tabular dataset CPU avoids MPS kernel/transfer overhead and
    # was the stable reproducible choice on the detected M1 Pro.
    return torch.device("cpu")


@dataclass
class MLPConfig:
    hidden_layers: tuple[int, ...] = (256, 128, 64)
    activation: str = "relu"
    dropout: float = 0.1
    normalization: str = "none"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    optimizer: str = "adamw"
    max_epochs: int = 500
    patience: int = 45
    scheduler_patience: int = 12
    gradient_clip: float = 5.0
    seed: int = 42


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, config: MLPConfig):
        super().__init__()
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": lambda: nn.LeakyReLU(0.1),
            "silu": nn.SiLU,
        }
        if config.activation not in activations:
            raise ValueError(config.activation)
        layers: list[nn.Module] = []
        previous = input_dim
        for width in config.hidden_layers:
            layers.append(nn.Linear(previous, width))
            if config.normalization == "batch":
                layers.append(nn.BatchNorm1d(width))
            elif config.normalization == "layer":
                layers.append(nn.LayerNorm(width))
            layers.append(activations[config.activation]())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def predict_scaled(model, X, device="cpu", batch_size=512):
    device = torch.device(device)
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            output.append(model(batch).detach().cpu().numpy())
    return np.concatenate(output)


def train_mlp(X_train, y_train_scaled, X_val, y_val_original, target_transformer,
              config: MLPConfig, device="auto"):
    set_seed(config.seed)
    device = resolve_device(device)
    model = MLPRegressor(X_train.shape[1], config).to(device)
    optimizer_cls = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=config.learning_rate,
                              weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config.scheduler_patience, min_lr=1e-6)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                      torch.as_tensor(y_train_scaled, dtype=torch.float32)),
        batch_size=min(config.batch_size, len(X_train)), shuffle=True,
        generator=generator,
        drop_last=(config.normalization == "batch" and len(X_train) % min(config.batch_size, len(X_train)) == 1),
    )
    best_rmse = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if config.gradient_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_scaled = predict_scaled(model, X_train, device)
        val_scaled = predict_scaled(model, X_val, device)
        train_pred = target_transformer.inverse_transform(train_scaled)
        val_pred = target_transformer.inverse_transform(val_scaled)
        train_true = target_transformer.inverse_transform(y_train_scaled)
        train_rmse = float(np.sqrt(np.mean((train_true - train_pred) ** 2)))
        val_rmse = float(np.sqrt(np.mean((np.asarray(y_val_original) - val_pred) ** 2)))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        scheduler.step(val_rmse)
        if val_rmse < best_rmse - 1.0:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, history, best_rmse


def train_mlp_fixed(X_train, y_train_scaled, config: MLPConfig, epochs: int, device="auto"):
    """Retrain on all available rows for a CV-derived number of epochs."""
    set_seed(config.seed)
    device = resolve_device(device)
    model = MLPRegressor(X_train.shape[1], config).to(device)
    optimizer_cls = torch.optim.AdamW if config.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=config.learning_rate,
                              weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(config.seed)
    batch_size = min(config.batch_size, len(X_train))
    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                      torch.as_tensor(y_train_scaled, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, generator=generator,
        drop_last=(config.normalization == "batch" and len(X_train) % batch_size == 1),
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if config.gradient_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    return model, history


def save_checkpoint(model, config: MLPConfig, input_dim: int, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "config": {**asdict(config), "hidden_layers": list(config.hidden_layers)},
    }, path)


def load_checkpoint(path: Path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    cfg = payload["config"]
    cfg["hidden_layers"] = tuple(cfg["hidden_layers"])
    config = MLPConfig(**cfg)
    model = MLPRegressor(payload["input_dim"], config)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, config, payload["input_dim"]
