from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .interface import ModelHandle


def auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TabularMLPHandle(ModelHandle):
    name = "mlp"

    def __init__(
        self,
        hidden: list[int] | None = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 20,
        batch_size: int = 512,
        device: str | torch.device | None = None,
        seed: int = 42,
    ):
        self.hidden = hidden or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device) if device else auto_device()
        self.seed = seed
        self.model: _MLP | None = None
        self.in_dim: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabularMLPHandle:
        torch.manual_seed(self.seed)
        self.in_dim = X.shape[1]
        self.model = _MLP(self.in_dim, self.hidden, self.dropout).to(self.device)

        ds = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "fit() must be called before predict_proba"
        self.model.eval()
        out: list[np.ndarray] = []
        x = torch.as_tensor(X, dtype=torch.float32)
        for i in range(0, len(x), self.batch_size):
            xb = x[i : i + self.batch_size].to(self.device)
            logits = self.model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        assert self.model is not None and self.in_dim is not None
        meta = {
            "hidden": self.hidden,
            "dropout": self.dropout,
            "in_dim": self.in_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
        }
        torch.save({"state_dict": self.model.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path: str | Path) -> TabularMLPHandle:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        meta: dict[str, Any] = blob["meta"]
        h = cls(
            hidden=meta["hidden"],
            dropout=meta["dropout"],
            lr=meta["lr"],
            weight_decay=meta["weight_decay"],
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            seed=meta["seed"],
        )
        h.in_dim = meta["in_dim"]
        h.model = _MLP(h.in_dim, h.hidden, h.dropout).to(h.device)
        h.model.load_state_dict(blob["state_dict"])
        return h
