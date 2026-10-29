from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size: int = 12):
        super().__init__()
        self.recurrent = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, values):
        sequence, _ = self.recurrent(values)
        return self.readout(sequence[:, -1, :]).squeeze(-1)


def fit_temporal_candidate(values: np.ndarray, lookback: int, path: Path) -> dict:
    """Chronological train/validation/test. Scaling and early stopping never see test labels."""
    torch.manual_seed(19)
    torch.set_num_threads(1)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("LSTM requires finite nonnegative observed quantities")
    train_end, validation_end = int(len(values) * 0.65), int(len(values) * 0.8)
    logged = np.log1p(values)
    mean, scale = float(logged[:train_end].mean()), float(logged[:train_end].std())
    scale = max(scale, 1e-6)
    scaled = (logged - mean) / scale
    origins = np.arange(lookback, len(values))
    x = torch.tensor(np.array([scaled[i - lookback:i] for i in origins]), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(scaled[origins], dtype=torch.float32)
    train = torch.from_numpy(origins < train_end)
    validation = torch.from_numpy((origins >= train_end) & (origins < validation_end))
    test = origins >= validation_end
    if not train.any() or not validation.any() or not test.any():
        raise ValueError("Insufficient chronological windows")
    model = LSTMForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    best_state, best_loss, patience = None, float("inf"), 0
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x[train]), y[train])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x[validation]), y[validation]))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 20:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_predictions = np.maximum(0, np.expm1(model(x[test]).numpy() * scale + mean))
        validation_predictions = np.maximum(0, np.expm1(model(x[validation]).numpy() * scale + mean))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "scale": scale,
                "lookback": lookback, "hidden_size": 12}, path)
    return {"test_predictions": test_predictions, "test_actual": values[origins[test]],
            "test_origins": origins[test], "validation_predictions": validation_predictions,
            "validation_actual": values[origins[validation]], "epochs": epoch + 1,
            "train_end": train_end, "validation_end": validation_end}


def predict_next(path: Path, observed_history: np.ndarray) -> float:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = LSTMForecaster(checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    lookback = checkpoint["lookback"]
    if len(observed_history) < lookback:
        raise ValueError("Not enough observed history for saved LSTM")
    scaled = (np.log1p(observed_history[-lookback:]) - checkpoint["mean"]) / checkpoint["scale"]
    with torch.no_grad():
        prediction = model(torch.tensor(scaled, dtype=torch.float32).reshape(1, lookback, 1)).item()
    return float(max(0, np.expm1(prediction * checkpoint["scale"] + checkpoint["mean"])))
