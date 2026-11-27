"""Learned payload detector (``b4``): a small dual-head 1D-CNN on servo telemetry.

Input windows of 59 samples (~1 s), stride 29, with 5 channels per joint:
current, position, velocity, goal - position, temperature (z-scored with source-task
statistics). Two heads share the trunk: payload presence (BCE) and payload mass
(MSE on grams / 500). Trajectory score = mean presence probability over its windows.
Trained for 60 epochs, Adam 1e-3, batch 32, 3 seeds, ~6 s per seed on CPU.

Evaluation protocol (run B6t): train and calibrate on every split of one task family,
score every trajectory of the other family. ``threshold_transfer.py`` analyses the
resulting scores, which are shipped in ``results/load_scores.json``.

Requires PyTorch. Loading the NAD CSVs into ``(n_frames, n_joints*5)`` matrices is
left to the caller (columns ``current{j}``, ``pos{j}``, ``vel{j}``, ``goal_pos{j}``,
``temp{j}``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

WINDOW, STRIDE = 59, 29
MASS_SCALE = 500.0


def feature_matrix(cols: dict, joints) -> np.ndarray:
    feats = []
    for j in joints:
        pos = np.asarray(cols[f"pos{j}"], float)
        feats += [cols[f"current{j}"], pos, cols[f"vel{j}"], np.asarray(cols[f"goal_pos{j}"], float) - pos,
                  cols[f"temp{j}"]]
    return np.column_stack(feats).astype(float)


def windows_of(mat: np.ndarray, window: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    if len(mat) < window:
        mat = np.concatenate([mat, np.repeat(mat[-1:], window - len(mat), axis=0)])
    return np.stack([mat[s : s + window] for s in range(0, len(mat) - window + 1, stride)])


class DualHeadCNN(nn.Module):
    def __init__(self, n_feat: int, width: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(n_feat, width, kernel_size=5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.presence = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 1))
        self.mass = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, x):  # x: (batch, window, features)
        h = self.trunk(x.transpose(1, 2))
        return self.presence(h).squeeze(-1), self.mass(h).squeeze(-1)


def train(X: np.ndarray, y: np.ndarray, mass_g: np.ndarray, seed: int, epochs: int = 60,
          batch: int = 32, lr: float = 1e-3) -> DualHeadCNN:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DualHeadCNN(X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    mt = torch.from_numpy((mass_g / MASS_SCALE).astype(np.float32))
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(Xt))
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            logit, m = model(Xt[idx])
            loss = bce(logit, yt[idx]) + torch.mean((m - mt[idx]) ** 2)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def score_trajectory(model: DualHeadCNN, mat: np.ndarray, mu, sd) -> tuple[float, float]:
    """(presence probability, mass in grams), averaged over the trajectory's windows."""
    model.eval()
    X = torch.from_numpy(((windows_of(mat) - mu) / sd).astype(np.float32))
    logit, m = model(X)
    return float(torch.sigmoid(logit).mean()), float(m.mean() * MASS_SCALE)
