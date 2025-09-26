# ExoWaveNet — Full Guide

> A complete, runnable guide to build an AstroWaveNet-style exoplanet detection project using a WaveNet-like 1D convolutional model (PyTorch). This guide covers project structure, data preparation, model design, training, evaluation, Colab usage, and best practices.

---

## 1. Project overview

**Goal:** build a reproducible pipeline to detect exoplanet transits (or other periodic signals) from time-series light-curve data using a WaveNet-inspired 1D convolutional neural network.

**What you will get:**
- A clean project layout and `requirements.txt`.
- Data loader + preprocessing code for common light-curve formats (CSV, numpy arrays, or simple FITS-derived tables).
- A WaveNet-like PyTorch model with residual dilated 1D conv blocks.
- Training loop with checkpointing, logging, and validation metrics (precision, recall, AUC).
- Evaluation / inference scripts with plotting utilities.
- Optional Colab notebook instructions for quick experimentation.

---

## 2. Prerequisites

- Python 3.9+ (3.8 is usually OK).
- GPU recommended (Colab GPU or local CUDA).
- Basic familiarity with PyTorch, numpy, pandas, matplotlib.

Suggested packages (we include exact list further down): `torch`, `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `tqdm`, `h5py` (optional), `astropy` (optional).

---

## 3. Project structure

```
exowavenet/
├── README.md
├── requirements.txt
├── setup.py (optional)
├── data/
│   ├── raw/            # raw files you download (kepler/tess/...)
│   ├── processed/      # preprocessed arrays / hdf5
│   └── synthetic/      # synthetic data generator
├── notebooks/          # colab/demo notebooks
├── exowavenet/
│   ├── __init__.py
│   ├── data.py         # Dataset & preprocessing utilities
│   ├── model.py        # WaveNet-like model (PyTorch)
│   ├── train.py        # Training loop
│   ├── evaluate.py     # Evaluation + plotting
│   └── utils.py        # helpers (metrics, checkpoints)
├── scripts/
│   ├── prepare_data.py
│   └── run_inference.sh
└── outputs/
    ├── checkpoints/
    └── logs/
```

---

## 4. Installation

Create a virtual environment, then install dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` (starter):
```
torch>=1.12
numpy
pandas
scipy
matplotlib
scikit-learn
tqdm
h5py
astropy
```

(Adjust torch version to match your CUDA.)

---

## 5. Data: formats & preprocessing

### 5.1 Data sources

You can use any light-curve source: Kepler, K2, TESS, or simulated light curves. Typically you'll have time (`t`) and flux (`f`) (and optionally flux_err). The pipeline below assumes a 1D evenly-sampled or resampled signal segment input (e.g., 1024 samples per segment).

### 5.2 Preprocessing steps

1. **Load** raw data (CSV/npz/FITS).
2. **Clean**: remove NaNs, detrend long-term variability (polynomial or median filter), optionally normalize.
3. **Segment**: extract windows around candidate transits or use sliding windows. Resample or interpolate to a fixed length (e.g., 1024).
4. **Labeling**: binary label per-window (1 if contains transit, 0 otherwise). For regression (depth/period), store targets accordingly.
5. **Augment**: jitter in time, add gaussian noise, vary transit depth/duration for synthetic augmentation.

**Important**: keep a validation and test split by target star or time ranges to avoid leakage.

---

## 6. Synthetic data generator (quick way to test)

Implement a simple generator that injects a box-shaped transit into white noise. This is useful to debug the model quickly without real data.

Key parameters: sample length (L), transit depth, duration, ingress/egress smoothing, SNR.

---

## 7. Model: WaveNet-style 1D conv (PyTorch)

High-level design:
- Input: (batch, 1, L)
- Initial causal 1D conv or non-causal conv (for classification non-causal is fine)
- Stack of **residual blocks** with exponentially increasing dilation (1, 2, 4, 8...) and gated activation (tanh * sigmoid) or simple ReLU residual blocks
- Skip connections summed to a small head
- Final global pooling and classifier (FC -> sigmoid) or per-sample output for segmentation

Key hyperparameters: base channels (e.g., 32), kernel size (3), number of stacks, residual channels, dilation cycles.

**Why WaveNet-like?** Dilated convolutions increase receptive field efficiently so the model can learn patterns at multiple time scales (good for transits of different durations).

---

## 8. Example code snippets (PyTorch)

> The following are compact, copy-pasteable code snippets. Put them into `exowavenet/model.py`, `exowavenet/data.py`, etc.

### `exowavenet/model.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, res_channels, kernel_size,
                              padding=(kernel_size-1)//2 * dilation, dilation=dilation)
        self.conv_res = nn.Conv1d(res_channels, in_channels, 1)
        self.conv_skip = nn.Conv1d(res_channels, skip_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        skip = self.conv_skip(out)
        res = self.conv_res(out)
        return (x + res), skip

class WaveLikeNet(nn.Module):
    def __init__(self, in_channels=1, res_channels=32, skip_channels=64,
                 n_blocks=3, n_layers=8, kernel_size=3):
        super().__init__()
        self.initial = nn.Conv1d(in_channels, res_channels, kernel_size=1)
        self.res_blocks = nn.ModuleList()
        for b in range(n_blocks):
            for i in range(n_layers):
                dilation = 2 ** i
                self.res_blocks.append(ResidualBlock(res_channels, res_channels,
                                                     skip_channels, kernel_size, dilation))
        self.relu = nn.ReLU()
        self.conv_post = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(skip_channels, 1)

    def forward(self, x):
        x = self.initial(x)
        skip_connections = []
        for block in self.res_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        s = sum(skip_connections)
        s = self.conv_post(s)
        s = s.squeeze(-1)
        out = self.fc(s)
        return torch.sigmoid(out).squeeze(-1)
```

Notes: this is a simple, robust variant — you can replace `ResidualBlock` with a gated activation implementation for the canonical WaveNet behavior.

### `exowavenet/data.py` (Dataset skeleton)

```python
from torch.utils.data import Dataset
import numpy as np

class LightCurveDataset(Dataset):
    def __init__(self, X, y, transform=None):
        # X: numpy array (N, L) or (N, 1, L)
        if X.ndim == 2:
            X = X[:, None, :]
        self.X = X.astype('float32')
        self.y = y.astype('float32')
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
```

### `exowavenet/train.py` (training loop)

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from exowavenet.model import WaveLikeNet
from exowavenet.data import LightCurveDataset

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        ys.append(y.numpy())
        preds.append(pred.cpu().numpy())
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    try:
        auc = roc_auc_score(ys, preds)
    except Exception:
        auc = float('nan')
    return auc, ys, preds


def main():
    # placeholder: load your preprocessed arrays X_train, y_train, X_val, y_val
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaveLikeNet().to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    train_ds = LightCurveDataset(X_train, y_train)
    val_ds = LightCurveDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    best_val = -1
    for epoch in range(1, 101):
        train_loss = train_epoch(model, train_loader, criterion, opt, device)
        val_auc, _, _ = eval_model(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_auc={val_auc:.4f}")
        if val_auc > best_val:
            best_val = val_auc
            torch.save(model.state_dict(), f"outputs/checkpoints/model_best.pth")

if __name__ == '__main__':
    main()
```

---

## 9. Evaluation and metrics

- **Binary classification**: AUC-ROC, precision/recall at chosen thresholds, F1, confusion matrix.
- **Localization**: if model outputs per-sample probability, use IoU-style metrics or event-based scoring (did you detect the transit event?).
- **Calibration**: reliability diagrams, Brier score.

Plot ROC curve, precision-recall curve, and example light curves with predictions overlayed.

---

## 10. Colab quickstart

1. Open a new Colab notebook.
2. Install requirements (`pip install torch numpy pandas matplotlib scikit-learn tqdm astropy` — use `--upgrade` as needed).
3. Upload a small sample dataset or mount Google Drive.
4. Paste the model/data/training code cells from above.
5. Train for a few epochs using the GPU runtime.

### Small tips for Colab
- Keep batch size small (32–64) if GPU memory is limited.
- Save checkpoints to `drive/MyDrive/` to persist across sessions.

---

## 11. Tips for success

- **Start small**: first verify your pipeline on synthetic data, then move to real light curves.
- **Normalize consistently**: use the same normalization for train/val/test.
- **Avoid leakage**: split by star or time ranges to prevent contamination.
- **Monitor overfitting**: if training accuracy >> validation metrics, add dropout, augment more, or reduce model capacity.
- **Hyperparameter search**: vary learning rate, number of dilation layers, residual channels, batch size.

---

## 12. Advanced ideas

- Use **gated activations** and **causal convolutions** for sequence modeling and generative tasks.
- Replace global pooling + FC with an **attention** head to focus on relevant regions.
- Use **multi-task learning** to predict transit depth and period along with classification.
- Train using **contrastive learning** or **self-supervised pretraining** on many unlabeled light curves.

---

## 13. Troubleshooting

- **Loss not decreasing**: check data pipeline (labels aligned with segments); try lower LR.
- **AUC stuck near 0.5**: model may be guessing; verify that synthetic positives genuinely differ from negatives.
- **Model very slow**: reduce dilation depth or number of channels.

---

## 14. Next steps I can do for you (pick any):
- Scaffold the full repo (create files and code).
- Provide a runnable Colab notebook (complete).
- Implement a full training run on synthetic data and provide plots.
- Translate model to TensorFlow/Keras.

---

## Appendix A — Suggested hyperparameters

- `sample_length = 1024`
- `batch_size = 32 or 64`
- `res_channels = 32`
- `skip_channels = 64`
- `n_blocks = 2` (stack cycles)
- `n_layers = 8` (per block)
- `lr = 1e-3` with scheduler (ReduceLROnPlateau or CosineAnnealing)
- `epochs = 50-200` depending on dataset size

---

## Appendix B — References & further reading

- WaveNet architecture and dilated convolutions (look up original WaveNet paper)
- AstroWaveNet README for project-specific design notes (use it as inspiration for dataset splits and evaluation schemes)


---

*End of guide.*

