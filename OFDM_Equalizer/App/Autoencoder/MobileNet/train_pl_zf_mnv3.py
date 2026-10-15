# train_pl_zf_mnv3.py
# Zero-Forcing phase corrector w/ MobileNetV3 (Lightning)
# Input:  H (48x48 complex) -> [mag,phase] => (2,48,48)
# Output: theta_hat (48,) per subcarrier; loss = wrapped MSE on phase(Y * e^{jθ̂}) vs angle(X)

import os
import glob
import argparse
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from scipy.io import loadmat

import pytorch_lightning as pl


# ---------------- utils ----------------

def complex_to_magphase(C: np.ndarray) -> np.ndarray:
    mag = np.abs(C)
    phs = np.angle(C)
    return np.stack([mag, phs], axis=0).astype(np.float32)

def to_torch_complex_ri(v: np.ndarray) -> torch.Tensor:
    # (N,) complex -> (N,2) real/imag
    return torch.from_numpy(
        np.stack([v.real.astype(np.float32), v.imag.astype(np.float32)], axis=-1)
    )

def complex_mul_by_phase(y_ri: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # y_ri: (B,48,2), theta: (B,48)
    c, s = torch.cos(theta), torch.sin(theta)
    yr, yi = y_ri[..., 0], y_ri[..., 1]
    out_r = yr * c - yi * s
    out_i = yr * s + yi * c
    return torch.stack([out_r, out_i], dim=-1)

def angle_from_ri(ri: torch.Tensor) -> torch.Tensor:
    # (B,48,2) -> (B,48)
    return torch.atan2(ri[..., 1], ri[..., 0])

def wrapped_phase_mse(pred_phase: torch.Tensor, tgt_phase: torch.Tensor) -> torch.Tensor:
    d = pred_phase - tgt_phase
    d_wrapped = torch.atan2(torch.sin(d), torch.cos(d))
    return torch.mean(d_wrapped ** 2)


# ---------------- dataset ----------------

class HEqualizerDataset(Dataset):
    """
    .mat must contain:
      - H_all:   (48,48,frames) complex
      - y_n_all: (48,frames)    complex
      - X_all:   (48,frames)    complex
    Y is noisy Y; X is QPSK ground truth symbols.
    """
    def __init__(self, data_dir: str):
        super().__init__()
        self.files: List[str] = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if not self.files:
            raise FileNotFoundError(f"No .mat files found in {data_dir}")
        self.index_map: List[Tuple[int, int]] = []
        self.cache = []
        for fi, f in enumerate(self.files):
            mat = loadmat(f)
            H_all = mat["H_all"]
            y_all = mat["y_n_all"]
            X_all = mat["X_all"]
            frames = H_all.shape[2]
            self.cache.append((H_all, y_all, X_all))
            for k in range(frames):
                self.index_map.append((fi, k))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fi, k = self.index_map[idx]
        H_all, y_all, X_all = self.cache[fi]
        H = H_all[:, :, k]                 # (48,48) complex
        Y = y_all[:, k]                    # (48,)   complex
        X = X_all[:, k]                    # (48,)   complex
        H_mp = complex_to_magphase(H)      # (2,48,48)
        Y_ri = to_torch_complex_ri(Y).float()    # (48,2)
        X_phase = torch.from_numpy(np.angle(X).astype(np.float32))  # (48,)
        return torch.from_numpy(H_mp), Y_ri, X_phase


# ---------------- model ----------------

class MobileNetV3PhaseHead(nn.Module):
    def __init__(self, in_ch: int = 2, out_dim: int = 48):
        super().__init__()
        m = mobilenet_v3_small(weights=None)
        # Patch first conv to accept 2 channels (mag, phase)
        first_conv = m.features[0][0]
        m.features[0][0] = nn.Conv2d(
            in_ch, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        # Replace classifier head
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, out_dim)
        self.backbone = m

    def forward(self, x):  # x: (B,2,48,48)
        return self.backbone(x)  # (B,48) radians (unbounded)


# ---------------- LightningModule ----------------

class ZFMobileNetLightning(pl.LightningModule):
    def __init__(self, lr: float = 2e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = MobileNetV3PhaseHead(in_ch=2, out_dim=48)

    def forward(self, H_mp: torch.Tensor) -> torch.Tensor:
        # returns theta_hat (B,48)
        return self.model(H_mp)

    def common_step(self, batch):
        H_mp, Y_ri, X_phase = batch        # (B,2,48,48), (B,48,2), (B,48)
        theta_hat = self(H_mp)             # (B,48)
        Y_rot_ri = complex_mul_by_phase(Y_ri, theta_hat)
        pred_phase = angle_from_ri(Y_rot_ri)
        loss = wrapped_phase_mse(pred_phase, X_phase)
        return loss

    # ---- steps using common_step ----
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        losses = [x['val_loss'] for x in outputs]
        concatenated = torch.stack([l.view(-1) for l in losses]).flatten()
        avg_loss = concatenated.mean()
        self.log("avg_val_loss", avg_loss, prog_bar=True)
        return {'val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return {'test_loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# ---------------- DataModule ----------------

class HEqualizerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, num_workers: int = 4, val_split: float = 0.1, seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

    def setup(self, stage=None):
        ds = HEqualizerDataset(self.data_dir)
        n = len(ds)
        n_val = max(1, int(n * self.val_split))
        n_train = n - n_val
        gen = torch.Generator().manual_seed(self.seed)
        self.ds_train, self.ds_val = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)
        self.ds_test = self.ds_val  # reuse for quick testing, or split separately

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--precision", type=str, default="32-true")  # e.g., "16-mixed" if you want AMP
    args = parser.parse_args()

    dm = HEqualizerDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )
    model = ZFMobileNetLightning(lr=args.lr)

    ckpt_dir = "checkpoints_pl"
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir, filename="zf_mnv3-{epoch:02d}-{val_loss:.6f}",
            save_top_k=3, monitor="val_loss", mode="min"
        )
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=".",
        callbacks=callbacks,
        log_every_n_steps=25,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()
