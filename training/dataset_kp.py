import csv
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _pad_or_trunc(x: np.ndarray, T: int) -> np.ndarray:
    """
    x: [T0, J, C]
    """
    T0 = x.shape[0]
    if T0 == T:
        return x
    if T0 > T:
        return x[:T]

    # pad with zeros
    pad_width = [(0, T - T0), (0, 0), (0, 0)]
    return np.pad(x, pad_width, mode="constant", constant_values=0.0)


def _normalize_keypoints(x: np.ndarray, use_conf: bool = True) -> np.ndarray:
    """
    Generic normalization to improve cross-user/scale robustness.
    x: [T, J, C] where C may be 2 (xy) or 3 (x,y,conf).

    We:
      - center xy by per-frame mean
      - scale xy by per-frame RMS
    confidence (if present) is kept as-is.
    """
    if x.shape[-1] < 2:
        return x

    xy = x[..., :2]
    conf = x[..., 2:3] if (x.shape[-1] >= 3) else None

    mean = xy.mean(axis=1, keepdims=True)  # [T,1,2]
    xy0 = xy - mean
    rms = np.sqrt((xy0 ** 2).mean(axis=(1, 2), keepdims=True) + 1e-6)  # [T,1,1]
    xy1 = xy0 / rms

    if conf is not None and use_conf:
        return np.concatenate([xy1, conf], axis=-1)
    return xy1


@dataclass
class ManifestRow:
    path: str
    label: int
    score: Optional[float]


class KeypointWindowDataset(Dataset):
    """
    Expected .npz file schema:
      - keypoints: float array shaped [T, J, C]
      - (optional) score: float

    Expected manifest.csv header:
      path,label,score
    where score may be empty.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str = "",
        T: int = 48,
        num_joints: int = 17,
        num_channels: int = 3,
        normalize: bool = True,
        use_conf: bool = True,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.data_root = data_root
        self.T = T
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.normalize = normalize
        self.use_conf = use_conf

        self.rows: list[ManifestRow] = []
        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"path", "label"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"manifest.csv must contain columns: {required}")

            for r in reader:
                path = r["path"]
                label = int(r["label"])
                score = None
                if "score" in r and r["score"] not in (None, "", "nan"):
                    score = float(r["score"])
                self.rows.append(ManifestRow(path=path, label=label, score=score))

        if len(self.rows) == 0:
            raise ValueError("Empty manifest.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        row = self.rows[idx]
        npz_path = row.path
        if self.data_root:
            npz_path = os.path.join(self.data_root, row.path)

        data = np.load(npz_path)
        if "keypoints" not in data:
            raise KeyError(f"{npz_path} missing keypoints array")

        kp = data["keypoints"].astype(np.float32)
        # Try to coerce to [T,J,C]
        if kp.ndim != 3:
            raise ValueError(f"keypoints must be 3D [T,J,C], got shape {kp.shape} in {npz_path}")

        # If kp looks like [J,C,T], transpose to [T,J,C]
        if kp.shape[0] == self.num_joints and kp.shape[2] != self.num_joints:
            # heuristic: [J, C, T] -> [T, J, C]
            kp = np.transpose(kp, (2, 0, 1))

        kp = _pad_or_trunc(kp, self.T)

        # Ensure joint/channel dims are consistent (soft check)
        if kp.shape[1] != self.num_joints:
            # If mismatch, we can't reliably fix it here; raise to avoid silent bugs.
            raise ValueError(f"{npz_path}: expected num_joints={self.num_joints}, got {kp.shape[1]}")
        if kp.shape[2] != self.num_channels:
            raise ValueError(f"{npz_path}: expected num_channels={self.num_channels}, got {kp.shape[2]}")

        if self.normalize:
            kp = _normalize_keypoints(kp, use_conf=self.use_conf)

        x = torch.from_numpy(kp)  # [T,J,C]
        y = row.label
        s = None
        if row.score is not None:
            s = torch.tensor([row.score], dtype=torch.float32)
        return x, y, s

