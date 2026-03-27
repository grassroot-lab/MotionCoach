# MotionCoach Training (PyTorch)

This folder contains a minimal PyTorch training pipeline for:
**keypoint sequence -> action type classification (serve / forehand / backhand)**  
Optionally: **keypoint sequence -> score regression** (multi-task).

## Expected dataset format

Each sample is a `.npz` file containing:
- `keypoints`: float array shaped **[T, J, C]**
  - `T`: number of frames (window length)
  - `J`: number of joints (default 17)
  - `C`: channels (default 3): `(x, y, conf)`  

You also provide a `manifest.csv` per split (`train` and `val`):

Header (required):
- `path` (string): path to the `.npz` (absolute or relative to `data_root`)
- `label` (int): class index `0..num_classes-1`

Optional columns:
- `score` (float): if you want regression enabled

## Model

`model_tcn.py` defines `TCNMultiTask`:
- Temporal Convolutional Network (TCN) over time
- classification head (cross-entropy)
- optional regression head (SmoothL1)

## Train

Example:

```bash
python3 training/train.py \
  --train_manifest /path/to/train_manifest.csv \
  --val_manifest /path/to/val_manifest.csv \
  --data_root /path/to/samples_root \
  --num_classes 3 \
  --num_joints 17 \
  --num_channels 3 \
  --T 48 \
  --batch_size 16 \
  --epochs 30 \
  --use_regression \
  --lambda_reg 0.5
```

## Notes

- This code assumes you already generated *windowed* keypoint sequences according to your
  labeling/cropping rules (Route B).
- If your `.npz` keypoints are stored as `[J, C, T]`, the dataset loader includes a heuristic
  transpose to `[T, J, C]`.
- Normalization is applied by default:
  - per-frame xy centering
  - per-frame xy scaling (RMS)

