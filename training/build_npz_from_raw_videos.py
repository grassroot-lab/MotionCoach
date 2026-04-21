import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


def read_raw_manifest(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("raw manifest is empty")
    return rows


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "score"])
        writer.writeheader()
        writer.writerows(rows)

def build_windows(arr: np.ndarray, window_size: int, stride: int):
    """
    arr: [T, J, C]
    Returns list of [window_size, J, C]
    """
    T = arr.shape[0]
    if T <= window_size:
        padded = np.zeros((window_size, arr.shape[1], arr.shape[2]), dtype=np.float32)
        padded[:T] = arr
        return [padded]

    windows = []
    start = 0
    while start + window_size <= T:
        windows.append(arr[start : start + window_size])
        start += stride
    if start < T:
        # include one last tail-aligned window for better coverage
        windows.append(arr[T - window_size : T])
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_manifest", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--manifest_out", type=Path, required=True)
    parser.add_argument("--train_manifest_out", type=Path, required=True)
    parser.add_argument("--val_manifest_out", type=Path, required=True)
    parser.add_argument("--swift_script", type=Path, required=True)
    parser.add_argument("--window_size", type=int, default=48)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    rows = read_raw_manifest(args.raw_manifest)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    by_source = []
    for r in rows:
        file_name = r["file_name"]
        label = int(r["label_id"])
        src = args.raw_dir / file_name
        if not src.exists():
            print(f"[WARN] missing: {src}")
            continue

        tmp_json = args.processed_dir / f"{src.stem}.json"
        out_npz = args.processed_dir / f"{src.stem}.npz"

        cmd = [
            "xcrun",
            "swift",
            str(args.swift_script),
            str(src),
            str(tmp_json),
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)

        with tmp_json.open("r", encoding="utf-8") as f:
            arr = np.array(json.load(f), dtype=np.float32)  # [T, J, C]

        np.savez_compressed(out_npz, keypoints=arr)
        tmp_json.unlink(missing_ok=True)
        windows = build_windows(arr, window_size=args.window_size, stride=args.stride)
        source_rows = []
        for i, w in enumerate(windows):
            win_npz = args.processed_dir / f"{src.stem}_w{i:03d}.npz"
            np.savez_compressed(win_npz, keypoints=w)
            row = {"path": str(win_npz), "label": label, "score": ""}
            processed.append(row)
            source_rows.append(row)
        by_source.append((label, source_rows))
        print(f"[OK] {file_name} -> {len(windows)} windows, raw={out_npz.name}, shape={arr.shape}")

    if not processed:
        raise RuntimeError("No processed samples.")

    # Split policy:
    # - Prefer source-level split when possible (>=2 videos in class)
    # - If only 1 source in class, split windows by tail ratio (not strict source split, but keeps class in train/val)
    grouped_sources = {}
    for label, rows_in_source in by_source:
        grouped_sources.setdefault(label, []).append(rows_in_source)

    train, val = [], []
    for label in sorted(grouped_sources):
        sources = grouped_sources[label]
        if len(sources) >= 2:
            # pick one source for val, others for train
            val_source_idx = 0
            for i, src_rows in enumerate(sources):
                if i == val_source_idx:
                    val.extend(src_rows)
                else:
                    train.extend(src_rows)
        else:
            src_rows = sources[0]
            n_val = max(1, int(round(len(src_rows) * args.val_ratio)))
            n_val = min(n_val, len(src_rows) - 1) if len(src_rows) > 1 else 1
            if n_val <= 0:
                train.extend(src_rows)
            else:
                train.extend(src_rows[:-n_val])
                val.extend(src_rows[-n_val:])

    write_manifest(args.manifest_out, processed)
    write_manifest(args.train_manifest_out, train)
    write_manifest(args.val_manifest_out, val)

    print(f"[DONE] processed={len(processed)} train={len(train)} val={len(val)} window={args.window_size} stride={args.stride}")
    print(f"[FILE] {args.manifest_out}")
    print(f"[FILE] {args.train_manifest_out}")
    print(f"[FILE] {args.val_manifest_out}")


if __name__ == "__main__":
    main()
