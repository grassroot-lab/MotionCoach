import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


def read_raw_manifest(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"file_name", "label_name", "label_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} missing required columns: {required}")
        return list(reader)


def extract_keypoints_from_video(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        # [J=33, C=3] -> x, y, visibility
        arr = np.zeros((33, 3), dtype=np.float32)
        if result.pose_landmarks is not None:
            for i, lm in enumerate(result.pose_landmarks.landmark):
                arr[i, 0] = lm.x
                arr[i, 1] = lm.y
                arr[i, 2] = lm.visibility
        frames.append(arr)

    cap.release()
    pose.close()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded: {video_path}")

    # [T, 33, 3]
    return np.stack(frames, axis=0).astype(np.float32)


def write_manifest(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "score"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_manifest", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--manifest_out", type=Path, required=True)
    parser.add_argument("--train_manifest_out", type=Path, required=True)
    parser.add_argument("--val_manifest_out", type=Path, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_raw_manifest(args.raw_manifest)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    processed_rows = []
    for row in rows:
        file_name = row["file_name"]
        label_id = int(row["label_id"])
        src = args.raw_dir / file_name
        if not src.exists():
            print(f"[WARN] missing file: {src}")
            continue

        keypoints = extract_keypoints_from_video(src)
        out_name = f"{src.stem}.npz"
        out_path = args.processed_dir / out_name
        np.savez_compressed(out_path, keypoints=keypoints)
        processed_rows.append({"path": str(out_path), "label": label_id, "score": ""})
        print(f"[OK] {file_name} -> {out_name}, shape={keypoints.shape}")

    if not processed_rows:
        raise RuntimeError("No processed rows generated.")

    # deterministic split by per-label round-robin
    by_label = {}
    for r in processed_rows:
        by_label.setdefault(r["label"], []).append(r)

    rng = np.random.default_rng(args.seed)
    train_rows, val_rows = [], []
    for label, group in sorted(by_label.items(), key=lambda x: x[0]):
        group = list(group)
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * args.val_ratio))) if len(group) > 1 else 1
        n_val = min(n_val, len(group))
        val_rows.extend(group[:n_val])
        train_rows.extend(group[n_val:] if n_val < len(group) else group[:])

    write_manifest(args.manifest_out, processed_rows)
    write_manifest(args.train_manifest_out, train_rows)
    write_manifest(args.val_manifest_out, val_rows)

    print(f"[DONE] processed={len(processed_rows)} train={len(train_rows)} val={len(val_rows)}")
    print(f"[FILE] {args.manifest_out}")
    print(f"[FILE] {args.train_manifest_out}")
    print(f"[FILE] {args.val_manifest_out}")


if __name__ == "__main__":
    main()

