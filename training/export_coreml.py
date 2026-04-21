import argparse
import os
import sys
from pathlib import Path

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from model_tcn import TCNMultiTask


class ClassificationOnlyWrapper(torch.nn.Module):
    """
    Wrap TCNMultiTask and output classification logits only,
    which is easier to convert to Core ML.
    """

    def __init__(self, model: TCNMultiTask):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .mlpackage path")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})

    model = TCNMultiTask(
        num_classes=int(train_args.get("num_classes", 3)),
        num_frames=int(train_args.get("T", 48)),
        num_joints=int(train_args.get("num_joints", 17)),
        num_channels=int(train_args.get("num_channels", 3)),
        hidden_channels=int(train_args.get("hidden_channels", 128)),
        num_levels=int(train_args.get("num_levels", 4)),
        kernel_size=int(train_args.get("kernel_size", 3)),
        dropout=float(train_args.get("dropout", 0.1)),
        enable_regression=False,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    wrapped = ClassificationOnlyWrapper(model).eval()

    # [B, T, J, C]
    sample_input = torch.randn(
        args.batch_size,
        int(train_args.get("T", 48)),
        int(train_args.get("num_joints", 17)),
        int(train_args.get("num_channels", 3)),
    )

    traced = torch.jit.trace(wrapped, sample_input)

    import coremltools as ct

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="keypoints",
                shape=sample_input.shape,
            )
        ],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    print(f"[DONE] Core ML model saved to: {args.output}")


if __name__ == "__main__":
    main()

