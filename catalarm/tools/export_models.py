"""Export detector and embedder models to ONNX (needs the ``export`` extra: torch etc.).

    python -m catalarm.tools.export_models --out /models yolo11n dinov2_vits14_reg
    python -m catalarm.tools.export_models --out /models --optional dinov3_vits16

Models listed after ``--optional`` may fail (e.g. DINOv3 without an HF_TOKEN) without
failing the build. Every embedder is checked for ONNX-vs-torch parity.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

SIZE = 224
PARITY_MIN_COSINE = 0.999


def export_yolo(name: str, out: Path) -> None:
    from ultralytics import YOLO

    exported = YOLO(f"{name}.pt").export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
    shutil.move(exported, out / f"{name}.onnx")


def _dinov2(name: str):
    import torch

    model = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)  # returns normed CLS token
    # The *_reg models resize their position embedding with antialiased bicubic interpolation,
    # which has no ONNX op. Our input size is fixed, so resize once here and store the result:
    # DINOv2 skips interpolation when the patch grid already matches.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, SIZE, SIZE)
        tokens = model.patch_embed(dummy)
        cls = model.cls_token.expand(1, -1, -1)
        pos = model.interpolate_pos_encoding(torch.cat([cls, tokens], dim=1), SIZE, SIZE)
    model.pos_embed = torch.nn.Parameter(pos.detach().clone())
    return model


def _dinov3(name: str):
    import torch
    from transformers import AutoModel

    repo = {"dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m"}[name]
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("DINOv3 is gated on Hugging Face, set HF_TOKEN (and accept the license)")
    model = AutoModel.from_pretrained(repo, token=token)

    class Pooled(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(pixel_values=x).pooler_output

    return Pooled(model)


def _megadescriptor(name: str):
    import timm

    repo = {"megadescriptor_t_224": "hf-hub:BVRA/MegaDescriptor-T-224"}[name]
    return timm.create_model(repo, num_classes=0, pretrained=True)


EMBEDDERS = {
    "dinov2_vits14_reg": _dinov2,
    "dinov2_vitb14_reg": _dinov2,
    "dinov3_vits16": _dinov3,
    "megadescriptor_t_224": _megadescriptor,
}


def export_embedder(name: str, out: Path) -> None:
    import onnxruntime as ort
    import torch

    class SingleInput(torch.nn.Module):
        # hides optional forward() arguments (e.g. DINOv2 "masks") from the exporter
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)

    model = SingleInput(EMBEDDERS[name](name)).eval()
    path = out / f"{name}.onnx"
    dummy = torch.randn(2, 3, SIZE, SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    check = torch.randn(3, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = model(check).numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"input": check.numpy()}
    )[0]
    cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
    print(f"{name}: dim={got.shape[1]} parity cosine min={cos.min():.6f}")
    if cos.min() < PARITY_MIN_COSINE:
        path.unlink()
        raise RuntimeError(f"{name}: ONNX output differs from torch (cosine {cos.min():.4f})")


def export(name: str, out: Path) -> None:
    if name.startswith("yolo"):
        export_yolo(name, out)
    elif name in EMBEDDERS:
        export_embedder(name, out)
    else:
        raise ValueError(f"unknown model {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("models", nargs="*")
    ap.add_argument("--optional", nargs="*", default=[])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        export(name, args.out)
    for name in args.optional:
        try:
            export(name, args.out)
        except Exception as e:  # optional models must not break the build
            print(f"skipping optional model {name}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
