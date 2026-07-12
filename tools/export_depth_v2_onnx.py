"""Export a Depth Anything V2 model to ONNX, with a torch-vs-ONNX parity check.

CV-audit item 11: the V2 depth models run through transformers/PyTorch on
every solve. An ONNX export enables deployment through ONNX Runtime /
TensorRT / OpenVINO (2-4x typical speedup at FP16, smaller install
footprint). This tool only EXPORTS and VERIFIES — runtime integration into
atlas_camera.inference.depth_estimator is documented future work; the
estimator keeps its transformers backend.

Usage:
    python tools/export_depth_v2_onnx.py                       # relative small model
    python tools/export_depth_v2_onnx.py --model depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
    python tools/export_depth_v2_onnx.py --output depth_v2.onnx --image path/to/photo.png

Requires:  pip install -e .[neural] onnx onnxruntime

Notes:
- The export is FIXED-RESOLUTION (--size, default 518 = the processor's own
  canonical inference size): the traced graph bakes the DPT head's final
  upsample target, so a different input H/W silently produces output at the
  export size (verified live — a 392x518 input yielded a 518x518 depth map).
  Only the batch axis is dynamic. Export one file per resolution you need,
  and feed processor output (it resizes to 518x518 by default anyway).
- The export wraps the HF model so the ONNX graph's single output is the
  predicted_depth tensor (B, H, W), not the full ModelOutput dict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_SIZE = 518  # the processor's canonical inference resolution (37 x 14px patches)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model id (any Depth-Anything-V2-*-hf variant)")
    ap.add_argument("--output", default=None,
                    help="Output .onnx path (default: <model-name>.onnx in CWD)")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE,
                    help="Export/verify resolution (must be a multiple of 14)")
    ap.add_argument("--image", default=None,
                    help="Optional real image for the parity check (else random input)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tolerance", type=float, default=1e-3,
                    help="Max allowed relative depth deviation torch vs ONNX")
    args = ap.parse_args()

    if args.size % 14:
        ap.error(f"--size {args.size} is not a multiple of the 14px ViT patch")

    try:
        import numpy as np
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        print("Needs torch + transformers:  pip install -e .[neural]", file=sys.stderr)
        return 2
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError:
        print("Needs the ONNX toolchain:  pip install onnx onnxruntime", file=sys.stderr)
        return 2

    out_path = Path(args.output or (args.model.rsplit("/", 1)[-1] + ".onnx"))
    print(f"Loading {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).eval()

    class DepthOnly(torch.nn.Module):
        """Unwrap the HF ModelOutput so ONNX sees one clean depth tensor."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).predicted_depth

    wrapped = DepthOnly(model)

    if args.image:
        from PIL import Image
        pil = Image.open(args.image).convert("RGB")
        pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"]
        print(f"Parity input: {args.image} -> {tuple(pixel_values.shape)}")
        # The processor decides the shape here, and the traced graph bakes it
        # (see the fixed-resolution note above) — an explicit --size would be
        # silently ignored, so say so rather than let the flag mislead.
        if args.size != DEFAULT_SIZE:
            print(f"NOTE: --size {args.size} ignored — with --image the export "
                  f"resolution is the processor's output shape above.")
    else:
        torch.manual_seed(0)
        pixel_values = torch.randn(1, 3, args.size, args.size)
        print(f"Parity input: random {tuple(pixel_values.shape)}")

    print(f"Exporting (opset {args.opset}) -> {out_path} ...")
    torch.onnx.export(
        wrapped, (pixel_values,), str(out_path),
        input_names=["pixel_values"], output_names=["predicted_depth"],
        # Batch only — H/W must stay fixed: the traced DPT head bakes its
        # final upsample size, so a dynamic-H/W graph would accept any input
        # yet always emit depth at the export resolution (verified live).
        dynamic_axes={"pixel_values": {0: "batch"},
                      "predicted_depth": {0: "batch"}},
        opset_version=args.opset,
        # torch >= 2.6 defaults to the dynamo exporter, which needs the extra
        # onnxscript dependency; the legacy TorchScript exporter handles this
        # ViT fine and keeps the tool's requirements to onnx + onnxruntime.
        dynamo=False,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")

    print("Parity check: torch vs onnxruntime ...")
    with torch.inference_mode():
        ref = wrapped(pixel_values).numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"pixel_values": pixel_values.numpy()})[0]

    scale = float(np.abs(ref).max()) or 1.0
    max_rel = float(np.abs(got - ref).max()) / scale
    mean_rel = float(np.abs(got - ref).mean()) / scale
    print(f"  max relative deviation:  {max_rel:.2e}")
    print(f"  mean relative deviation: {mean_rel:.2e}")
    if max_rel > args.tolerance:
        print(f"FAIL: exceeds --tolerance {args.tolerance}", file=sys.stderr)
        return 1
    print("PASS: ONNX output matches the torch model.")
    print("\nNext steps (not automated here):")
    print("  TensorRT:  trtexec --onnx={} --saveEngine=depth_v2.engine --fp16".format(out_path))
    print("  OpenVINO:  mo --input_model {}".format(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
