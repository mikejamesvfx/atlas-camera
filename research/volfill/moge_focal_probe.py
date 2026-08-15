"""Does MoGe's own intrinsics/pointmap disagree with the Atlas solve? (RESEARCH)

Atlas FEEDS its solve focal into MoGe (`fov_x`, focal_source="solve"), so MoGe's
depth silently inherits any focal error the solve has. Running MoGe FREE (no
fov_x) returns an INDEPENDENT (focal, metric pointmap) pair. The gap between
them is a trust signal Atlas can already consume: scene_health.py:457 cross-checks
`metadata["predicted_focal_px"]`, but only DepthPro / DA3-Metric emit it — the
MoGe path emits neither `points` nor `intrinsics`.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, str(Path(__file__).resolve().parent / "repos" / "volfill"))
from third_party.moge.model.v2 import MoGeModel

dev = "cuda"
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(dev).eval()

rows = []
for name, path, solve_fx in json.loads(Path(sys.argv[1]).read_text()):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    if max(W, H) > 2048:
        s = 2048 / max(W, H)
        im = im.resize((round(W*s), round(H*s)), Image.LANCZOS)
    t = torch.tensor(np.asarray(im, np.float32)/255., device=dev).permute(2,0,1)
    with torch.inference_mode():
        out = model.infer(t, resolution_level=9)          # FREE: no fov_x
    K = out["intrinsics"].cpu().numpy()                    # normalized (fx in image widths)
    fx_norm = float(K[0, 0])
    fx_px = fx_norm * W                                    # back to SOURCE pixels
    pts = out["points"].cpu().numpy()
    msk = out["mask"].cpu().numpy().astype(bool)
    z = pts[..., 2][msk]
    rows.append({
        "plate": name, "source_w": W,
        "moge_fx_px": round(fx_px, 1),
        "solve_fx_px": solve_fx,
        "ratio_moge_over_solve": (round(fx_px/solve_fx, 3) if solve_fx else None),
        "moge_hfov_deg": round(float(np.degrees(2*np.arctan(0.5/fx_norm))), 2),
        "median_depth_m": round(float(np.median(z)), 3),
        "p95_depth_m": round(float(np.percentile(z, 95)), 2),
    })
    r = rows[-1]
    print(f"{name:<18} moge_fx {r['moge_fx_px']:>9.1f}px  solve_fx "
          f"{str(r['solve_fx_px']):>9}  ratio {str(r['ratio_moge_over_solve']):>6}  "
          f"hfov {r['moge_hfov_deg']:5.1f}deg  med_depth {r['median_depth_m']:8.2f} m")
Path("out/moge_focal_probe.json").write_text(json.dumps(rows, indent=2))
