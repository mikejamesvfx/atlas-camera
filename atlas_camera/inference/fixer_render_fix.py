"""NVIDIA Fixer render repair (EXPERIMENTAL) — Docker-container inference.

Fixer (github.com/nv-tlabs/Fixer, the production successor to Difix3D+,
CVPR 2025) is a single-step image diffusion model trained to repair the
artifacts of rendered novel views — stretched texels, torn silhouettes,
hard tear-holes — exactly the classes Atlas's projected camera-move renders
produce. Spike-verified on this repo's own baked frames 2026-07-10: fills
~1/3 of hard black tear pixels on a bare relief mesh, softens smears on the
full DMP rig, adds no temporal flicker, ~0.3–0.45 s/frame on an RTX 5090.
Known limits: mild overall softening (single-step regeneration, internal
576×1024 working resolution), and it does NOT outpaint large frame-edge
reveals (the band-layer ``frame_outpaint_px`` track remains the answer
there).

LICENSING: the Fixer repository is Apache-2.0 and the ``nvidia/Fixer``
weights ship under the NVIDIA Open Model License (commercial use permitted)
— unlike the LaRI/WT hidden-geometry backends this track has no
research-only restriction, but it follows the same "user clones the
upstream repo, atlas_camera bundles none of it" pattern.

WHY DOCKER (the one structural difference from the LaRI/WT pattern): the
model depends on ``cosmos-predict2`` + ``transformer_engine``, which have no
native Windows builds — in-process inference in ComfyUI's venv is
impossible. Inference instead shells out to a Docker container. The working
image recipe (public NGC PyTorch base + cosmos wheel + three container-quirk
fixes) lives in ``docker/fixer/Dockerfile``; a tiny ``sitecustomize.py``
shim (bundled as package data in ``fixer_shim/``) is mounted into every run
to bridge a transformer-engine 2.x API rename.

Frame exchange is by directory: the caller writes input PNGs, the container
reads/writes via two bind mounts, the caller reads the fixed PNGs back.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

FIXER_DEFAULT_IMAGE = "fixer-spike-env"
FIXER_MODEL_RELPATH = Path("models/pretrained/pretrained_fixer.pkl")
FIXER_SCRIPT_RELPATH = Path("src/inference_pretrained_model.py")


def shim_dir() -> Path:
    """Directory holding the bundled sitecustomize.py TE-compat shim."""
    return Path(__file__).resolve().parent / "fixer_shim"


def resolve_fixer_root(fixer_path: str | None) -> Path:
    """Locate the user's Fixer clone (widget value wins over ATLAS_FIXER_PATH).

    The clone must contain the inference script and the downloaded weights
    (both ``models/base/*`` and ``models/pretrained/pretrained_fixer.pkl`` —
    the model code hard-expects the base files next to the pickle).
    """
    p = (fixer_path or "").strip() or os.environ.get("ATLAS_FIXER_PATH", "").strip()
    root = Path(p) if p else None
    if root is None or not root.is_dir():
        raise RuntimeError(
            "Fixer repository not found. This EXPERIMENTAL node needs a local\n"
            "clone of NVIDIA Fixer (Apache-2.0) with its weights downloaded:\n"
            "    git clone https://github.com/nv-tlabs/Fixer.git\n"
            "    cd Fixer && hf download nvidia/Fixer --local-dir models\n"
            "then set the node's fixer_path widget (or the ATLAS_FIXER_PATH\n"
            "env var) to the clone. See INSTALL.md 'Experimental: Fixer\n"
            "Render Repair'."
        )
    missing = [str(rel) for rel in (FIXER_MODEL_RELPATH, FIXER_SCRIPT_RELPATH)
               if not (root / rel).exists()]
    if missing:
        raise RuntimeError(
            f"Fixer clone at {root} is incomplete — missing: {', '.join(missing)}.\n"
            "Download the weights into the clone:\n"
            "    hf download nvidia/Fixer --local-dir models"
        )
    return root


def build_docker_command(
    fixer_root: Path,
    exchange_dir: Path,
    docker_image: str = FIXER_DEFAULT_IMAGE,
    timestep: int = 250,
) -> list[str]:
    """The exact spike-proven invocation, as an argv list (no shell quoting).

    Mounts: the Fixer clone at /work (script + weights), the exchange dir at
    /exchange (in/ and out/ frame folders), and the bundled TE shim at
    /atlas_shim (PYTHONPATH points there so Python auto-imports the
    sitecustomize compat patch before cosmos loads).
    """
    inner = (
        f"python /work/{FIXER_SCRIPT_RELPATH.as_posix()} "
        f"--model /work/{FIXER_MODEL_RELPATH.as_posix()} "
        f"--input /exchange/in --output /exchange/out "
        f"--timestep {int(timestep)}"
    )
    return [
        "docker", "run", "--rm", "--gpus=all", "--ipc=host",
        "-e", "PYTHONPATH=/atlas_shim",
        "-v", f"{fixer_root}:/work",
        "-v", f"{exchange_dir}:/exchange",
        "-v", f"{shim_dir()}:/atlas_shim",
        docker_image,
        "-c", inner,  # image ENTRYPOINT is /bin/bash
    ]


def run_fixer_on_dir(
    in_dir: Path,
    out_dir: Path,
    fixer_root: Path,
    docker_image: str = FIXER_DEFAULT_IMAGE,
    timestep: int = 250,
    timeout_s: int = 900,
) -> str:
    """Run Fixer over every PNG in ``in_dir``; fixed frames land in ``out_dir``.

    ``in_dir`` and ``out_dir`` must be ``<exchange>/in`` and ``<exchange>/out``
    for the same parent (one mount carries both). Returns the container log
    tail for the node's report output; raises RuntimeError with actionable
    context on every failure mode observed during the spike (docker missing,
    engine down/wedged, image not built, model error).
    """
    exchange = in_dir.parent
    if out_dir.parent != exchange:
        raise ValueError("in_dir and out_dir must share the same parent")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_docker_command(fixer_root, exchange, docker_image, timestep)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The `docker` CLI was not found on PATH. Fixer inference runs in a\n"
            "Docker container (its cosmos/transformer_engine stack has no native\n"
            "Windows build). Install/start Docker Desktop and retry."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Fixer container exceeded {timeout_s}s. First run after a reboot\n"
            "pulls nothing but does warm up the model (~1 min); if this keeps\n"
            "happening, check `docker ps` / Docker Desktop health and raise the\n"
            "node's timeout_s."
        ) from exc
    log_tail = "\n".join(
        (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-25:]
    )
    if proc.returncode != 0:
        hint = ""
        joined = (proc.stdout or "") + (proc.stderr or "")
        if "Unable to find image" in joined:
            hint = (
                "\nThe inference image is not built. Build it once:\n"
                "    docker build -t {img} -f docker/fixer/Dockerfile docker/fixer/\n"
                "(see INSTALL.md 'Experimental: Fixer Render Repair')".format(
                    img=docker_image)
            )
        elif "error during connect" in joined or "Internal Server Error" in joined:
            hint = (
                "\nThe Docker engine looks down or wedged — start Docker Desktop\n"
                "(a full restart incl. `wsl --shutdown` cleared a wedged engine\n"
                "during the spike)."
            )
        raise RuntimeError(
            f"Fixer container failed (exit {proc.returncode}).{hint}\n"
            f"--- log tail ---\n{log_tail}"
        )
    n_in = len(list(in_dir.glob("*.png")))
    n_out = len(list(out_dir.glob("*.png")))
    if n_out < n_in:
        raise RuntimeError(
            f"Fixer wrote {n_out}/{n_in} frames — container log tail:\n{log_tail}"
        )
    return log_tail
