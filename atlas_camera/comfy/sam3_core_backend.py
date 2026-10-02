"""SAM3 through ComfyUI's OWN model stack (e.g. ``sam3.1_multiplex_fp16``).

``AtlasSAM3Mask`` historically loaded SAM3 from Hugging Face via
``transformers`` (``facebook/sam3``: gated, needs ``hf auth login``). Core
ComfyUI now ships SAM3 natively: a single checkpoint in ``models/checkpoints``
(Comfy-Org's ``sam3.1_multiplex_fp16.safetensors``) loaded by
``CheckpointLoaderSimple`` into a MODEL + a SAM3 text encoder, and run by the
core ``SAM3_Detect`` node. This module drives that same path from inside an
Atlas node, so an install with the checkpoint needs no gated download, no
``[sam3]`` extra and no transformers pin.

Prompt syntax is core's: ``concept:N`` asks for up to N detections of one
concept; a bare ``concept`` returns ONE (``comfy/text_encoders/sam3_clip.py``
``_parse_prompts``). Atlas's contract is "every instance of every concept",
so each concept is encoded separately as ``concept:N``.

Host-coupled by design (imports ``folder_paths``, ``comfy.sd`` and
``comfy_extras.nodes_sam3`` lazily), so it lives in ``comfy/`` and nothing
outside ``comfy/`` may import it.
"""

from __future__ import annotations

from typing import Any

#: The transformers / Hugging Face path. Stays the default so every saved
#: workflow keeps its behaviour.
HF_BACKEND = "hf:facebook/sam3"

#: Upper bound on detections per concept for the ``concept:N`` prompt.
DEFAULT_MAX_PER_CONCEPT = 64

_LOADED: dict[str, tuple[Any, Any]] = {}


def sam3_checkpoint_choices() -> list[str]:
    """Combo values: the HF default, then every SAM3 file in ``checkpoints``.

    Outside ComfyUI (tests, CLI) ``folder_paths`` is absent and only the HF
    default is offered.
    """
    names: list[str] = []
    try:
        import folder_paths  # type: ignore[import-not-found]
        names = [n for n in folder_paths.get_filename_list("checkpoints")
                 if "sam3" in n.lower() and "sam3d" not in n.lower()]
    except Exception:  # noqa: BLE001 - not inside ComfyUI
        names = []
    return [HF_BACKEND, *sorted(names)]


def is_core_checkpoint(choice: str | None) -> bool:
    return bool(choice) and str(choice) != HF_BACKEND


def core_sam3_available() -> bool:
    """True when this ComfyUI has the core SAM3 node and at least one checkpoint."""
    try:
        import comfy_extras.nodes_sam3  # type: ignore[import-not-found]  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return len(sam3_checkpoint_choices()) > 1


def _load(ckpt_name: str) -> tuple[Any, Any]:
    if ckpt_name in _LOADED:
        return _LOADED[ckpt_name]
    import comfy.sd  # type: ignore[import-not-found]
    import folder_paths  # type: ignore[import-not-found]

    path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
    out = comfy.sd.load_checkpoint_guess_config(
        path, output_vae=False, output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"))
    model, clip = out[0], out[1]
    if model is None or clip is None:
        raise RuntimeError(f"{ckpt_name} did not load as a SAM3 model + text encoder")
    # One SAM3 at a time: a second checkpoint replaces the first rather than
    # pinning two copies (ComfyUI's model management still owns VRAM).
    _LOADED.clear()
    _LOADED[ckpt_name] = (model, clip)
    return model, clip


def split_concepts(concepts: str) -> list[str]:
    """Atlas concepts are comma-separated; strip any core ``:N`` suffix."""
    out = []
    for part in (concepts or "").split(","):
        name = part.split(":")[0].strip().strip("()")
        if name:
            out.append(name)
    return out


def core_sam3_instances(
    image: Any,
    concepts: str,
    *,
    ckpt_name: str,
    confidence_threshold: float = 0.5,
    max_per_concept: int = DEFAULT_MAX_PER_CONCEPT,
    refine_iterations: int = 2,
) -> tuple[list[Any], list[str]]:
    """Per-instance masks for every concept, via core ``SAM3_Detect``.

    ``image`` is a ComfyUI IMAGE tensor (B,H,W,C; frame 0 is used). Returns
    ``(masks, matched)``: a list of (H,W) bool numpy masks, LARGEST FIRST, and
    the concept each came from (parallel list). Empty lists when nothing
    matched.
    """
    from comfy_extras.nodes_sam3 import SAM3_Detect  # type: ignore[import-not-found]

    model, clip = _load(ckpt_name)
    frame = image[:1]
    masks: list[Any] = []
    matched: list[str] = []
    for concept in split_concepts(concepts):
        tokens = clip.tokenize(f"{concept}:{int(max(1, max_per_concept))}")
        cond = clip.encode_from_tokens_scheduled(tokens)
        res = SAM3_Detect.execute(
            model, frame, conditioning=cond, threshold=float(confidence_threshold),
            refine_iterations=int(refine_iterations), individual_masks=True)
        stack = res.result[0] if hasattr(res, "result") else res[0]
        for m in stack:
            arr = m.detach().cpu().numpy() > 0.5
            if arr.any():
                masks.append(arr)
                matched.append(concept)
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    return [masks[i] for i in order], [matched[i] for i in order]
