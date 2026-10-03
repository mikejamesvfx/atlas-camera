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

#: The core-ComfyUI path with no file named: the FIRST of the sorted ``*sam3*``
#: checkpoints (``sam3d`` excluded). Deterministic, and the chosen file is
#: named in the node report.
CORE_AUTO = "core:auto"

#: The FIXED combo values (F-8 / D30). They used to be read from disk, so a
#: graph saved on one machine failed validation on another that lacked the
#: same file. An exact file now goes in the ``sam3_checkpoint_override``
#: STRING (ComfyUI rejects STRING->combo links, the ``*_override`` pattern).
#: Saved-workflow contract: APPEND-ONLY from here on.
SAM3_CHECKPOINT_CHOICES: tuple[str, ...] = (HF_BACKEND, CORE_AUTO)

#: Upper bound on detections per concept for the ``concept:N`` prompt.
DEFAULT_MAX_PER_CONCEPT = 64

#: One SAM3 at a time: {ckpt_name: (model_patcher, clip)}. Released by
#: :func:`release_sam3` -- on a checkpoint switch, explicitly, and on
#: ComfyUI's own unload (see :func:`_install_unload_hook`).
_LOADED: dict[str, tuple[Any, Any]] = {}

_UNLOAD_HOOK_ATTR = "_atlas_sam3_release_hooked"


class Sam3CheckpointMissing(LookupError):
    """The requested core SAM3 checkpoint is not on disk.

    The ONE core-path failure that degrades to an empty mask + report (like
    the HF path's gated repo): it is a per-machine install gap, not a broken
    graph. Every other core failure raises, naming the checkpoint.
    """


def sam3_checkpoint_choices() -> list[str]:
    """The fixed combo values: ``["hf:facebook/sam3", "core:auto"]``."""
    return list(SAM3_CHECKPOINT_CHOICES)


def _all_checkpoints() -> list[str]:
    """Every file in ComfyUI's ``checkpoints`` folder; [] outside ComfyUI."""
    try:
        import folder_paths  # type: ignore[import-not-found]
        return list(folder_paths.get_filename_list("checkpoints"))
    except Exception:  # noqa: BLE001 - not inside ComfyUI
        return []


def core_sam3_checkpoints() -> list[str]:
    """Sorted ``*sam3*`` checkpoints (``sam3d`` excluded) -- core:auto's pool."""
    return sorted(n for n in _all_checkpoints()
                  if "sam3" in n.lower() and "sam3d" not in n.lower())


def wants_core(choice: str | None, override: str | None = None) -> bool:
    """True when the core ComfyUI SAM3 path is requested.

    A non-empty override always selects core (it names a core file); otherwise
    any combo value other than the HF default does.
    """
    if (override or "").strip():
        return True
    return bool(choice) and str(choice) != HF_BACKEND



def resolve_core_checkpoint(choice: str | None, override: str | None = None) -> str:
    """The exact checkpoint file the core path will load.

    * non-empty ``override`` -> that exact file (path separators normalised);
    * ``core:auto`` -> the first sorted ``*sam3*`` checkpoint;
    * any other non-HF value (programmatic callers) -> that exact file.

    Raises :class:`Sam3CheckpointMissing` naming what was looked for.
    """
    ov = (override or "").strip()
    if ov:
        names = _all_checkpoints()
        want = ov.replace("\\", "/")
        for n in names:
            if n == ov or n.replace("\\", "/") == want:
                return n
        raise Sam3CheckpointMissing(
            f"sam3_checkpoint_override '{ov}' not found in models/checkpoints")
    if str(choice) == CORE_AUTO:
        pool = core_sam3_checkpoints()
        if not pool:
            raise Sam3CheckpointMissing(
                "core:auto found no *sam3* checkpoint in models/checkpoints "
                "(e.g. sam3.1_multiplex_fp16.safetensors; sam3d files excluded)")
        return pool[0]
    name = str(choice)
    if name in _all_checkpoints():
        return name
    raise Sam3CheckpointMissing(f"core SAM3 checkpoint '{name}' not found in models/checkpoints")


def core_checkpoint_label(choice: str | None, override: str | None,
                          resolved: str) -> str:
    """The report label for a resolved core checkpoint.

    ``core:auto -> <file>`` when the auto pick chose it (so the report names
    the file it landed on), otherwise just the file. One builder for
    ``AtlasSAM3Mask`` and ``AtlasInput`` so their reports cannot drift.
    """
    if not (override or "").strip() and str(choice) == CORE_AUTO:
        return f"{CORE_AUTO} -> {resolved}"
    return resolved


def validate_checkpoint_choice(choice: Any) -> bool | str:
    """``VALIDATE_INPUTS`` body for the ``sam3_checkpoint`` combo.

    Graphs saved before the combo values were fixed (F-8) carry a bare
    checkpoint filename. ComfyUI's built-in list check would reject it as
    "value not in list" before :func:`resolve_core_checkpoint`'s legacy branch
    ever ran, so the nodes take over validation of THIS input: the fixed
    values pass, an installed checkpoint filename passes (at run time it acts
    exactly like ``sam3_checkpoint_override``), anything else is an error
    string naming the value. ``None`` (linked / absent) passes.
    """
    if choice is None or str(choice) in SAM3_CHECKPOINT_CHOICES:
        return True
    name = str(choice)
    want = name.replace("\\", "/")
    if any(n == name or n.replace("\\", "/") == want for n in _all_checkpoints()):
        return True
    return (f"sam3_checkpoint '{name}' is neither {HF_BACKEND}, {CORE_AUTO} nor an "
            f"installed file in models/checkpoints; pick a listed value, or put the "
            f"exact file in sam3_checkpoint_override")


def _checkpoint_mtime_ns(name: str) -> int | None:
    try:
        import os

        import folder_paths  # type: ignore[import-not-found]
        getter = (getattr(folder_paths, "get_full_path", None)
                  or folder_paths.get_full_path_or_raise)
        path = getter("checkpoints", name)
        return os.stat(path).st_mtime_ns if path else None
    except Exception:  # noqa: BLE001 - not inside ComfyUI / vanished mid-call
        return None


def checkpoint_cache_token(choice: str | None, override: str | None = None) -> str:
    """``IS_CHANGED`` token: which core checkpoint a run would load, and when
    it was written.

    The HF path returns a constant, so its caching is unchanged. A core
    request returns ``<file>:<mtime_ns>`` -- or ``missing:<what was asked>``
    -- so installing (or replacing) the checkpoint after a run that degraded
    to an empty mask invalidates ComfyUI's cached result. Stats only; never
    hashes a multi-GB file.
    """
    if not wants_core(choice, override):
        return "hf"
    try:
        name = resolve_core_checkpoint(choice, override)
    except Sam3CheckpointMissing:
        asked = (override or "").strip() or str(choice)
        return f"missing:{asked}"
    mtime = _checkpoint_mtime_ns(name)
    return f"{name}:{mtime if mtime is not None else '?'}"


def core_sam3_available() -> bool:
    """True when this ComfyUI has the core SAM3 node and at least one checkpoint."""
    try:
        import comfy_extras.nodes_sam3  # type: ignore[import-not-found]  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return bool(core_sam3_checkpoints())


def release_sam3() -> bool:
    """Drop the cached SAM3 model + text encoder. True if one was held.

    The cache holds the ONLY strong reference to the ModelPatcher once the
    node returns (ComfyUI's ``current_loaded_models`` holds it weakly), so
    clearing it lets the weights be collected.
    """
    held = bool(_LOADED)
    _LOADED.clear()
    return held


def _install_unload_hook() -> bool:
    """Chain :func:`release_sam3` onto ``comfy.model_management.unload_all_models``.

    ComfyUI has no unload-callback registry (checked against V135:
    ``model_management`` keeps only WEAK refs in ``current_loaded_models``,
    and RAM is freed by dropping the execution cache). Every "unload" path --
    the ``/free`` endpoint (``unload_models`` / ``free_memory``), the OOM
    handler, ``--disable-smart-memory`` -- goes through ``unload_all_models``
    via the module attribute, so wrapping that attribute is the one hook that
    sees them all. Idempotent; a no-op (False) outside ComfyUI.
    """
    try:
        import comfy.model_management as mm  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - not inside ComfyUI
        return False
    current = getattr(mm, "unload_all_models", None)
    if current is None:
        return False
    if getattr(current, _UNLOAD_HOOK_ATTR, False):
        return True

    def unload_all_models(*args, **kwargs):
        release_sam3()
        return current(*args, **kwargs)

    setattr(unload_all_models, _UNLOAD_HOOK_ATTR, True)
    unload_all_models.__wrapped__ = current  # type: ignore[attr-defined]
    mm.unload_all_models = unload_all_models
    return True


def _load(ckpt_name: str) -> tuple[Any, Any]:
    if ckpt_name in _LOADED:
        return _LOADED[ckpt_name]
    # One SAM3 at a time: a different checkpoint releases the first BEFORE
    # loading, so two copies are never resident together.
    release_sam3()
    import comfy.sd  # type: ignore[import-not-found]
    import folder_paths  # type: ignore[import-not-found]

    path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
    out = comfy.sd.load_checkpoint_guess_config(
        path, output_vae=False, output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"))
    model, clip = out[0], out[1]
    if model is None or clip is None:
        raise RuntimeError(f"{ckpt_name} did not load as a SAM3 model + text encoder")
    _install_unload_hook()
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
