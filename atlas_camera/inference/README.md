# Atlas Camera Inference Helpers

Inference helpers sit around the deterministic Atlas core.

Planned helpers:

- Object detector: identify common scale anchors such as people, cars, buses,
  signs, and doors.
- Small multimodal LLM: describe visible scene scale cues, suggest likely
  `reference_id` entries, and explain uncertainty in artist-readable language.

Rules:

- These helpers suggest references; they do not solve camera scale by
  themselves.
- Model outputs must be stored with confidence, source, and uncertainty.
- The core solver must remain usable without inference dependencies.

