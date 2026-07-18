# Gate state table — the ExecutionBlocker family's full scenario matrix

The staged pipeline has four gates built on the same native pause mechanism
(`comfy_execution.graph.ExecutionBlocker`), each with a persisted approval
widget that must be identity-scoped ("any persisted widget that gates
execution needs an identity fingerprint" — the rule every live-found gate bug
reduced to). This table is the specification the tests mirror; if you change
gate behavior, update the row AND its test.

## Gate 1 — `AtlasAssessImage` (image output)

State variables: `auto_continue` (bool widget, default ON), `proceed` (bool
widget, set by ▶ Continue), `approved_for` (fingerprint stamped by ▶),
current image fingerprint.

| # | Given                                   | When queued            | Then                                    | Test |
|---|-----------------------------------------|------------------------|-----------------------------------------|------|
| 1 | auto_continue ON (default)              | any image, any state   | image FLOWS; assessment runs; report shows; the ✅ solve gate is the first stop | `test_node_pauses_image_until_proceed` (first block) |
| 2 | auto_continue OFF, proceed OFF          | fresh image            | image BLOCKED; report + fingerprint emitted | `test_node_pauses_image_until_proceed` |
| 3 | auto_continue OFF, proceed ON, approved_for empty | any image    | image FLOWS (manual unconditional override) | `test_stale_approval_from_a_different_image_rearms_the_gate` (manual case) |
| 4 | auto_continue OFF, proceed ON, approved_for == image fp | same image | image FLOWS (▶-approved)             | same test (ok case) |
| 5 | auto_continue OFF, proceed ON, approved_for != image fp | image swapped | image BLOCKED; "GATE RE-ARMED" banner | same test (stale case) |
| 6 | row 5, then ▶ Continue clicked          | re-queue               | ▶ re-stamps approved_for from ui.fingerprint → row 4 | frontend path (atlas_assess.js), backend covered by row 4 |
| 7 | assessment FAILED (provider down / unparseable) | any            | never cached → every queue retries; gate per rows above | `test_node_never_caches_a_failed_assessment`, `test_unparseable_reply_fails_visibly_and_uncached` |
| 8 | outside ComfyUI (no ExecutionBlocker importable) | any           | degrades to pass-through                | `test_node_falls_back_to_passthrough_outside_comfy` |

Non-blocking outputs: report/settings/sam_prompt_*/geom_*/band_* always flow
(everything they feed also consumes the gated image, so the image blocker is
the single pause point).

## Gate 2 — `AtlasSolveGate` (solve output)

Identity = solve camera + source image fingerprint (a re-solve with different
settings OR a new photo re-arms). Same row structure as rows 2–5 above with
`proceed`/`approved_for`; no auto_continue (the cheap-preview/heavy-stack
split is its whole purpose). Tests: `tests/test_solve_gate.py` (6 tests).

## Gate 3 — 📐 patch-branch outputs (`AtlasBlockoutViewport`)

`patch_*`/`patch_exact` outputs return ExecutionBlocker until a 📐 extraction
exists in `client_data` with a fingerprint matching the CURRENT solve+image;
mismatch re-arms and the frontend clears the stale entry + shows a HUD hint.
Tests: viewport/extraction suites (`test_exact_patch_view.py` and friends).

## Gate 4 — `AtlasSceneHealthGate` 🩺 (solve output, before exporters)

State variables: `pass_through_on_pass` (bool widget, default ON), `proceed`
(bool widget, set by ✅ Acknowledge & Continue), `approved_for` (fingerprint
stamped by ✅), current solve+image fingerprint (`_solve_fingerprint` — same
identity as Gate 2), and the computed health level (pass/warn/fail from
`core.scene_health.evaluate_scene_health`).

The semantic difference from Gate 2: this one is an ACKNOWLEDGEMENT gate —
the artist may override a warn/fail report, but the report is stamped into
`debug_metadata["scene_health"]` (with `acknowledged` + `evaluated_at`) on
EVERY execution, blocked or flowing, so the warning survives into exporter
summaries, review report.md, and the project manifest. Overridable, never
losable.

| # | Given                                        | When queued | Then                                                     | Test |
|---|----------------------------------------------|-------------|----------------------------------------------------------|------|
| 1 | level PASS, pass_through_on_pass ON (default)| any         | solve FLOWS, zero clicks; stamp level=pass, acknowledged=false | `test_pass_level_flows_without_click`, `test_pass_stamp_is_not_marked_acknowledged` |
| 2 | level PASS, pass_through_on_pass OFF         | any         | solve BLOCKED until ✅ (deliberate manual checkpoint)     | `test_pass_through_off_still_gates` |
| 3 | level WARN/FAIL, proceed OFF                 | fresh scene | solve BLOCKED; per-flag report (✖ fail / ⚠ warn) + fingerprint emitted; stamp acknowledged=false | `test_warn_level_ships_closed`, `test_stamp_is_indelible_both_states` |
| 4 | level WARN/FAIL, proceed ON, approved_for == fp | same scene | solve FLOWS; stamp acknowledged=true                     | `test_acknowledge_with_matching_fingerprint_flows` |
| 5 | level WARN/FAIL, proceed ON, approved_for != fp | scene changed since ✅ | solve BLOCKED; "GATE RE-ARMED" banner          | `test_stale_fingerprint_rearms` |
| 6 | level WARN/FAIL, proceed ON, approved_for empty | any       | solve FLOWS (manual unconditional override)              | `test_manual_unconditional_override` |
| 7 | outside ComfyUI (no ExecutionBlocker)        | any         | degrades to pass-through (stamp still applied)           | `test_pass_through_outside_comfy` |

Downstream consumers of the stamp: `_health_summary_suffix` (Nuke/Maya layer
summaries), review `report.md` "## Scene health" section, and the
`atlas_project.json` manifest (`scene_health` key).

## Shared invariants

- Approval identity is CONTENT-based, never positional: a serialized-open
  gate must not sail through on reload with the same content (the v2 staged
  master shipped an approved gate by accident — ships CLOSED since v3).
- A failed upstream step must never consume an approval.
- Every silent branch-skip needs a visible explanation (HUD hint, report
  banner, or 🎯/🔍 status line).
