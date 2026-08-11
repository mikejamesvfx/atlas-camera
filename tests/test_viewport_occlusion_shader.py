"""Regression contracts for filtered projection occlusion and alpha handling."""

from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "atlas_camera"
    / "comfy"
    / "web"
    / "atlas_blockout.js"
).read_text(encoding="utf-8")
UI_SOURCE = (
    Path(__file__).parents[1] / "ui" / "src" / "ProjectionMaterial.ts"
).read_text(encoding="utf-8")


def test_occlusion_is_edge_gated_and_derivative_filtered_not_a_hard_cut():
    """A mismatched depth model must not erase broad, smooth surfaces."""
    assert "float atlasUnpackMetricDepth" in SOURCE
    assert "float relativeDepthJump" in SOURCE
    assert "depthEdge = smoothstep(0.015, 0.08, relativeDepthJump)" in SOURCE
    assert "float depthProbeRadius = clamp" in SOURCE
    assert "float relativeDepthMismatch = abs(-vCamZ - storedZ)" in SOURCE
    assert "fwidth(relativeDepthMismatch)" in SOURCE
    assert "coverage *= 1.0 - depthEdge * depthMismatch" in SOURCE
    assert "if (-vCamZ > storedZ + uOccludeBias) discard" not in SOURCE


def test_grazing_texel_and_matte_edges_feed_the_same_coverage():
    assert "vec2 texelDx = dFdx(uv) * uImageSize" in SOURCE
    assert 'geo.setAttribute("atlasEdgeRisk"' in SOURCE
    assert "float topologyStretch = smoothstep(2.0, 8.0, majorFootprint)" in SOURCE
    assert "float topologyDilate = mix(0.38, 0.08, topologyStretch)" in SOURCE
    assert "topologyDilate, 1.0, topologyRisk" in SOURCE
    assert "float topologyCoverage = 1.0 - smoothstep" in SOURCE
    assert "coverage *= topologyCoverage" in SOURCE
    assert "float footprintRisk = smoothstep(uStretchStart - footprintFeather" in SOURCE
    assert "float grazingRisk = 1.0 - smoothstep(0.06, 0.30, facing)" in SOURCE
    assert "coverage *= 1.0 - footprintRisk * edgeRisk" in SOURCE
    assert "majorFootprint / minorFootprint" not in SOURCE
    assert "float matteFeather = clamp(0.5 * fwidth(matte)" in SOURCE
    assert "coverage *= smoothstep(0.5 - matteFeather" in SOURCE


def test_coverage_obeys_ocio_associated_alpha_rules():
    """Coverage is data: transform straight RGB, then blend straight alpha."""
    assert "transparent: true" in SOURCE
    assert "premultipliedAlpha: false" in SOURCE
    assert "depthWrite: true" in SOURCE
    assert "depthTest: true" in SOURCE
    assert "atlasLinearToSRGB(clamp(col.rgb * relight" in SOURCE
    assert "float finalAlpha = clamp(col.a * uOpacity * coverage" in SOURCE
    assert "gl_FragColor = vec4(outColor, finalAlpha)" in SOURCE
    assert "col.rgb * coverage" not in SOURCE
    color_output = SOURCE[
        SOURCE.index("vec4 col = texture2D(uTexture, uv)") :
        SOURCE.index("gl_FragColor = vec4(outColor, finalAlpha)")
    ]
    # The ban on neighbour taps is about the PRIMARY surface: a fragment shader
    # must never bleed adjacent RGB inward to fake coverage it does not have
    # (the "expands coverage only over fragments that already exist" rule).
    #
    # The transition ribbon is the one carve-out, and it is a carve-out rather
    # than an exception to the rule: it is dedicated geometry whose entire
    # stated job is to carry the subject's edge colour OUTWARD, its coverage is
    # real triangles rather than a bled matte, and its alpha comes from the
    # per-vertex ribbon_t and not from anything sampled. Each column is frozen
    # to a single texel, so without averaging across neighbouring columns the
    # skirt is a fan of flat radial streaks. The taps are therefore allowed
    # inside that branch and nowhere else, which is what this checks.
    smudge = color_output.index("if (isRibbon && uRibbonSmudge > 0.0)")
    smudge_end = color_output.index("// Relight normal", smudge)
    outside_ribbon = color_output[:smudge] + color_output[smudge_end:]
    assert "texture2D(uTexture, uv +" not in outside_ribbon
    assert "texture2D(uTexture, uv -" not in outside_ribbon
    assert "atlasLinearToSRGB(coverage" not in SOURCE


def test_metric_depth_texture_is_explicitly_unmanaged_data():
    block = SOURCE[SOURCE.index("if (data.primary_depth_b64)") :]
    assert "dTex.flipY = false" in block
    assert "dTex.colorSpace = THREE.NoColorSpace" in block
    assert "dTex.magFilter = THREE.NearestFilter" in block
    assert "dTex.minFilter = THREE.NearestFilter" in block
    assert "uniform vec2 uPrimaryDepthSize" in SOURCE
    assert "1.0 / max(uPrimaryDepthSize, vec2(1.0))" in SOURCE


def test_standalone_projection_shader_also_declares_straight_alpha():
    assert UI_SOURCE.count("premultipliedAlpha: false") == 2
    assert "vec4(col.rgb, col.a * uOpacity)" in UI_SOURCE


def test_projection_shader_body_cannot_terminate_its_javascript_template():
    start = SOURCE.index("const PROJECTION_FRAGMENT_SHADER = `")
    body_start = SOURCE.index("\n", start) + 1
    body_end = SOURCE.index("\n`;", body_start)
    assert "`" not in SOURCE[body_start:body_end]


def test_planar_patch_mask_is_reprojected_for_layer_color_and_proxy_pass():
    assert "const PLANAR_PATCH_DEBUG = 0xff2fd6" in SOURCE
    assert "uniform sampler2D uPatchMask" in SOURCE
    assert "uPatchViewMatrix * vec4(vWorldPos, 1.0)" in SOURCE
    assert "generated planar hole islands" in SOURCE
    assert "patch_render_mask: patchMaskB64" in SOURCE
    assert 'passes: ["shaded", "depth", "normal", "mask", "patch_render_mask"]' in SOURCE


def test_the_matte_feather_is_not_gated_on_the_occlusion_toggle():
    """A silhouette matte says nothing about depth occlusion.

    The feather used to sit inside `uOccludePrimary > 0.5` and fall back to a
    hard `discard` at 0.5 with the toggle off — and that toggle is off by
    default, so the shipped path put a binary cut back on the exact edge the
    matte exists to soften.
    """
    assert "} else if (matte < 0.5) {" not in SOURCE, (
        "the matte's hard-discard fallback is back")
    block = SOURCE.split("if (uHasMatte > 0.5) {", 1)[1].split("\n    }", 1)[0]
    # Comments in this block explain the gate that was REMOVED, so strip them
    # before looking for the gate itself.
    code = "\n".join(line for line in block.splitlines()
                     if not line.strip().startswith("//"))
    assert "uOccludePrimary" not in code
    assert "discard" not in code


def test_offscreen_render_targets_are_multisampled_except_the_measuring_probe():
    """The canvas is antialias:true but WebGLRenderTarget defaults to samples:0,
    so every offscreen render came back aliased while the viewport looked smooth.

    The 160px coverage probe is the one deliberate exception: it is MEASURED,
    and resolving samples would blend partial coverage into the counts.
    """
    import re
    targets = re.findall(r"new THREE\.WebGLRenderTarget\((.*?)\);", SOURCE)
    assert targets, "no render targets found — the pin has drifted"
    sampled = [t for t in targets if "samples: 4" in t]
    unsampled = [t for t in targets if "samples: 4" not in t]
    assert len(sampled) == 3, f"expected 3 multisampled targets, got {sampled}"
    assert len(unsampled) == 1 and "W, H" in unsampled[0]


def test_the_preview_backbuffer_is_supersampled_and_bounded():
    """The canvas is CSS-scaled, so rendering 1:1 into it wastes real pixels.

    This is DISPLAY supersampling and is unrelated to the `resolution` widget,
    which sets node._atlasW/_atlasH — the dimensions Render Proxy Passes and
    baked Camera Path frames use. The two were conflated once and the
    supersample was wrongly refused as a rule violation.
    """
    # Raised from 2048/2x: that bound was set against a canvas displayed at
    # roughly preview size and inverts on a big one — an 8K display stretches
    # the canvas across more than 2048 device pixels, so the buffer becomes an
    # UNDERsample. 3x from the 1280 logical preview reaches 3840, ~1:1 on 8K.
    assert "ATLAS_VIEWPORT_BACKBUFFER_MAX_SCALE = 3" in SOURCE
    assert "ATLAS_VIEWPORT_BACKBUFFER_MAX_LONG_EDGE = 3840" in SOURCE
    # Bounded, so a dense relief mesh cannot be handed a raw-DPR-sized buffer.
    # Assert on the APIs, not the word — the comment explaining why DPR was
    # rejected necessarily contains it.
    assert "setPixelRatio" not in SOURCE
    assert "window.devicePixelRatio" not in SOURCE
    # The LOGICAL preview size must keep its meaning: it is reported to Python
    # and drives node layout.
    assert "node._atlasPreviewW = previewW; node._atlasPreviewH = previewH;" in SOURCE
    assert "renderer.setSize(previewW, previewH, false)" not in SOURCE


def test_click_tolerances_stay_in_screen_pixels_under_supersampling():
    """`2 * px / canvas.height` converts a screen-pixel tolerance to NDC.

    canvas.height is the BACKBUFFER, so supersampling it without dividing the
    scale back out silently tightens every hit target by the supersample factor
    — a 12px grab handle becoming a 6px one.
    """
    import re
    bare = re.findall(r"2 \* \d+ / Math\.max\(canvas\.height, 1\)", SOURCE)
    assert not bare, f"{len(bare)} tolerance(s) still divide by the raw backbuffer"
    scaled = re.findall(r"2 \* \d+ / Math\.max\(canvas\.height / previewScale, 1\)",
                        SOURCE)
    assert len(scaled) == 7


def test_a_soft_visibility_matte_is_multiplied_not_thresholded():
    """Soft layering ships a CONTINUOUS field A = exp(-beta*|grad D|^2).

    Running it through the 0.5 smoothstep would re-binarize the exact gradient
    it exists to carry and put the hard edge straight back — the artifact the
    whole approach removes.
    """
    assert "uniform float uMatteSoft;" in SOURCE
    assert "if (uMatteSoft > 0.5) {" in SOURCE
    assert "coverage *= clamp(matte, 0.0, 1.0);" in SOURCE
    # The cut path must survive untouched for hand-authored mattes.
    assert "coverage *= smoothstep(0.5 - matteFeather, 0.5 + matteFeather, matte);" in SOURCE


def test_the_soft_stretch_fade_acts_alone():
    """An untorn mesh keeps its rubber-band triangles, and their texel footprint
    explodes — an independent detector of exactly those fragments.

    It must NOT be multiplied by edgeRisk the way the ordinary stretch term is
    (`coverage *= 1.0 - footprintRisk * edgeRisk`), or a camera-facing smear in
    a smooth depth region fades by nothing at all.
    """
    assert "uniform float uSoftStretch;" in SOURCE
    block = SOURCE.split("if (uSoftStretch > 0.0) {", 1)[1].split("}", 1)[0]
    assert "majorFootprint" in block
    assert "edgeRisk" not in block, "the soft stretch fade must not be gated by edgeRisk"


def test_the_soft_terms_are_not_gated_on_the_occlusion_toggle():
    """`uOccludePrimary` is off by default AND ANDs in the topologyDilate hard
    cut. Silhouette coverage is not occlusion semantics."""
    for marker in ("if (uMatteSoft > 0.5) {", "if (uSoftStretch > 0.0) {"):
        before = SOURCE.split(marker, 1)[0]
        # the nearest enclosing occlusion branch must have closed already
        assert before.count("if (uOccludePrimary > 0.5 && uHasPrimaryDepth > 0.5) {") \
            <= before.count("uOccludePrimary")
    assert "uMatteSoft: { value: options.matteSoft ? 1.0 : 0.0 }," in SOURCE
    assert "uSoftStretch: { value: options.matteSoft ? 1.0 : 0.0 }," in SOURCE
