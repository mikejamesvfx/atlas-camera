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
    assert "texture2D(uTexture, uv +" not in color_output
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
