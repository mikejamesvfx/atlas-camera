"""ui_to_api DynamicCombo-V3 widget mapping (found live 2026-08-13)."""
from atlas_camera.mcp.comfy_http import widget_inputs

OI = {"ResizeImageMaskNode": {"input": {"required": {
    "input": ["COMFY_MATCHTYPE_V3", {}],
    "resize_type": ["COMFY_DYNAMICCOMBO_V3", {"options": [
        {"key": "scale longer dimension",
         "inputs": {"required": {"longer_size": ["INT", {"default": 512}]}}},
        {"key": "match size",
         "inputs": {"required": {"match": ["IMAGE,MASK", {}],
                                 "crop": ["COMBO", {"options": ["disabled", "center"]}]}}},
    ]}],
    "scale_method": ["COMBO", {"options": ["area", "lanczos"]}],
}}}}


def test_dynamiccombo_sub_widgets_namespaced():
    out = widget_inputs(OI, "ResizeImageMaskNode",
                        ["scale longer dimension", 1536, "lanczos"])
    assert out == {"resize_type": "scale longer dimension",
                   "resize_type.longer_size": 1536,
                   "scale_method": "lanczos"}


def test_dynamiccombo_link_subinputs_skipped():
    out = widget_inputs(OI, "ResizeImageMaskNode",
                        ["match size", "center", "area"])
    assert out == {"resize_type": "match size",
                   "resize_type.crop": "center",
                   "scale_method": "area"}
