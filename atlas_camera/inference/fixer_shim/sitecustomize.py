# transformer-engine 2.x compat shim for cosmos-predict2 1.0.9 (written for
# TE 1.13). Mounted into the Fixer inference container at /atlas_shim and put
# on PYTHONPATH, so Python auto-imports it before cosmos_predict2 loads.
#
# TE 2.x moved apply_rotary_pos_emb to transformer_engine.pytorch.attention.rope;
# cosmos 1.0.9 imports it from the old location. Re-export it there.
try:
    import transformer_engine.pytorch.attention as _te_att
    if not hasattr(_te_att, "apply_rotary_pos_emb"):
        from transformer_engine.pytorch.attention.rope import (
            apply_rotary_pos_emb as _arpe,
        )
        _te_att.apply_rotary_pos_emb = _arpe
except Exception:
    pass
