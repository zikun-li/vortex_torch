"""sglang adapter for vortex_torch.

Re-exports the engine entry points from ``api`` so existing callers using
``from vortex_torch.engine.sgl import get_engine`` continue to work. The
re-export is **lazy** (PEP 562): importing this package — e.g. to reach the
light ``config`` submodule — does not pull in ``api`` (and thus sglang's
engine) until one of the names below is actually accessed.
"""

__all__ = [
    "DEFAULT_SCHEDULE_POLICY",
    "EngineConfigError",
    "MODEL_PATH",
    "check_engine_config",
    "get_engine",
    "get_engine_from_json",
    "PrefillPatchConfig",
    "PrefillPatchTrace",
]


def __getattr__(name):
    if name == "PrefillPatchConfig":
        from vortex_torch.engine.sgl.config import PrefillPatchConfig
        return PrefillPatchConfig
    if name == "PrefillPatchTrace":
        from vortex_torch.engine.sgl.prefill_patch import PrefillPatchTrace
        return PrefillPatchTrace
    if name in __all__:
        from vortex_torch.engine.sgl import api
        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
