# Keep this package init minimal to avoid import-order issues.
# Do NOT import heavy modules here.

__all__ = []  # optionally re-export selectively below

# If you want convenience re-exports without breaking tests, keep them lazy:
def __getattr__(name):
    if name in {"NoisePredictionHead", "MultiModalNoiseHead"}:
        from .heads.noise_heads import NoisePredictionHead, MultiModalNoiseHead
        return {"NoisePredictionHead": NoisePredictionHead,
                "MultiModalNoiseHead": MultiModalNoiseHead}[name]
    raise AttributeError(name)
