from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_morph_analyzer():
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError as exc:
        raise RuntimeError(
            "NLP mode requested, but optional dependency is not installed. "
            "Install with: uv sync --extra nlp"
        ) from exc
    return MorphAnalyzer()


def normalize_token(token: str, enabled: bool) -> str:
    if not enabled:
        return token.lower()
    if not token:
        return token
    analyzer = get_morph_analyzer()
    parsed = analyzer.parse(token)
    if not parsed:
        return token.lower()
    return parsed[0].normal_form.lower()
