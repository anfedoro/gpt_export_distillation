from __future__ import annotations

import hashlib


def stable_id(*parts: object, prefix: str | None = None) -> str:
    text = "\x1f".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}" if prefix else digest
