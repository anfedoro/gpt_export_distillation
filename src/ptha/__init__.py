"""PTHA product orchestration layer."""

from importlib.metadata import PackageNotFoundError, version


def application_version() -> str:
    try:
        return version("gpt-export-distillation")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__all__ = ["application_version"]
