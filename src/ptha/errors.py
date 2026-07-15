"""Stable user-facing PTHA errors."""

from __future__ import annotations


class PthaError(Exception):
    code = "internal_error"
    exit_code = 1


class ConfigurationError(PthaError):
    code = "configuration_error"
    exit_code = 2


class DatabaseNotFoundError(PthaError):
    code = "database_not_found"
    exit_code = 4


class DatabaseExistsError(PthaError):
    code = "database_exists"
    exit_code = 5
