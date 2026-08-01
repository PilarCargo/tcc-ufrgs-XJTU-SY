"""Application-specific exception types."""


class XJTUSYTCCError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(XJTUSYTCCError, ValueError):
    """Raised when a project configuration is missing, malformed, or inconsistent."""
