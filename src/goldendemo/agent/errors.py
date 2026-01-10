"""Error classification for agent tool execution."""


class FatalToolError(Exception):
    """Raised when a tool encounters an unrecoverable error that should abort the agent.

    Examples: connection refused, service unavailable, authentication failures.
    These errors indicate infrastructure problems, not data issues.
    """

    pass


# Patterns that indicate fatal infrastructure errors
FATAL_ERROR_PATTERNS = [
    "connection refused",
    "connection reset",
    "connection timed out",
    "failed to connect",
    "service unavailable",
    "no route to host",
    "name resolution failed",
    "dns lookup failed",
    "authentication failed",
    "unauthorized",
    "permission denied",
]


def is_fatal_error(error: Exception) -> bool:
    """Determine if an error is fatal (infrastructure) vs recoverable (data).

    Args:
        error: The exception to classify.

    Returns:
        True if the error is fatal and agent should abort,
        False if the error is recoverable and agent should continue.
    """
    error_str = str(error).lower()

    # Check for fatal patterns in error message
    for pattern in FATAL_ERROR_PATTERNS:
        if pattern in error_str:
            return True

    # Check for specific exception types
    error_type = type(error).__name__.lower()

    # Connection/network errors are fatal
    if any(term in error_type for term in ["connection", "timeout", "network", "socket", "dns"]):
        return True

    # HTTP 5xx errors are fatal (server-side)
    if hasattr(error, "status_code"):
        status = getattr(error, "status_code", 0)
        if isinstance(status, int) and status >= 500:
            return True

    # Everything else is recoverable
    return False
