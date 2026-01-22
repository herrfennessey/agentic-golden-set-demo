"""Error classification for agent tool execution."""

import openai
import weaviate.exceptions as we


class FatalToolError(Exception):
    """Unrecoverable error that should abort the agent.

    Examples: connection refused, service unavailable, authentication failures.
    These indicate infrastructure problems, not data issues the agent can fix.
    """

    pass


# Exception types that indicate fatal infrastructure errors.
# Uses isinstance() to respect inheritance hierarchies.
FATAL_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    # Python stdlib - network/connection issues
    ConnectionError,  # Base for ConnectionRefused, ConnectionReset, BrokenPipe, etc.
    TimeoutError,
    # OpenAI SDK - infrastructure/auth errors
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.AuthenticationError,
    openai.InternalServerError,
    openai.PermissionDeniedError,
    # Weaviate - connection/infrastructure errors
    we.WeaviateConnectionError,
    we.WeaviateGRPCUnavailableError,
    we.WeaviateTimeoutError,
    we.WeaviateClosedClientError,
    we.WeaviateStartUpError,
    we.AuthenticationFailedError,
)


def is_fatal_error(error: Exception) -> bool:
    """Determine if an error is fatal (infrastructure) vs recoverable (data).

    Fatal errors abort the agent immediately. Recoverable errors are returned
    to the agent as tool failures so it can adjust its approach.

    Args:
        error: The exception to classify.

    Returns:
        True if fatal (agent should abort), False if recoverable (agent continues).
    """
    # Check against known fatal exception types
    if isinstance(error, FATAL_EXCEPTION_TYPES):
        return True

    # Check HTTP status for any error with status_code attribute (covers edge cases)
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True

    # Everything else is recoverable - let the agent handle it
    return False
