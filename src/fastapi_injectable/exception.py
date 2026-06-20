class DependencyCleanupError(Exception):
    """Custom error for dependency cleanup issues."""


class RunCoroutineSyncMaxRetriesError(TimeoutError):
    """Raised when ``run_coroutine_sync`` exhausts its retries waiting for a result.

    Subclasses the builtin :class:`TimeoutError` so existing ``except TimeoutError``
    handlers keep working.
    """
