class DependencyCleanupError(Exception):
    """Custom error for dependency cleanup issues."""


class RunCoroutineSyncMaxRetriesError(TimeoutError):
    """Raised when ``run_coroutine_sync`` exhausts its retries waiting for a result.

    Subclasses the builtin :class:`TimeoutError` so existing ``except TimeoutError``
    handlers keep working.
    """


class FastAPICompatibilityError(RuntimeError):
    """Raised at import time when the installed FastAPI's private dependency API has drifted.

    ``fastapi-injectable`` resolves dependencies through undocumented FastAPI
    internals (``fastapi.dependencies.utils.get_dependant`` / ``solve_dependencies``).
    When an allowed-but-incompatible FastAPI release changes that API, this error
    names the installed version and the supported range instead of letting a raw
    ``TypeError`` surface deep inside resolution.
    """
