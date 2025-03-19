import logging

# Create a logger for the entire package
logger = logging.getLogger("fastapi_injectable")


def reset_logger_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def configure_logging(
    level: int | str | None = None,
    format_: str | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """Configure logging for fastapi-injectable.

    Args:
        level: The logging level to set. Can be a string (e.g., 'INFO', 'DEBUG')
               or a logging constant (e.g., logging.INFO).
        format_: Custom format string for logs. If None, a default format is used.
        handler: Custom logging handler. If None, a StreamHandler is used.
    """
    if handler is None:
        handler = logging.StreamHandler()
        if format_ is None:
            format_ = "%(levelname)s:%(name)s:%(message)s"
        handler.setFormatter(logging.Formatter(format_))

    if level is not None:
        logger.setLevel(level)

    # Only add the handler if it's not already there
    for existing_handler in logger.handlers:
        if existing_handler == handler:
            return

    reset_logger_handlers(logger)
    logger.addHandler(handler)
