import logging
from collections.abc import Generator
from unittest.mock import Mock, patch

import pytest

from src.fastapi_injectable.logging import configure_logging, logger


@pytest.fixture
def mock_logger() -> Generator[Mock, None, None]:
    with patch("src.fastapi_injectable.logging.logger") as mock:
        yield mock


@pytest.fixture
def mock_formatter() -> Generator[Mock, None, None]:
    with patch("src.fastapi_injectable.logging.logging.Formatter") as mock:
        yield mock


def test_configure_logging_default(mock_logger: Mock, mock_formatter: Mock) -> None:
    configure_logging()

    # Default formatter should be created with default format
    mock_formatter.assert_called_once_with("%(levelname)s:%(name)s:%(message)s")

    # A handler should be added
    mock_logger.addHandler.assert_called_once()

    # Level shouldn't be set when not specified
    mock_logger.setLevel.assert_not_called()


def test_configure_logging_with_level(mock_logger: Mock) -> None:
    configure_logging(level=logging.DEBUG)

    mock_logger.setLevel.assert_called_once_with(logging.DEBUG)
    mock_logger.addHandler.assert_called_once()


def test_configure_logging_with_string_level(mock_logger: Mock) -> None:
    configure_logging(level="INFO")

    mock_logger.setLevel.assert_called_once_with("INFO")
    mock_logger.addHandler.assert_called_once()


def test_configure_logging_with_custom_format(mock_formatter: Mock) -> None:
    custom_format = "%(asctime)s - %(levelname)s - %(message)s"
    configure_logging(format_=custom_format)

    mock_formatter.assert_called_once_with(custom_format)


def test_configure_logging_with_custom_handler(mock_logger: Mock, mock_formatter: Mock) -> None:
    custom_handler = logging.FileHandler("test.log")

    configure_logging(handler=custom_handler)

    # Custom handler should be used directly without creating a new one
    mock_logger.addHandler.assert_called_once_with(custom_handler)
    # Format shouldn't be set on custom handler
    mock_formatter.assert_not_called()


def test_configure_logging_handler_already_exists(mock_logger: Mock) -> None:
    handler = logging.StreamHandler()
    mock_logger.handlers = [handler]

    # When handlers are compared, they should match
    with patch("src.fastapi_injectable.logging.logging.StreamHandler") as mock_stream_handler:
        mock_stream_handler.return_value = handler
        configure_logging()

    # No handler should be added since it already exists
    mock_logger.addHandler.assert_not_called()


def test_configure_logging_real_logger() -> None:
    # Test with the actual logger to verify behavior
    # First clear any existing handlers
    logger.handlers = []

    # Configure with default settings
    configure_logging()
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

    # Configuring again shouldn't add a duplicate handler
    configure_logging()
    assert len(logger.handlers) == 1


def test_logger_name() -> None:
    assert logger.name == "fastapi_injectable"
