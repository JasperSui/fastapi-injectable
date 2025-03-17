from unittest.mock import patch

import pytest

from src.fastapi_injectable.concurrency import run_coroutine_sync


def test_run_coroutine_sync_success() -> None:
    async def coro() -> str:
        return "test"

    assert run_coroutine_sync(coro()) == "test"


def test_run_coroutine_sync_failed_to_get_event_loop() -> None:
    async def coro() -> str:
        return "test"

    with patch("asyncio.get_event_loop", side_effect=RuntimeError("No running event loop")):  # noqa: SIM117
        with pytest.raises(RuntimeError, match="Failed to get a runnable event loop"):
            run_coroutine_sync(coro())
