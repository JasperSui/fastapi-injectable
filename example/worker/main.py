# ruff: noqa: T201, S101, SLF001

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends

from fastapi_injectable.concurrency import run_coroutine_sync
from fastapi_injectable.util import (
    cleanup_exit_stack_of_func,
    clear_dependency_cache,
    get_injected_obj,
    setup_graceful_shutdown,
)


class Mayor:
    def __init__(self) -> None:
        self._is_cleaned_up = False

    def cleanup(self) -> None:
        print("[Mayor] cleanup called")
        self._is_cleaned_up = True


class Capital:
    def __init__(self, mayor: Mayor) -> None:
        self.mayor = mayor
        self._is_cleaned_up = False

    def cleanup(self) -> None:
        print("[Capital] cleanup called")
        self._is_cleaned_up = True
        self.mayor.cleanup()


class Country:
    def __init__(self, capital: Capital) -> None:
        self.capital = capital

    def do_something(self, message: str) -> None:
        print(f"[Country] do_something for message: {message}")


async def get_mayor() -> Mayor:
    return Mayor()


def get_capital(mayor: Annotated[Mayor, Depends(get_mayor)]) -> Generator[Capital, None, None]:
    capital = Capital(mayor)
    yield capital
    capital.cleanup()  # This will be called only when `cleanup_all_exit_stacks` or `cleanup_exit_stack_of_func(get_country)` is called.  # noqa: E501


def get_country(capital: Annotated[Capital, Depends(get_capital)]) -> Country:
    return Country(capital)


class CountryWorker:
    def _init_as_consumer(self) -> None:
        self.country: Country = get_injected_obj(get_country)

    def do_something(self, message: str) -> None:
        self.country.do_something(message)

    def process(self, messages: list[str]) -> None:
        """This is a simple worker that processes messages in a loop.

        # NOTE(Jasper Sui):
        # I personally recommend to use initialize the injected object before processing each message,
        # and do the cleanup after processing each message,
        # because it's easier to understand and maintain, and it's quite similar to FastAPI's http request lifecycle.
        """
        for id_, message in enumerate(messages, 1):
            # For each message, reinitialize the injected object as a consumer.
            print(f"[CountryWorker] Processing message {id_} of {len(messages)}")
            self._init_as_consumer()
            assert self.country is not None
            assert self.country.capital is not None
            assert self.country.capital._is_cleaned_up is False
            assert self.country.capital.mayor is not None
            assert self.country.capital.mayor._is_cleaned_up is False

            # Do something with the injected object.
            print(f"[CountryWorker] Message: {message}")
            self.do_something(message)

            # Do the cleanup after each message to make sure all generators are closed.
            print("[CountryWorker] post message cleanup called")
            self._post_message_cleanup()
            assert self.country is not None
            assert self.country.capital._is_cleaned_up is True
            assert self.country.capital.mayor._is_cleaned_up is True  # type: ignore[unreachable]

            print("-" * 30)

    def _post_message_cleanup(self) -> None:
        # Clear the async exit stack of the injected object to run the rest of code of the generators in stack.
        run_coroutine_sync(cleanup_exit_stack_of_func(get_country))

        # Clear the dependency cache to free up memory or reset state in scenarios where dependencies
        # might have changed dynamically.
        run_coroutine_sync(clear_dependency_cache())

        # If you have multiple injected objects, you can use `cleanup_all_exit_stacks` to clean up all of them at once.
        # run_coroutine_sync(cleanup_all_exit_stacks())  # noqa: ERA001


if __name__ == "__main__":
    messages = ["Hello", "World"]

    # Setup graceful shutdown before running the worker to clean up all exit stacks and dependency cache when the program is interrupted.  # noqa: E501
    setup_graceful_shutdown()

    country_worker = CountryWorker()
    country_worker.process(messages)
