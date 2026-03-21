# type: ignore  # noqa: PGH003
"""Shared dependency providers for integration tests."""


class DbSession:
    def __init__(self) -> None:
        self.connected = True


class EmailService:
    def __init__(self) -> None:
        self.ready = True


def get_db() -> DbSession:
    return DbSession()


def get_email_service() -> EmailService:
    return EmailService()
