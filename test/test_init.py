import fastapi_injectable


def test_all_public_names_are_importable() -> None:
    # Every name advertised in __all__ must be resolvable from the package root,
    # otherwise `from fastapi_injectable import *` raises AttributeError and
    # documented top-level imports raise ImportError.
    for name in fastapi_injectable.__all__:
        assert hasattr(fastapi_injectable, name), name


def test_configure_logging_is_exported_from_package_root() -> None:
    # Exercises the public re-export documented in the README's logging section.
    assert callable(fastapi_injectable.configure_logging)
