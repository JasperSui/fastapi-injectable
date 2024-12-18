"""Sphinx configuration."""

project = "Fastapi Injectable"
author = "Jasper Sui"
module = "fastapi_injectable"
copyright = "2024, Jasper Sui"  # noqa: A001
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
