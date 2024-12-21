# Theme and basic settings
project = "FastAPI Injectable"
author = "Jasper Sui"
module = "fastapi_injectable"
copyright = "2024, Jasper Sui"  # noqa: A001

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Theme settings
html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#009688",
        "color-brand-content": "#009688",
    },
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
}

# AutoDoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"
