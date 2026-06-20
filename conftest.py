"""Root pytest configuration for fastapi-injectable's own test suite.

The shipped pytest plugin loads automatically via its ``pytest11`` entry point (the
same way adopters get it), so the suite dogfoods its own test-isolation behaviour.
Here we only enable ``pytester`` for the plugin's end-to-end tests.
"""

pytest_plugins = ["pytester"]
