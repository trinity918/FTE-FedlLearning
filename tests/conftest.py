"""pytest configuration: register custom markers."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "realdata: mark test as requiring real datasets on disk (deselect with -m 'not realdata')",
    )
