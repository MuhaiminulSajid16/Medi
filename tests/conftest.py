"""
conftest.py – pytest configuration for the Medi test suite.

Mocks heavy optional dependencies (cv2, easyocr) that are not installed
in the lightweight CI test environment, allowing the endpoint tests to
import ``main`` without errors.
"""

import sys
import types
from unittest.mock import MagicMock


def _mock_module(name: str) -> MagicMock:
    mock = MagicMock()
    mock.__name__ = name
    mock.__spec__ = None
    return mock


# Mock cv2 if not installed
if "cv2" not in sys.modules:
    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.modules["cv2"] = _mock_module("cv2")

# Mock easyocr if not installed
if "easyocr" not in sys.modules:
    try:
        import easyocr  # noqa: F401
    except ImportError:
        easyocr_mock = _mock_module("easyocr")
        # Reader needs to return a list of tuples when called
        easyocr_mock.Reader.return_value.readtext.return_value = []
        sys.modules["easyocr"] = easyocr_mock

# Mock torch if not installed (pulled in by some easyocr paths)
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.modules["torch"] = _mock_module("torch")
