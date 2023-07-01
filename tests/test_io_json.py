import json
from unittest.mock import mock_open, patch

import pytest

from io_utils.io_json import dump_json, read_json


def test_read_json_valid_file(tmp_path):
    """
    Test case for reading a valid JSON file.

    Args:
        tmp_path: pytest fixture for creating a temporary directory.
    """
    data = {"name": "John Doe", "age": 30, "city": "New York"}
    file_path = tmp_path / "data.json"
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file)

    result = read_json(file_path)

    assert isinstance(result, dict)
    assert result == data


def test_read_json_invalid_file(tmp_path):
    """
    Test case for reading an invalid JSON file.

    Args:
        tmp_path: pytest fixture for creating a temporary directory.
    """
    file_path = tmp_path / "invalid.json"
    with open(file_path, "w", encoding="utf-8") as json_file:
        json_file.write("Invalid JSON data")

    with pytest.raises(json.JSONDecodeError):
        read_json(file_path)


def test_dump_json(tmp_path):
    """
    Test case for dumping a dictionary to a JSON file.

    Args:
        tmp_path: pytest fixture for creating a temporary directory.
    """
    data = {"name": "John Doe", "age": 30, "city": "New York"}
    file_path = tmp_path / "output.json"

    with patch("io_utils.io_json.open", mock_open()) as mock_file:
        dump_json(data, file_path)
        mock_file.assert_called_once_with(file_path, "w", encoding="utf-8")
