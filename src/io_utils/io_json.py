import json
from pathlib import Path
from typing import Any, Dict, Union


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Reads a json file and returns a dictionary with the data."""
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def dump_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Dumps a dictionary to a json file."""
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)
