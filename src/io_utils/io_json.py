import json
from typing import Any, Dict


def read_json(path) -> Dict[str, Any]:
    """Reads a json file and returns a dictionary with the data."""
    with open(path, "r") as f:
        return json.load(f)


def dump_json(data: Dict[str, Any], path: str) -> None:
    """Dumps a dictionary to a json file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
