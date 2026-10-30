"""Native absolute paths for libraries that convert Windows filenames to bytes."""
import os
from pathlib import Path


def windows_io_path(path):
    value = str(Path(path).resolve())
    if os.name != 'nt' or value.startswith('\\\\?\\'):
        return value
    if value.startswith('\\\\'):
        return '\\\\?\\UNC\\' + value[2:]
    return '\\\\?\\' + value
