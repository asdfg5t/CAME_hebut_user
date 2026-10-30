"""Cooperative pause requests apply only to children of the resumable suite."""
import os
from pathlib import Path


class TrainingPaused(Exception):
    pass


def pause_requested():
    path = os.environ.get('CAME_PAUSE_FILE')
    parent = os.environ.get('CAME_SUITE_PARENT_PID')
    if parent:
        import psutil
        if not psutil.pid_exists(int(parent)):
            return True
    return bool(path and Path(path).is_file())


def pause_if_requested():
    if pause_requested():
        raise TrainingPaused('Safe pause requested; resume from the latest consistent checkpoint/cache cursor')
