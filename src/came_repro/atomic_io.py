"""Atomic persistence with bounded retries for Windows sharing/access conflicts."""
import json
import logging
import os
from pathlib import Path
import time
import uuid
from .filesystem_paths import windows_io_path


def retry_windows_io(operation, path, timeout=5.0):
    deadline = time.monotonic() + timeout
    delay, warned = .01, False
    while True:
        try:
            return operation()
        except OSError as error:
            # Access denied may also be permanent: retry briefly, then surface
            # the original error. Never delete a valid destination as a fallback.
            if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            if not warned:
                logging.getLogger(__name__).warning('Windows file access conflict; retrying atomically for at most %.1fs: %s', timeout, path)
                warned = True
            time.sleep(min(delay, remaining))
            delay = min(.2, delay * 2)


def unique_temporary_path(path):
    path = Path(path)
    return path.with_name(path.name + '.' + uuid.uuid4().hex[:12] + '.tmp')


def atomic_replace(source, target, timeout=5.0):
    retry_windows_io(lambda: os.replace(windows_io_path(source), windows_io_path(target)), target, timeout)


def atomic_write_json(path, value, timeout=5.0):
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    temporary = unique_temporary_path(path)
    def write():
        with open(windows_io_path(temporary), 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    retry_windows_io(write, temporary, timeout)
    atomic_replace(temporary, path, timeout)


def read_json_file(path, timeout=5.0):
    def read():
        with open(windows_io_path(path), encoding='utf-8-sig') as stream:
            return json.load(stream)
    return retry_windows_io(read, path, timeout)
