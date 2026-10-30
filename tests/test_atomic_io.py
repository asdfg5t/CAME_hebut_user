import ctypes
from ctypes import wintypes
import errno
import json
import os
import threading
import time
import pytest
from came_repro.atomic_io import atomic_write_json, retry_windows_io, read_json_file


def test_json_serialization_failure_preserves_previous_valid_file(tmp_path):
    target = tmp_path / 'state.json'
    atomic_write_json(target, {'cursor': 32})
    previous = target.read_bytes()
    with pytest.raises(ValueError):
        atomic_write_json(target, {'cursor': float('nan')})
    assert target.read_bytes() == previous
    assert not list(tmp_path.glob('*.tmp'))


def test_disk_full_is_not_hidden_or_retried():
    calls = []
    def fail():
        calls.append(1)
        raise OSError(errno.ENOSPC, 'disk full')
    with pytest.raises(OSError):
        retry_windows_io(fail, 'fixture')
    assert len(calls) == 1


@pytest.mark.skipif(os.name != 'nt', reason='Actual Windows sharing semantics')
@pytest.mark.parametrize('release_lock', [True, False])
def test_windows_reader_lock_retry_preserves_atomic_cursor_updates(tmp_path, release_lock):
    target = tmp_path / 'inference_state.json'
    atomic_write_json(target, {'cursor': 32})
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    # Permit reads/writes, but deny delete/replace while the handle is open.
    handle = create(str(target), 0x80000000, 3, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    released = threading.Event()
    observed = []
    thread = None
    def release():
        time.sleep(.12)
        observed.append(read_json_file(target)['cursor'])
        assert close(handle)
        released.set()
    try:
        if release_lock:
            thread = threading.Thread(target=release)
            thread.start()
            atomic_write_json(target, {'cursor': 64}, timeout=2)
            thread.join(timeout=2)
            assert observed == [32] and released.is_set()
            assert read_json_file(target)['cursor'] == 64
            assert not list(tmp_path.glob('*.tmp'))
        else:
            started = time.monotonic()
            with pytest.raises(PermissionError):
                atomic_write_json(target, {'cursor': 64}, timeout=.04)
            assert time.monotonic() - started < 1
            assert read_json_file(target)['cursor'] == 32
    finally:
        if thread:
            thread.join(timeout=2)
        if not released.is_set():
            close(handle)
