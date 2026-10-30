"""Range assembly must preserve a saved prefix and reject corrupt or ignored ranges."""
import hashlib
import importlib.util
import io
import re

import pytest
from came_repro.author_loader import ROOT


def downloader(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('dataset_downloader_test', ROOT / 'scripts/download_datasets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.time, 'sleep', lambda duration: None)
    return module


class RangeResponse(io.BytesIO):
    def __init__(self, data, start, end, total, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': f'bytes {start}-{end}/{total}'}


def test_range_assembly_preserves_prefix_and_checks_publisher_hash(tmp_path, monkeypatch):
    module = downloader(tmp_path, monkeypatch)
    payload = bytes(range(256)) * 32
    expected = hashlib.sha256(payload).hexdigest()
    item = {'path': 'fixture/asset.pt', 'size': len(payload), 'lfs': {'oid': expected}}
    partial = tmp_path / 'fixture/asset.pt.part'
    partial.parent.mkdir()
    partial.write_bytes(payload[:317])

    def response(request, timeout):
        start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', request.get_header('Range')).groups())
        assert start == 317
        return RangeResponse(payload[start:end + 1], start, end, len(payload))

    monkeypatch.setattr(module.urllib.request, 'urlopen', response)
    digest = module.ranged_download(item, 'pinned-fixture-revision', tmp_path, 2)
    assert digest == expected
    assert (tmp_path / item['path']).read_bytes() == payload
    assert not partial.exists()


def test_range_download_refuses_server_ignoring_range(tmp_path, monkeypatch):
    module = downloader(tmp_path, monkeypatch)
    payload = b'not a valid partial response'
    item = {'path': 'fixture/asset.pt', 'size': len(payload), 'lfs': {'oid': hashlib.sha256(payload).hexdigest()}}
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda request, timeout: RangeResponse(payload, 0, len(payload)-1, len(payload), status=200))
    with pytest.raises(RuntimeError, match='Range not honored'):
        module.ranged_download(item, 'fixture', tmp_path, 2)
    assert not (tmp_path / item['path']).exists()


def test_range_download_refuses_file_hash_mismatch(tmp_path, monkeypatch):
    module = downloader(tmp_path, monkeypatch)
    payload = b'publisher mismatch'
    item = {'path': 'fixture/asset.pt', 'size': len(payload), 'lfs': {'oid': '0' * 64}}
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda request, timeout: RangeResponse(payload, 0, len(payload)-1, len(payload)))
    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        module.ranged_download(item, 'fixture', tmp_path, 2)
    assert not (tmp_path / item['path']).exists()
