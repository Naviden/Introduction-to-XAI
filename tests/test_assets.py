"""Asset integrity and offline tests use synthetic bytes, never public downloads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
from urllib.error import URLError

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assets = load_module("test_xai_assets", ROOT / "xai_assets.py")
cli = load_module("test_manage_assets", ROOT / "scripts/manage_assets.py")


@pytest.fixture
def registered(tmp_path, monkeypatch):
    payload = b"Reviewed teaching bytes\n"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "files": {"sample": {"url": "https://example.invalid/sample", "filename": "sample.dat", "sha256": digest}},
        "huggingface": {"model": {"repo_id": "example/model", "revision": "a" * 40,
                                    "files": {"config.json": digest, "model.safetensors": digest}}},
        "decoded_datasets": {},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(assets, "MANIFEST_PATH", manifest_path)
    monkeypatch.setenv("XAI_ASSET_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("XAI_OFFLINE", raising=False)
    monkeypatch.setattr(cli, "xai_assets", assets)
    return payload, tmp_path / "cache/sample/sample.dat", manifest


def forbid_network(*args, **kwargs):
    pytest.fail("This operation must not access the network")


@pytest.mark.parametrize("offline_environment", [False, True])
def test_missing_offline_asset_never_downloads(registered, monkeypatch, offline_environment):
    _, target, _ = registered
    monkeypatch.setattr(assets, "urlopen", forbid_network)
    if offline_environment:
        monkeypatch.setenv("XAI_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="prefetch"):
        assets.asset_file("sample", offline=not offline_environment)
    assert not target.parent.exists()


def test_existing_cache_is_verified_without_network_and_tampering_is_rejected(registered, monkeypatch):
    payload, target, _ = registered
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    monkeypatch.setattr(assets, "urlopen", forbid_network)
    assert assets.asset_file("sample", offline=True) == target
    target.write_bytes(payload + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        assets.asset_file("sample")
    assert target.read_bytes().endswith(b"tampered")  # never silently replace it


def test_download_publishes_only_matching_bytes_and_removes_temporary_files(registered, monkeypatch):
    payload, target, _ = registered
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    assert assets.asset_file("sample") == target
    assert target.read_bytes() == payload
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("failure", ["checksum", "network", "interrupted_stream"])
def test_failed_download_leaves_no_usable_or_partial_cached_file(registered, monkeypatch, failure):
    _, target, _ = registered
    class BrokenStream(io.BytesIO):
        def read(self, *args, **kwargs):
            raise OSError("interrupted stream")
    def response(*args, **kwargs):
        if failure == "network":
            raise URLError("unavailable")
        return BrokenStream(b"partial") if failure == "interrupted_stream" else io.BytesIO(b"wrong bytes")
    monkeypatch.setattr(assets, "urlopen", response)
    expected = ValueError if failure == "checksum" else OSError
    with pytest.raises(expected):
        assets.asset_file("sample")
    assert not target.exists()
    assert not list(target.parent.glob("download-*"))


@pytest.mark.parametrize("environment", [False, True])
def test_hub_pins_commit_allowlist_and_offline_flag_and_checks_every_file(registered, monkeypatch, tmp_path, environment):
    payload, _, manifest = registered
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in manifest["huggingface"]["model"]["files"]:
        (snapshot / name).write_bytes(payload)
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=download))
    if environment:
        monkeypatch.setenv("XAI_OFFLINE", "1")
    assert assets.hf_model_path("model", offline=not environment) == snapshot
    assert calls[-1] == {"repo_id": "example/model", "revision": "a" * 40,
                         "allow_patterns": ["config.json", "model.safetensors"], "local_files_only": True}
    (snapshot / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        assets.hf_model_path("model", offline=True)
    (snapshot / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        assets.hf_model_path("model", offline=True)


def test_hub_cache_miss_is_not_retried_online(registered, monkeypatch):
    calls = []
    def missing(**kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("missing snapshot")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=missing))
    with pytest.raises(FileNotFoundError, match="missing snapshot"):
        assets.hf_model_path("model", offline=True)
    assert len(calls) == 1 and calls[0]["local_files_only"] is True


def test_decoded_fingerprint_preserves_order_shape_and_dtype_but_normalizes_endianness():
    values = np.array([1, 2, 3, 4], dtype="<i4")
    reference = assets.fingerprint(values)
    assert assets.fingerprint(values.astype(">i4")) == reference
    assert assets.fingerprint(values[::-1]) != reference
    assert assets.fingerprint(values.reshape(2, 2)) != reference
    assert assets.fingerprint(values.astype("f8")) != reference
    assert assets.fingerprint(["ab", "c"]) != assets.fingerprint(["a", "bc"])


def test_decoded_dataset_change_is_rejected(registered, monkeypatch):
    _, _, manifest = registered
    values = np.array([1., 2.])
    manifest["decoded_datasets"]["tiny"] = {"sha256": assets.fingerprint(values)}
    monkeypatch.setattr(assets, "manifest", lambda: manifest)
    assert assets.verify_dataset("tiny", values) == assets.fingerprint(values)
    with pytest.raises(ValueError, match="Decoded dataset tiny changed"):
        assets.verify_dataset("tiny", values[::-1])


def test_cli_prefetch_requires_explicit_selection_and_validates_all_keys_first(registered, monkeypatch):
    monkeypatch.setattr(assets, "urlopen", forbid_network)
    with pytest.raises(SystemExit) as missing:
        cli.main(["prefetch"])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as unknown:
        cli.main(["prefetch", "sample", "unknown"])
    assert unknown.value.code == 2
    with pytest.raises(SystemExit):
        cli.main(["prefetch", "sample", "--all"])


def test_cli_verify_is_offline_and_returns_failure_for_missing_cache(registered, monkeypatch, capsys):
    monkeypatch.setattr(assets, "urlopen", forbid_network)
    assert cli.main(["verify", "sample"]) == 1
    assert "FAILED file:sample" in capsys.readouterr().err


def test_cli_lists_without_loading_or_fetching_and_prefetches_only_selected_asset(registered, monkeypatch, capsys):
    payload, target, _ = registered
    monkeypatch.setattr(assets, "urlopen", forbid_network)
    assert cli.main(["list", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output["assets"]) == {"file:sample", "hf:model"}
    assert "does not populate" in output["cache_note"]
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    assert cli.main(["prefetch", "file:sample"]) == 0
    assert target.read_bytes() == payload


def test_cli_all_is_explicit_and_bundled_files_are_verified_without_fetching(registered, monkeypatch, tmp_path):
    payload, _, manifest = registered
    bundled = tmp_path / "review.txt"
    bundled.write_bytes(payload)
    manifest["bundled"] = {"review": {"path": "review.txt", "sha256": hashlib.sha256(payload).hexdigest()}}
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "manifest", lambda: manifest)
    monkeypatch.setattr(assets, "urlopen", forbid_network)
    assert cli.resolve_keys(cli.inventory(manifest), [], True) == ["file:sample", "hf:model", "bundled:review"]
    assert cli.main(["verify", "bundled:review"]) == 0
    bundled.write_bytes(b"changed")
    assert cli.main(["prefetch", "bundled:review"]) == 1
