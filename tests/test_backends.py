import pytest

from targ import backends


@pytest.fixture(autouse=True)
def fresh_config():
    backends.config.cache_clear()
    yield
    backends.config.cache_clear()


def test_local_defaults():
    assert backends.get_framework() == "torch"
    assert type(backends.get_storage()).__name__ == "LocalStorage"
    assert type(backends.get_compute()).__name__ == "LocalRunner"


def test_env_override_switches_framework(monkeypatch):
    monkeypatch.setenv("TARG_FRAMEWORK", "mindspore")
    assert backends.get_framework() == "mindspore"


def test_local_storage_roundtrip(tmp_path):
    from targ.backends.storage import LocalStorage

    s = LocalStorage(tmp_path)
    s.write_bytes("runs/a/x.bin", b"hi")
    assert s.exists("runs/a/x.bin") and s.read_bytes("runs/a/x.bin") == b"hi"
    assert s.list("runs") == ["runs/a/x.bin"]


def test_modelarts_fails_loudly_until_available(monkeypatch):
    monkeypatch.setenv("TARG_COMPUTE", "modelarts")
    with pytest.raises(NotImplementedError, match="not yet available"):
        backends.get_compute().submit("train.py", [])
