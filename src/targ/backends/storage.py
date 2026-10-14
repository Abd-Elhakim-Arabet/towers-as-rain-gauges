"""Storage backends with one tiny interface: read_bytes / write_bytes / exists / list."""

from __future__ import annotations

import os
from pathlib import Path


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _p(self, key: str) -> Path:
        return self.root / key

    def read_bytes(self, key: str) -> bytes:
        return self._p(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def list(self, prefix: str) -> list[str]:
        base = self._p(prefix)
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())


class OBSStorage:
    """Huawei OBS. Needs `pip install esdk-obs-python` and AK/SK env vars."""

    def __init__(self, bucket: str, endpoint: str):
        try:
            from obs import ObsClient  # esdk-obs-python
        except ImportError as e:
            raise RuntimeError("OBS backend selected but esdk-obs-python is not installed; "
                               "`uv pip install esdk-obs-python` or set storage: local") from e
        ak, sk = os.environ.get("HUAWEICLOUD_SDK_AK"), os.environ.get("HUAWEICLOUD_SDK_SK")
        if not (ak and sk):
            raise RuntimeError("set HUAWEICLOUD_SDK_AK and HUAWEICLOUD_SDK_SK for OBS")
        self.bucket = bucket
        self.client = ObsClient(access_key_id=ak, secret_access_key=sk, server=endpoint)

    def _check(self, resp):
        if resp.status >= 300:
            raise IOError(f"OBS error {resp.status}: {resp.errorMessage}")
        return resp

    def read_bytes(self, key: str) -> bytes:
        resp = self._check(self.client.getObject(self.bucket, key, loadStreamInMemory=True))
        return resp.body.buffer

    def write_bytes(self, key: str, data: bytes) -> None:
        self._check(self.client.putContent(self.bucket, key, content=data))

    def exists(self, key: str) -> bool:
        return self.client.getObjectMetadata(self.bucket, key).status < 300

    def list(self, prefix: str) -> list[str]:
        resp = self._check(self.client.listObjects(self.bucket, prefix=prefix))
        return sorted(c.key for c in resp.body.contents)
