"""Compute backends: run a training entry point locally or as a ModelArts training job."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class LocalRunner:
    def __init__(self, root: Path):
        self.root = Path(root)

    def submit(self, script: str, args: list[str]) -> int:
        """Run `python <script> <args>` in this environment; returns the exit code."""
        return subprocess.call([sys.executable, str(self.root / script), *args], cwd=self.root)


class ModelArtsRunner:
    """Submit the same entry point as a ModelArts training job.

    Not wired up yet: the account has no Huawei Cloud access. The job spec below is
    what will be sent; the code is uploaded to obs_code_dir and outputs land in
    obs_output_dir, which the OBS storage backend reads back.
    """

    def __init__(self, region: str, flavor: str, image: str, obs_code_dir: str,
                 obs_output_dir: str):
        self.spec = dict(region=region, flavor=flavor, image=image,
                         code_dir=obs_code_dir, output_dir=obs_output_dir)

    def submit(self, script: str, args: list[str]) -> int:
        raise NotImplementedError(
            "ModelArts compute is selected but not yet available. Job spec that would be "
            f"submitted: {self.spec | {'boot_file': script, 'args': args}}. "
            "Set compute: local in configs/backend.yaml to run on this machine.")
