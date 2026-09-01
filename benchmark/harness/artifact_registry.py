from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactRegistry:
    def __init__(self, index_path: Path):
        self.index_path = index_path

    def register(self, path: Path, job_id: str, role: str) -> dict[str, Any]:
        record = {"job_id": job_id, "role": role, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    @staticmethod
    def verify(record: dict[str, Any]) -> bool:
        return sha256(Path(record["path"])) == record["sha256"]
