from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen


OFFICIAL_TEST_ARCHIVE = (
    "https://huggingface.co/datasets/SII-LiMing/ReWeaver-GCD-TS/resolve/"
    "main/test/test.tar.gz.part-0000?download=true"
)


class CountingReader:
    def __init__(self, source):
        self.source = source
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self.source.read(size)
        self.bytes_read += len(data)
        return data

    def close(self) -> None:
        self.source.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_parts(member_name: str) -> tuple[str, tuple[str, ...]] | None:
    parts = PurePosixPath(member_name).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    for index, part in enumerate(parts):
        if part.startswith("rand_"):
            return part, tuple(parts[index + 1 :])
    return None


def complete(files: list[str]) -> bool:
    rgb = [name for name in files if "/rgb/" in f"/{name}" and name.lower().endswith((".png", ".jpg", ".jpeg"))]
    return (
        len(rgb) >= 4
        and any(name.endswith("_2d_panel.json") for name in files)
        and any(name.endswith("_3d_geo.npz") for name in files)
    )


def acquire(output_root: Path, manifest_path: Path, count: int) -> dict[str, object]:
    request = Request(OFFICIAL_TEST_ARCHIVE, headers={"User-Agent": "game-garment-benchmark/1.0"})
    selected: list[str] = []
    extracted: dict[str, list[str]] = defaultdict(list)

    response = urlopen(request, timeout=60)
    reader = CountingReader(response)
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                parsed = sample_parts(member.name)
                if parsed is None:
                    continue
                sample_name, relative_parts = parsed

                if sample_name not in selected:
                    if len(selected) >= count:
                        if all(complete(extracted[name]) for name in selected):
                            break
                        continue
                    selected.append(sample_name)

                if sample_name not in selected or not relative_parts:
                    continue
                if member.issym() or member.islnk() or member.isdev():
                    raise RuntimeError(f"unsafe tar member type: {member.name}")

                destination = output_root / sample_name / Path(*relative_parts)
                resolved = destination.resolve()
                if output_root.resolve() not in resolved.parents:
                    raise RuntimeError(f"unsafe tar path: {member.name}")
                if member.isdir():
                    resolved.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue

                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read tar member: {member.name}")
                resolved.parent.mkdir(parents=True, exist_ok=True)
                with resolved.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                extracted[sample_name].append("/".join(relative_parts))
    finally:
        reader.close()

    incomplete = [name for name in selected if not complete(extracted[name])]
    if len(selected) != count or incomplete:
        raise RuntimeError(f"official subset incomplete: selected={selected}, incomplete={incomplete}")

    files = []
    for sample_name in selected:
        for path in sorted((output_root / sample_name).rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(output_root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    manifest = {
        "source": OFFICIAL_TEST_ARCHIVE.split("?", 1)[0],
        "provenance": "ReWeaver authors' official Hugging Face dataset repository",
        "license": "CC-BY-NC-4.0",
        "stream_bytes_read": reader.bytes_read,
        "selected_samples": selected,
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a bounded official ReWeaver GCD-TS test subset.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    result = acquire(args.output, args.manifest, args.count)
    print(json.dumps({key: result[key] for key in ("stream_bytes_read", "selected_samples")}, indent=2))


if __name__ == "__main__":
    main()
