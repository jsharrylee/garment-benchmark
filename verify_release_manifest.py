"""Verify the public release manifest against committed and local bytes.

In a Git checkout, declared hashes are checked against both ``HEAD`` blob
contents and working-tree files.  The tracked path set must equal the manifest
path set plus the non-recursive root manifest itself.  In an extracted release
without ``.git``, declared hashes are checked against filesystem bytes only.

This verifier intentionally computes SHA-256 over blob *contents*.  A Git
object ID from ``git hash-object`` is not the same digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
MANIFEST_NAME = "RELEASE_MANIFEST.sha256.json"
MANIFEST = ROOT / MANIFEST_NAME
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ManifestVerificationError(RuntimeError):
    """Raised when the manifest schema or Git inspection cannot be trusted."""


@dataclass(frozen=True)
class Entry:
    path: str
    size_bytes: int
    sha256: str


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestVerificationError("manifest entry path must be a non-empty string")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ManifestVerificationError(f"unsafe manifest path: {value!r}")
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ManifestVerificationError(f"unsafe manifest path: {value!r}")
    if value == MANIFEST_NAME:
        raise ManifestVerificationError("the release manifest must not recursively hash itself")
    return value


def _load_manifest(root: Path) -> tuple[bytes, list[Entry]]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ManifestVerificationError(f"missing manifest: {MANIFEST_NAME}")
    raw = manifest_path.read_bytes()
    try:
        payload: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestVerificationError(f"invalid UTF-8 JSON manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestVerificationError("manifest root must be an object")
    if payload.get("schema_version") != 1:
        raise ManifestVerificationError("manifest schema_version must equal 1")
    if payload.get("algorithm") != "sha256":
        raise ManifestVerificationError("manifest algorithm must equal 'sha256'")
    if payload.get("archive_root") != "game-garment-benchmark/":
        raise ManifestVerificationError(
            "manifest archive_root must equal 'game-garment-benchmark/'"
        )
    files = payload.get("files")
    if not isinstance(files, list):
        raise ManifestVerificationError("manifest files must be an array")

    entries: list[Entry] = []
    seen: set[str] = set()
    seen_casefold: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ManifestVerificationError(f"manifest files[{index}] must be an object")
        path = _validate_relative_path(item.get("path"))
        size_bytes = item.get("size_bytes")
        sha256 = item.get("sha256")
        if not _is_nonnegative_int(size_bytes):
            raise ManifestVerificationError(f"invalid size_bytes for {path}")
        if not isinstance(sha256, str) or SHA256_PATTERN.fullmatch(sha256) is None:
            raise ManifestVerificationError(f"invalid SHA-256 for {path}")
        if path in seen or path.casefold() in seen_casefold:
            raise ManifestVerificationError(f"duplicate or case-colliding manifest path: {path}")
        seen.add(path)
        seen_casefold.add(path.casefold())
        entries.append(Entry(path, size_bytes, sha256))

    if payload.get("file_count") != len(entries):
        raise ManifestVerificationError(
            f"file_count mismatch: manifest {payload.get('file_count')!r}, actual {len(entries)}"
        )
    total_bytes = sum(entry.size_bytes for entry in entries)
    if payload.get("total_bytes") != total_bytes:
        raise ManifestVerificationError(
            f"total_bytes mismatch: manifest {payload.get('total_bytes')!r}, actual {total_bytes}"
        )
    expected_order = sorted((entry.path for entry in entries), key=str.casefold)
    if [entry.path for entry in entries] != expected_order:
        raise ManifestVerificationError("manifest file paths are not in deterministic order")
    return raw, entries


def _digest(data: bytes) -> tuple[int, str]:
    return len(data), hashlib.sha256(data).hexdigest()


def _compare_bytes(label: str, path: str, data: bytes, entry: Entry) -> list[str]:
    size_bytes, sha256 = _digest(data)
    problems: list[str] = []
    if size_bytes != entry.size_bytes:
        problems.append(
            f"{label}_SIZE {path}: manifest {entry.size_bytes}, actual {size_bytes}"
        )
    if sha256 != entry.sha256:
        problems.append(
            f"{label}_SHA256 {path}: manifest {entry.sha256}, actual {sha256}"
        )
    return problems


def _working_tree_bytes(root: Path, entry: Entry) -> tuple[bytes | None, list[str]]:
    target = root / entry.path
    if not target.is_file():
        return None, [f"WORKTREE_MISSING {entry.path}"]
    if target.is_symlink():
        return None, [f"WORKTREE_SYMLINK {entry.path}"]
    try:
        target.resolve(strict=True).relative_to(root.resolve())
    except ValueError:
        return None, [f"WORKTREE_ESCAPE {entry.path}"]
    return target.read_bytes(), []


def _run_git(root: Path, arguments: Sequence[str], *, input_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ManifestVerificationError(
            f"Git inspection failed for {' '.join(arguments)}: {detail or exc}"
        ) from exc
    return result.stdout


def _git_blob_bytes(root: Path, paths: Sequence[str]) -> dict[str, bytes | None]:
    specifications = [f"HEAD:{path}" for path in paths]
    output = _run_git(
        root,
        ["cat-file", "--batch"],
        input_bytes=("\n".join(specifications) + "\n").encode("utf-8"),
    )
    offset = 0
    blobs: dict[str, bytes | None] = {}
    for path, specification in zip(paths, specifications, strict=True):
        newline = output.find(b"\n", offset)
        if newline < 0:
            raise ManifestVerificationError("truncated git cat-file header")
        header = output[offset:newline]
        offset = newline + 1
        if header.endswith(b" missing"):
            blobs[path] = None
            continue
        parts = header.split()
        if len(parts) != 3 or parts[1] != b"blob":
            raise ManifestVerificationError(
                f"unexpected git cat-file response for {specification}: {header!r}"
            )
        try:
            size_bytes = int(parts[2])
        except ValueError as exc:
            raise ManifestVerificationError(
                f"invalid git blob size for {specification}: {parts[2]!r}"
            ) from exc
        end = offset + size_bytes
        if end >= len(output) or output[end : end + 1] != b"\n":
            raise ManifestVerificationError(f"truncated git blob for {specification}")
        blobs[path] = output[offset:end]
        offset = end + 1
    if offset != len(output):
        raise ManifestVerificationError("unexpected trailing bytes from git cat-file")
    return blobs


def _tracked_paths(root: Path) -> set[str]:
    raw = _run_git(root, ["ls-files", "-z"])
    return {
        value.decode("utf-8", errors="surrogateescape")
        for value in raw.split(b"\0")
        if value
    }


def verify(root: Path = ROOT) -> tuple[str, int, list[str]]:
    root = root.resolve()
    manifest_bytes, entries = _load_manifest(root)
    problems: list[str] = []
    for entry in entries:
        data, path_problems = _working_tree_bytes(root, entry)
        problems.extend(path_problems)
        if data is not None:
            problems.extend(_compare_bytes("WORKTREE", entry.path, data, entry))

    git_mode = (root / ".git").exists()
    if not git_mode:
        return "filesystem", len(entries), problems

    declared = {entry.path for entry in entries}
    tracked = _tracked_paths(root)
    for path in sorted(declared - tracked, key=str.casefold):
        problems.append(f"TRACKED_MISSING {path}")
    for path in sorted(tracked - declared - {MANIFEST_NAME}, key=str.casefold):
        problems.append(f"TRACKED_UNDECLARED {path}")
    if MANIFEST_NAME not in tracked:
        problems.append(f"TRACKED_MISSING {MANIFEST_NAME}")

    paths = [MANIFEST_NAME, *(entry.path for entry in entries)]
    blobs = _git_blob_bytes(root, paths)
    root_manifest_blob = blobs[MANIFEST_NAME]
    if root_manifest_blob is None:
        problems.append(f"GIT_BLOB_MISSING {MANIFEST_NAME}")
    elif root_manifest_blob != manifest_bytes:
        problems.append(f"GIT_BLOB_MISMATCH {MANIFEST_NAME}")
    for entry in entries:
        blob = blobs[entry.path]
        if blob is None:
            problems.append(f"GIT_BLOB_MISSING {entry.path}")
        else:
            problems.extend(_compare_bytes("GIT_BLOB", entry.path, blob, entry))
    return "git", len(entries), problems


def main() -> int:
    try:
        mode, declared_count, problems = verify(ROOT)
    except ManifestVerificationError as exc:
        print(f"Manifest verification failed: {exc}")
        return 1
    if problems:
        for problem in problems:
            print(problem)
        print(
            f"\n{len(problems)} release-manifest inconsistencies "
            f"across {declared_count} declared files ({mode} mode)."
        )
        print(
            "Regenerate with: python -m benchmark.scripts.build_github_release_zip "
            "--overwrite"
        )
        return 1
    print(
        f"Release manifest verified: {declared_count} declared files match "
        f"({mode} mode)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
