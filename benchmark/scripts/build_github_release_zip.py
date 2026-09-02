"""Build a fail-closed, GitHub-public source and document release ZIP.

The archive is selected from an explicit allowlist. It never copies Git
history, runtime data, model weights, external repositories, caches, or local
environments. Every included file is content-scanned and recorded in a
SHA256 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import struct
import tempfile
from typing import Iterable, Sequence
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = "game-garment-benchmark"
MANIFEST_NAME = "RELEASE_MANIFEST.sha256.json"
DEFAULT_OUTPUT = Path("output/releases/game-garment-benchmark-public.zip")
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_METADATA_CHUNK_BYTES = 1024 * 1024
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

ROOT_FILES = (
    Path(".gitattributes"),
    Path(".gitignore"),
    Path("pyproject.toml"),
    Path("README.md"),
    Path("GITHUB_UPLOAD_GUIDE.md"),
    Path("LICENSE_NOTICE.md"),
    Path("THIRD_PARTY_NOTICES.md"),
    Path("THIRD_PARTY_LICENSES/GarmentParticles-MIT.txt"),
)
FINAL_DOCX = Path("output/docx/semantic_pattern_bridge_portfolio_en.docx")
PUBLIC_REPORT_FILES = (
    Path("reports/figures/pattern_semantic_parser_schematic_en.png"),
    Path("reports/figures/pattern_dsl_semantic_example_en.png"),
)

EXCLUDED_PUBLIC_BENCHMARK_PATHS = {
    "benchmark/README.md",
    "benchmark/configs/synbody_samples.json",
    "benchmark/scripts/build_portfolio_technical_report.py",
}
EXCLUDED_PUBLIC_MANIFEST_NAMES = {
    "drafting_counterfactual_pairs.json",
    "final_benchmark.json",
    "freesewing_teagan_diverse.json",
    "freesewing_teagan_holdout.json",
    "freesewing_teagan_training.json",
    "garment_particles_checkpoint_5.json",
    "retrieval_anchored_v2_corpus.json",
    "reweaver_checkpoint_4.json",
    "reweaver_gcd_ts_official_subset.json",
    "source_report.json",
    "synbody_local_acquisition.json",
    "synbody_processed_samples.json",
}
EXCLUDED_PUBLIC_REPORT_NAMES = {
    "checkpoint_2_discovery.md",
    "checkpoint_2_synbody_local_acquisition.md",
    "checkpoint_3_synbody_preprocessing.md",
    "checkpoint_4_reweaver.md",
    "checkpoint_5_garment_particles.md",
    "final_benchmark.md",
    "portfolio_case_study_ko.md",
    "generative_pattern_pipeline_implementation.md",
    "retrieval_anchored_v2_minimal_evaluation.md",
    "reweaver_official_gcd_ts_evaluation.md",
}

BENCHMARK_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".md", ".mjs", ".gitkeep"}
MANIFEST_SUFFIXES = {".json", ".yaml", ".yml", ".md", ".gitkeep"}
REPORT_SUFFIXES = {".md", ".png", ".svg", ".json", ".gitkeep"}
BINARY_RELEASE_SUFFIXES = {".docx", ".png"}
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx"}
SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".netrc",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----", re.IGNORECASE),
    ),
    ("GitHub token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("OpenAI-style secret", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    (
        "assigned credential",
        re.compile(
            r"(?:api[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password|passwd)"
            r"\s*[:=]\s*[\"'][^\"'\r\n]{6,}[\"']",
            re.IGNORECASE,
        ),
    ),
)

PERSONAL_METADATA_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Windows absolute workspace path",
        re.compile(
            r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/](?:Users|work|projects|repos)[\\/]"
            r"[^\s\"'<>]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "POSIX home path",
        re.compile(r"(?<![A-Za-z0-9])/(?:home|Users)/[^/\s\"'<>]+(?:/[^\s\"'<>]*)?"),
    ),
    (
        "local file URI",
        re.compile(r"\bfile:" + r"/{3}[^\s\"'<>]+", re.IGNORECASE),
    ),
    (
        "consumer email address",
        re.compile(
            r"\b[A-Z0-9._%+-]+@(?:gmail|naver|kakao|outlook|hotmail|icloud)\.com\b",
            re.IGNORECASE,
        ),
    ),
)


class ReleaseValidationError(RuntimeError):
    """Raised when a candidate would violate the public-release policy."""


@dataclass(frozen=True)
class ReleaseFile:
    source: Path
    relative: PurePosixPath
    size_bytes: int
    sha256: str

    @property
    def archive_name(self) -> str:
        return f"{ARCHIVE_ROOT}/{self.relative.as_posix()}"


def _is_nova_human(relative: Path) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", relative.as_posix().lower())
    return "novahuman" in compact


def _is_obsolete_schematic(relative: Path) -> bool:
    return (
        relative.parent.as_posix().lower() == "reports/figures"
        and relative.stem.lower() == "structure_prior_pattern_bridge_schematic"
    )


def _has_ignored_part(relative: Path) -> bool:
    return any(part.lower() in IGNORED_PARTS for part in relative.parts)


def _iter_allowed_tree(base: Path, suffixes: set[str]) -> Iterable[Path]:
    if not base.is_dir():
        raise ReleaseValidationError(f"required directory is missing: {base.relative_to(ROOT)}")
    for path in sorted(base.rglob("*"), key=lambda item: item.as_posix().lower()):
        relative = path.relative_to(ROOT)
        if _has_ignored_part(relative):
            continue
        if path.is_symlink():
            raise ReleaseValidationError(f"symbolic links are not allowed: {relative}")
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def collect_candidate_paths() -> list[Path]:
    candidates: list[Path] = []

    for relative in (*ROOT_FILES, FINAL_DOCX):
        path = ROOT / relative
        if not path.is_file():
            raise ReleaseValidationError(f"required release file is missing: {relative}")
        candidates.append(path)

    for path in _iter_allowed_tree(ROOT / "benchmark", BENCHMARK_SUFFIXES):
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED_PUBLIC_BENCHMARK_PATHS:
            continue
        candidates.append(path)

    for path in _iter_allowed_tree(ROOT / "data" / "manifests", MANIFEST_SUFFIXES):
        relative = path.relative_to(ROOT)
        if _is_nova_human(relative) or path.name in EXCLUDED_PUBLIC_MANIFEST_NAMES:
            continue
        candidates.append(path)

    for relative in PUBLIC_REPORT_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise ReleaseValidationError(f"required public report artifact is missing: {relative}")
        candidates.append(path)

    by_relative: dict[str, Path] = {}
    by_casefold: dict[str, str] = {}
    for path in candidates:
        relative = path.relative_to(ROOT).as_posix()
        folded = relative.casefold()
        if relative in by_relative:
            continue
        if folded in by_casefold and by_casefold[folded] != relative:
            raise ReleaseValidationError(
                "case-insensitive archive path collision: "
                f"{by_casefold[folded]!r} and {relative!r}"
            )
        by_relative[relative] = path
        by_casefold[folded] = relative

    return [by_relative[key] for key in sorted(by_relative, key=str.casefold)]


def _validate_text(text: str, label: str) -> None:
    for description, pattern in (*SECRET_PATTERNS, *PERSONAL_METADATA_PATTERNS):
        if pattern.search(text):
            raise ReleaseValidationError(f"{description} detected in {label}")


def _bounded_zlib_decompress(payload: bytes, label: str) -> bytes:
    decompressor = zlib.decompressobj()
    result = decompressor.decompress(payload, MAX_METADATA_CHUNK_BYTES + 1)
    if (
        len(result) > MAX_METADATA_CHUNK_BYTES
        or decompressor.unconsumed_tail
        or not decompressor.eof
    ):
        raise ReleaseValidationError(f"oversized compressed metadata in {label}")
    return result


def _scan_png_metadata(data: bytes, label: str) -> None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ReleaseValidationError(f"invalid PNG signature: {label}")

    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + length
        next_offset = payload_end + 4
        if payload_end > len(data) - 4:
            raise ReleaseValidationError(f"truncated PNG chunk in {label}")

        payload = data[payload_start:payload_end]
        expected_crc = struct.unpack(">I", data[payload_end:next_offset])[0]
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ReleaseValidationError(f"PNG CRC mismatch in {label}")

        metadata = b""
        if chunk_type == b"tEXt":
            metadata = payload
        elif chunk_type == b"zTXt":
            if b"\x00" not in payload:
                raise ReleaseValidationError(f"malformed zTXt chunk in {label}")
            _, compressed = payload.split(b"\x00", 1)
            if len(compressed) < 2 or compressed[:1] != b"\x00":
                raise ReleaseValidationError(f"unsupported zTXt compression in {label}")
            metadata = _bounded_zlib_decompress(compressed[1:], label)
        elif chunk_type == b"iTXt":
            if b"\x00" not in payload:
                raise ReleaseValidationError(f"malformed iTXt chunk in {label}")
            _, remainder = payload.split(b"\x00", 1)
            if len(remainder) >= 2 and b"\x00" in remainder[2:]:
                compression_flag = remainder[:1]
                compression_method = remainder[1:2]
                if compression_flag not in {b"\x00", b"\x01"} or compression_method != b"\x00":
                    raise ReleaseValidationError(f"unsupported iTXt compression in {label}")
                remainder = remainder[2:]
                _, remainder = remainder.split(b"\x00", 1)
                if b"\x00" in remainder:
                    _, text_payload = remainder.split(b"\x00", 1)
                    metadata = (
                        _bounded_zlib_decompress(text_payload, label)
                        if compression_flag == b"\x01"
                        else text_payload
                    )
                else:
                    raise ReleaseValidationError(f"malformed iTXt translated keyword in {label}")
            else:
                raise ReleaseValidationError(f"malformed iTXt language tag in {label}")

        if metadata:
            _validate_text(metadata.decode("utf-8", errors="ignore"), f"PNG metadata: {label}")
        offset = next_offset
        if chunk_type == b"IEND":
            return

    raise ReleaseValidationError(f"PNG has no valid IEND chunk: {label}")


def _scan_docx(data: bytes, label: str) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data), "r")
    except zipfile.BadZipFile as exc:
        raise ReleaseValidationError(f"invalid DOCX container: {label}") from exc

    with archive:
        total_uncompressed = sum(info.file_size for info in archive.infolist())
        if total_uncompressed > MAX_DOCX_UNCOMPRESSED_BYTES:
            raise ReleaseValidationError(
                f"DOCX expands beyond {MAX_DOCX_UNCOMPRESSED_BYTES} bytes: {label}"
            )
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ReleaseValidationError(f"unsafe DOCX member {info.filename!r}: {label}")
            if info.is_dir():
                continue
            if info.file_size > MAX_FILE_BYTES:
                raise ReleaseValidationError(f"oversized DOCX member {info.filename!r}: {label}")
            if member.suffix.lower() in {".xml", ".rels", ".txt", ".json", ".html", ".htm"}:
                payload = archive.read(info)
                _validate_text(
                    payload.decode("utf-8", errors="ignore"),
                    f"{label}!/{info.filename}",
                )


def _validate_file(path: Path) -> ReleaseFile:
    try:
        relative_native = path.relative_to(ROOT)
    except ValueError as exc:
        raise ReleaseValidationError(f"candidate is outside repository root: {path}") from exc

    relative = PurePosixPath(relative_native.as_posix())
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseValidationError(f"unsafe candidate path: {relative}")
    if path.is_symlink():
        raise ReleaseValidationError(f"symbolic links are not allowed: {relative}")

    resolved_root = ROOT.resolve()
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ReleaseValidationError(f"resolved path leaves repository root: {relative}") from exc

    lower_name = path.name.lower()
    if lower_name in SENSITIVE_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES:
        raise ReleaseValidationError(f"sensitive filename is not allowed: {relative}")

    size_bytes = path.stat().st_size
    if size_bytes > MAX_FILE_BYTES:
        raise ReleaseValidationError(
            f"file exceeds the 10 MiB public-release limit: {relative} ({size_bytes} bytes)"
        )

    data = path.read_bytes()
    if len(data) != size_bytes:
        raise ReleaseValidationError(f"file changed while being read: {relative}")

    suffix = path.suffix.lower()
    if suffix not in BINARY_RELEASE_SUFFIXES:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReleaseValidationError(
                f"public text file is not valid UTF-8: {relative}"
            ) from exc
        if "\r" in text:
            raise ReleaseValidationError(
                f"public text file contains a carriage return; use canonical LF: {relative}"
            )
        _validate_text(text, relative.as_posix())
    else:
        _validate_text(data.decode("utf-8", errors="ignore"), relative.as_posix())

    if suffix == ".docx":
        _scan_docx(data, relative.as_posix())
    elif suffix == ".png":
        _scan_png_metadata(data, relative.as_posix())

    return ReleaseFile(
        source=path,
        relative=relative,
        size_bytes=size_bytes,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def validate_release_files(paths: Sequence[Path]) -> list[ReleaseFile]:
    release_files = [_validate_file(path) for path in paths]
    archive_names = [item.archive_name for item in release_files]
    if len(archive_names) != len(set(archive_names)):
        raise ReleaseValidationError("duplicate archive paths detected")
    return release_files


def _manifest_bytes(release_files: Sequence[ReleaseFile]) -> bytes:
    manifest = {
        "schema_version": 1,
        "algorithm": "sha256",
        "archive_root": f"{ARCHIVE_ROOT}/",
        "manifest_note": "The manifest does not recursively hash itself.",
        "policy": {
            "maximum_file_bytes": MAX_FILE_BYTES,
            "excluded": [
                "Git history and local environments",
                "runtime data, artifacts, caches, checkpoints, external sources, logs, and temporary files",
                "raw, processed, restricted, and private-evaluation data payloads",
                "restricted NOVA-Human acquisition payloads and live storage links",
                "granular ReWeaver GCD-TS and SynBody configuration, manifests, and reports",
                "exact FreeSewing and GarmentCode generated-geometry manifests",
                "obsolete structure_prior_pattern_bridge schematic",
                "secrets, personal absolute paths, sensitive metadata, and symbolic links",
            ],
        },
        "file_count": len(release_files),
        "total_bytes": sum(item.size_bytes for item in release_files),
        "files": [
            {
                "path": item.relative.as_posix(),
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in release_files
        ],
    }
    return (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    info.flag_bits |= 0x800
    return info


def _write_archive(path: Path, release_files: Sequence[ReleaseFile]) -> bytes:
    manifest_bytes = _manifest_bytes(release_files)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in release_files:
            data = item.source.read_bytes()
            if len(data) != item.size_bytes or hashlib.sha256(data).hexdigest() != item.sha256:
                raise ReleaseValidationError(f"source changed during archive build: {item.relative}")
            archive.writestr(_zip_info(item.archive_name), data)
        archive.writestr(
            _zip_info(f"{ARCHIVE_ROOT}/{MANIFEST_NAME}"),
            manifest_bytes,
        )
    return manifest_bytes


def _verify_archive(
    path: Path,
    release_files: Sequence[ReleaseFile],
    manifest_bytes: bytes,
) -> None:
    expected = {item.archive_name: item for item in release_files}
    manifest_archive_name = f"{ARCHIVE_ROOT}/{MANIFEST_NAME}"
    expected_names = set(expected) | {manifest_archive_name}

    with zipfile.ZipFile(path, "r") as archive:
        actual_names = set(archive.namelist())
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ReleaseValidationError(f"archive entry mismatch; missing={missing}, extra={extra}")
        if archive.read(manifest_archive_name) != manifest_bytes:
            raise ReleaseValidationError("archive manifest content mismatch")
        for archive_name, item in expected.items():
            payload = archive.read(archive_name)
            if len(payload) != item.size_bytes:
                raise ReleaseValidationError(f"archive size mismatch: {item.relative}")
            if hashlib.sha256(payload).hexdigest() != item.sha256:
                raise ReleaseValidationError(f"archive SHA256 mismatch: {item.relative}")


def _resolve_output(value: Path) -> Path:
    path = value if value.is_absolute() else ROOT / value
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ReleaseValidationError("output must remain inside the repository root") from exc
    if path.suffix.lower() != ".zip":
        raise ReleaseValidationError("output filename must end in .zip")
    return path


def _resolve_manifest_output(value: Path) -> Path:
    path = value if value.is_absolute() else ROOT / value
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ReleaseValidationError("manifest output must remain inside the repository root") from exc
    if path.suffix.lower() != ".json":
        raise ReleaseValidationError("manifest output filename must end in .json")
    expected = (ROOT / MANIFEST_NAME).resolve()
    if path != expected:
        raise ReleaseValidationError(
            f"manifest output must be the non-recursive root manifest: {MANIFEST_NAME}"
        )
    return path


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_archive(output: Path, release_files: Sequence[ReleaseFile], overwrite: bool) -> bytes:
    if output.exists() and not overwrite:
        raise ReleaseValidationError(
            f"output already exists: {output.relative_to(ROOT)}; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        manifest_bytes = _write_archive(temporary_path, release_files)
        _verify_archive(temporary_path, release_files, manifest_bytes)
        if output.exists() and not overwrite:
            raise ReleaseValidationError(f"output appeared during build: {output.relative_to(ROOT)}")
        os.replace(temporary_path, output)
        return manifest_bytes
    finally:
        temporary_path.unlink(missing_ok=True)


def _format_mib(byte_count: int) -> str:
    return f"{byte_count / (1024 * 1024):.2f} MiB"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"ZIP path relative to repository root (default: {DEFAULT_OUTPUT.as_posix()})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the public allowlist without creating a ZIP",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the exact output ZIP if it already exists",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(MANIFEST_NAME),
        help=(
            "external manifest path relative to the repository root "
            f"(default: {MANIFEST_NAME})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = _resolve_output(args.output)
        manifest_output = _resolve_manifest_output(args.manifest_output)
        if manifest_output == output:
            raise ReleaseValidationError("ZIP output and manifest output must be different files")
        release_files = validate_release_files(collect_candidate_paths())
        total_bytes = sum(item.size_bytes for item in release_files)
        print(f"Validated {len(release_files)} public files ({_format_mib(total_bytes)}).")
        if args.check_only:
            print("Check-only mode: no archive was created.")
            return 0
        if manifest_output.exists() and not args.overwrite:
            raise ReleaseValidationError(
                f"manifest output already exists: {manifest_output.relative_to(ROOT)}; "
                "pass --overwrite to replace it"
            )
        manifest_bytes = build_archive(output, release_files, overwrite=args.overwrite)
        _write_atomic_bytes(manifest_output, manifest_bytes)
        if manifest_output.read_bytes() != manifest_bytes:
            raise ReleaseValidationError("external manifest content mismatch after write")
        print(f"Created {output.relative_to(ROOT)}")
        print(f"Archive root: {ARCHIVE_ROOT}/")
        print(f"Manifest: {ARCHIVE_ROOT}/{MANIFEST_NAME}")
        print(f"External manifest: {manifest_output.relative_to(ROOT)}")
        return 0
    except (OSError, ReleaseValidationError, zipfile.BadZipFile, zlib.error) as exc:
        print(f"Release build failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
