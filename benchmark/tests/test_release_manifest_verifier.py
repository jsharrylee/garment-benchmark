from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import verify_release_manifest as verifier


def _write_manifest(root: Path, files: dict[str, bytes]) -> None:
    entries = []
    for path in sorted(files, key=str.casefold):
        data = files[path]
        entries.append(
            {
                "path": path,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    payload = {
        "schema_version": 1,
        "algorithm": "sha256",
        "archive_root": "game-garment-benchmark/",
        "file_count": len(entries),
        "total_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
    }
    (root / verifier.MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


class ReleaseManifestVerifierTests(unittest.TestCase):
    def test_valid_extracted_tree_passes_in_filesystem_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"canonical LF\n"
            (root / "sample.txt").write_bytes(data)
            _write_manifest(root, {"sample.txt": data})

            mode, count, problems = verifier.verify(root)

            self.assertEqual(mode, "filesystem")
            self.assertEqual(count, 1)
            self.assertEqual(problems, [])

    def test_git_mode_separates_head_blob_from_worktree_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b"canonical LF\n"
            target = root / "sample.txt"
            target.write_bytes(original)
            _write_manifest(root, {"sample.txt": original})
            _git(root, "init")
            _git(root, "config", "user.name", "Verifier Test")
            _git(root, "config", "user.email", "verifier@example.invalid")
            _git(root, "add", "sample.txt", verifier.MANIFEST_NAME)
            _git(root, "commit", "-m", "fixture")

            mode, _, initial = verifier.verify(root)
            self.assertEqual(mode, "git")
            self.assertEqual(initial, [])

            target.write_bytes(b"changed value\n")
            _, _, mutated = verifier.verify(root)
            self.assertTrue(any(value.startswith("WORKTREE_") for value in mutated))
            self.assertFalse(any(value.startswith("GIT_BLOB_") for value in mutated))

    def test_git_mode_rejects_unmanifested_tracked_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"one\n"
            (root / "sample.txt").write_bytes(data)
            _write_manifest(root, {"sample.txt": data})
            _git(root, "init")
            _git(root, "config", "user.name", "Verifier Test")
            _git(root, "config", "user.email", "verifier@example.invalid")
            _git(root, "add", "sample.txt", verifier.MANIFEST_NAME)
            _git(root, "commit", "-m", "fixture")
            (root / "extra.txt").write_bytes(b"extra\n")
            _git(root, "add", "extra.txt")

            _, _, problems = verifier.verify(root)

            self.assertIn("TRACKED_UNDECLARED extra.txt", problems)

    def test_unsafe_windows_absolute_path_is_rejected(self):
        with self.assertRaises(verifier.ManifestVerificationError):
            verifier._validate_relative_path("C:/escape.txt")

    def test_unsafe_windows_drive_relative_path_is_rejected(self):
        with self.assertRaises(verifier.ManifestVerificationError):
            verifier._validate_relative_path("C:escape.txt")


if __name__ == "__main__":
    unittest.main()
