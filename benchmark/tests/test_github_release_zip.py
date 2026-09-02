from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from benchmark.scripts import build_github_release_zip as release


class GithubReleaseZipTests(unittest.TestCase):
    def test_crlf_text_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "sample.json"
            candidate.write_bytes(b'{\r\n  "ok": true\r\n}\r\n')

            with patch.object(release, "ROOT", root):
                with self.assertRaisesRegex(release.ReleaseValidationError, "canonical LF"):
                    release._validate_file(candidate)

    def test_root_manifest_matches_embedded_manifest_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "sample.json"
            candidate.write_bytes(b'{\n  "ok": true\n}\n')
            archive_path = root / "release.zip"
            manifest_path = root / release.MANIFEST_NAME

            with (
                patch.object(release, "ROOT", root),
                patch.object(release, "collect_candidate_paths", return_value=[candidate]),
            ):
                result = release.main(
                    [
                        "--output",
                        str(archive_path),
                        "--manifest-output",
                        str(manifest_path),
                    ]
                )

            self.assertEqual(result, 0)
            external = manifest_path.read_bytes()
            with zipfile.ZipFile(archive_path, "r") as archive:
                embedded = archive.read(f"{release.ARCHIVE_ROOT}/{release.MANIFEST_NAME}")
            self.assertEqual(external, embedded)
            parsed = json.loads(external.decode("utf-8"))
            self.assertNotIn(release.MANIFEST_NAME, {item["path"] for item in parsed["files"]})

    def test_phase_one_allowlist_includes_gitattributes_but_not_root_manifest(self):
        self.assertIn(Path(".gitattributes"), release.ROOT_FILES)
        self.assertNotIn(Path(release.MANIFEST_NAME), release.ROOT_FILES)


if __name__ == "__main__":
    unittest.main()
