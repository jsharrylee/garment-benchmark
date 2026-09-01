# Publishing the Verified GitHub Release

This workflow publishes only the validated contents of the public ZIP. It does not expose the working directory or its existing Git history.

## 1. Review the final portfolio

Open and review:

```text
output/docx/semantic_pattern_bridge_portfolio_en.docx
```

Confirm that the document describes the current 2D semantic parser separately from four-view retrieval, guarded editing, and future parametric generation.

## 2. Run the non-writing release check

From the project root:

```powershell
python -m benchmark.scripts.build_github_release_zip --check-only
```

This validates the explicit allowlist, file sizes, symbolic links, secret patterns, personal absolute paths, and DOCX metadata without creating an archive.

## 3. Build the public ZIP

```powershell
python -m benchmark.scripts.build_github_release_zip
```

The default output is:

```text
output/releases/game-garment-benchmark-public.zip
```

If the file already exists, the builder refuses to overwrite it. Replace it intentionally only after reviewing the new contents:

```powershell
python -m benchmark.scripts.build_github_release_zip --overwrite
```

The builder verifies every archive member against `RELEASE_MANIFEST.sha256.json` before moving the completed ZIP into place.

## 4. Create an empty GitHub repository

Create a new public repository on GitHub. Do not initialize it with a README, `.gitignore`, or license, because those files are already present in the release.

Extract the ZIP into a new local directory **outside this working repository**. Do not upload the ZIP itself as the repository contents, and do not create a nested Git repository under the private working tree.

For example, from the current project root on Windows:

```powershell
New-Item -ItemType Directory -Path ..\game-garment-benchmark-public-upload
python -m zipfile -e output/releases/game-garment-benchmark-public.zip ..\game-garment-benchmark-public-upload
Set-Location ..\game-garment-benchmark-public-upload\game-garment-benchmark
```

Choose another empty sibling directory if that example path already exists.

## 5. Create a clean public Git history

Use the GitHub noreply address shown in your GitHub email settings if you do not want a personal email embedded in the commit metadata.

```powershell
git init
git branch -M main
git config user.name "<GitHub display name>"
git config user.email "<GitHub noreply email>"
git add .
git status --short
git commit -m "Initial public release"
```

Before committing, verify that `git status --short` does not show datasets, checkpoints, environments, caches, `tmp`, `external`, or a nested `.git` directory.

## 6. Push the new repository

Replace the placeholders below with the intended GitHub account and repository name.

```powershell
git remote add origin https://github.com/<USER>/<REPOSITORY>.git
git push -u origin main
```

After the first push, verify on GitHub that:

- the English `README.md` renders correctly;
- `output/docx/` contains the English portfolio;
- the two public figures use English labels;
- `RELEASE_MANIFEST.sha256.json` is present at the repository root;
- no `data/raw`, `data/processed`, `data/restricted`, `checkpoints`, `external`, `artifacts`, `cache`, or `tmp` directory is present;
- the commit author uses the intended noreply address.

## 7. Understand the license notice

`LICENSE_NOTICE.md` describes the current rights status; it is not itself an open-source license. `THIRD_PARTY_NOTICES.md` documents upstream sources, license terms, changes, and exclusions. Add a separate project license only after confirming which project-authored material you have the right to license. Upstream conditions continue to govern third-party material.
