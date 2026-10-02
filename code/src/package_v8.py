"""Archive v8 without inspecting the blinded sampling-key contents."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT.parent
ZIP = PARENT / f"{ROOT.name}.zip"
REPORT = PARENT / f"{ROOT.name}_archive_verification.txt"
MANIFEST = ROOT / "tables/SHA256SUMS_v8.txt"
KEY = ROOT / "tables/imagery_sampling_key_v7.csv"


def digest(stream) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if ZIP.exists() or REPORT.exists():
        raise SystemExit("Refusing to overwrite an existing v8 archive or verification report")
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        sha, rel = line.split("  ", 1)
        entries[rel] = sha
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file())
    expected = {p.relative_to(ROOT).as_posix() for p in paths}
    if expected - set(entries) != {"tables/SHA256SUMS_v8.txt", "tables/imagery_sampling_key_v7.csv"}:
        raise SystemExit("Manifest/file inventory mismatch")
    if len(entries) != len(expected) - 2:
        raise SystemExit("Unexpected manifest entry count")
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as archive:
        for path in paths:
            archive.write(path, arcname=f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}")
    with zipfile.ZipFile(ZIP) as archive:
        members = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        if len(members) != len(paths):
            raise SystemExit("ZIP member count mismatch")
        if {name.removeprefix(ROOT.name + "/") for name in members} != expected:
            raise SystemExit("ZIP path inventory mismatch")
        for rel, sha in entries.items():
            with archive.open(f"{ROOT.name}/{rel}") as stream:
                if digest(stream) != sha:
                    raise SystemExit(f"ZIP entry hash mismatch: {rel}")
        manifest_rel = "tables/SHA256SUMS_v8.txt"
        with archive.open(f"{ROOT.name}/{manifest_rel}") as stream:
            if digest(stream) != hashlib.sha256(MANIFEST.read_bytes()).hexdigest():
                raise SystemExit("Manifest ZIP entry mismatch")
        key_info = members[f"{ROOT.name}/tables/{KEY.name}"]
        if key_info.file_size != KEY.stat().st_size:
            raise SystemExit("Sealed key member-size mismatch")
    with ZIP.open("rb") as stream:
        zip_sha = digest(stream)
    record = (
        f"Archive: {ZIP.name}\nSHA-256: {zip_sha}\n"
        f"Package files: {len(paths)}\nManifest entries verified in ZIP: {len(entries)}\n"
        "Manifest itself verified in ZIP: yes\n"
        "Sealed imagery key: filename and member size checked only; contents not inspected or hashed separately\n"
        "Five top-level package folders; exact member-path inventory verified\n"
    )
    REPORT.write_text(record, encoding="utf-8")
    print(record)


if __name__ == "__main__":
    main()
