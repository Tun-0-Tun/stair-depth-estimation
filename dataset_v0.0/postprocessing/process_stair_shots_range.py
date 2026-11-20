#!/usr/bin/env python3
"""Process a selected inclusive shot-ID range from shots.xlsx.

For each shot (excluding lost shot_id=1):
  1. Ensure raw_data_dir/stair{stairs_id}.7z is extracted to
     extraction_dir/stair{stairs_id}/.
  2. Run fast_bag_extractor.py with --name s{shot_id}.
  3. Remove extraction_dir/stair{stairs_id} only after every non-lost shot
     for that stairs_id in the workbook has completed successfully.

Requires openpyxl. For .7z extraction, either a 7-Zip executable (7z/7zz/7za)
available on PATH, or the optional Python package py7zr.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

LOST_SHOT_ID = 1
EXPECTED_HEADERS = {
    "shot id": ("shot id", "shot_id", "shotid"),
    "file name": ("file name", "filename", "file_name"),
    "stairs id": ("stairs id", "stairs_id", "stair id", "stair_id"),
}


def norm_header(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def parse_int(value: Any, label: str, excel_row: int) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Excel row {excel_row}: missing {label}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        try:
            number_float = float(value)
            if not number_float.is_integer():
                raise ValueError
            number = int(number_float)
        except (TypeError, ValueError):
            raise ValueError(f"Excel row {excel_row}: {label} must be an integer, got {value!r}")
    return number


def load_shots(xlsx_path: Path) -> list[dict[str, Any]]:
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"shots.xlsx not found: {xlsx_path}")
    workbook = load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    try:
        header_row = next(rows)
    except StopIteration:
        raise ValueError(f"Workbook is empty: {xlsx_path}")

    header_to_index = {norm_header(v): i for i, v in enumerate(header_row) if v is not None}
    col_indices: dict[str, int] = {}
    for required, aliases in EXPECTED_HEADERS.items():
        match = next((header_to_index[a] for a in aliases if a in header_to_index), None)
        if match is None:
            raise ValueError(
                f"Missing required column {required!r}. Found headers: {list(header_to_index)}"
            )
        col_indices[required] = match

    shots: list[dict[str, Any]] = []
    for excel_row, values in enumerate(rows, start=2):
        shot_value = values[col_indices["shot id"]] if col_indices["shot id"] < len(values) else None
        if shot_value is None or str(shot_value).strip() == "":
            continue
        shot_id = parse_int(shot_value, "shot id", excel_row)
        if shot_id == LOST_SHOT_ID:
            print(f"Skipping lost shot id {LOST_SHOT_ID} (Excel row {excel_row}).")
            continue
        file_value = values[col_indices["file name"]] if col_indices["file name"] < len(values) else None
        stairs_value = values[col_indices["stairs id"]] if col_indices["stairs id"] < len(values) else None
        if file_value is None or not str(file_value).strip():
            raise ValueError(f"Excel row {excel_row}: missing file name for shot {shot_id}")
        stairs_id = parse_int(stairs_value, "stairs id", excel_row)
        # Workbook entries should be basenames, not arbitrary paths.
        file_name = Path(str(file_value).strip()).name
        if file_name != str(file_value).strip().replace("\\", "/").split("/")[-1]:
            raise ValueError(f"Excel row {excel_row}: invalid file name {file_value!r}")
        shots.append({
            "shot_id": shot_id,
            "file_name": file_name,
            "stairs_id": stairs_id,
            "excel_row": excel_row,
        })
    workbook.close()
    shots.sort(key=lambda item: item["shot_id"])
    ids = [s["shot_id"] for s in shots]
    if len(ids) != len(set(ids)):
        duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
        raise ValueError(f"Duplicate shot id(s) in shots.xlsx: {duplicates}")
    return shots


def find_7zip(explicit: str | None) -> str | None:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p)
        found = shutil.which(explicit)
        if found:
            return found
        raise FileNotFoundError(f"Could not find 7-Zip executable: {explicit}")
    for name in ("7z", "7zz", "7za", "7z.exe", "7za.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def extract_7z(archive: Path, destination: Path, seven_zip: str | None) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    if seven_zip:
        cmd = [seven_zip, "x", "-y", f"-o{destination}", str(archive)]
        print("Extracting:", archive.name)
        subprocess.run(cmd, check=True)
        return
    try:
        import py7zr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "No 7-Zip executable found on PATH and py7zr is not installed. "
            "Install 7-Zip or run: python -m pip install py7zr; alternatively pass --seven-zip."
        ) from exc
    print("Extracting:", archive.name)
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        zf.extractall(path=destination)


def bag_path_for(extraction_dir: Path, shot: dict[str, Any]) -> Path:
    stair = f"stair{shot['stairs_id']}"
    return extraction_dir / stair / stair / shot["file_name"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", required=True, help="Path to shots.xlsx")
    parser.add_argument("--raw-data-dir", required=True, help="Directory containing stair{stairs_id}.7z archives")
    parser.add_argument("--extraction-dir", required=True, help="Parent directory for extracted stair archives")
    parser.add_argument("--output-dir", required=True, help="Output directory passed to fast_bag_extractor.py")
    parser.add_argument("--extractor", default=None, help="Path to fast_bag_extractor.py (default: next to this script)")
    parser.add_argument("--seven-zip", default=None, help="Optional path/name of 7z/7zz/7za executable")
    parser.add_argument("--first-shot-id", type=int, default=None, help="Inclusive lower shot ID limit")
    parser.add_argument("--last-shot-id", type=int, default=None, help="Inclusive upper shot ID limit")
    parser.add_argument("--keep-extracted", action="store_true", help="Do not delete extracted stair directories after processing")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without extracting/running/deleting")
    args = parser.parse_args()
    if args.first_shot_id is not None and args.last_shot_id is not None and args.first_shot_id > args.last_shot_id:
        parser.error("--first-shot-id must be <= --last-shot-id")

    xlsx_path = Path(args.xlsx).expanduser().resolve()
    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    extraction_dir = Path(args.extraction_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    extractor = Path(args.extractor).expanduser().resolve() if args.extractor else Path(__file__).resolve().with_name("fast_bag_extractor.py")

    if not raw_data_dir.is_dir():
        raise NotADirectoryError(f"raw-data directory does not exist: {raw_data_dir}")
    if not args.dry_run and not extractor.is_file():
        raise FileNotFoundError(f"fast_bag_extractor.py not found: {extractor}; pass --extractor")

    all_shots = load_shots(xlsx_path)
    if not all_shots:
        print("No shots to process after excluding shot id 1.")
        return 0

    selected = [s for s in all_shots
                if (args.first_shot_id is None or s["shot_id"] >= args.first_shot_id)
                and (args.last_shot_id is None or s["shot_id"] <= args.last_shot_id)]
    if not selected:
        print("No shots match the requested shot-id range.")
        return 0

    all_by_stair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    selected_by_stair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for shot in all_shots:
        all_by_stair[shot["stairs_id"]].append(shot)
    for shot in selected:
        selected_by_stair[shot["stairs_id"]].append(shot)

    seven_zip = find_7zip(args.seven_zip) if not args.dry_run else args.seven_zip
    output_dir.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
    succeeded: set[int] = set()
    failed: set[int] = set()

    print(f"Workbook shots (excluding shot id {LOST_SHOT_ID}): {len(all_shots)}")
    print(f"Shots selected for this run: {len(selected)}")
    print(f"Output directory: {output_dir}")

    for stairs_id in sorted(selected_by_stair):
        stair_name = f"stair{stairs_id}"
        stair_extract_dir = extraction_dir / stair_name
        archive = raw_data_dir / f"{stair_name}.7z"
        stair_shots = selected_by_stair[stairs_id]
        expected_paths = [bag_path_for(extraction_dir, s) for s in all_by_stair[stairs_id]]
        if not all(p.is_file() for p in expected_paths):
            if args.dry_run:
                print(f"[DRY RUN] Would extract {archive} -> {stair_extract_dir}")
            else:
                try:
                    extract_7z(archive, stair_extract_dir, seven_zip)
                except Exception as exc:
                    print(f"ERROR extracting {archive}: {exc}", file=sys.stderr)
                    failed.update(s["shot_id"] for s in stair_shots)
                    continue
        else:
            print(f"Already extracted: {stair_name} (all expected bag files exist).")

        for shot in stair_shots:
            bag = bag_path_for(extraction_dir, shot)
            if not bag.is_file():
                print(f"ERROR: expected bag file not found: {bag}", file=sys.stderr)
                failed.add(shot["shot_id"])
                continue
            command = [sys.executable, str(extractor), "--bag", str(bag),
                       "--out", str(output_dir), "--name", f"s{shot['shot_id']}"]
            if args.dry_run:
                print("[DRY RUN]", subprocess.list2cmdline(command))
                succeeded.add(shot["shot_id"])
                continue
            print(f"\nProcessing shot {shot['shot_id']} (stairs {stairs_id}): {bag.name}")
            try:
                subprocess.run(command, check=True)
                succeeded.add(shot["shot_id"])
            except subprocess.CalledProcessError as exc:
                print(f"ERROR: extractor failed for shot {shot['shot_id']} (exit {exc.returncode})", file=sys.stderr)
                failed.add(shot["shot_id"])

        # Delete only if all non-lost shots for this stair are included in this
        # run and all of them succeeded. Never delete data after a partial-range run.
        all_ids = {s["shot_id"] for s in all_by_stair[stairs_id]}
        selected_ids = {s["shot_id"] for s in stair_shots}
        all_processed_this_run = all_ids == selected_ids and all_ids.issubset(succeeded)
        if args.keep_extracted:
            print(f"Keeping extracted directory by request: {stair_extract_dir}")
        elif all_processed_this_run:
            if args.dry_run:
                print(f"[DRY RUN] Would delete extracted directory: {stair_extract_dir}")
            elif stair_extract_dir.exists():
                shutil.rmtree(stair_extract_dir)
                print(f"Deleted extracted directory: {stair_extract_dir}")
        else:
            remaining = sorted(all_ids - selected_ids)
            if remaining:
                print(f"Keeping {stair_extract_dir}: this run did not include all shots for this stair (not selected: {remaining}).")
            elif all_ids & failed:
                print(f"Keeping {stair_extract_dir}: at least one shot failed.")

    print("\nFinished.")
    print(f"Succeeded: {len(succeeded)}; failed: {len(failed)}")
    if failed:
        print("Failed shot IDs:", ", ".join(map(str, sorted(failed))))
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
