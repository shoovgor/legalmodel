"""
Removes all .txt case files in the cases folder whose "Маргааны төрөл" field
is NOT "Бусад зээл".  Files with no dispute-type field are listed as warnings
and left untouched.

Run with --dry-run first to preview what would be deleted.
"""

import os
import sys

CASE_DIR = "/Users/purevjavgerel/Desktop/analysis/research/analysis/cases"
KEEP_DISPUTE = "Бусад зээл"
DRY_RUN = "--dry-run" in sys.argv

deleted = 0
kept = 0
no_field = []

files = [f for f in os.listdir(CASE_DIR) if f.endswith(".txt")]
total = len(files)
print(f"Scanning {total} .txt files in {CASE_DIR} ...")
print(f"Mode: {'DRY RUN (no files deleted)' if DRY_RUN else 'LIVE (files will be deleted)'}")
print("=" * 60)

for i, fname in enumerate(files, 1):
    filepath = os.path.join(CASE_DIR, fname)
    dispute_type = None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for j, line in enumerate(f):
                if j > 40:
                    break
                if line.startswith("Маргааны төрөл"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        # Strip trailing comma/space that the site sometimes adds
                        dispute_type = parts[-1].strip().rstrip(",").strip()
                    break
    except Exception as e:
        no_field.append(f"{fname}: read error – {e}")
        continue

    if dispute_type is None:
        no_field.append(f"{fname}: no 'Маргааны төрөл' field found")
        continue

    if dispute_type == KEEP_DISPUTE:
        kept += 1
    else:
        if not DRY_RUN:
            os.remove(filepath)
        else:
            print(f"  [would delete] {fname}  ({dispute_type})")
        deleted += 1

    if i % 1000 == 0:
        print(f"  Progress: {i}/{total} …")

print("=" * 60)
print(f"Kept  (Бусад зээл):  {kept}")
print(f"{'Would delete' if DRY_RUN else 'Deleted'} (other types): {deleted}")
print(f"Warnings (no field): {len(no_field)}")
if no_field:
    for w in no_field[:20]:
        print(f"  ⚠ {w}")
    if len(no_field) > 20:
        print(f"  … and {len(no_field) - 20} more")
