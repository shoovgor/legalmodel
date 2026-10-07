"""
Debug export: for first_instance loan cases where civil_outcome came
back None, dump source_file + parse_warnings + point1 text so we can
see the real phrasing and figure out where denied/partial cases are
hiding (vs. genuinely not occurring in this data).

Run this from the same folder as your (fixed) shuukh_parser_v2.py.
"""
import csv
import sys
from pathlib import Path
from shuukh_parser_v2 import (
    load_text, extract_metadata, get_case_level,
    get_operative_section, get_point1_text, classify_outcome,
)

INPUT_FOLDER = Path("/Users/purevjavgerel/Desktop/analysis/research/analysis/cases")  # <- set this, same folder you feed shuukh_parser_v2.py
OUTPUT_FILE = "unclassified_debug.csv"

if not INPUT_FOLDER.exists():
    sys.exit(f"INPUT_FOLDER does not exist: {INPUT_FOLDER.resolve()}  -- fix the path above.")

total = loan_first_instance = unclassified = 0
rows = []

for f in INPUT_FOLDER.rglob("*.txt"):
    total += 1
    text = load_text(f)
    warnings = []
    meta = extract_metadata(text, warnings)
    level = get_case_level(meta["court"], warnings)

    d_type = meta.get("dispute_type") or ""
    if "Бусад зээл" not in d_type:
        continue
    if level != "first_instance":
        continue
    loan_first_instance += 1

    op_text, is_settlement_doc = get_operative_section(text)
    point1 = get_point1_text(op_text)
    outcome = classify_outcome(point1, d_type, is_settlement_doc, warnings)

    if outcome is None:
        unclassified += 1
        rows.append({
            "source_file": f.name,
            "warnings": "|".join(warnings),
            "point1_text": point1 or "",
        })

print(f"scanned:               {total} .txt files")
print(f"loan + first_instance: {loan_first_instance}")
print(f"unclassified:          {unclassified}")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as out:
    w = csv.DictWriter(out, fieldnames=["source_file", "warnings", "point1_text"])
    w.writeheader()
    w.writerows(rows)

print(f"-> wrote {OUTPUT_FILE}")