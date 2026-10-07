"""
Debug export: point1_text for first_instance loan cases from 2019-2020
classified as 'partial', to check whether the broader partial-detection
pattern (олгож/олгон + хэрэгсэхгүй anywhere in the text) is misfiring
on older document phrasing rather than catching genuine partial grants.

45.7% of 2019 cases came back 'partial' vs 2.5-6.7% for 2024-2026 -
too large a gap to be a real shift in how courts ruled; more likely the
pattern catching something specific to older-era phrasing that recent
documents don't have.

Run this from the same folder as your (fixed) shuukh_parser_v2.py.
"""
import csv
import sys
from pathlib import Path
from shuukh_parser_v2 import (
    load_text, extract_metadata, get_case_level,
    get_operative_section, get_point1_text, classify_outcome,
)

INPUT_FOLDER = Path("/Users/purevjavgerel/Desktop/analysis/research/analysis/cases")  # <- set this
OUTPUT_FILE = "partial_2019_debug.csv"

if not INPUT_FOLDER.exists():
    sys.exit(f"INPUT_FOLDER does not exist: {INPUT_FOLDER.resolve()}  -- fix the path above.")

rows = []
for f in INPUT_FOLDER.rglob("*.txt"):
    text = load_text(f)
    warnings = []
    meta = extract_metadata(text, warnings)
    level = get_case_level(meta["court"], warnings)
    d_type = meta.get("dispute_type") or ""
    decision_date = meta.get("decision_date") or ""

    if "Бусад зээл" not in d_type or level != "first_instance":
        continue
    if not (decision_date.startswith("2019") or decision_date.startswith("2020")):
        continue

    op_text, is_settlement_doc = get_operative_section(text)
    point1 = get_point1_text(op_text) if op_text else None
    outcome = classify_outcome(point1, d_type, is_settlement_doc, warnings) if point1 else None

    if outcome == "partial":
        rows.append({
            "source_file": f.name,
            "decision_date": decision_date,
            "point1_text": point1,
        })

print(f"2019-2020 cases classified as partial: {len(rows)}")
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as out:
    w = csv.DictWriter(out, fieldnames=["source_file", "decision_date", "point1_text"])
    w.writeheader()
    w.writerows(rows)
print(f"-> wrote {OUTPUT_FILE}")