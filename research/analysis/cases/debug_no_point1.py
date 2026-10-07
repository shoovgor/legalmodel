"""
Debug export for the operative-section/point1 extraction failures (the
76 "no_point1" first_instance cases). Splits failures into two distinct
stages, since they need different fixes:

  stage 1: get_operative_section finds neither 'ТОГТООХ нь:' nor
           'ЗАХИРАМЖЛАХ нь:' anywhere in the document at all
  stage 2: one of those markers IS found, but get_point1_text can't
           find a '1.' right after it (op_text isn't None, point1 is)

Run this from the same folder as your (fixed) shuukh_parser_v2.py.
"""
import csv
import sys
from pathlib import Path
from shuukh_parser_v2 import (
    load_text, extract_metadata, get_case_level,
    get_operative_section, get_point1_text,
)

INPUT_FOLDER = Path("/Users/purevjavgerel/Desktop/analysis/research/analysis/cases")  # <- set this, same folder you feed shuukh_parser_v2.py
OUTPUT_FILE = "no_point1_debug.csv"

if not INPUT_FOLDER.exists():
    sys.exit(f"INPUT_FOLDER does not exist: {INPUT_FOLDER.resolve()}  -- fix the path above.")

total = loan_first_instance = 0
stage1_rows = []
stage2_rows = []

for f in INPUT_FOLDER.rglob("*.txt"):
    total += 1
    text = load_text(f)
    warnings = []
    meta = extract_metadata(text, warnings)
    level = get_case_level(meta["court"], warnings)

    d_type = meta.get("dispute_type") or ""
    if "Бусад зээл" not in d_type or level != "first_instance":
        continue
    loan_first_instance += 1

    op_text, is_settlement_doc = get_operative_section(text)

    if op_text is None:
        # stage 1: neither marker found anywhere in the whole document.
        # Show the tail of the raw text - if the doc is complete, the
        # ruling should be near the end even if it's phrased differently
        # than either marker this function currently looks for.
        stage1_rows.append({
            "source_file": f.name,
            "stage": "1_no_operative_marker",
            "contains_ТОГТООХ_substring": "ТОГТООХ" in text.upper(),
            "contains_ЗАХИРАМЖЛАХ_substring": "ЗАХИРАМЖЛАХ" in text.upper(),
            "text_length": len(text),
            "text_tail": text[-600:],
        })
        continue

    point1 = get_point1_text(op_text)
    if point1 is None:
        # stage 2: marker was found, but no '1.' shows up right after it.
        # Show the start of op_text - whatever's sitting where '1.'
        # should be is the actual phrasing/numbering variant to handle.
        stage2_rows.append({
            "source_file": f.name,
            "stage": "2_no_point1_after_marker",
            "is_settlement_doc_marker": is_settlement_doc,
            "op_text_start": op_text[:400],
        })

print(f"scanned:                       {total} .txt files")
print(f"loan + first_instance:         {loan_first_instance}")
print(f"stage 1 (no marker at all):    {len(stage1_rows)}")
print(f"stage 2 (marker, no '1.'):     {len(stage2_rows)}")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as out:
    fieldnames = [
        "source_file", "stage",
        "contains_ТОГТООХ_substring", "contains_ЗАХИРАМЖЛАХ_substring",
        "text_length", "text_tail",
        "is_settlement_doc_marker", "op_text_start",
    ]
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(stage1_rows)
    w.writerows(stage2_rows)

print(f"-> wrote {OUTPUT_FILE}")