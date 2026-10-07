"""
Debug export for the remaining stage-1 no_point1 failures (get_operative
_section finds neither 'ТОГТООХ нь:' nor 'ЗАХИРАМЖЛАХ нь:' anywhere).

The previous debug script's blind last-600-chars tail was too short for
many of these documents (text_length ranged up to 47k chars) - this
version searches for the actual position of either bare word (even
without the full 'нь:' match) and shows real context around it, so we
can see exactly why the full pattern doesn't match. For the few cases
where neither word appears anywhere at all, it shows a much bigger tail
(2000 chars) instead of guessing.

Run this from the same folder as your (fixed) shuukh_parser_v2.py.
"""
import csv
import re
import sys
from pathlib import Path
from shuukh_parser_v2 import load_text, extract_metadata, get_case_level, get_operative_section

INPUT_FOLDER = Path("/Users/purevjavgerel/Desktop/analysis/research/analysis/cases")  # <- set this, same folder you feed shuukh_parser_v2.py
OUTPUT_FILE = "stage1_debug.csv"

if not INPUT_FOLDER.exists():
    sys.exit(f"INPUT_FOLDER does not exist: {INPUT_FOLDER.resolve()}  -- fix the path above.")

rows = []
for f in INPUT_FOLDER.rglob("*.txt"):
    text = load_text(f)
    warnings = []
    meta = extract_metadata(text, warnings)
    level = get_case_level(meta["court"], warnings)
    d_type = meta.get("dispute_type") or ""
    if "Бусад зээл" not in d_type or level != "first_instance":
        continue

    op_text, _ = get_operative_section(text)
    if op_text is not None:
        continue  # already resolved by get_operative_section, not a stage-1 case

    # find the actual position of either bare word, case-insensitive -
    # regardless of whether the full 'нь:' pattern matches
    m = re.search(r"ТОГТООХ|ЗАХИРАМЖЛАХ", text, re.I)
    if m:
        ctx_start = max(0, m.start() - 150)
        context = text[ctx_start:m.start() + 400]
    else:
        context = text[-2000:]  # neither word anywhere - show a much bigger tail

    rows.append({
        "source_file": f.name,
        "word_found": m.group(0) if m else "NEITHER",
        "context": context,
    })

print(f"stage 1 cases found: {len(rows)}")
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as out:
    w = csv.DictWriter(out, fieldnames=["source_file", "word_found", "context"])
    w.writeheader()
    w.writerows(rows)
print(f"-> wrote {OUTPUT_FILE}")