"""
Reconcile raw .txt file count against what actually makes it into
output.csv. Buckets every file into exactly one category so we can see
precisely where the gap between "files on disk" and "rows in the
table" is coming from, instead of guessing.

Run this from the same folder as your (fixed) shuukh_parser_v2.py.
"""
from pathlib import Path
from collections import Counter, defaultdict
from shuukh_parser_v2 import load_text, extract_metadata

INPUT_FOLDER = Path("/Users/purevjavgerel/Desktop/analysis/research/analysis/cases")  # <- set this: the TOP folder containing all raw .txt files (rglob searches subfolders too, so point it above case_details/ if you've got multiple dispute-type/level subfolders under there)

if not INPUT_FOLDER.exists():
    raise SystemExit(f"INPUT_FOLDER does not exist: {INPUT_FOLDER.resolve()}  -- fix the path above.")

total = 0
bucket_counts = Counter()
seen_case_ids = {}
dupe_examples = []
wrong_dispute_examples = defaultdict(list)
no_metadata_examples = []

for f in INPUT_FOLDER.rglob("*.txt"):
    total += 1
    text = load_text(f)
    warnings = []
    meta = extract_metadata(text, warnings)
    case_id = meta.get("case_id")
    d_type = meta.get("dispute_type") or ""

    if not case_id:
        bucket_counts["no_case_id_extracted"] += 1
        no_metadata_examples.append(f.name)
        continue

    if "Бусад зээл" not in d_type:
        bucket_counts["wrong_dispute_type"] += 1
        wrong_dispute_examples[d_type or "(empty)"].append(f.name)
        continue

    if case_id in seen_case_ids:
        bucket_counts["duplicate_case_id"] += 1
        dupe_examples.append((case_id, seen_case_ids[case_id], f.name))
        continue

    seen_case_ids[case_id] = f.name
    bucket_counts["kept"] += 1

print(f"total .txt files scanned:     {total}")
for k, v in bucket_counts.most_common():
    print(f"  {k:24s} {v}")
print()
print(f"expected rows in output.csv:  {bucket_counts['kept']}")
print("(compare this to 7821 - if it matches, the gap is fully explained below;")
print(" if 'kept' is higher than 7821, something else is happening downstream")
print(" of parsing, not during it.)")
print()

if wrong_dispute_examples:
    print("--- dispute_type values seen among excluded files, by frequency ---")
    for dt, files in sorted(wrong_dispute_examples.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {len(files):5d}  {dt!r}  (e.g. {files[0]})")
    print()

if dupe_examples:
    print("--- duplicate case_id examples (same real case, different files) ---")
    for cid, first_file, dupe_file in dupe_examples[:10]:
        print(f"  {cid}: kept {first_file}, duplicate {dupe_file}")
    print()

if no_metadata_examples:
    print("--- files with no case_id extracted at all (sample) ---")
    for name in no_metadata_examples[:10]:
        print(f"  {name}")