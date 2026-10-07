"""
Rename case_XXXXX.txt files to their proper case numbers.
Reads the "Дугаар" (Number) line from each file and uses that as the filename.
"""
import os
import re

CASE_DIR = "/Users/purevjavgerel/Desktop/analysis/research/analysis/cases"

renamed = 0
skipped = 0
errors = []

for fname in sorted(os.listdir(CASE_DIR)):
    if not fname.startswith("case_") or not fname.endswith(".txt"):
        continue

    filepath = os.path.join(CASE_DIR, fname)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Read first 20 lines to find the case number
            for i, line in enumerate(f):
                if i > 25:
                    break
                if "Дугаар" in line:
                    # Format: "Дугаар\t001/ХТ2022/00508"
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        case_num = parts[-1].strip()
                        # Clean for filename: replace / with _
                        clean_name = case_num.replace("/", "_").replace(" ", "_")
                        new_filename = f"{clean_name}.txt"
                        new_filepath = os.path.join(CASE_DIR, new_filename)

                        if os.path.exists(new_filepath):
                            print(f"  SKIP (exists): {fname} -> {new_filename}")
                            skipped += 1
                        else:
                            os.rename(filepath, new_filepath)
                            print(f"  RENAMED: {fname} -> {new_filename}")
                            renamed += 1
                    break
            else:
                errors.append(f"{fname}: No 'Дугаар' line found")
    except Exception as e:
        errors.append(f"{fname}: {e}")

print(f"\n{'='*50}")
print(f"Renamed: {renamed}")
print(f"Skipped (already exists): {skipped}")
print(f"Errors: {len(errors)}")
for err in errors:
    print(f"  ⚠ {err}")
