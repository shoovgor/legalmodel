import os
import re

CASE_DIR = "/Users/purevjavgerel/Desktop/analysis/research/analysis/cases"

# Standard pattern for proper case filenames
std_pattern = re.compile(r'^\d+[_][А-Яа-яЁёA-Za-z]{2,}\d{4}[_]\d+\.txt$|^\d+[_]\d{4}[А-Яа-яЁёA-Za-z]{2,}[_]\d+\.txt$', re.IGNORECASE)

# Pattern to check if a string is a standard case number (like 001/ХТ2022/00508)
case_num_pattern = re.compile(r'^\d+[/_][А-Яа-яЁёA-Za-z]{2,}\d{4}[/_]\d+$|^\d+[/_]\d{4}[А-Яа-яЁёA-Za-z]{2,}[/_]\d+$', re.IGNORECASE)

renamed = 0
deleted_duplicates = 0
deleted_empty = 0
errors = []

for fname in os.listdir(CASE_DIR):
    if not fname.endswith(".txt"):
        continue

    # Skip files that already have proper standard names
    if std_pattern.match(fname):
        continue

    filepath = os.path.join(CASE_DIR, fname)
    
    # If the file is completely empty, just delete it
    if os.path.getsize(filepath) == 0:
        os.remove(filepath)
        deleted_empty += 1
        continue

    dugaar = None
    index = None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 30:
                    break
                if line.startswith("Дугаар"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        dugaar = parts[-1].strip()
                if line.startswith("Хэргийн индекс"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        index = parts[-1].strip()
        
        target_name_raw = None
        
        # Prefer standard Дугаар
        if dugaar and case_num_pattern.match(dugaar):
            target_name_raw = dugaar
        # Fallback to Хэргийн индекс if Дугаар is not standard (e.g. just a number)
        elif index:
            target_name_raw = index
        elif dugaar:
            target_name_raw = dugaar # Ultimate fallback

        if target_name_raw:
            clean_name = target_name_raw.replace("/", "_").replace(" ", "_")
            new_filename = f"{clean_name}.txt"
            new_filepath = os.path.join(CASE_DIR, new_filename)

            if new_filename == fname:
                continue # Nothing to do

            if os.path.exists(new_filepath):
                # Duplicate found! The proper file already exists.
                # Delete this badly named / duplicate file to clean up.
                os.remove(filepath)
                deleted_duplicates += 1
            else:
                # Proper file doesn't exist, so rename it
                os.rename(filepath, new_filepath)
                renamed += 1
        else:
            errors.append(f"{fname}: No Дугаар or Хэргийн индекс found.")

    except Exception as e:
        errors.append(f"{fname}: Error reading file - {e}")

print(f"{'='*50}")
print(f"Renamed badly named files: {renamed}")
print(f"Deleted duplicates (proper file already existed): {deleted_duplicates}")
print(f"Deleted completely empty files: {deleted_empty}")
print(f"Errors: {len(errors)}")
if errors:
    for err in errors[:10]:
        print(f"  ⚠ {err}")
