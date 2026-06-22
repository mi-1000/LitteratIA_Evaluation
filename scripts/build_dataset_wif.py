import glob
import json
import os
import re

from tqdm import tqdm

OUTPUT_DIR = os.path.join('..', 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'wif_goteborg_dataset.json')
SOURCE_DIR = os.path.join(OUTPUT_DIR, 'franska-elevtexter', 'txt')
files = glob.glob(os.path.join(SOURCE_DIR, '**', '*.txt'), recursive=True)

if not os.path.exists(SOURCE_DIR):
    raise FileNotFoundError(f"Directory {SOURCE_DIR} does not exist.")

def get_student_grade_number(file_path: str) -> int:
    """Extract the grade number of the student from the file path. Assumes that the grade is indicated in the second-to-last directory name (e.g., 'ak6' for grade 6)."""
    try:
        grade = int(file_path.split('/')[-2][-1])
        return grade
    except (ValueError, IndexError):
        raise ValueError(f"Could not extract grade number from file path: {file_path}")

def extract_written_production(file_path: str) -> str:
    """Extract the meaningful textual content of a written production."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.replace('\ufeff', '') # Remove BOM
        text = re.sub(r'^\[.*\]$', '', text, flags = re.MULTILINE) # Remove metadata lines between brackets
        text = text.strip()

    return text

def save_dataset():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        file_list = []
        for file in tqdm(files):
            content = extract_written_production(file)
            grade = get_student_grade_number(file)
            file_list.append({file: {"grade": grade, "content": content}})
        json.dump(file_list, f_out, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(file_list)} files in {OUTPUT_FILE}.")

if __name__ == "__main__":
    save_dataset()