from collections import defaultdict
import json
import numpy as np
import pandas as pd
import os

SOURCE_FILE_PATH = os.path.join("..", "profs_phase1.csv")
OUTPUT_FILE_PATH = os.path.join("..", "data", "profs_phase1.json")

INPUT_STUDENT_PATH = os.path.join("..", "data", "reactions.json"),
INPUT_TEACHER_PATH = os.path.join("..", "data", "profs_phase1.json"),
OUTPUT_PATH = os.path.join("..", "data", "students_teacher_gold.json")

LABELS_COLUMNS = [
    "complete",
    "correct",
    "relevant",
    "concise",
    "scaffolding",
    "understandable"
]

SAVED_COLUMNS_STUDENTS = [ # We're not keeping every column from the database dump; some data is redundant and some is irrelevant for our purposes
    "id",
    "timestamp",
    "model_a_name",
    "model_b_name",
    "msg_index",
    "conversation_a",
    "conversation_b",
    "model_pos",
    "conv_turns",
    "system_prompt",
    "conv_a_id",
    "conv_b_id",
    "ip",
    "response_content",
    "question_content",
    "comment",
    "msg_rank",
    "rating",
    "device_type",
    "interface_lang"
] + LABELS_COLUMNS

SAVED_COLUMNS_TEACHERS = [ # We're not keeping every column from the database dump; some data is redundant and some is irrelevant for our purposes
    "id",
    "timestamp",
    "source_reaction_id",
    "annotator_id",
    "preferred_model",
    "rating",
    "comment"
] + LABELS_COLUMNS

def format_student_data_csv_to_json(input_path: str = SOURCE_FILE_PATH, output_path: str = OUTPUT_FILE_PATH, saved_columns: list[str] = SAVED_COLUMNS_STUDENTS) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found at: {input_path}")

    df = pd.read_csv(input_path)
    df = df[saved_columns]
    
    # french_mask = df["question_content"].astype(str).apply(is_french)
    # french_mask = french_mask.fillna(False)
    
    # print(f"Dropping {len(df[~french_mask])} non-French rows.")
    # df = df[french_mask] # Only keep rows where the question content is in French
        
    df = df.replace({np.nan: None}) # Replace NaN values with None for JSON compatibility

    # Rename columns for clarity
    df.rename(columns={"model_pos": "preferred_model", "ip": "user_id", "id": "reaction_id"}, inplace=True)

    out = df.to_dict(orient="records")
    
    hash_to_user_id_map = {}
    user_counter = 1

    for item in out:
        # Parsing JSON-like data records
        item["conversation_a"] = json.loads(item["conversation_a"])
        item["conversation_b"] = json.loads(item["conversation_b"])
        
        # We fully anonymise hashed user IDs by mapping them to sequential integers in the output data
        hash_user = item["user_id"]
        if hash_user not in hash_to_user_id_map.keys():
            hash_to_user_id_map[hash_user] = user_counter
            user_counter += 1
        
        item["user_id"] = hash_to_user_id_map[hash_user]  # Replace hashed user ID with sequential integer

        # Mapping raw values ("a"/"b"/"both_equal") to the actual model names or None
        preferred_model = item["preferred_model"]
        if preferred_model == "a":
            item["preferred_model"] = item["model_a_name"]
        elif preferred_model == "b":
            item["preferred_model"] = item["model_b_name"]
        elif preferred_model == "both_equal":
            item["preferred_model"] = (
                None # No model is preferred; we use None to facilitate data processing and standardise values
            )
        else: # Should not happen due to strict SQL constraints at the database level
            raise ValueError(f"Unexpected value for preferred_model: {preferred_model}")
    
        item["labels"] = {label: item[label] for label in LABELS_COLUMNS} # Merging label columns
        
        for label in LABELS_COLUMNS:
            del item[label] # Removing individual label columns after merging

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print("Data file successfully saved at ", output_path)


def format_teacher_data_csv_to_json(input_path: str = SOURCE_FILE_PATH, output_path: str = OUTPUT_FILE_PATH, saved_columns: list[str] = SAVED_COLUMNS_TEACHERS) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found at: {input_path}")

    df = pd.read_csv(input_path)
    df = df[saved_columns]

    df = df.replace({np.nan: None}) # Replace NaN values with None for JSON compatibility
    
    out = df.to_dict(orient="records")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

    print("Data file successfully saved at ", output_path)

def merge_student_and_teacher_data(input_student_path: str = INPUT_STUDENT_PATH, input_teacher_path: str = INPUT_TEACHER_PATH, output_path: str = OUTPUT_PATH) -> None:
    if not os.path.exists(input_student_path):
        raise FileNotFoundError(f"File not found at: {input_student_path}")
    if not os.path.exists(input_teacher_path):
        raise FileNotFoundError(f"File not found at: {input_teacher_path}")

    with open(input_student_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)

    with open(input_teacher_path, "r", encoding="utf-8") as f:
        teacher_data = json.load(f)

    # Create a mapping from a student conversation id to all teacher annotations for that conversation
    teacher_mapping = defaultdict(list)
    for item in teacher_data:
        teacher_mapping[item["source_reaction_id"]].append(item)
        del item["source_reaction_id"]  # Remove the source_reaction_id from the teacher data to avoid redundancy

    # Merge teacher data into student data based on reaction_id
    for student_item in student_data:
        reaction_id = student_item["reaction_id"]
        student_item["teacher_annotation_data"] = teacher_mapping.get(reaction_id, [])  # Assign all corresponding teacher data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(student_data, f, indent=4, ensure_ascii=False)

    print("Merged data file successfully saved at ", output_path)
    

if __name__ == "__main__":
    # format_student_data_csv_to_json()
    # format_teacher_data_csv_to_json()
    merge_student_and_teacher_data()