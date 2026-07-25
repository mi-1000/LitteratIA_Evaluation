import json
import os

from tqdm import tqdm

from utils import is_french

SOURCE_JSONL_PATH = "synthetic_outputs.jsonl"
OUTPUT_JSON_PATH = os.path.join("..", "data", "synthetic_conversations.json")

def format_synthetic_to_reactions(source_path: str, output_path: str) -> None:
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"File not found at: {source_path}")

    formatted_records = []
    
    lines = []
    
    with open(source_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines), desc="Processing synthetic data", total=len(lines)):
            if not line.strip():
                continue
                
            item = json.loads(line)
            
            conversation_a = [
                {
                    "role": "user", 
                    "content": item["user_prompt"]
                },
                {
                    "role": "assistant", 
                    "content": item["model_a"]["llm_response"]
                }
            ]
            
            conversation_b = [
                {
                    "role": "user", 
                    "content": item["user_prompt"]
                },
                {
                    "role": "assistant", 
                    "content": item["model_b"]["llm_response"]
                }
            ]
            
            reaction_record = {
                "reaction_id": i + 1,
                "msg_index": 1,
                "conversation_a": conversation_a,
                "conversation_b": conversation_b,
                "model_a_name": item["model_a"]["model_name"],
                "model_b_name": item["model_b"]["model_name"],
                "system_prompt": item.get("system_prompt", ""),
                "question_content": item["user_prompt"],
                "metadata": {
                    "original_item_id": item["item_id"],
                    "prompt_configuration": item["prompt_configuration"],
                    "category": item["category"],
                    "source_dataset": item.get("source_dataset")
                }
            }
            
            if is_french(item["user_prompt"]) and item["model_a"]["finish_reason"] == "stop" and item["model_b"]["finish_reason"] == "stop": # Only keep French data where both models finished properly without interruption
                formatted_records.append(reaction_record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_records, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(formatted_records)}/{len(lines)} entries ({len(lines) - len(formatted_records)} filtered out) to {output_path}.")

if __name__ == "__main__":
    format_synthetic_to_reactions(SOURCE_JSONL_PATH, OUTPUT_JSON_PATH)