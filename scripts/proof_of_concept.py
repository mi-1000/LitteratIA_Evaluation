import json
import os

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from system_prompt import PromptType, PersonaType, ToneType, get_system_prompt

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in environment variables.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

DATASET_PATH = os.path.join('..', 'data', 'wif_goteborg', 'wif_goteborg_dataset.json')
OUTPUT_PATH = "synthetic_outputs.jsonl"

PROMPT_CONFIGURATIONS = {
    "original": PromptType(None, None),
    "teacher_casual": PromptType(PersonaType.TEACHER, ToneType.CASUAL),
    "teacher_guided": PromptType(PersonaType.TEACHER, ToneType.GUIDED),
    "advisor_casual": PromptType(PersonaType.ADVISOR, ToneType.CASUAL),
    "advisor_guided": PromptType(PersonaType.ADVISOR, ToneType.GUIDED),
}

def format_dialogue_to_prompt(dialogue_turns: list) -> str:
    formatted_turns = []
    
    for turn in dialogue_turns:
        speaker = turn["speaker"]["name"]
        utterance = turn["utterance"].strip()
        formatted_turns.append(f"{speaker} : {utterance}")
        
    transcription_text = "\n".join(formatted_turns)
    
    introduction = "Voici la transcription d'une conversation que j'ai eue pour pratiquer le français (j'ai le rôle 'Student'). J'aimerais savoir comment faire pour améliorer mon dialogue en français."

    return f"{introduction}\n\n{transcription_text}"

def run_prompt_generation(n = 1):
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")
        
    print(f"Loading dataset: {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
        
    generated_records = []
    
    for item in tqdm(corpus_data[:n]):
        # Extracts key and value since root objects are formatted as {"path/to/file.cha": [...]}
        file_path = list(item.keys())[0]
        dialogue_turns = item[file_path]
        file_id = os.path.basename(file_path).replace('.cha', '')
        
        user_prompt = format_dialogue_to_prompt(dialogue_turns)
        
        for config_name, prompt_type in PROMPT_CONFIGURATIONS.items():
            system_prompt = get_system_prompt(prompt_type, seed=2026)
            
            print(f"Calling OpenRouter [{config_name}] for file: {file_id}...")
            
            try:
                response = client.chat.completions.create(
                    model="mistralai/ministral-3b-2512",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                llm_output = response.choices[0].message.content
            except Exception as e:
                print(f"Error calling API for {file_id} ({config_name}): {e}")
                llm_output = f"[API ERROR: {str(e)}]"
                
            generated_records.append({
                "item_id": file_id,
                "source_file": file_path,
                "category": "f",
                "prompt_configuration": config_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "llm_response": llm_output
            })
            
        print(f"Saving results to: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'a+', encoding='utf-8') as out_f:
            for record in generated_records:
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
    print("Generation completed.")

if __name__ == "__main__":
    run_prompt_generation()