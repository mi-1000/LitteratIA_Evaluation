import json
import os

from enum import Enum
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from system_prompt import PromptType, PersonaType, ToneType, get_system_prompt

load_dotenv()

SEED = 2026
MODEL_NAME = "mistralai/ministral-3b-2512"
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in environment variables.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

INPUT_DIR = os.path.join('..', 'data')
SPOKEN_DATASET_PATH = os.path.join(INPUT_DIR, 'spoken_dataset.json')
STACKEXCHANGE_DATASET_PATH = os.path.join(INPUT_DIR, 'french_dataset.jsonl')
WIF_DATASET_PATH = os.path.join(INPUT_DIR, 'wif_goteborg_dataset.json')
OUTPUT_PATH = "synthetic_outputs.jsonl"

PROMPT_CONFIGURATIONS = {
    "original": PromptType(None, None),
    "teacher_casual": PromptType(PersonaType.TEACHER, ToneType.CASUAL),
    "teacher_guided": PromptType(PersonaType.TEACHER, ToneType.GUIDED),
    "advisor_casual": PromptType(PersonaType.ADVISOR, ToneType.CASUAL),
    "advisor_guided": PromptType(PersonaType.ADVISOR, ToneType.GUIDED),
}

class DatasetType(Enum):
    SPOKEN = "spoken"
    STACKEXCHANGE = "stackexchange"
    WIF = "wif"

def get_llm_response(system_prompt: str, user_prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling API for prompt: {user_prompt[:50]}…: {e}")

def format_dialogue_to_prompt(dialogue_turns: list) -> str:
    formatted_turns = []
    
    for turn in dialogue_turns:
        speaker = turn["speaker"]["name"]
        utterance = turn["utterance"].strip()
        formatted_turns.append(f"{speaker} : {utterance}")
        
    transcription_text = "\n".join(formatted_turns)
    
    introduction = "Voici la transcription d'une conversation que j'ai eue pour pratiquer le français (j'ai le rôle 'Student'). J'aimerais savoir comment faire pour améliorer mon dialogue en français."

    return f"{introduction}\n\n{transcription_text}"

def format_question_to_prompt(question_data: dict) -> str:
    question_text = question_data.get("question", "").strip()
    body_text = question_data.get("body", "").strip()
    
    return f"{question_text}\n{body_text}"

def format_written_production_to_prompt(text: str, grade: int) -> str:
    return f"Je viens de Suède en {grade}ème classe. Peux-tu corriger mon texte ?\n\n{text}"

def run_prompt_generation_spoken(n = 1):
    if not os.path.exists(SPOKEN_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {SPOKEN_DATASET_PATH}")
        
    print(f"Loading dataset: {SPOKEN_DATASET_PATH}")
    with open(SPOKEN_DATASET_PATH, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
        
    generated_records = []
    
    for item in tqdm(corpus_data[:n]):
        # Extracts key and value since root objects are formatted as {"path/to/file.cha": [...]}
        file_path = list(item.keys())[0]
        dialogue_turns = item[file_path]
        file_id = os.path.basename(file_path).replace('.cha', '')
        
        user_prompt = format_dialogue_to_prompt(dialogue_turns)
        
        for config_name, prompt_type in PROMPT_CONFIGURATIONS.items():
            system_prompt = get_system_prompt(prompt_type, seed=SEED)
            llm_output = get_llm_response(system_prompt, user_prompt)

            generated_records.append({
                "item_id": "spoken_"+file_id,
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
    
def run_generation_stackexchange(n = 1):
    if not os.path.exists(STACKEXCHANGE_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {STACKEXCHANGE_DATASET_PATH}")
        
    print(f"Loading dataset: {STACKEXCHANGE_DATASET_PATH}")
    
    with open(STACKEXCHANGE_DATASET_PATH, 'r', encoding='utf-8') as f:
        corpus_data = []
        for line in f.readlines():
            data = json.loads(line)
            corpus_data.append(data)
        
    generated_records = []
    
    for item in tqdm(corpus_data[:n]):
        question_id = int(item.get("id", "0"))
        tags = item.get("tags", [])
        
        user_prompt = format_question_to_prompt(item)
        
        for config_name, prompt_type in PROMPT_CONFIGURATIONS.items():
            system_prompt = get_system_prompt(prompt_type, seed=SEED)
            llm_output = get_llm_response(system_prompt, user_prompt)

            generated_records.append({
                "item_id": f"fse_{question_id}",
                "tags": tags,
                "category": "c",
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

def run_generation_wif(n = 1):
    if not os.path.exists(WIF_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {WIF_DATASET_PATH}")
    
    print(f"Loading dataset: {WIF_DATASET_PATH}")
    with open(WIF_DATASET_PATH, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    generated_records = []

    for item in tqdm(corpus_data[:n]):
        # Extracts key and value since root objects are formatted as {"path/to/file.txt": [...]}
        file_path = list(item.keys())[0]
        file_id = os.path.basename(file_path).replace('.txt', '')
        contents = item[file_path]
        
        grade: int = contents.get("grade", 0)
        text: str = contents.get("content", "").strip()
        
        user_prompt = format_written_production_to_prompt(text, grade)
        
        for config_name, prompt_type in PROMPT_CONFIGURATIONS.items():
            system_prompt = get_system_prompt(prompt_type, seed=SEED)
            llm_output = get_llm_response(system_prompt, user_prompt)

            generated_records.append({
                "item_id": "wif_"+file_id,
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

def run_prompt_generation(dataset: DatasetType = DatasetType.SPOKEN, n: int = 1):
    if dataset == DatasetType.SPOKEN:
        run_prompt_generation_spoken(n)
    elif dataset == DatasetType.STACKEXCHANGE:
        run_generation_stackexchange(n)
    elif dataset == DatasetType.WIF:
        run_generation_wif(n)
    else: # Should not happen
        raise ValueError(f"Unsupported dataset type: {dataset}")

if __name__ == "__main__":
    # pbar = tqdm(total=len(PROMPT_CONFIGURATIONS) * len(corpus_data[:n]), desc='Generating prompts', leave=True)
    # Dans la boucle : pbar.update()
    # À la toute toute fin : pbar.close()
    # nb : pour remplacer les print() => pbar.write()
    run_prompt_generation(DatasetType.WIF, n=1)