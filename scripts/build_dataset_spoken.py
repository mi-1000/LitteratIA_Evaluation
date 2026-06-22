import glob
import json
import os
import re

from tqdm import tqdm

SPEAKER = "speaker"
UTTERANCE = "utterance"
ID = "id"
NAME = "name"

OUTPUT_DIR = os.path.join('..', 'data', 'spoken')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'spoken_dataset.json')
SOURCE_DIR = os.path.join(OUTPUT_DIR, 'transcripts')
chat_files = glob.glob(os.path.join(SOURCE_DIR, '**', '*.cha'), recursive=True)

if not os.path.exists(SOURCE_DIR):
    raise FileNotFoundError(f"Directory {SOURCE_DIR} does not exist.")

def extract_dialogue_lines(file_path: str):
    dialogue_lines = []
    current_speaker = None
    current_text = ""
    
    roles = {}
    
    speaker_regex = re.compile(r'^\*([A-Z0-9]{3}):\s*(.*)')
    participants_regex = re.compile(r'([A-Z0-9]{3})\s+([A-Za-z]+)')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('@Participants:'):
                participants_data = line.replace('@Participants:', '').strip()
                matches = participants_regex.findall(participants_data)
                for code, role in matches:
                    roles[code] = role
            
            if not line or line.startswith('@') or line.startswith('%'): # Skip other metadata lines
                continue

            match = speaker_regex.match(line)
            
            if match:
                if current_speaker and current_text:
                    resolved_speaker = roles.get(current_speaker, current_speaker)
                    dialogue_lines.append({SPEAKER: {ID: current_speaker, NAME: resolved_speaker}, UTTERANCE: current_text.strip()})

                current_speaker = match.group(1)
                current_text = match.group(2)

            elif current_speaker and current_text: # Continuation of the current speaker's utterance on a new line
                current_text += " " + line

    if current_speaker and current_text: # Append the last speaker's text if it exists
        resolved_speaker = roles.get(current_speaker, current_speaker)
        dialogue_lines.append({SPEAKER: {ID: current_speaker, NAME: resolved_speaker}, UTTERANCE: current_text.strip()})

    return dialogue_lines

def save_dataset():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        files = []
        for chat_file in tqdm(chat_files):
            lines = extract_dialogue_lines(chat_file)
            files.append({chat_file: lines})
        json.dump(files, f_out, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(chat_files)} files in {OUTPUT_FILE}.")

if __name__ == "__main__":
    save_dataset()