import codecs
import glob
import os
import pylangacq
import re

SOURCE_DIR = os.path.join('..', 'data', 'wif_goteborg', 'transcripts')
chat_files = glob.glob(os.path.join(SOURCE_DIR, '**', '*.cha'), recursive=True)

if not os.path.exists(SOURCE_DIR):
    raise FileNotFoundError(f"Directory {SOURCE_DIR} does not exist.")

def clean_corpus(force=False) -> None:
    """- Re-encode all .cha files to utf-8 (see _StackOverflow_ source: https://stackoverflow.com/a/191043)
    - Fix capitalization of elements that prevent the parser from correctly recognizing participants and their roles
    - Remove unnecessary tokens
    - Fix typography
    """
    if not force:
        print("Re-encoding corpus is disabled by default to prevent accidental data loss. Pass force=True to enable it.")
        return
    
    for source_file_name in chat_files:
        target_file_name=source_file_name.replace('.cha', '.txt')
        with codecs.open(source_file_name, "r", "cp1252") as source_file:
            with codecs.open(target_file_name, "w", "utf-8") as target_file:
                while True:
                    contents = source_file.read()
                    if not contents:
                        break
                    contents = contents.replace('@participants:', '@Participants:') # Fix capitalization of @Participants line, that prevents the parser from recognizing participants
                    contents = contents.replace('investigator', 'Investigator') # Fix capitalization of roles
                    contents = contents.replace('subject', 'Subject')
                    contents = contents.replace('Investigator/Partner', 'Investigator')
                    contents = re.sub(r'\[[^\]]*?\]', '', contents) # Remove between brackets
                    lines = contents.splitlines()
                    for line in lines:
                        words = []
                        if line.startswith(('*', '%')):
                            words = line.split()
                            words = [word for word in words if not word.startswith(('@', '%', '+', '[', ']', '#'))] # Remove metadata tokens
                        line = ' '.join(words)
                    contents = '\n'.join(lines)
                    # Fix typography quirks
                    contents = contents.replace('<', '')
                    contents = contents.replace('>', '')
                    contents = contents.replace('(', '')
                    contents = contents.replace(')', '')
                    contents = contents.replace('#', '')
                    contents = re.sub(r'\b\w+@\w+\b', '', contents) # Remove noisy tokens containing @
                    contents = re.sub(r'[^\S\r\n]{2,}', ' ', contents).strip() # Remove extra whitespace, except for newlines
                    contents = contents.replace("' ", "'")
                    contents = contents.replace(' .', '.')
                    contents = contents.replace('+', '-')
                    contents = '\n'.join([line for line in contents.splitlines() if not line or any(char.islower() for char in line)]) # Remove lines that don't contain any lowercase letters, as they are most likely not meaningful, but keep empty lines that structure the file
                    target_file.write(contents)
                    os.remove(source_file_name) # Remove the original .cha file after re-encoding
                    os.rename(target_file_name, source_file_name) # Rename the .txt file back to .cha

    print("Succesfully cleaned corpus.")

if __name__ == "__main__":
    clean_corpus()
    
    dataset = pylangacq.read_chat(SOURCE_DIR, strict=False)

    for i in range(1):
        file = dataset.pop_left()
        path = file.file_paths[0]
        participants = file.participants()
        with open(path, 'r', encoding='utf-8') as f:
            print(f"File: {path}, Participants: {participants}\n{f.read()}")
