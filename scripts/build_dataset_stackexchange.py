import xml.etree.ElementTree as ET
import json
import os
import regex

from bs4 import BeautifulSoup
from tqdm import tqdm

from utils import is_french

INPUT_FILE = os.path.join("..", "data", "StackExchange", "Posts.xml")
OUTPUT_FILE = os.path.join("..", "data", "StackExchange", "french_dataset.jsonl")

def parse_html(html_content: str) -> str:
    """Parse HTML content and return clean text."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text()
    
    # Normalize newlines
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove Unicode characters that are not meaningful for the dataset
    text = regex.sub(r'(?>[\p{Cf}\p{Cc}](?<!\n))+', '', text) # Remove invisible control characters, except for newlines
    text = regex.sub(r'\p{Z}', ' ', text) # Replace all kinds of whitespace with a single space
    
    return text

def build_dataset() -> None:    
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline='\n') as f_out:
        n_lines = 0
        filtered = 0
        for _event, elem in tqdm(ET.iterparse(INPUT_FILE, events=("end",)), desc="Processing posts"):
            if elem.tag == "row":
                post_type = elem.get("PostTypeId")
                score = int(elem.get("Score", 0))

                if (post_type == "1" and score >= 2): # Questions (PostTypeId="1") with score >= 2 to ensure minimal relevance
                
                    question_id = elem.get("Id")
                    title = elem.get("Title")
                    body = elem.get("Body")
                    tags = elem.get("Tags", "").strip('|').split('|')

                    title = parse_html(title)
                    body = parse_html(body)
                    
                    if not is_french(title) or not is_french(body):
                        filtered += 1
                        continue # Skip non-French questions

                    data = {
                        "id": question_id,
                        "question": title,
                        "body": body,
                        "score": score,
                        "tags": tags,
                    }
                    f_out.write(json.dumps(data, indent=0, ensure_ascii=False))
                    n_lines += 1

                elem.clear() # Manage memory

    print(f"Succesfully saved {n_lines} lines in {OUTPUT_FILE}. Filtered out {filtered} non-French questions.")

if __name__ == "__main__":
    build_dataset()