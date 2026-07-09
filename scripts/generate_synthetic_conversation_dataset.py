import json
import os
import random
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Tuple
from openai import APIConnectionError, AuthenticationError, OpenAI, PermissionDeniedError, RateLimitError
from dotenv import load_dotenv
from tqdm import tqdm

from system_prompt import PromptType, PersonaType, ToneType, get_system_prompt

load_dotenv()

SEED = 2026
random.seed(SEED)

WRITE_LOCK = threading.Lock()
MAX_WORKERS = 15 # Number of threads for concurrent API calls

MODEL_LIST = [
    "tencent/hy3:free",
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-120b:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "poolside/laguna-xs.2:free"
    # "google/gemini-3-flash-preview",
    # "google/gemini-3.1-flash-lite-preview",
    # "openai/gpt-5.4",
    # "openai/gpt-5.4-mini",
    # "deepseek/deepseek-v4-pro",
    # "deepseek/deepseek-v4-flash",
    # "mistralai/mistral-large-2512",
    # "mistralai/mistral-medium-3"
]

WEIGHTS = [
    1.0,  # google/gemini-3-flash-preview
    1.0,  # google/gemini-3.1-flash-lite-preview
    # 0.5,  # openai/gpt-5.4 (expensive 🥲)
    1.0,  # openai/gpt-5.4-mini
    1.0,  # deepseek/deepseek-v4-pro
    1.0,  # deepseek/deepseek-v4-flash
    1.0,  # mistralai/mistral-large-2512
    # 1.0   # mistralai/mistral-medium-3
]

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

def get_llm_response(system_prompt: str, user_prompt: str, model_name: str, max_completion_tokens: int = 4096, seed: int = SEED, max_retries: int = 3) -> dict[str, Any]:
    """
    Get a response from a given LLM.
    
    Args:
        system_prompt (str): The system prompt to set the context for the LLM.
        user_prompt (str): The user prompt to which the LLM should respond.
        model_name (str): The identifier of the LLM model to use.
        max_completion_tokens (int): The maximum number of tokens to generate in the completion. This number should be relatively high, because responses will be cut sharp past it.
        seed (int): Seed for reproducibility.
        max_retries (int): Maximum number of retries for API calls in case of rate limits or connection errors.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=max_completion_tokens,
                seed=seed
            )
            choice = response.choices[0]
            return {
                "content": choice.message.content,
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage else None
            }
        
        except (AuthenticationError, PermissionDeniedError) as e:
            raise e
        except (RateLimitError, APIConnectionError) as e:
            print(f"Rate limit or connection error occurred: {e}.\n\nRetrying ({attempt + 1}/{max_retries})…")
            time.sleep(2 ** (attempt + 1)) # Exponential backoff
        except Exception as e:
            print(f"Error generating response for prompt: {user_prompt[:50]}…: {e}")
            time.sleep(1)
            return {}
    
    print(f"No output written: Could not to get a response after {max_retries} attempts for prompt: {user_prompt[:50]}…")
    return {}

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

def get_balanced_model_assignments(n_processed_items: int, seed: int = SEED, model_list: list[str] = MODEL_LIST, weights: list[float] | None = WEIGHTS) -> list[str]:
    """
    Get a list of model names assigned in a balanced way, in order to perform (uniform or weighted) stratified random sampling.
    If weights are provided, they will be used to determine the weight of each model in the output list (_i.e._ one can determine some to appear more or less frequently).
    
    Unlike regular random sampling, this method allows to reduce most of the variance in the number of occurences per model across the dataset.
    
    Args:
        n_processed_items (int): Total number of items to process.
        seed (int): Seed for reproducibility.
        model_list (list[str]): List of model names to assign.
        weights (list[float] | None): Optional list of weights corresponding to each model in model_list. If None, all models will be assigned equally.
        
    Returns:
        list[str]: A list of model names assigned in a balanced way.
        
    Raises:
        ValueError: If the length of `weights` does not match the length of `model_list` (lists should map one-to-one).
    """
    rng = random.Random(seed)

    if weights is None:
        n_list_repetitions = (n_processed_items // len(model_list)) + 1
        out = (model_list * n_list_repetitions)[:n_processed_items]
    else:
        if len(weights) != len(model_list):
            raise ValueError("Length of weights must match length of model list.")
        
        # Counting how much times each model appear based on the weights
        total_weight = sum(weights)
        proportions = [w / total_weight for w in weights]
        counts = [int(p * n_processed_items) for p in proportions] # Round down figures and leave out the remainders for now
        
        # Distributing the remaining remainders among models using the largest remainder method
        # See https://en.wikipedia.org/wiki/Quota_method
        remainder = n_processed_items - sum(counts)
        if remainder > 0:
            # Calculate the fractional parts of the proportions for each model
            remainders = [(p * n_processed_items) - int(p * n_processed_items) for p in proportions]
            # Sort models by their fractional parts in descending order
            indices = list(range(len(remainders)))
            rng.shuffle(indices) # Shuffle indices to break ties randomly
            model_indices = sorted(indices, key=lambda i: remainders[i], reverse=True)
            # Assign the remaining items to the models with the largest fractional parts
            for i in range(remainder):
                counts[model_indices[i]] += 1
        
        out = []
        for model, count in zip(model_list, counts):
            out.extend([model] * count)

    rng.shuffle(out)
    return out

def _ipf_balance_pair_matrix(
    weights: list[float],
    n_processed_items: int,
    tol: float = 1e-9,
    max_iter: int = 1000
) -> list[list[float]]:
    """
    Rescales the pairwise weight matrix (product of individual weights, zero diagonal) via [Iterative Proportional Fitting](https://en.wikipedia.org/wiki/Iterative_proportional_fitting) so that both row sums and column sums converge to the target marginal frequency per model (proportional to `weights`), rather than the quadratic `w_i` * (`W_total` - `w_i`) approximation of the raw product matrix.

    Returns:
        A matrix of rescaled pair weights.
    """
    # If all weights are equal, we directly return the product matrix (with zero diagonal)
    if all(w == weights[0] for w in weights):
        k = len(weights)
        return [[weights[i] * weights[j] if i != j else 0.0 for j in range(k)] for i in range(k)]
    
    k = len(weights)
    total_weight = sum(weights)
    
    # Target: number of times model i should appear in position A (and B likewise)
    target = [w / total_weight * n_processed_items for w in weights]

    # Initial matrix: product of weights, null diagonal (no self-pairing)
    M = [[weights[i] * weights[j] if i != j else 0.0 for j in range(k)] for i in range(k)]

    for _ in range(max_iter):
        # Rescaling of rows (position A)
        row_sums = [sum(row) for row in M]
        for i in range(k):
            if row_sums[i] > 0:
                factor = target[i] / row_sums[i]
                M[i] = [x * factor for x in M[i]]

        # Rescaling of columns (position B)
        col_sums = [sum(M[i][j] for i in range(k)) for j in range(k)]
        for j in range(k):
            if col_sums[j] > 0:
                factor = target[j] / col_sums[j]
                for i in range(k):
                    M[i][j] *= factor

        # Convergence: row sums should be close to target
        row_sums_check = [sum(row) for row in M]
        max_diff = max(abs(row_sums_check[i] - target[i]) for i in range(k))
        if max_diff < tol:
            break

    return M

def get_balanced_model_pairs(n_processed_items: int, seed: int = SEED, model_list: list[str] = MODEL_LIST, weights: list[float] | None = WEIGHTS) -> Tuple[list[str], list[str]]:
    """
    Get a 2-tuple of lists of model names assigned in a balanced way, in order to perform (uniform or weighted) stratified random sampling.
    If weights are provided, they will be used to determine the weight of each model in the output list (_i.e._ one can determine some to appear more or less frequently).
    
    Unlike regular random sampling, this method allows to reduce most of the variance in the number of occurences per model and co-occurrences between models across the dataset.
    
    Args:
        n_processed_items (int): Total number of items to process.
        seed (int): Seed for reproducibility.
        model_list (list[str]): List of model names to assign.
        weights (list[float] | None): Optional list of weights corresponding to each model in model_list. If None, all models will be assigned equally.
        
    Returns:
        Tuple[list[str], list[str]]: A 2-tuple of lists of model names assigned in a balanced way.
        
    Raises:
        ValueError: If the length of `weights` does not match the length of `model_list` (lists should map one-to-one).
    """
    rng = random.Random(seed)
    
    if weights is None:
        weights = [1.0] * len(model_list)
    
    # Calculate weights for all possible pairs of models
    M = _ipf_balance_pair_matrix(weights, n_processed_items)
    
    all_possible_pairs = []
    pair_weights = []
    
    for i, m1 in enumerate(model_list):
        for j, m2 in enumerate(model_list):
            if i != j: # No match against oneself
                all_possible_pairs.append((m1, m2))
                pair_weights.append(M[i][j]) # The pair's weight is the rescaled weight from the IPF matrix
    
    # Counting how much times each model pair appears based on the weights
    total_pair_weight = sum(pair_weights)
    proportions = [w / total_pair_weight for w in pair_weights]
    counts = [int(p * n_processed_items) for p in proportions] # Round down figures and leave out the remainders for now
    
    # Distributing the remaining remainders among models using the largest remainder method
    # See https://en.wikipedia.org/wiki/Quota_method
    remainder = n_processed_items - sum(counts)
    if remainder > 0:
        # Calculate the fractional parts of the proportions for each model
        remainders = [(p * n_processed_items) - int(p * n_processed_items) for p in proportions]
        # Sort models by their fractional parts in descending order
        indices = list(range(len(remainders)))
        rng.shuffle(indices) # Shuffle indices to break ties randomly
        pair_indices = sorted(indices, key=lambda i: remainders[i], reverse=True)
        # Assign the remaining items to the models with the largest fractional parts
        for i in range(remainder):
            counts[pair_indices[i]] += 1
    
    final_pairs = []
    for pair, count in zip(all_possible_pairs, counts):
        final_pairs.extend([pair] * count)
    
    rng.shuffle(final_pairs)
    
    model_a_list = [pair[0] for pair in final_pairs]
    model_b_list = [pair[1] for pair in final_pairs]
    return model_a_list, model_b_list

def process_single_response(item_id: str, category: str, config_name: str, prompt_type: PromptType, user_prompt: str, model_name: str, seed: int = SEED, **kwargs) -> dict[str, str]:
    """
    Process a given prompt configuration.
    
    Args:
        item_id (str): Unique identifier for the item being processed.
        category (str): Category of the question.
        config_name (str): Name of the prompt configuration.
        prompt_type (PromptType): The type of prompt configuration to use.
        user_prompt (str): The user prompt to which the LLM should respond.
        model_name (str): The identifier of the LLM model to use.
        seed (int): Seed for reproducibility.
        **kwargs: Additional keyword arguments for dataset-specific fields.
    
    Returns:
        dict[str, str]: A dictionary containing the processed record with all relevant fields.
    """
    system_prompt = get_system_prompt(prompt_type, seed=seed)
    llm_response = get_llm_response(system_prompt, user_prompt, model_name=model_name, seed=seed)
    llm_output = llm_response.get("content", "")
    
    out = {
        "item_id": item_id,
        "category": category,
        "prompt_configuration": config_name,
        "model_name": model_name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "llm_response": llm_output,
        "llm_usage": llm_response.get("usage", None),
        "finish_reason": llm_response.get("finish_reason", None)
    }
    
    out.update({k: v for k, v in kwargs.items() if k not in out}) # Add any additional, dataset-specific fields
    
    return out

def write_record(record: dict[str, Any], output_path: str = OUTPUT_PATH) -> None:
    """Thread-safe record writing"""
    with WRITE_LOCK:
        with open(output_path, 'a+', encoding='utf-8') as out_f:
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

def run_prompt_generation_spoken(n: int = 1, start_item: int = 0, sample: int = 1000) -> None:
    """
    Run prompt generation for the spoken dataset.
    
    Args:
        n (int): Number of items to process. If it exceeds the number of items in the dataset, it will be capped.
        start_item (int): Index of the first item to process in the original dataset.
        sample (int): Number of items from the input dataset to randomly sample from.
    """
    if not os.path.exists(SPOKEN_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {SPOKEN_DATASET_PATH}")
        
    print(f"Loading dataset: {SPOKEN_DATASET_PATH}")
    with open(SPOKEN_DATASET_PATH, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    corpus_data = random.sample(corpus_data, min(sample, len(corpus_data)))
    
    items = corpus_data[start_item:start_item + n]
    
    model_a_assignments, model_b_assignments = get_balanced_model_pairs(len(items)) # Not using 'n' here to avoid an IndexError
    
    tasks = []
    
    for i, item in enumerate(items):
        # Extracts key and value since root objects are formatted as {"path/to/file.cha": [...]}
        file_path = list(item.keys())[0]
        dialogue_turns = item[file_path]
        file_id = os.path.basename(file_path).replace('.cha', '')
        
        user_prompt = format_dialogue_to_prompt(dialogue_turns)
        
        model_a, model_b = model_a_assignments[i], model_b_assignments[i]
        
        for config_name, prompt_type in PROMPT_CONFIGURATIONS.items():
            tasks.append((f"spoken_{file_id}", "f", config_name, prompt_type, user_prompt, model_a, {"source_file": file_path, "source_dataset": DatasetType.SPOKEN.value}))
            tasks.append((f"spoken_{file_id}", "f", config_name, prompt_type, user_prompt, model_b, {"source_file": file_path, "source_dataset": DatasetType.SPOKEN.value}))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_response, *task[:-1], **task[-1]) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating prompts"):
            record = future.result()
            write_record(record)
        
    print("Generation completed.")

def run_prompt_generation_stackexchange(n: int = 1, start_item: int = 0, sample: int = 1000) -> None:
    """
    Run prompt generation for the StackExchange dataset.
    
    Args:
        n (int): Number of items to process. If it exceeds the number of items in the dataset, it will be capped.
        start_item (int): Index of the first item to process in the original dataset.
        sample (int): Number of items from the input dataset to randomly sample from.
    """
    if not os.path.exists(STACKEXCHANGE_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {STACKEXCHANGE_DATASET_PATH}")
        
    print(f"Loading dataset: {STACKEXCHANGE_DATASET_PATH}")
    
    with open(STACKEXCHANGE_DATASET_PATH, 'r', encoding='utf-8') as f:
        corpus_data = []
        lines = f.readlines()
        lines = random.sample(lines, min(sample, len(lines)))
        
        for line in lines:
            data = json.loads(line)
            corpus_data.append(data)
    
    model_a_assignments, model_b_assignments = get_balanced_model_pairs(len(corpus_data[start_item:start_item + n]))
    
    tasks = []
    
    for i, item in enumerate(corpus_data[start_item:start_item + n]):
        question_id = int(item.get("id", "0"))
        tags = item.get("tags", [])
        
        user_prompt = format_question_to_prompt(item)
        
        model_a, model_b = model_a_assignments[i], model_b_assignments[i]
        
        for config_name, prompt_type in PROMPT_CONFIGURATIONS.items():
            tasks.append((f"fse_{question_id}", "c", config_name, prompt_type, user_prompt, model_a, {"tags": tags, "source_dataset": DatasetType.STACKEXCHANGE.value}))
            tasks.append((f"fse_{question_id}", "c", config_name, prompt_type, user_prompt, model_b, {"tags": tags, "source_dataset": DatasetType.STACKEXCHANGE.value}))
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_response, *task[:-1], **task[-1]) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating prompts"):
            record = future.result()
            write_record(record)

    print("Generation completed.")

def run_prompt_generation_wif(n: int = 1, start_item: int = 0, sample: int = 1000) -> None:
    """
    Run prompt generation for the WIF dataset.
    
    Args:
        n (int): Number of items to process. If it exceeds the number of items in the dataset, it will be capped.
        start_item (int): Index of the first item to process in the original dataset.
        sample (int): Number of items from the input dataset to randomly sample from.
    """
    if not os.path.exists(WIF_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {WIF_DATASET_PATH}")
    
    print(f"Loading dataset: {WIF_DATASET_PATH}")
    
    with open(WIF_DATASET_PATH, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    corpus_data = random.sample(corpus_data, min(sample, len(corpus_data)))

    model_a_assignments, model_b_assignments = get_balanced_model_pairs(len(corpus_data[start_item:start_item + n]))

    tasks = []
    
    for i, item in enumerate(corpus_data[start_item:start_item + n]):
        # Extracts key and value since root objects are formatted as {"path/to/file.txt": [...]}
        file_path = list(item.keys())[0]
        file_id = os.path.basename(file_path).replace('.txt', '')
        contents = item[file_path]
        
        grade: int = contents.get("grade", 0)
        text: str = contents.get("content", "").strip()
        
        user_prompt = format_written_production_to_prompt(text, grade)
        
        model_a, model_b = model_a_assignments[i], model_b_assignments[i]
        
        for config_name, prompt_type in PROMPT_CONFIGURATIONS.items():
            tasks.append((f"wif_{file_id}", "f", config_name, prompt_type, user_prompt, model_a, {"source_file": file_path, "source_dataset": DatasetType.WIF.value}))
            tasks.append((f"wif_{file_id}", "f", config_name, prompt_type, user_prompt, model_b, {"source_file": file_path, "source_dataset": DatasetType.WIF.value}))
            
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_response, *task[:-1], **task[-1]) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating prompts"):
            record= future.result()
            write_record(record)
        
    print("Generation completed.")

def run_prompt_generation(n: int | Tuple[int, int, int] = 1, start_item: int | Tuple[int, int, int] = 0) -> None:
    """
    Launch the prompt generation process for all datasets.
    
    Args:
        n (int | Tuple[int, int, int]): Number of items to process for each dataset. If a single integer is provided, it will be used for all datasets. If a tuple of three integers is provided, they will be used for the spoken, StackExchange, and WIF datasets respectively.
        start_item (int | Tuple[int, int, int]): Index of the first item to process in the original dataset for each dataset. If a single integer is provided, it will be used for all datasets. If a tuple of three integers is provided, they will be used for the spoken, StackExchange, and WIF datasets respectively.
    """
    
    if isinstance(n, tuple):
        n_spoken, n_stackexchange, n_wif = n
    else:
        n_spoken = n_stackexchange = n_wif = n
    if isinstance(start_item, tuple):
        start_item_spoken, start_item_stackexchange, start_item_wif = start_item
    else:
        start_item_spoken = start_item_stackexchange = start_item_wif = start_item
    
    run_prompt_generation_spoken(n_spoken, start_item_spoken, sample = 500)
    run_prompt_generation_stackexchange(n_stackexchange, start_item_stackexchange, sample = 1000)
    run_prompt_generation_wif(n_wif, start_item_wif, sample = 104)

def check_balanced_dataset() -> None:
    """
    Returns:
        A dictionary with the counts of each model output in the generated dataset.
    """
    
    from collections import Counter
    
    counts = Counter()
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            record = json.loads(line)
            counts[record["model_name"]] += 1
    print(counts)

def check_marginals_and_cooccurrence(computed_pairs: tuple[list[str], list[str]], model_list: list[str] = MODEL_LIST) -> None:
    """Checks the marginal frequencies and co-occurrence counts of model pairs in the generated dataset."""
    from collections import Counter

    pairs = list(zip(*computed_pairs))
    marginal = Counter(m for pair in pairs for m in pair)
    cooccurrence = Counter(pairs)
    
    print("Marginal frequency per model:")
    for model, count in marginal.most_common():
        print(f"  {model}: {count} ({count / sum(marginal.values()):.2%})")
    print(f"Number of distinct pairs covered: {len(cooccurrence)} / {len(model_list) * (len(model_list) - 1)}")
    print(f"Standard deviation of pair counts: {(sum((c - sum(cooccurrence.values())/len(cooccurrence))**2 for c in cooccurrence.values()) / len(cooccurrence)) ** 0.5:.2f}")

if __name__ == "__main__":
    run_prompt_generation()