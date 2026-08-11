import argparse
import json
import os
import random

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm
from typing import Literal, Any, Tuple, TypedDict

SEED = 2026

DEFAULT_MODEL = "phi4"
DEFAULT_PROVIDER = "ollama"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT_PATH, "data", "reactions.json")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT_PATH, "data", "llm_as_a_judge_pairwise_preferences.jsonl")

GOLD_DATASET_PATH = os.path.join(PROJECT_ROOT_PATH, "data", "students_teacher_gold.json")

class JudgeOutput(BaseModel):
    comment: str = Field(description="Short rationale explaining the choice in English")
    labels: list[Literal["complete", "correct", "relevant", "concise", "scaffolding", "understandable"]] = Field(
        description="Applicable pedagogical labels"
    )
    preferred_model: Literal["a", "b", "both_equal"] = Field(description="Preferred response slot")
    rating: int = Field(ge=1, le=5, description="Discrete rating from 1 to 5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean LLM-as-a-Judge script.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to source reactions JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to output JSONL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge model name.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["ollama", "openrouter"], help="API provider.")
    parser.add_argument("--limit", type=int, default=None, help="Max items to process.")
    parser.add_argument("--force", action="store_true", help="Overwrite output file.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    return parser.parse_args()


def build_client(provider: str) -> OpenAI:
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENROUTER_API_KEY.")
        return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    
    return OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL),
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    )


def extract_assistant_message(conversation: Any) -> str:
    if not isinstance(conversation, list):
        return ""
    for message in reversed(conversation):
        if isinstance(message, dict) and message.get("role") in ("assistant", "model"):
            content = message.get("content", "")
            return content.strip() if isinstance(content, str) else ""
    return ""

def format_conversation(conversation: list[dict[str, str]]) -> str:
    """
    Format a conversation into a string representation for the system prompt.
    """
    formatted_lines = []
    for message in conversation:
        role = message.get("role", "")
        content = message.get("content", "")
        
        # Format depending on the role
        if role == "user":
            formatted_lines.append(f"########\nLEARNER:\n########\n\n{content}\n\n########\n")
        elif role in ("assistant", "model"):
            formatted_lines.append(f"########\nTUTOR:\n########\n\n{content}\n\n########\n")
            
    return "\n".join(formatted_lines)

def format_example(example: dict[str, str]) -> str:
    conversation_a, conversation_b = example.get("conversation_a"), example.get("conversation_b")
    preferred_model = example.get("preferred_model", "")
    
    if not preferred_model:
        raise ValueError("Example must contain a 'preferred_model' key.")
    
    if preferred_model not in ("a", "b", "both_equal"):
        raise ValueError("Preferred model must be one of 'a', 'b', or 'both_equal'.")
    
    if not conversation_a or not conversation_b:
        raise ValueError("Example must contain both 'conversation_a' and 'conversation_b'.")

    out = ""
    out += f"\n========\nPREFERRED MODEL:{preferred_model if preferred_model == 'both_equal' else preferred_model.upper()}\n========\n\n"
    
    out += "\n===========\nRESPONSE A:\n===========\n\n"
    out += format_conversation(conversation_a)
    out += "\n===========\nRESPONSE B:\n===========\n\n"
    out += format_conversation(conversation_b)
    
    return out

def get_formatted_randomised_few_shot_examples(*examples, seed: int | None = SEED) -> str:
    rng = random.Random(seed)
    examples = list(examples)
    rng.shuffle(examples)
    return "\n--------\n".join(format_example(example) for example in examples)

def select_few_shot_example(input_gold_dataset_path: str = GOLD_DATASET_PATH, outcome: Literal["both_equal", "winner"] = "both_equal", most_annotators_agreeing: bool = True, seed: int | None = None) -> dict[str, Any]:
    """
    Find relevant examples for few-shot prompting (just a random one per call).
    We care about examples where there is a consensus among annotators on a given item, i.e. everybody agrees on what the best answer is.
    
    Args:
        input_gold_dataset_path (str): Path to the gold dataset JSON file.
        outcome (Literal["both_equal", "winner"]): The desired outcome to filter examples by. "both_equal" means all annotators agreed that both responses are equal, while "winner" means all annotators agreed that one response is better than the other (either a, or b, which ultimately doesn't matter).
        longest (bool): Whether to select conversations only from those having the highest number of annotators agreeing.
        seed (int | None): Random seed for reproducibility. If None, the random selection will be non-deterministic.

    Returns:
        dict[str, Any]: A randomly selected example from the gold dataset that matches the specified outcome
    
    """
    
    if not os.path.exists(input_gold_dataset_path):
        raise FileNotFoundError(f"Gold dataset not found at: {input_gold_dataset_path}")

    with open(input_gold_dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Select all examples where there is a consensus of "both_equal" / "a" / "b" among the annotators
    teacher_annotation_data = []
    for item in data:
        teacher_annotation_data.append({"model_a_name": item["model_a_name"], "model_b_name": item["model_b_name"], "conversation_a": item["conversation_a"], "conversation_b": item["conversation_b"], "teacher_annotation_data": item.get("teacher_annotation_data", [])})
    
    if most_annotators_agreeing:
        # Sort the teacher_annotation_data by the number of annotators agreeing (length of teacher_annotation_data) in descending order and keep only the items with the maximum number of annotators agreeing
        teacher_annotation_data.sort(key=lambda x: len(x.get("teacher_annotation_data", [])), reverse=True)
        max_annotators = len(teacher_annotation_data[0].get("teacher_annotation_data", []))
        teacher_annotation_data = [item for item in teacher_annotation_data if len(item.get("teacher_annotation_data", [])) == max_annotators]
    
    contenders = []
    rng = random.Random(seed)
    
    for annotations in teacher_annotation_data:
        items = annotations.get("teacher_annotation_data", [])
        preferred_models = [ann.get("preferred_model") for ann in items if ann.get("preferred_model") in ("a", "b", "both_equal")]
        # If all elements of the list are equal, and we have at least 2 annotations, we can consider it a consensus:
        if len(preferred_models) >= 2 and all(x == preferred_models[0] for x in preferred_models):
            if most_annotators_agreeing:
                # Keep series with the most annotators agreeing (i.e. the longest series of identical preferred_model values)
                max_agreeing_count = max(preferred_models.count(x) for x in set(preferred_models))
            
            is_match = False
            if outcome == "both_equal" and preferred_models[0] == "both_equal":
                is_match = True
            elif outcome == "winner" and preferred_models[0] in ("a", "b"):
                is_match = True
            
            if is_match:
                contenders.append({"conversation_a": annotations.get("conversation_a"), "conversation_b": annotations.get("conversation_b"), "model_a_name": annotations.get("model_a_name"), "model_b_name": annotations.get("model_b_name"), "preferred_model": preferred_models[0]})
            
    if not contenders:
        raise ValueError(f"No examples found with consensus outcome '{outcome}' in the gold dataset.")
    
    return rng.choice(contenders)

JUDGE_SYSTEM_PROMPT = f"""You are an impartial judge for pairwise responses produced by a French-learning assistant.
Compare Response A and Response B for the user's request based on pedagogical value.

1. Prefer the response that provides better pedagogical value (choose "a", "b", or "both_equal").
2. Rate the preferred response on a discrete scale of 1 to 5.
3. Select all applicable labels from this exact list: ["complete", "correct", "relevant", "concise", "scaffolding", "understandable"].
4. Provide a brief rationale explaining your choice in English.

You can rely on the following examples:

{get_formatted_randomised_few_shot_examples(
    select_few_shot_example(outcome="both_equal", most_annotators_agreeing=False, seed=2143),
    select_few_shot_example(outcome="winner", most_annotators_agreeing=False, seed=5316),
    select_few_shot_example(outcome="winner", most_annotators_agreeing=True, seed=4182)
)}
"""

def call_judge(client: OpenAI, model: str, question: str, response_a: str, response_b: str, seed: int = SEED) -> dict[str, Any]:
    prompt = f"========USER QUESTION:========\n{question}\n\n========RESPONSE A:========\n{response_a}\n\n========RESPONSE B:========\n{response_b}"

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=JudgeOutput,
        temperature=0,
        seed=seed,
    )
    
    result: JudgeOutput = response.choices[0].message.parsed
    return result.model_dump()

def main() -> None:
    args = parse_args()
    load_dotenv()
    client = build_client(args.provider)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Process items based on dataset type
    processed_items = []
    label_list = ["complete", "correct", "relevant", "concise", "scaffolding", "understandable"]

    if "profs_phase2" in args.input:
        synthetic_conversations_path = os.path.join(PROJECT_ROOT_PATH, "data", "synthetic_conversations.json")
        if not os.path.exists(synthetic_conversations_path):
            raise FileNotFoundError(f"Synthetic conversations file not found for joining at: {synthetic_conversations_path}")
        
        with open(synthetic_conversations_path, "r", encoding="utf-8") as f_react:
            reactions_data = json.load(f_react)
        
        reactions_map = {r.get("reaction_id") or r.get("id"): r for r in reactions_data}
        
        # Unroll each teacher annotation so each line in output corresponds to a unique human evaluation
        for ann in items:
            src_id = ann.get("source_reaction_id")
            if src_id in reactions_map:
                merged = dict(reactions_map[src_id])
                
                pref = ann.get("preferred_model")
                if pref == "a":
                    human_pref = merged.get("model_a_name")
                elif pref == "b":
                    human_pref = merged.get("model_b_name")
                else:
                    human_pref = "both_equal"
                
                merged["annotator_id"] = ann.get("annotator_id")
                merged["human_preferred_model"] = human_pref
                merged["human_labels"] = [label for label in label_list if ann.get(label) is True]
                merged["human_rating"] = ann.get("rating")
                merged["human_rationale"] = ann.get("comment")

                processed_items.append(merged)
    else:
        for item in items:
            pref = item.get("preferred_model")
            if pref == "a":
                item["human_preferred_model"] = item.get("model_a_name")
            elif pref == "b":
                item["human_preferred_model"] = item.get("model_b_name")
            else:
                item["human_preferred_model"] = "both_equal"
            
            labels_dict = item.get("labels", {})
            if isinstance(labels_dict, dict):
                item["human_labels"] = [label for label in label_list if labels_dict.get(label) is True]
            else:
                item["human_labels"] = []
                
            item["human_rating"] = item.get("rating")
            item["human_rationale"] = item.get("comment")
            processed_items.append(item)

    rng = random.Random(args.seed)
    rng.shuffle(processed_items)
    
    if args.limit:
        processed_items = processed_items[:args.limit]

    if args.force and os.path.exists(args.output):
        os.remove(args.output)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "a", encoding="utf-8") as out_file:
        for item in tqdm(processed_items, desc="Evaluating pairs"):
            resp_a = extract_assistant_message(item.get("conversation_a"))
            resp_b = extract_assistant_message(item.get("conversation_b"))
            question = item.get("question_content") or item.get("opening_msg", "")

            if not resp_a or not resp_b or not question:
                continue

            for is_swapped in (False, True):
                current_a = resp_b if is_swapped else resp_a
                current_b = resp_a if is_swapped else resp_b
                name_a = item.get("model_b_name") if is_swapped else item.get("model_a_name")
                name_b = item.get("model_a_name") if is_swapped else item.get("model_b_name")

                try:
                    evaluation = call_judge(client, args.model, question, current_a, current_b, args.seed)
                except Exception as e:
                    print(f"Error evaluating item {item.get('reaction_id')}: {e}")
                    continue

                pref = evaluation["preferred_model"]
                if pref == "a":
                    final_pref_model = name_a
                elif pref == "b":
                    final_pref_model = name_b
                else:
                    final_pref_model = "both_equal"

                record = {
                    "reaction_id": item.get("reaction_id"),
                    "annotator_id": item.get("annotator_id"),
                    "is_swapped": is_swapped,
                    "question_content": question,
                    "response_a": current_a,
                    "response_b": current_b,
                    "model_a_name": item.get("model_a_name"),
                    "model_b_name": item.get("model_b_name"),
                    "presented_model_a_name": name_a,
                    "presented_model_b_name": name_b,
                    "human_preferred_model": item.get("human_preferred_model"),
                    "human_rationale": item.get("human_rationale"),
                    "human_labels": item.get("human_labels"),
                    "human_rating": item.get("human_rating"),
                    "judge_model": args.model,
                    "judge_preferred_model": final_pref_model,
                    "judge_rationale": evaluation["comment"],
                    "judge_labels": evaluation["labels"],
                    "judge_rating": evaluation["rating"],
                }

                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()

    print(f"Evaluation complete. Saved to {args.output}")
    
if __name__ == "__main__":
    main()