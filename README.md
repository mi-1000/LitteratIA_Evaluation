This repository contains scripts to aggregate and format data for synthetic language conversational data generation, and analyse the resulting data.

# Installation

This project was developed and tested with Python 3.11.14. Although it may work with other versions, it is recommended to use the same version to avoid compatibility issues.

To install, run the following commands in your terminal:

```bash
git clone https://github.com/mi-1000/LitteratIA_Evaluation.git
cd LitteratIA_Evaluation
python -m venv .venv
source .venv/bin/activate  # On Windows, run instead: `.venv\Scripts\activate`
pip install -r requirements.txt
```

# Usage

To run the full pipeline, follow these steps:

## Ecological dataset construction

1. Download SQL data collected from the [data collection platform](https://github.com/mi-1000/LitteratIA) as a CSV dump (preferably using PgAdmin) -- one file from learners, and one from teachers.
2. Clean, format and merge the data using [`scripts/format_crowdsourced_data.py`](scripts/format_crowdsourced_data.py). This will generate a JSON file containing the formatted data and save it to the [`data/`](data/) folder.

## Synthetic dataset construction

1. Download the source data (see [Sources](#sources) below) and place it in the [`data/`](data/) folder.
    - For FLLOC, as the original data relies on the *.cha format and contains specific artifacts, you can use [`scripts/clean_corpus_spoken.py`](scripts/clean_corpus_spoken.py) to clean the corpus before building the dataset.
2. Build the datasets using the following scripts:
    - [`scripts/build_dataset_spoken.py`](scripts/build_dataset_spoken.py) for the spoken dataset (FLLOC)
    - [`scripts/build_dataset_stackexchange.py`](scripts/build_dataset_stackexchange.py) for the Stack Exchange dataset
    - [`scripts/build_dataset_wif.py`](scripts/build_dataset_wif.py) for the WIF Göteborg dataset
3. Generate synthetic conversations using [`scripts/generate_synthetic_conversation_dataset.py`](scripts/generate_synthetic_conversation_dataset.py).
    - You can tweak the following variables:
        - `SEED`: A fixed random seed for reproducibility;
        - `MAX_WORKERS`: The number of parallel workers to use for generation (the more workers, the faster it will be, but it will also consume more resources and be more likely to hit API rate limits);
        - `MODEL_LIST`: The list of models to use for generation. In our case, the same were used as on the [data collection platform](https://github.com/mi-1000/LitteratIA), but you can add or remove models as you wish;
        - `WEIGHTS`: The weights to use for each model in the generation process. The weights must match the order of the models in `MODEL_LIST` and have the same length. Weights determine how often each model is randomly drawn for generation relative to others. A model with a weight of 2.0 will be drawn twice as often as a model with a weight of 1.0. If you want to use uniform weights, you can set all weights to 1.0;
        - `API_KEY`: The API key for output generation. It is highly advised to set it as an environment variable in an `.env` file that you can create in the [`scripts/`](scripts/) folder, by writing inside the following line: `OPENROUTER_API_KEY=<your_api_key>`. The script will then automatically load it. You can also set it directly in the script, but this is not recommended for security reasons.
    - You can also tweak the system prompt and other generation parameters in [`scripts/system_prompt.py`](scripts/system_prompt.py).
4. Format the synthetic outputs using [`scripts/format_synthetic_data.py`](scripts/format_synthetic_data.py). This will generate a JSON file containing the formatted data and save it to the [`data/`](data/) folder.

## Evaluation

Run the LLM-as-a-judge evaluation pipeline for chosen datasets and models:
- On Linux/MacOS:
```bash
chmod +x scripts/run_all_judge_pipelines.sh # Make the script executable
./scripts/run_all_judge_pipelines.sh
```
- You can tweak the input dataset variables at the beginning as well as the models in the section "*Launch jobs*". This will launch multiple judge runs in parallel for the chosen datasets and models and save the results in the [`data/judge_runs/`](data/judge_runs/) folder. The process IDs are saved in `pids/` (if you want to kill a process, retrieve the ID and run `sudo kill <PID>`, or `sudo kill -9 <PID>` to forcefully end it). Generation logs are stored in the `logs/` folder so you can make sure everything is going right 🙂 You can quickly check logs using `tail -f logs/*.log`.
- If you are on Windows, you can run each command in the script manually with [`scripts/generate_llm_as_a_judge_pairwise_preferences.py`](scripts/generate_llm_as_a_judge_pairwise_preferences.py), or use WSL to run the script as above.

## Analysis

- To compute general descriptive statistics on the data, run the notebook [`scripts/dataset_analysis.ipynb`](scripts/dataset_analysis.ipynb).
- To compute metrics on the data, generate plots and tables, run the notebook [`scripts/compute_metrics.ipynb`](scripts/compute_metrics.ipynb).

# Provided data

All the data used in this project is provided in the [`data/`](data/) folder. It contains:
- The original ecological datasets used for the evaluation (learner conversations + learner and teacher feedback), as well as intermediate files.
- The synthetic outputs generated by the LLMs.
- The results of the LLM-as-a-judge evaluation.

> [!NOTE]
> In the file [`data/synthetic_conversations.json`](data/synthetic_conversations.json), metadata include a key `category` set to either `c` or `f`. These correspond to the taxonomy used in [Nassau and Molle (2025)](https://hal.science/hal-05313736), whereby `c` corresponds to questions about understanding the language, and `f` corresponds to questions about help regarding language production.
> The full taxonomy is as follows:
> - `a`: Guided/casual conversation style (*see key `prompt_configuration`*)
> - `b`: Teacher/language advisor persona (*see key `prompt_configuration`*)
> - `c`: Understanding the language (e.g., grammar, vocabulary, _etc._)
> - `d`: Creating learning activities (_e.g._, drills)
> - `e`: Creating texts (as a model, for comprehension/expression, _etc._)
> - `f`: Helping regarding language production (e.g., drafting, correcting, improving, explaining, _etc._)
> - `g`: Giving ideas

# Sources

We thank the following people and organisations for providing the data used in this project:

- WIF Corpus -- "The written production in learner French (WIF) corpus: a new resource for examining children’s writing" [(Lindqvist, 2026)](https://doi.org/10.1016/j.acorp.2026.100201)
- French Learner Language Oral Corpora (FLLOC) [(Myles, 2006)](https://llds.ling-phil.ox.ac.uk/llds/xmlui/handle/20.500.14106/2495)
- French Stack Exchange (https://french.stackexchange.com) (last free-access data dump from June 2024) -- Data was downloaded from https://archive.org/download/stackexchange (french.stackexchange.com.7z) on the 2nd of June 2026 @ 14h CEST

# Conditions of use

Data including content from the French Stack Exchange is licensed under CC-BY-SA 4.0 and redistributed under the same license.

Everything else, including the scripts, collected data, and other materials in this repository, is licensed under CC-BY-NC-SA 4.0. You are free to use, modify and redistribute the scripts provided in this repository for research purposes, or use and modify them for personal use, provided you cite this repository and the original sources of the data.