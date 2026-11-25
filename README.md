# LLM Response Comparer

![tests](https://github.com/techiscode/llm-response-comparer/actions/workflows/tests.yml/badge.svg)

Send the same prompt to two or more large language models, then score their answers side by side against a rubric, without knowing which model wrote which answer. Results are saved as JSON and as a readable Markdown report.

I use it to compare Claude and other models on the same task, and to practice structured, written evaluation of model output.

## Why blind scoring

If you know an answer came from your favorite model, it's hard not to score it higher. This tool shuffles the responses and labels them **Response A**, **Response B**, and so on. You score each one on every rubric criterion, write a short note, pick the better answer, and explain why. Only then does it reveal which model wrote each response.

## Features

- Sends one prompt to any mix of models: Anthropic (Claude) and OpenAI, plus any OpenAI-compatible API such as a local Ollama server
- Blind A/B scoring in the terminal, with model names hidden until you finish
- Custom rubrics in JSON, with example rubrics for code review and factual answers included
- Requires a written reason for the final preference, not just a number
- Saves every run as JSON (for analysis) and Markdown (for reading or sharing)
- Ask now and score later with `--no-score`, then `score` the saved file
- Records response time and token counts for each model
- Calls the APIs with plain HTTP, so there are no vendor SDKs to install
- Covered by unit tests, with all API calls mocked so the tests run offline

## Setup

```bash
git clone https://github.com/techiscode/llm-response-comparer.git
cd llm-response-comparer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API keys to `.env`. You only need keys for the providers you use.

```
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

## Usage

Models are written as `provider:model-name`. Use current model names from each provider's documentation.

```bash
python -m llm_compare run "Write a Python function that checks if a string is a palindrome" \
  -m anthropic:claude-sonnet-5-5 \
  -m openai:your-openai-model \
  --rubric rubrics/code_review.json
```

The tool shows the prompt and each response by label, then walks you through scoring:

```
Score response A
  correctness (1-5): 5
  security (1-5): 5
  readability (1-5): 4
  explanation (1-5): 3
  Note (what was good or wrong): Handles case and spaces, but doesn't explain why

Score response B
  ...

Which response is better overall? (A/B): A
Why is it better? B fails on mixed-case input like "Racecar"

Revealing models:
  A = anthropic:claude-sonnet-5-5  (17/20)  <- preferred
  B = openai:your-openai-model  (14/20)

Saved results/2026-11-19_214512.json and results/2026-11-19_214512.md
```

### Commands

| Command | What it does |
|---|---|
| `run PROMPT -m MODEL -m MODEL` | Ask the models, score the answers, and save the results |
| `run --prompt-file prompt.txt ...` | Read a longer prompt from a file |
| `run ... --no-score` | Save the answers without scoring |
| `score results/FILE.json` | Score a saved comparison |
| `report results/FILE.json` | Rebuild the Markdown report |

Other options for `run`: `--rubric PATH`, `--max-tokens N` (default 1024), and `--out-dir DIR` (default `results`).

## Rubrics

With no `--rubric` option, answers are scored 1 to 5 on accuracy, completeness, and clarity. To use your own rubric, write a JSON file:

```json
{
  "name": "Code review",
  "min_score": 1,
  "max_score": 5,
  "criteria": [
    {"name": "correctness", "description": "Code runs and produces the right result, including edge cases"},
    {"name": "security", "description": "No injection risks, leaked secrets, or unsafe input handling"}
  ]
}
```

Ready-made rubrics are in [`rubrics/`](rubrics/):

- `code_review.json`: correctness, security, readability, explanation
- `factual_answer.json`: accuracy, admitting uncertainty, relevance

## Using an OpenAI-compatible API

The `openai` provider works with any service that follows the OpenAI chat completions format. Set `OPENAI_BASE_URL` in `.env`. For example, for a local Ollama server:

```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
```

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest -v
ruff check .
```

The tests mock every API call, so they need no keys and cost nothing to run.

## Project structure

```
llm_compare/
  providers.py    Calls the Anthropic and OpenAI-compatible APIs
  rubric.py       Loads and validates rubrics
  comparison.py   Blind labeling, saving, and loading results
  scoring.py      Interactive scoring in the terminal
  report.py       Markdown report output
  cli.py          Command-line interface
rubrics/          Example rubrics
tests/            pytest suite for each module
```

## License

MIT
