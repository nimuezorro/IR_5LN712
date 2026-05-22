# PantryPal

PantryPal is a modular cooking assistant built for an Information Retrieval assignment. It helps users decide what to cook from pantry ingredients by combining persistent memory, local recipe retrieval, ingredient substitution retrieval, web-search fallback, and optional OpenAI-compatible LLM answer synthesis.

The project is intentionally retrieval-first: the LLM is not the source of truth. It receives retrieved evidence from PantryPal tools and turns that evidence into a grounded final answer.

Report and video demo are in `reports/`

## Overview

PantryPal supports:

- Pantry memory for ingredients the user has available
- Preference memory for dietary preferences, disliked ingredients, and favorite cuisines
- Local recipe retrieval from `data/recipes.json`
- Ingredient substitution retrieval from `data/substitutions.json`
- Web-search fallback when local retrieval is weak or the user asks outside the local corpus
- OpenAI-compatible LLM synthesis, including Berget.AI-compatible configuration
- Interactive CLI chat with a lightweight ReAct-style loop

## Architecture

```text
pantrypal/
  agent/       Planner, prompts, and ReAct-style agent loop
  tools/       Modular tool interface and tool implementations
  retrieval/   Recipe and substitution retrieval logic
  memory/      JSON-backed pantry and preference memory
  llm/         OpenAI-compatible LLM client wrapper
  cli/         Interactive command-line entry point
  utils/       Environment configuration and logging
reports/       PDF report for the assignment and video demo
data/          Recipes, substitutions, and local memory JSON files
tests/         Unit and smoke tests
```

The main separation of concerns is:

- `memory/` stores user state and does not depend on the agent.
- `retrieval/` ranks recipes and substitutions and does not depend on the LLM.
- `tools/` expose memory, retrieval, and web search through a shared interface.
- `agent/` decides which tools to call and builds a grounded final prompt.
- `llm/` handles provider configuration, retries, timeout handling, and chat completion calls.

## IR Design Rationale

PantryPal models cooking assistance as an information retrieval problem:

1. Parse the user need: recipe search, pantry update, substitution, technique, or fallback lookup.
2. Retrieve evidence from local structured corpora and memory.
3. Rank candidate recipes or substitutions using relevance signals.
4. Estimate confidence and decide whether local evidence is sufficient.
5. Use web search only when local retrieval is weak or out of scope.
6. Generate a final answer grounded in retrieved context.

Local recipe retrieval uses a BM25-style scoring component combined with cooking-specific ranking signals:

- ingredient overlap
- partial ingredient matches
- pantry ingredient boosts
- missing ingredient penalties
- dietary preference filtering
- retrieval confidence scores

This makes the system inspectable: the assistant can report which local recipes matched, which ingredients matched, what was missing, and when fallback search was needed.

## Memory

PantryPal loads memory automatically on startup from:

```text
data/pantry_memory.json
data/preferences_memory.json
```

Pantry memory stores normalized ingredient names:

```text
I usually keep chickpeas and feta.
add rice, eggs, soy sauce
remove eggs
list pantry
```

Preference memory stores:

- dietary preferences, such as `vegetarian` or `vegan`
- disliked ingredients, such as `mushrooms`
- favorite cuisines, such as `Greek`

Examples:

```text
I hate mushrooms.
I am vegetarian.
My favorite cuisine is Greek.
list preferences
```

Use `PANTRYPAL_MEMORY_DIR` to point PantryPal at a different memory directory.

## Local Recipe Retrieval

Recipes are loaded from:

```text
data/recipes.json
```

Each recipe uses this schema:

```json
{
  "title": "",
  "ingredients": [],
  "instructions": [],
  "dietary_tags": [],
  "cuisine": "",
  "prep_time": ""
}
```

The retriever returns:

- top-k ranked recipes
- confidence scores
- matched ingredients
- missing ingredients
- a `use_web_fallback` signal

If confidence is below `PANTRYPAL_RETRIEVAL_CONFIDENCE_THRESHOLD` or fewer than three relevant recipes are found, the agent automatically triggers web-search fallback.

## Substitution Retrieval

Substitutions are loaded from:

```text
data/substitutions.json
```

Each entry contains:

- missing ingredient
- substitute candidates
- flavor similarity
- cooking-use notes

The substitution retriever ranks substitutes using:

- match quality for the missing ingredient
- flavor similarity
- whether the substitute is already in the pantry

Example structured tool input:

```python
{
    "missing": "heavy cream",
    "available": ["milk", "butter", "greek yogurt"],
}
```

The output includes the best substitute, alternatives, confidence score, tradeoff explanation, and structured evidence.

## Web Search Fallback

The web-search tool is provider-based and lives in:

```text
pantrypal/tools/web_search_tool.py
```

Supported search types:

- `recipe`
- `substitution`
- `technique`

The default provider is `mock`, which is deterministic and works without network access. A generic HTTP provider can be configured through environment variables.

Web search is used automatically when:

- local recipe retrieval confidence is low
- fewer than three relevant local recipes are found
- the user asks for an unfamiliar cuisine
- the user asks a cooking technique question

Search results include titles, snippets, URLs, provider name, result type, and confidence scores where available.

## Running Locally

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local environment file if you want LLM access or custom settings:

```bash
cp .env.example .env
```

Edit `.env` with your own API key and model. PantryPal loads `assignment_2/.env` automatically on startup. The checked-in `.env.example` contains placeholders only; real `.env` files are ignored by git.

If you prefer exporting the variables into your shell manually, you can also run:

```bash
set -a
source .env
set +a
```

Run the CLI:

```bash
python -m pantrypal.cli.main
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Environment Variables

See [.env.example](.env.example) for a complete template.

LLM configuration:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.berget.ai/v1"
export OPENAI_MODEL="your-model"
export OPENAI_TIMEOUT_SECONDS="30"
export OPENAI_MAX_RETRIES="2"
```

Retrieval and memory configuration:

```bash
export PANTRYPAL_MEMORY_DIR="data"
export PANTRYPAL_DATA_DIR="data"
export PANTRYPAL_RETRIEVAL_CONFIDENCE_THRESHOLD="0.35"
```

Web-search configuration:

```bash
export PANTRYPAL_WEB_SEARCH_PROVIDER="mock"      # mock, disabled, or http
export PANTRYPAL_WEB_SEARCH_ENDPOINT="https://example-search.local/search"
export PANTRYPAL_WEB_SEARCH_API_KEY="your-search-key"
```

Logging:

```bash
export PANTRYPAL_LOG_LEVEL="INFO"
```

Without `OPENAI_API_KEY`, PantryPal still runs and prints a grounded local fallback response using retrieved tool context.

## Example Commands

```text
add rice, eggs, soy sauce
I usually keep chickpeas and feta.
I hate mushrooms.
I am vegetarian.
My favorite cuisine is Greek.
what can I cook with rice and eggs?
substitute heavy cream
how to temper chocolate
Suggest a Peruvian dinner recipe
list pantry
list preferences
```

## Example Transcript

```text
you> I usually keep rice and eggs.
pantrypal> Added: rice, eggs

you> what can I cook with rice and eggs?
pantrypal> Final Answer:
I retrieved context for: what can I cook with rice and eggs?
Pantry memory: eggs, rice.
Local recipe matches: Egg Fried Rice, Spinach Feta Omelette, Black Bean Breakfast Tacos.

Sources used:
pantry memory, preferences memory, local recipe retrieval

Tool trace:
Thought: Check pantry memory before recommending food.
Action: pantry.run
Observation: Pantry: eggs, rice

Thought: Retrieve local recipes relevant to the question.
Action: recipe_search.run
Observation: - Egg Fried Rice ...
```

With an API key configured, the same retrieved context is passed to the LLM for a more natural final answer.

## Extensibility

Tools share a small interface:

```python
name: str
description: str
run(input_data: dict) -> dict
```

This makes it straightforward to add future tools, such as:

- nutrition lookup
- shopping-list generation
- vector retrieval
- real web-search providers
- meal planning
- user feedback logging

Retrievers can also be swapped independently. For example, the BM25-style recipe retriever could be replaced by BM25 plus embeddings without changing memory, tools, or the LLM wrapper.

## Assignment Requirement Mapping

| Requirement | PantryPal implementation |
| --- | --- |
| Interactive chatbot | `pantrypal/cli/main.py` |
| Agent loop | ReAct-style loop in `pantrypal/agent/loop.py` |
| Pantry memory | JSON-backed `PantryMemory` |
| Preferences memory | JSON-backed `PreferencesMemory` |
| Local recipe retrieval | BM25-style `RecipeRetriever` over `data/recipes.json` |
| Substitution retrieval | Ranked `SubstitutionRetriever` over `data/substitutions.json` |
| Web fallback | Provider-based `WebSearchTool` |
| OpenAI-compatible LLM | `LLMClient` using OpenAI Python SDK |
| Environment config | `pantrypal/utils/config.py` |
| Modular tool system | Shared `Tool` interface in `pantrypal/tools/base.py` |
| Grounded answers | Agent combines retrieved context before LLM synthesis |
| Tests | `tests/` unit and smoke tests |
