# Composer LLM Memory Interface

This repository demonstrates how to combine a structured "memory" database with
an LLM that emits validated, schema-constrained queries.

## What is included?

* `memory_llm_interface.py` — Python script that
  * builds the SQLite schema described in the prompt,
  * seeds a few example records, and
  * asks an Ollama-served model to translate natural language into a structured
    query.
* `memory.db` — Automatically created the first time the script runs.

The LLM is guided with a JSON schema (via Pydantic) so its response can be
validated before the application touches the database.  This mirrors the
"FriendList" example supplied in the prompt but adapts it to the memory schema.

## Requirements

* Python 3.11+
* [Ollama](https://ollama.com/) running locally with access to the
  `llama3.3:70b` model (or adjust the model name in the script).
* `pip install pydantic ollama`

## Running the demo

```bash
python memory_llm_interface.py
```

The script will:

1. Ensure the SQLite database and tables exist (creating them if necessary).
2. Ask the LLM to emit a `NodeQuery` JSON payload that satisfies a plain-English
   request.
3. Execute the generated query and print the results.

Feel free to modify the `natural_language_request` string or extend the schema
and seed data to fit your needs.
