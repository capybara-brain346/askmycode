# askmycode

askmycode is a Streamlit app for asking questions about local code repositories with a Groq-powered agent. It can inspect whitelisted repos, search code, read files, and synthesize grounded answers from tool results.

## What it does

- Lists available repos from `config.json` and `repos/`
- Lets you scope questions with `@repo_name`
- Uses these tools:
  - `list_repos`
  - `get_file_tree`
  - `read_file_tool`
  - `search_code`
  - `get_repo_metadata`
- Stops after a bounded number of reasoning hops
- Writes app logs to `logs/askmycode.log`

## Setup

1. Create a `.env` file with your Groq key:

   ```env
   OPENROUTER_API_KEY=your_key_here
   ```

2. Configure repositories in `config.json`:

   ```json
   {
     "repos": {
       "capybaradb": "capybara-brain346/capybaradb",
       "knowflow": "capybara-brain346/knowflow"
     }
   }
   ```

   Each value can be a local path, a GitHub URL, or a short `owner/repo` spec. Any directories already present under `repos/` are also added automatically.

3. Install dependencies and run the app:

   ```bash
   uv sync
   uv run streamlit run src/app.py
   ```

## Usage

Ask a question in the chat box. Tag repos to narrow the scope:

```text
How is auth handled in @capynodes-backend?
```

## Tests

The repo includes eval tests under `tests/evals/`:

- `T1` checks tool sequencing
- `G2` checks grounding
- `E2E` runs the full golden set

Run them with:

```bash
uv run pytest
```

The LLM-backed tests require `OPENROUTER_API_KEY`; they are skipped if the key is not set.
