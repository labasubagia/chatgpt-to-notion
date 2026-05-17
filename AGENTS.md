# Repository Guidelines

## Project Structure & Module Organization

Top-level Python modules hold the application code: `main.py` contains the Typer CLI, `chatgpt.py` handles ChatGPT history/download flows, `notion.py` wraps Notion uploads, `img.py` writes PNG metadata, `util.py` contains shared helpers, `models.py` defines Pydantic models, and `db.py` manages SQLite storage. Tests live under `tests/` and mirror the module layout with files such as `tests/test_main.py` and `tests/test_chatgpt.py`. Example configuration lives in `config.toml.example`.

## Build, Test, and Development Commands

- `uv sync`: install runtime and dev dependencies into `.venv`.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/test_main.py`: run a focused test file.
- `uv run mypy .`: run static type checking.
- `uv run ruff check .`: run lint checks.
- `make upload_to_notion`: run the CLI upload flow with the default debug image folder.
- `python main.py --help`: inspect the current CLI surface.

## Coding Style & Naming Conventions

Use Python 3.13, 4-space indentation, and type hints on public functions. Follow the existing flat-module structure; do not introduce new packages or abstractions unless the change clearly requires them. Use `snake_case` for functions, variables, and test names. Keep CLI-facing names simple and user-oriented, matching the current command style such as `upload-to-notion`. Ruff enforces import sorting and core lint rules; match existing formatting before relying on the formatter.

## Testing Guidelines

Pytest is the test runner, with `pytest-asyncio` for async code and `pytest-mock` for mocking. Add or update tests in the matching `tests/test_*.py` file for every behavioral change. Prefer focused unit/integration tests that explain intent, not just output. Keep test names descriptive, e.g. `test_upload_to_notion_from_history_flag`. Use the `isolated_db` fixture for tests that need database access. Run `uv run pytest` before opening a PR.

## Agent Workflow

When using coding agents in this repository, prefer small, verifiable changes. State assumptions when behavior is ambiguous, avoid speculative abstractions, and keep edits limited to the files required by the task. Read the relevant module and immediate callers before changing shared helpers such as `util.py` or `main.py`. For bug fixes and refactors, define success in terms of checks you can run here, usually `uv run pytest` or a focused test file. If a change creates unused imports, variables, or tests, remove only the orphans introduced by that change.

## Commit & Pull Request Guidelines

Recent history follows concise Conventional Commit-style subjects such as `refactor: simplify cli naming` and `fix: improve conversation deletion logic`. Keep commits scoped to one logical change. PRs should include a short summary, the user-visible impact, and a testing section listing commands run, e.g. `uv run pytest`. Link the relevant issue or context when available.

## Security & Configuration Tips

Do not commit `config.toml`, tokens, cookies, or downloaded output. Start from `config.toml.example`, keep secrets local, and use per-account configuration only when needed for the current task.
