# Repository Guidelines

## Project Structure & Module Organization

The application is structured as a clean, layered pythonic package under `src/chatgpt_to_notion/`:
- `cli/`: Typer CLI application entry point and command routers (`app.py`, `commands/`).
- `services/`: Main business logic pipeline coordinators (e.g. `history_service.py`, `account_status_service.py`).
- `adapters/`: External integration bridges (e.g. `chatgpt_api.py` for ChatGPT, `notion_api.py` for Notion, and `sqlite_store.py` for SQLite datastore).
- `shared/`: Utility helper tools (e.g. `logging.py`, `http.py`, `time.py`).

Tests reside under `tests/` and mirror this modular package architecture (e.g. `tests/adapters/test_chatgpt_api.py`, `tests/shared/test_logging.py`).
Example configuration is provided in `config.toml.example`.

For detailed information on Product and Technical specifications:
- Refer to [docs/requirement.md](docs/requirement.md) for the Product Requirement Document (PRD).
- Refer to [docs/technical.md](docs/technical.md) for the Technical Requirements Document (TRD).
- Refer to [docs/user_guide.md](docs/user_guide.md) for the User Playbook & Deployment Guide (Credential acquisition and scheduling).

## Build, Test, and Development Commands

- `uv sync`: install runtime and dev dependencies into `.venv`.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/test_main.py`: run a focused test file.
- `uv run pyrefly check src/`: run static type checking.
- `uv run ruff check .`: run lint checks.
- `make upload_to_notion`: run the CLI upload flow with the default debug image folder.
- `uv run chatgpt-to-notion --help`: inspect the current CLI surface.

## Coding Style & Naming Conventions

Use Python 3.13, 4-space indentation, and type hints on public functions. Follow the existing flat-module structure; do not introduce new packages or abstractions unless the change clearly requires them. Use `snake_case` for functions, variables, and test names. Keep CLI-facing names simple and user-oriented, matching the current command style such as `upload-to-notion`. Ruff enforces import sorting and core lint rules; match existing formatting before relying on the formatter.

## Testing Guidelines

Pytest is the test runner, with `pytest-asyncio` for async code and `pytest-mock` for mocking. Add or update tests in the matching `tests/test_*.py` file for every behavioral change. Prefer focused unit/integration tests that explain intent, not just output. Keep test names descriptive, e.g. `test_upload_to_notion_from_history_flag`. Use the `isolated_db` fixture for tests that need database access. Run `uv run pytest` before opening a PR.

## Agent Workflow

When using coding agents in this repository, prefer small, verifiable changes. State assumptions when behavior is ambiguous, avoid speculative abstractions, and keep edits limited to the files required by the task. Read the relevant module and immediate callers before changing shared helpers such as `util.py` or `main.py`. For bug fixes and refactors, define success in terms of checks you can run here, usually `uv run pytest` or a focused test file. If a change creates unused imports, variables, or tests, remove only the orphans introduced by that change.

## Commit & Pull Request Guidelines

Recent history follows concise Conventional Commit-style subjects such as `refactor: simplify cli naming` and `fix: improve conversation deletion logic`. Keep commits scoped to one logical change. PRs should include a short summary, the user-visible impact, and a testing section listing commands run, e.g. `uv run pytest`. Link the relevant issue or context when available.

## Development Process

**All changes must go through git worktrees and Pull Requests (using GitHub CLI):**

1. **Create worktree** → `git worktree add ./tree/<branch> -b <branch> main`
2. **Code changes** → implement in worktree directory
3. **Verify locally** → `uv run pytest`, `uv run ruff check .`, `uv run pyrefly check src/`
4. **Push & PR** → `gh pr create --title "..." --body-file ... --base main --head <branch>`
5. **Monitor CI** → `gh pr checks <number> --watch` wait for all checks (pre-commit, tests, security) to pass
6. **Verify PR** → `gh pr view <number>` confirm no conflicts, branch up-to-date, mergeable, squash mode
7. **Squash merge** → `./scripts/merge-pr.sh <number>` (merges, deletes local+remote branch, removes worktree) only after CI passes **and explicit user approval**
8. **Cleanup** → Handled by `merge-pr.sh`; verify with `git worktree list`

**Branch naming:** `fix/...`, `feat/...`, `refactor/...`, `chore/...`
**Direct pushes to `main` are prohibited.** Branch protection rules enforce:
- Required status checks (pre-commit, tests)
- No force pushes
- No direct commits
**Never auto-merge PRs without explicit user permission.** Wait for confirmation before merging.
**Never commit:** `config.toml`, tokens, cookies, downloaded output, `.venv/`, `__pycache__/`.

## PR Format

**Ensure good markdown formatting** in PR descriptions. Use this template:

### Summary
- Brief description of the changes.

### Changes Made
- Change 1
- Change 2

### Testing
- [ ] Tested locally
- [ ] Tests passed (`uv run pytest`)
- **Do not include test logs/output in PR description**
### Notes
- Any additional information (optional)
## Security & Configuration Tips

Do not commit `config.toml`, tokens, cookies, or downloaded output. Start from `config.toml.example`, keep secrets local, and use per-account configuration only when needed for the current task.
