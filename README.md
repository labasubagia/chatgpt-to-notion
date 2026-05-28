# ChatGPT To Notion

A command-line tool for downloading ChatGPT image generations, embedding prompts in PNG metadata, and uploading the results to Notion.

## Features

- Download recent ChatGPT image generations
- Save prompt text into PNG metadata
- Upload images to a Notion database
- Store image generations in a local SQLite database
- Optionally delete uploaded conversations from ChatGPT

## Installation

### Prerequisites

- Python 3.13+
- `uv` (recommended) or `pip`

### Setup

```bash
uv sync
cp config.toml.example config.toml
```

Edit `config.toml` with your Notion credentials and one or more ChatGPT accounts:

```toml
[shared]
user_agent = "your_user_agent"

[notion]
api_key = "your_notion_integration_token"
database_id = "your_database_id"

[accounts.personal]
authorization_token = "your_auth_token"

[accounts.work]
authorization_token = "your_other_auth_token"
notion_database_id = "optional_override_database_id"
```

## Usage

Show CLI help:

```bash
uv run chatgpt-to-notion --help
```

Upload recent generations:

```bash
uv run chatgpt-to-notion upload-to-notion \
  --account personal \
  --limit 100
```

Run every configured account in sequence:

```bash
uv run chatgpt-to-notion upload-to-notion
```

Replay from saved history instead of fetching live data (today's data only):

```bash
uv run chatgpt-to-notion upload-to-notion --from-history
```

Load all data from history (not just today's):

```bash
uv run chatgpt-to-notion upload-to-notion --from-history --all
```

Verify saved history against Notion directly (today's data only):

```bash
uv run chatgpt-to-notion upload-to-notion --verify-history
```

Load all data and verify against Notion:

```bash
uv run chatgpt-to-notion upload-to-notion --verify-history --all
```

Check which accounts are ready:

```bash
uv run chatgpt-to-notion account-status --timezone Asia/Singapore
```

Clean generated output:

```bash
uv run chatgpt-to-notion clean-output-path
```

Backup the local SQLite database:

```bash
uv run chatgpt-to-notion backup-db ./backups/chatgpt.db
```

Restore the local SQLite database from a backup:

```bash
uv run chatgpt-to-notion restore-db ./backups/chatgpt.db
```

## CLI Notes

- Image generations are stored in a local SQLite database (`output/chatgpt.db`)
- `uploaded_at` is used to skip repeated Notion uploads
- `--check-notion-api` bypasses the local DB cache and checks Notion directly
- `--image-folder` defaults to `output/images/`, override with `--image-folder <path>`
- `--mode single` (default) processes files one-by-one; `--mode batch` processes in parallel
- `--remove` deletes uploaded conversations after verification

## Make Targets

```bash
make upload_to_notion
make upload_to_notion_remove
make clean-output-path
```

## Project Structure

```text
project/
├── src/
│   └── chatgpt_to_notion/
│       ├── cli/
│       ├── domain/
│       ├── services/
│       ├── adapters/
│       └── shared/
├── config.toml.example
├── tests/
└── output/
    ├── images/
    └── chatgpt.db
```

## Getting Credentials

1. Open browser developer tools.
2. Visit `https://chatgpt.com`.
3. Inspect authenticated network requests.
4. Copy the bearer token and user agent.

## Testing

```bash
uv run pytest
uv run pyrefly check src/
uv run ruff check .
```

## Documentation

Comprehensive specification and architecture documentation can be found in the [docs](docs) directory:
- [Product Requirement Document (PRD)](docs/requirement.md): Vision, user goals, and functional sync pillars.
- [Technical Requirements Document (TRD)](docs/technical.md): Asynchronous tech stack, SQLite index schema, technology-agnostic synchronization logic flowcharts, and account status/cooldown calculation algorithms.
- [User Playbook & Deployment Guide](docs/user_guide.md): In-depth guide on Notion connections, harvesting browser authorization tokens, config parameters, and background cron/systemd scheduling.

