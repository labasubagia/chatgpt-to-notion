# ChatGPT Image CLI

A command-line tool for downloading ChatGPT image generations, embedding prompts in PNG metadata, and uploading the results to Notion.

## Features

- Download recent ChatGPT image generations
- Save prompt text into PNG metadata
- Upload images to a Notion database
- Keep per-account history CSVs for replay and verification
- Optionally delete uploaded conversations from ChatGPT

## Installation

### Prerequisites

- Python 3.13+
- `uv` (recommended) or `pip`

### Setup

```bash
git clone <repository-url>
cd sora
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
python main.py --help
```

Upload recent ChatGPT generations:

```bash
python main.py chatgpt-upload-to-notion \
  --account personal \
  --image-folder images \
  --limit 100
```

Run every configured account in sequence:

```bash
python main.py chatgpt-upload-to-notion
```

Replay from saved history instead of fetching live data:

```bash
python main.py chatgpt-upload-to-notion --from-history
```

Verify saved history against Notion directly:

```bash
python main.py chatgpt-upload-to-notion --verify-history
```

Check which accounts are ready:

```bash
python main.py account-status --timezone Asia/Singapore
```

Clean generated output:

```bash
python main.py clean-output-path
```

## CLI Notes

- History is stored at `output/history/<account>_chatgpt.csv`
- `uploaded_at` is used to skip repeated Notion uploads
- `--check-notion-api` bypasses the CSV shortcut and checks Notion directly
- `--remove-in-chatgpt` deletes uploaded conversations after verification

## Make Targets

```bash
make chatgpt_upload_to_notion
make chatgpt_upload_to_notion_remove
make clean-output-path
```

## Project Structure

```text
sora/
├── main.py
├── chatgpt.py
├── notion.py
├── img.py
├── util.py
├── models.py
├── config.toml.example
└── output/
```

## Getting Credentials

1. Open browser developer tools.
2. Visit `https://chatgpt.com`.
3. Inspect authenticated network requests.
4. Copy the bearer token and user agent.

## Testing

```bash
uv run pytest
```
