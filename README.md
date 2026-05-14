# Sora CLI Tool

A command-line tool for managing and backing up AI-generated images from **Sora** and **ChatGPT**, with integration to store them in **Notion**.

## Features

- 📥 Download generated images from Sora and ChatGPT
- 📝 Embed prompts as PNG metadata
- 📤 Upload images to Notion database
- 🗑️ Cleanup/trash generations after upload
- ⚡ Concurrent downloads and uploads for speed
- 🔄 Automatic retry with exponential backoff for failed requests

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd sora
```

2. **Install dependencies**

Using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

3. **Configure credentials**

Multi-account setup is now supported through `config.toml` and is the recommended path.

```bash
cp config.toml.example config.toml
```

Edit `config.toml` with your Notion credentials and one or more named accounts:

```toml
[shared]
user_agent = "your_user_agent"
cookie_string_base64 = "base64_encoded_cookie_string"

[notion]
api_key = "your_notion_integration_token"
database_id = "your_database_id"

[accounts.personal]
authorization_token = "your_auth_token"

[accounts.work]
authorization_token = "your_other_auth_token"
notion_database_id = "optional_override_database_id"
```

> **Security Note**: Never commit your credential files to version control.

## Usage

### CLI Commands

Run the CLI with `--help` to see all available commands:

```bash
python main.py --help
```

#### Upload Sora Generations to Notion

```bash
python main.py sora-upload-to-notion \
  --account personal \
  --image-folder images \
  --upload-to-notion true \
  --trash-in-sora false \
  --remove-in-sora false
```

**Options:**
- `--image-folder`: Folder under `output/` to store downloaded images (default: `images`)
- `--db-id`: Notion database ID
- `--upload-to-notion`: Whether to upload to Notion (default: `true`)
- `--trash-in-sora`: Move uploaded items to trash (default: `false`)
- `--remove-in-sora`: Permanently delete uploaded items (default: `false`)

#### Upload ChatGPT Image Generations to Notion

```bash
python main.py chatgpt-upload-to-notion \
  --account work \
  --image-folder images \
  --limit 100
```

To run every configured account in sequence, omit `--account`:

```bash
python main.py chatgpt-upload-to-notion
python main.py sora-upload-to-notion
```

Check which accounts are ready before running uploads:

```bash
python main.py account-status --timezone Asia/Singapore
```

`account-status` follows the Colab cooldown logic: rows with `created_at` in the
last 24 hours are still waiting, and accounts with no recent rows are ready. It
reads the same single per-account CSV used by uploads:
`output/history/<account>_chatgpt.csv`.

Upload commands automatically write one merged CSV per account and service:
`output/history/<account>_chatgpt.csv`, `output/history/<account>_sora.csv`, or
`output/history/<account>_sora_trash.csv`. Each run merges new and old rows by
unique `id` and keeps only the last 2 days of data.

After an image is uploaded to Notion, the CSV row gets `uploaded_at`. Future runs
trust that value and skip the Notion "already exists" API check for that image.
Use `--check-notion-api` when you want to verify Notion directly and repair the
CSV state:

```bash
python main.py chatgpt-upload-to-notion --check-notion-api
```

If ChatGPT data has already been deleted, upload from the saved CSV instead of
fetching live ChatGPT generations:

```bash
python main.py chatgpt-upload-to-notion --from-history
```

For recovery/verification, use the shortcut that reads history and checks
Notion directly:

```bash
python main.py chatgpt-upload-to-notion --verify-history
```

**Options:**
- `--image-folder`: Folder under `output/` to store downloaded images (default: `images`)
- `--db-id`: Notion database ID
- `--config`: Path to `config.toml` if not using `./config.toml`
- `--account`: Named account inside `config.toml`; if omitted, all accounts run sequentially
- `--check-notion-api`: Check Notion directly even when `uploaded_at` is already set
- `--from-history`: Use `output/history/<account>_chatgpt.csv` as the generation source
- `--verify-history`: Shortcut for `--from-history --check-notion-api`
- `--limit`: Maximum number of generations to process (default: `100`)
- `--remove-in-chatgpt`: Delete conversations after upload (default: `false`)

#### Cleanup Commands

**Clean up trashed Sora generations:**
```bash
python main.py sora-cleanup-trash
```

**Delete empty Sora tasks:**
```bash
python main.py sora-cleanup-tasks
```

**Clean output directory:**
```bash
python main.py clean-output-path
```

## Project Structure

```
sora/
├── main.py          # CLI entry point with Typer commands
├── sora.py          # Sora API client and operations
├── chatgpt.py       # ChatGPT API client and operations
├── notion.py        # Notion API client and operations
├── img.py           # Image processing (add prompts to PNG metadata)
├── util.py          # Shared utilities (retry logic, path handling, etc.)
├── config.toml.example  # Multi-account configuration template
└── output/          # Default output directory for downloads
```

## Configuration

### TOML Config

| Key | Description | Required |
|-----|-------------|----------|
| `notion.api_key` | Notion integration token | For Notion operations |
| `notion.database_id` | Default Notion database ID | Optional if each account sets `notion_database_id` or `--db-id` is passed |
| `shared.user_agent` | Shared User-Agent header | Recommended |
| `shared.cookie_string_base64` | Shared base64-encoded cookies | Needed for ChatGPT image history |
| `accounts.<name>.authorization_token` | ChatGPT/Sora auth token | Yes |
| `accounts.<name>.user_agent` | Account-specific User-Agent override | Optional |
| `accounts.<name>.cookie_string_base64` | Account-specific cookie override | Optional |
| `accounts.<name>.notion_database_id` | Per-account Notion DB override | Optional |

### How to Get ChatGPT/Sora Credentials

1. Open your browser's Developer Tools (F12)
2. Go to the Network tab
3. Visit [Sora](https://sora.chatgpt.com) or [ChatGPT](https://chatgpt.com)
4. Look for XHR/fetch requests
5. Inspect request headers to find:
   - `Authorization` token
   - `User-Agent`
   - `Cookie` string (encode to base64 for `CHATGPT_COOKIE_STRING_BASE64`)

## Notion Database Setup

Create a Notion database with the following properties:

| Property Name | Type | Description |
|---------------|------|-------------|
| `Name` | Title | Image filename |
| `Image` | Files | The uploaded image |
| `Prompt` | Rich Text | Generation prompt |
| `Model` | Select | Model used (e.g., "Sora") |
| `Face` | Select | Face option used |

## Advanced Usage

### Reading PNG Metadata

After processing, prompts are embedded in PNG files. Use `exiftool` to read them:

```bash
exiftool -Prompt output/images/abc123.png
```

### Dataset CSV Format

Generated CSV files contain:
- `created_at`: Generation timestamp
- `id`: Generation/image ID
- `task_id`/`conversation_id`: Parent task/conversation
- `url`: Original image URL
- `prompt`: Generation prompt
- `uploaded_at`: UTC timestamp set after Notion upload or Notion existence check

## Error Handling

- **HTTP 429 (Rate Limit)**: Automatic retry with exponential backoff
- **HTTP 5xx (Server Error)**: Automatic retry with exponential backoff
- **File Exists**: Skips already downloaded images
- **Network Errors**: Retries up to 5 times before failing

## Performance

Default concurrency settings:
- **Downloads**: 10 concurrent downloads
- **API Requests**: 10 concurrent requests
- **HTTP Timeout**: 30 seconds

Adjust these in `util.py` if needed.

## Testing

This project uses pytest for testing with a pure mocking approach (no real API calls in CI).

### Run Tests

```bash
# Run all tests
uv run pytest

# Run unit tests only (faster, no integration tests)
uv run pytest -m "not integration"

# Run with coverage report
uv run pytest --cov=. --cov-report=html --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_util.py -v

# Run specific test function
uv run pytest tests/test_util.py::TestGetOutputPath::test_relative_path_allowed -v
```

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures and mock helpers
├── test_util.py         # Unit tests for utility functions (26 tests)
├── test_img.py          # Unit tests for image processing (9 tests)
├── test_notion.py       # Integration tests with mocking (11 tests)
├── test_chatgpt.py      # Integration tests with mocking (10 tests)
├── test_sora.py         # Integration tests with mocking (11 tests)
└── test_main.py         # CLI command tests (15 tests)
```

### Test Markers

- `@pytest.mark.integration` - Integration tests using mocked API responses
- `@pytest.mark.smoke` - Tests that require real API access (skip in CI)
- `@pytest.mark.slow` - Slow-running tests

### CI/CD

GitHub Actions runs on every push and pull request:

- **Lint**: `ruff check .`
- **Unit Tests**: pytest with HTML coverage report
- **Integration Tests**: Mocked API tests
- **Build**: Package build verification

Coverage reports are uploaded as GitHub artifacts (available for 7 days).
Download `coverage-report.zip` from the workflow run and open `htmlcov/index.html` in your browser.

## Troubleshooting

### "Missing required configuration values"

Ensure your `config.toml` exists and that the selected account plus shared settings contain all required values.

### "Notion database ID must be a valid ID"

The database ID should be a 32-character string (with or without hyphens).

### Images not uploading to Notion

1. Check that your Notion integration has access to the database
2. Verify the database ID is correct
3. Ensure the database has the required properties

### API rate limiting

The tool includes automatic retry logic, but if you hit rate limits frequently, consider reducing the `--limit` parameter.

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Set up pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```
4. Make your changes
5. Run tests and linting:
   ```bash
   uv run pytest -m "not integration"  # Run unit tests
   uv run ruff check .                 # Run linter
   ```
6. Commit your changes: `git commit -m "Add my feature"`
7. Push to the branch: `git push origin feature/my-feature`
8. Submit a pull request

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) with [ruff](https://github.com/astral-sh/ruff-pre-commit) for automatic code formatting and linting on commit.

Install the hooks:
```bash
uv run pre-commit install
```

The hooks will automatically run on every commit:
- **Format**: `ruff format .`
- **Lint**: `ruff check --extend-select I --fix .`

Run manually on all files:
```bash
uv run pre-commit run --all-files
```

### Testing Requirements

- All new features should include unit tests
- API interactions should be mocked in tests
- Maintain >80% code coverage
- All tests must pass before merging

## Makefile

A `Makefile` is provided to run the most common CLI workflows (these mirror the debug configurations in `.vscode/launch.json`). The Makefile prefers the project's virtualenv Python at `.venv/bin/python` when present; you can override the interpreter with the `PY` variable.

Common targets:

```bash
# Show available targets
make help

# ChatGPT upload to Notion (no remove)
make chatgpt_upload_to_notion

# ChatGPT upload to Notion (remove in ChatGPT)
make chatgpt_upload_to_notion_remove

# Sora cleanup and tasks
make sora_cleanup_trash
make sora_cleanup_tasks

# Sora upload to Notion variants
make sora_upload_to_notion
make sora_upload_to_notion_trash
make sora_upload_to_notion_remove

# Run the module-style utility target
make clean-output-path
```

Examples:

```bash
# Use the project's .venv python automatically (if present)
make chatgpt_upload_to_notion

# Force a specific Python interpreter
make PY=/usr/bin/python3 sora_upload_to_notion
```

If you want an activation-style shell (to export environment variables for multiple commands), activate the venv then run Make targets:

```bash
source .venv/bin/activate
make sora_upload_to_notion
```
