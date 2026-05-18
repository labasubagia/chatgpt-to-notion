# User Playbook & Deployment Guide

Welcome to the **ChatGPT to Notion Image Sync Tool** user playbook! This document will walk you through setting up your credentials, configuring your first Notion gallery database, running CLI commands, and setting up automated sync schedules (e.g., cron jobs).

---

## 1. Notion Workspace Connection & Database Setup

To sync your image generations, you must configure a private Notion integration and link it to a target database.

### Step 1: Create a Notion Integration Key
1. Go to the [Notion Integrations Portal](https://www.notion.so/my-integrations).
2. Click **+ New Integration**.
3. Choose the target workspace, set the name to **"ChatGPT Sync Tool"**, and grant the integration the following permissions:
   - **Read Content**
   - **Update Content**
   - **Insert Content**
4. Save the integration and copy your secret **Internal Integration Token** (e.g., `secret_...`).

### Step 2: Connection Type Configuration
Ensure that your integration has proper read/write connection access to your database. For an in-depth explanation of connection types and sharing configurations, refer to the [Official Notion Developer Guide](https://developers.notion.com/guides/get-started/overview#connection-types).

### Step 3: Link Integration to your Database
1. Open your target Notion database in the browser.
2. Click the three dots `...` in the top right corner of the database page.
3. Click **Add connections** (or **Connect to**) and search for your integration name (**"ChatGPT Sync Tool"**).
4. Click **Confirm** to grant full access.
5. Copy the **Database ID** from the database's browser URL:
   - URL Format: `https://www.notion.so/<long_hash_1>?v=<long_hash_2>`
   - The Database ID is the 32-character alphanumeric string represented by **`<long_hash_1>`** (the path parameter preceding the `?v=` view parameter).
   - For an in-depth explanation of this URL format and how to retrieve it, see this [StackOverflow Guide](https://stackoverflow.com/a/69860478).

---

## 2. Acquiring ChatGPT Credentials

Because this synchronizer fetches generation records from your account history directly, it requires your private session authentication token.

### How to Retrieve your Session Authorization Token:
1. Open your web browser (Chrome, Firefox, or Brave) and open **Developer Tools** (Press `F12` or `Ctrl + Shift + I` / `Cmd + Opt + I` on macOS).
2. Go to the **Network** tab inside Developer Tools.
3. Visit [https://chatgpt.com](https://chatgpt.com) and log in to your account.
4. Filter network requests by **Fetch/XHR**.
5. Click on any active network request (e.g., `conversations?offset=0&limit=20` or `bootstrap`).
6. Scroll down to the **Request Headers** section and locate the **`Authorization`** header.
7. Copy the entire token string **WITHOUT** the leading `'Bearer '` prefix.
   - *Example Header*: `Authorization: Bearer eyJhbGciOiJSUzI1...`
   - *Correct Value to Copy*: `eyJhbGciOiJSUzI1...`
8. Scroll further down the request headers and locate your **`User-Agent`** string (e.g., `Mozilla/5.0...`). Copy this value as well.
   - **Note**: The `User-Agent` is placed under the global `[shared]` configuration block. You only need to acquire **one** global User-Agent string to be shared across all of your pooled accounts!

---

## 3. Configuration Mapping (`config.toml`)

Create a `config.toml` file in the root directory by copying the template:

```bash
cp config.toml.example config.toml
```

Populate the configuration file with your acquired tokens and database credentials:

```toml
[shared]
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."

[notion]
api_key = "secret_yourNotionIntegrationTokenHere"
database_id = "yourDefaultNotionDatabaseIDHere"

# Configure as many separate ChatGPT accounts as you want to pool
[accounts.personal]
authorization_token = "eyJhbGciOiJSUzI1..." # ChatGPT Auth token (excluding 'Bearer ')

[accounts.work]
authorization_token = "eyJhbGciOiJSUzI1_other..."
notion_database_id = "optional_override_database_id" # Sync work images to a separate database
```

---

## 4. Command Line Interface Reference

All CLI commands are handled via the Typer framework. The command router and business boundaries are cleanly organized under [src/chatgpt_to_notion/cli](requirement.md).

The command source files can be inspected inside the [src/chatgpt_to_notion/cli/commands](requirement.md) directory:

### 1. `upload-to-notion`
Synchronizes images from ChatGPT accounts into your Notion gallery database.
- **Source Module**: [src/chatgpt_to_notion/cli/commands/upload.py](requirement.md)
- **Execution Command**:
  ```bash
  uv run chatgpt-to-notion upload-to-notion [OPTIONS]
  ```
- **Key Options**:
  - `--account <name>`: Sync a specific account profile (e.g. `personal`). Defaults to processing all accounts sequentially.
  - `--limit <int>`: Max number of history items to crawl per run.
  - `--from-history`: Replays synchronization using saved offline local index files.
  - `--verify-history`: Audits and repairs missing Notion pages from local logs without fetching new live data.
  - `--all`: Used in combination with history options to process all records rather than just today's.
  - `--check-notion-api`: Actively query the Notion database directly via API to prevent duplicates (Enabled by default).
  - `--mode <single|batch>`: Choose single sequential execution (high stability) or batch execution (high bandwidth parallel workers).
  - `--remove`: Automatically hide/delete uploaded conversations from ChatGPT after successfully verifying Notion presence.

### 2. `account-status`
Audits daily image generation usage counts and remaining 24-hour cooldown periods across all configured accounts.
- **Source Module**: [src/chatgpt_to_notion/cli/commands/accounts.py](requirement.md)
- **Execution Command**:
  ```bash
  uv run chatgpt-to-notion account-status --timezone <name>
  ```
- **Options**:
  - `--timezone <name>`: Mappings for date boundaries (e.g., `Asia/Singapore` or `America/New_York`).

### 3. `clean-output-path`
Removes temporary local downloaded assets or index caches safely.
- **Source Module**: [src/chatgpt_to_notion/cli/commands/maintenance.py](requirement.md)
- **Execution Command**:
  ```bash
  uv run chatgpt-to-notion clean-output-path
  ```

---

## 5. Automated Scheduling & Background Operations

To run your synchronizations automatically in the background, you can schedule the CLI using standard Unix scheduler daemons.

### 1. Linux / macOS Cron Setup
To sync your accounts every hour automatically:
1. Open the crontab editor:
   ```bash
   crontab -e
   ```
2. Add the following entry (adjusting the paths to your workspace and `.venv` wrapper):
   ```cron
   0 * * * * cd /home/john/tmp/chatgpt-to-notion && /home/john/tmp/chatgpt-to-notion/.venv/bin/chatgpt-to-notion upload-to-notion >> /home/john/tmp/chatgpt-to-notion/output/cron.log 2>&1
   ```

### 2. Linux Systemd Services
For advanced system scheduling, create a dedicated systemd timer:

Create service file `/etc/systemd/user/chatgpt-sync.service`:
```ini
[Unit]
Description=ChatGPT to Notion Sync Service
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/home/john/tmp/chatgpt-to-notion
ExecStart=/home/john/tmp/chatgpt-to-notion/.venv/bin/chatgpt-to-notion upload-to-notion
```

Create timer file `/etc/systemd/user/chatgpt-sync.timer`:
```ini
[Unit]
Description=Run ChatGPT to Notion Sync every hour

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

Enable the scheduled execution:
```bash
systemctl --user daemon-reload
systemctl --user enable --now chatgpt-sync.timer
```
