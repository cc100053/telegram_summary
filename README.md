# Telegram Forum Summarizer

Summarizes Telegram forum topics (topics-enabled group) every 8 hours using Google Gemini, with output sent to your Saved Messages (test mode) or back into each topic.

## Features
- Fetches last 8 hours of text messages per topic; skips media.
- Handles forum topics; per-topic message caps with retry on blocked prompts.
- Gemini prompt tailored for a Chinese crypto-farming community with VIP handling for “笑苍生”.
- Test mode sends to Saved Messages; production mode replies in the topic.
- Optional topic filter to process a single topic by title substring.

## Requirements
- Python 3.11+
- Packages in `requirements.txt`
- Telegram API ID/Hash and StringSession
- Gemini API key

## Setup (local, macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Export env vars (adjust values):
```bash
export TG_API_ID=123456
export TG_API_HASH=your_api_hash
export TG_SESSION_STRING=your_string_session
export GEMINI_API_KEY=your_gemini_key
export TARGET_GROUP=https://t.me/your_group_link_or_id
export TEST_MODE=true          # false to post into topics
export TOPIC_FILTER=新手提问专区  # optional: process only titles containing this substring
```

Run:
```bash
python main.py
```

To generate a session string locally:
```bash
python get_session.py
```

## Push to GitHub
```bash
git init                     # if not already a repo
git remote add origin git@github.com:<user>/<repo>.git  # or https://...
git add .
git commit -m "Add Telegram forum summarizer"
git branch -M main
git push -u origin main
```

## GitHub Actions
- Workflow: `.github/workflows/summary.yml` (cron every 8 hours + manual dispatch).
- Set repository secrets: `TG_API_ID`, `TG_API_HASH`, `TG_SESSION_STRING`, `GEMINI_API_KEY`, `TARGET_GROUP`, optional `TEST_MODE`, `TOPIC_FILTER`.

## Notes
- Timezone: Asia/Hong_Kong; timeframe label included in prompts.
- Message caps: `MAX_MESSAGES_PER_TOPIC` and `FALLBACK_MESSAGES` in `main.py`.
- Safety: Gemini safety settings set to BLOCK_NONE; blocked prompts retry with fewer messages and mark the output.
