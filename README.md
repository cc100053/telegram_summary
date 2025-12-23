# Telegram Forum Summarizer

Summarizes Telegram forum topics (topics-enabled group) every 4 hours using Google Gemini, with output sent to your Saved Messages (test mode) or back into each topic.

## Features
- **Dynamic Time Window**: Automatically adjusts lookback period based on time of day (6 hours for morning run, 3.5 hours for daytime runs). Also supports `LAST_RUN_TIMESTAMP` from GitHub Actions cache.
- **Large Topic Handling**: Automatically splits topics with >1000 messages into chunks for summarization, then combines them.
- **Robust Error Handling**: Retries API calls up to 3 times on server errors; falls back to last 500 messages if full context fails.
- **Message Count**: Displays the number of processed messages in the summary header.
- **AI Disclaimer**: Each summary includes "⚠️ AI有幻觉，总结只作参考" to remind users of potential AI hallucinations.
- **Topic Filtering**: Filter specific topics via `TOPIC_FILTER` or exclude topics via `IGNORED_TOPICS`.
- **Tailored Prompt**: Specialized for a Chinese crypto-farming community with VIP handling for "笑苍生".
- **Test Mode**: Sends to Saved Messages; production mode replies in the topic.

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
export TEST_MODE=true            # false to post into topics
export TOPIC_FILTER=新手提问专区   # optional: process only titles containing this substring
export IGNORED_TOPICS=闲聊,灌水   # optional: comma-separated list of topic names to skip
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
- Workflow: `.github/workflows/summary.yml`
- **Schedule**: Runs at 07:00, 10:30, 14:00, 17:30, 21:00, 00:30 (HKT).
- Set repository secrets: `TG_API_ID`, `TG_API_HASH`, `TG_SESSION_STRING`, `GEMINI_API_KEYS` (comma-separated for rotation) or `GEMINI_API_KEY` (single key), `TARGET_GROUP`, optional `TEST_MODE`, `TOPIC_FILTER`, `IGNORED_TOPICS`.

## Notes
- **Timezone**: Asia/Hong_Kong.
- **Safety**: Gemini safety settings set to BLOCK_NONE.
- **Limits**: 
    - `CHUNK_SIZE`: 1000 messages (for splitting).
    - `FALLBACK_MESSAGES`: 500 messages (last resort).
    - `MAX_MESSAGES_PER_TOPIC`: 5000 messages (hard cap).
