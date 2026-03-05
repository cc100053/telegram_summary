import asyncio
import os
import signal
import time
from datetime import datetime, timedelta

import pytz
from google import genai
from google.genai import types as genai_types
from telethon import TelegramClient, functions, types
from telethon.sessions import StringSession
from telethon.errors.rpcerrorlist import ChatWriteForbiddenError, UserBannedInChannelError

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

HK_TZ = pytz.timezone("Asia/Hong_Kong")
TOPIC_LIMIT = 50
MAX_MESSAGES_PER_TOPIC = 5000
CHUNK_SIZE = 1000  # Messages per chunk for large topics
CHUNK_DELAY_SECONDS = 5  # Short delay between API calls
FALLBACK_MESSAGES = 500
DEFAULT_MODEL_CALL_TIMEOUT_SECONDS = 45

RUN_START_MONO = time.monotonic()
RUNTIME_STATE = {
    "phase": "boot",
    "topic": "",
    "topic_index": 0,
    "topics_total": 0,
    "summaries_sent": 0,
    "last_feedback": "",
}


def log(message: str) -> None:
    elapsed = time.monotonic() - RUN_START_MONO
    print(f"[T+{elapsed:7.2f}s] {message}", flush=True)


def update_runtime_state(**kwargs) -> None:
    RUNTIME_STATE.update(kwargs)


def dump_runtime_diagnostics(reason: str) -> None:
    log(
        "DIAGNOSTIC "
        f"reason={reason} phase={RUNTIME_STATE.get('phase')} "
        f"topic={RUNTIME_STATE.get('topic')} "
        f"topic_idx={RUNTIME_STATE.get('topic_index')}/{RUNTIME_STATE.get('topics_total')} "
        f"summaries_sent={RUNTIME_STATE.get('summaries_sent')} "
        f"last_feedback={RUNTIME_STATE.get('last_feedback')}"
    )


def install_signal_handlers() -> None:
    def _handle_signal(signum, _frame):
        dump_runtime_diagnostics(f"signal_{signum}")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

SAFETY_SETTINGS = [
    genai_types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF",
    ),
    genai_types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF",
    ),
    genai_types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF",
    ),
    genai_types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF",
    ),
]

MODELS_TO_TRY = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
]

SYSTEM_INSTRUCTION = (
    "You are an AI assistant that summarizes Telegram discussions. "
    "If any content violates safety guidelines, ignore that part and continue. Never refuse the entire task."
)


def parse_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required.")
    return value


def parse_target_group(raw: str):
    stripped = raw.strip()
    if stripped.lstrip("-").isdigit():
        value = int(stripped)
        if value < 0 and not stripped.startswith("-100"):
            # Channel/megagroup IDs typically include the -100 prefix.
            value = int(f"-100{abs(value)}")
        return value
    return stripped


def get_cutoff_time() -> datetime:
    """
    Get the cutoff time for fetching messages.
    Priority:
    1. Use LAST_RUN_TIMESTAMP from env (set by GitHub Actions cache)
    2. Fallback to dynamic window calculation
    """
    last_run = os.getenv("LAST_RUN_TIMESTAMP")
    if last_run:
        try:
            # Parse ISO format timestamp from GitHub Actions
            cutoff = datetime.fromisoformat(last_run.replace("Z", "+00:00"))
            print(f"Using cached last run time: {cutoff}")
            return cutoff
        except ValueError:
            print(f"Warning: Could not parse LAST_RUN_TIMESTAMP: {last_run}")
    
    # Fallback to dynamic window
    now_hk = datetime.now(HK_TZ)
    if 6 <= now_hk.hour < 8:
        window_hours = 6.0  # Morning run covers night
    else:
        window_hours = 3.5
    
    print(f"Using dynamic window: {window_hours} hours")
    return datetime.now(tz=pytz.UTC) - timedelta(hours=window_hours)


async def fetch_topics(client: TelegramClient, target) -> tuple[list[types.ForumTopic], int]:
    topics: list[types.ForumTopic] = []
    seen_topic_ids: set[int] = set()
    offset_date = None
    offset_id = 0
    offset_topic = 0
    total_count = 0

    while True:
        result = await client(
            functions.messages.GetForumTopicsRequest(
                peer=target,
                offset_date=offset_date,
                offset_id=offset_id,
                offset_topic=offset_topic,
                limit=TOPIC_LIMIT,
                q=None,
            )
        )
        if not total_count:
            total_count = getattr(result, "count", 0) or 0

        batch = [
            t
            for t in (result.topics or [])
            if isinstance(t, types.ForumTopic) and getattr(t, "top_message", None)
        ]
        if not batch:
            break

        new_batch = []
        for topic in batch:
            if topic.id in seen_topic_ids:
                continue
            seen_topic_ids.add(topic.id)
            new_batch.append(topic)

        if not new_batch:
            break

        topics.extend(new_batch)

        if total_count and len(topics) >= total_count:
            break
        if len(batch) < TOPIC_LIMIT:
            break

        last_topic = new_batch[-1]
        offset_id = last_topic.top_message or 0
        offset_topic = last_topic.id
        offset_date = None
        for message in getattr(result, "messages", []) or []:
            if getattr(message, "id", None) == last_topic.top_message:
                offset_date = getattr(message, "date", None)
                break

        if offset_id == 0 and offset_topic == 0:
            break

    return topics, total_count


def can_speak_in_topic(topic: types.ForumTopic) -> bool:
    return not getattr(topic, "closed", False)


async def fetch_messages_for_topic(
    client: TelegramClient, target, topic: types.ForumTopic, cutoff_utc: datetime
) -> tuple[list[dict], bool]:
    """Collect recent text/url messages for a single topic."""
    collected = []
    input_peer = await client.get_input_entity(target)
    entity_cache: dict[int, str] = {}

    def cache_entities(result) -> None:
        for user in getattr(result, "users", []) or []:
            label = "Unknown"
            if getattr(user, "username", None):
                label = f"@{user.username}"
            elif getattr(user, "first_name", None):
                label = user.first_name
            elif getattr(user, "id", None):
                label = str(user.id)
            entity_cache[user.id] = label

        for chat in getattr(result, "chats", []) or []:
            label = "Unknown"
            if getattr(chat, "title", None):
                label = chat.title
            elif getattr(chat, "username", None):
                label = f"@{chat.username}"
            elif getattr(chat, "id", None):
                label = str(chat.id)
            entity_cache[chat.id] = label

    def resolve_sender_label(message) -> str:
        peer = getattr(message, "from_id", None)
        if not peer:
            return "Unknown"
        peer_id = None
        if hasattr(peer, "user_id") and peer.user_id:
            peer_id = peer.user_id
        elif hasattr(peer, "channel_id") and peer.channel_id:
            peer_id = peer.channel_id
        elif hasattr(peer, "chat_id") and peer.chat_id:
            peer_id = peer.chat_id
        if peer_id is None:
            return "Unknown"
        return entity_cache.get(peer_id, str(peer_id))

    async def collect_for_top(top_reference: int) -> None:
        offset_id = 0
        page = 0
        max_pages = 50

        while page < max_pages:
            result = await client(
                functions.messages.SearchRequest(
                    peer=input_peer,
                    q="",
                    filter=types.InputMessagesFilterEmpty(),
                    min_date=None,
                    max_date=None,
                    offset_id=offset_id,
                    add_offset=0,
                    limit=100,
                    max_id=0,
                    min_id=0,
                    hash=0,
                    top_msg_id=top_reference,
                )
            )

            messages = result.messages or []
            if not messages:
                break

            cache_entities(result)

            for message in messages:
                if not message or not message.date:
                    continue

                message_time_utc = message.date.replace(tzinfo=pytz.UTC)
                if message_time_utc < cutoff_utc:
                    continue

                text = (getattr(message, "message", "") or "").strip()
                if not text:
                    continue

                # Skip previous bot summaries to avoid feedback loops
                if "[Summary]" in text or "#总结" in text:
                    continue

                collected.append(
                    {
                        "sender": resolve_sender_label(message),
                        "text": text,
                        "time": message_time_utc.astimezone(HK_TZ),
                    }
                )

            offset_id = messages[-1].id
            if messages[-1].date and messages[-1].date.replace(tzinfo=pytz.UTC) < cutoff_utc:
                break
            page += 1

    # Try with top_message first; if nothing collected, fall back to topic.id
    await collect_for_top(topic.top_message)
    if not collected and topic.id != topic.top_message:
        await collect_for_top(topic.id)

    collected.sort(key=lambda m: m["time"])
    truncated = False
    if len(collected) > MAX_MESSAGES_PER_TOPIC:
        truncated = True
        collected = collected[-MAX_MESSAGES_PER_TOPIC:]
    return collected, truncated


def format_messages(messages: list[dict], truncated: bool, timeframe_label: str) -> str:
    formatted_messages = "\n".join(
        f"[{m['time'].strftime('%Y-%m-%d %H:%M')}] {m['sender']}: {m['text']}"
        for m in messages
    )
    note = f"时间范围：{timeframe_label}"
    if truncated:
        note += f"\n(仅包含最近 {len(messages)} 条消息，因长度限制进行了截断)"
    return f"{note}\n{formatted_messages}"


def get_ai_summary(
    client: genai.Client, topic_name: str, text_data: str, timeframe_label: str, model_name: str
) -> tuple[str, str | None]:
    # 傳入 current_date 以便 AI 計算 "明天/下週" 的具體日期
    current_date = datetime.now(HK_TZ).strftime("%Y年%m月%d日 (%A)")
    
    prompt = f"""
    # Role
    你是由「Crypto Farming Group」指派的高級鏈上分析師與會議秘書。你具備深厚的 DeFi、Airdrop、MEV 及合約交互知識。
    你的任務是從雜亂的社群對話中，提煉出高價值的「Alpha 資訊」與「操作策略」。

    # Context
    * **今日日期**: {current_date}
    * **對話主題**: {topic_name}
    * **時間範圍**: {timeframe_label}
    * **核心術語庫**: Wash trading (自成交), Sybil (女巫), Gas Optimization, Binance Alpha, LP, Slippage.
    * **KOL**: 用户 "笑苍生" 是本群精神領袖。

    # Constraints (Critical)
    1.  **输出语言 (Output Language)**: **必须使用简体中文**输出所有内容。这是强制要求，无例外。
    2.  **絕對真實 (Zero Hallucination)**: 總結內容**必須嚴格基於**提供的 `{text_data}`。嚴禁編造未在對話中出現的項目名稱、價格預測或操作建議。
    3.  **來源歸屬 (Attribution)**: 
        * 每一條【重點摘要】和【待辦事項】都**必須**標註來源。
        * 格式：`— (@username)`。
        * 若該觀點由多人共同完善，標註主要發起人即可；若無法確定，標註 `— (多人討論)`。
        * **例外**：【笑苍生说】區塊**不需要**標註 @username（因他是已知 KOL）。
    4.  **待辦識別 (Actionable Intel)**: 
        * 僅提取**具有時效性**的群體任務（如：Snapshot 時間、Mint 截止、AMA 開始）。
        * **日期處理**：將「明天」、「這週日」轉換為具體日期（MM月DD日）。如果無法確定具體日期，請**保留原文描述**，不要強行猜測。
    5.  **時間標註 (Timestamp Attribution)**:
        * 每一條【重點摘要】都**必須**標註該討論發生的時間範圍，格式：`[HH:MM-HH:MM]`。
        * 使用**主要討論時段**的時間。若某議題在多個不連續時段被大量討論（如 14:00 和 18:00），請**拆分為兩條獨立的摘要**，各自標註其時間範圍。
    6.  **KOL 優先**: 只要 "笑苍生" 有發言，無論長短，必須在專屬區塊中精確轉述。
    7.  **噪音過濾**: 自動忽略 "GM", "GN", 表情包, 情緒宣洩 (FUD/FOMO) 及無關閒聊。

    # Workflow
    1.  **掃描與過濾**: 閱讀對話，剔除噪音。
    2.  **KOL 提取**: 鎖定 "笑苍生" 的所有指令與觀點。
    3.  **信息結構化**:
        * 提取熱門項目的核心爭議或亮點。
        * 提取具體技術細節（Gas 設置、路徑）並綁定發言者 ID。
    4.  **時效性掃描**: 尋找關鍵詞（截止、快照、claim、填表），生成待辦清單。
    5.  **輸出生成**: 按下方格式輸出。

    # Output Format
    請嚴格遵守以下格式，列表符號統一使用 "-"：

    🗓️ **时间范围**: {timeframe_label}

    🔥 **热门话题** (Top Discussed)
    - [項目/代幣名稱]: [一句話概括核心討論點]
    - (若無熱點則寫 "無特別熱點")

    🗣️ **笑苍生说** (KOL Insights)
    - [精確轉述他的觀點、指令或判斷]
    - (若此段時間他未發言，請直接移除此區塊)

    📝 **重点摘要** (Key Takeaways)
    - [HH:MM-HH:MM] [技術/策略]: [詳細說明] — *(@username)*
    - [HH:MM-HH:MM] [風險警示]: [例如：合約有後門、查女巫嚴格] — *(@username)*

    ⏰ **待辦事項** (Action Items)
    - 📅 [MM-DD 或 原文時間]: [具體行動，如：去 Galxe 領取 OAT] — *(@username 提醒)*
    - (若無時限性任務則不顯示此區塊)

    ---
    **Input Data**:
    {text_data}
    """

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                safety_settings=SAFETY_SETTINGS,
                temperature=0.3,
            ),
        )
    except Exception as exc:
        return "", f"api_error: {exc}"

    feedback = None

    # Check for blocked prompt
    if hasattr(response, "prompt_feedback") and response.prompt_feedback:
        pf = response.prompt_feedback
        if hasattr(pf, "block_reason") and pf.block_reason:
            return "", f"prompt_blocked: {pf.block_reason}"

    # Extract text from response (new SDK has .text property)
    if hasattr(response, "text") and response.text:
        return response.text.strip(), None

    # Fallback: try candidates
    candidates = getattr(response, "candidates", []) or []
    if not candidates:
        return "", "no_candidates"

    cand = candidates[0]
    finish_reason = getattr(cand, "finish_reason", None)
    
    # Try to get text from content parts
    content = getattr(cand, "content", None)
    if content:
        parts = getattr(content, "parts", None) or []
        text_parts = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
        summary_text = "\n".join(text_parts).strip()
        if summary_text:
            return summary_text, None

    if finish_reason is not None:
        feedback = f"finish_reason={finish_reason}"
    return "", feedback


def run_summary(
    client: genai.Client, topic_title: str, text_data: str, timeframe_label: str, model_name: str
) -> tuple[str, str | None]:
    return get_ai_summary(client, topic_title, text_data, timeframe_label, model_name)


def build_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def is_model_overloaded_error(feedback: str | None) -> bool:
    if not feedback:
        return False
    normalized = feedback.upper()
    return (
        "API_ERROR" in normalized
        and "503" in normalized
        and ("UNAVAILABLE" in normalized or "HIGH DEMAND" in normalized)
    )


async def send_summary(
    client: TelegramClient, target, topic: types.ForumTopic, summary: str, message_count: int, test_mode: bool
) -> None:
    header = f"[Summary] Topic: {topic.title} ({message_count} messages)"
    disclaimer = "⚠️ AI有幻觉，总结只作参考"
    payload = f"{header}\n{disclaimer}\n\n{summary}"
    
    # Debug logging
    print(f"  [DEBUG] send_summary called:")
    print(f"    - test_mode: {test_mode}")
    print(f"    - target: {target} (type: {type(target).__name__})")
    print(f"    - topic.id: {topic.id}, topic.top_message: {topic.top_message}")
    print(f"    - payload length: {len(payload)} chars")
    
    if test_mode:
        result = await client.send_message("me", payload)
        print(f"    - Sent to Saved Messages, msg_id: {result.id}")
    else:
        result = await client.send_message(target, payload, reply_to=topic.top_message)
        print(f"    - Sent to topic, msg_id: {result.id}")




async def run_summary_with_retry(
    api_keys: list[str],
    start_key_index: int,
    topic_title: str,
    text_data: str,
    timeframe_label: str,
    max_retries: int = 3,
    model_call_timeout_seconds: int = DEFAULT_MODEL_CALL_TIMEOUT_SECONDS,
) -> tuple[str, str | None, int]:
    """
    Attempts to generate a summary using multiple models and rotating keys.
    Returns: (summary, feedback, next_key_index)
    """
    last_feedback = None
    current_key_idx = start_key_index

    for model_name in MODELS_TO_TRY:
        print(f"  > [Model: {model_name}] Starting attempts...")

        for attempt in range(max_retries):
            # Rotate key potentially on every attempt if previous failed
            api_key = api_keys[current_key_idx % len(api_keys)]
            client = build_client(api_key)
            
            # Mask key for logging
            masked_key = f"...{api_key[-4:]}" if len(api_key) > 4 else "std"
            print(f"    - Attempt {attempt + 1}/{max_retries} using key {masked_key}")
            attempt_started = time.monotonic()

            try:
                summary, feedback = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_summary,
                        client,
                        topic_title,
                        text_data,
                        timeframe_label,
                        model_name,
                    ),
                    timeout=model_call_timeout_seconds,
                )
                log(
                    f"Gemini call finished model={model_name} "
                    f"attempt={attempt + 1}/{max_retries} "
                    f"elapsed={time.monotonic() - attempt_started:.2f}s "
                    f"input_chars={len(text_data)}"
                )
                if summary:
                    # Success! Force rotation for next call (Round Robin)
                    return summary, feedback, current_key_idx + 1

                last_feedback = feedback
                update_runtime_state(last_feedback=str(feedback or ""))
                print(f"    - Failed: {feedback}")

                # FAIL FAST: Prompt blocked (safety) - do not retry this model
                if feedback and "prompt_blocked" in feedback:
                    print(f"    - Prompt blocked. Skipping remaining retries for {model_name}.")
                    break  # Break attempt loop, move to next model (or stop if all blocked)

                # FAIL FAST: Model overloaded (503 UNAVAILABLE/high demand)
                if is_model_overloaded_error(feedback):
                    log(
                        f"Model overloaded for {model_name}; switching model immediately "
                        f"(skip remaining retries on this model)."
                    )
                    break

                # FALLBACK ROTATION: If API error/Quota, switch key immediately
                if feedback and ("api_error" in feedback or "finish_reason" in feedback):
                    if len(api_keys) > 1:
                        print("    - API/Quota issue. Rotating to next key immediately.")
                        current_key_idx += 1
                        continue  # Immediate retry with next key
                    else:
                        # Single key: must wait
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 * (attempt + 1))
                        continue

            except asyncio.TimeoutError:
                last_feedback = f"model_timeout_after_{model_call_timeout_seconds}s"
                update_runtime_state(last_feedback=last_feedback)
                log(
                    f"Gemini call timeout model={model_name} "
                    f"attempt={attempt + 1}/{max_retries} "
                    f"timeout={model_call_timeout_seconds}s "
                    f"input_chars={len(text_data)}"
                )
                if len(api_keys) > 1:
                    current_key_idx += 1
                elif attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
            except Exception as exc:
                print(f"    - Exception: {exc}")
                last_feedback = f"exception: {exc}"
                update_runtime_state(last_feedback=last_feedback)
                # Switch key on exception too? Yes, safest.
                if len(api_keys) > 1:
                    current_key_idx += 1
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 * (attempt + 1))

        print(f"  > [Model: {model_name}] Exhausted retries.")

    return None, f"all_models_failed: {last_feedback}", current_key_idx


async def run() -> None:
    if load_dotenv:
        load_dotenv()

    update_runtime_state(phase="load_config")

    api_id = int(require_env("TG_API_ID"))
    api_hash = require_env("TG_API_HASH")
    session_string = require_env("TG_SESSION_STRING")
    raw_keys = os.getenv("GEMINI_API_KEYS")
    gemini_api_keys = [k.strip() for k in raw_keys.split(",")] if raw_keys else []
    gemini_api_keys = [k for k in gemini_api_keys if k]
    if not gemini_api_keys:
        single_key = os.getenv("GEMINI_API_KEY")
        if single_key and single_key.strip():
            gemini_api_keys = [single_key.strip()]
    if not gemini_api_keys:
        raise RuntimeError(
            "Environment variable GEMINI_API_KEYS (or GEMINI_API_KEY) is required."
        )
    model_call_timeout_seconds = int(
        os.getenv("MODEL_CALL_TIMEOUT_SECONDS", str(DEFAULT_MODEL_CALL_TIMEOUT_SECONDS))
    )
    if model_call_timeout_seconds < 5:
        model_call_timeout_seconds = 5
    target_group = parse_target_group(require_env("TARGET_GROUP"))
    test_mode = parse_bool(os.getenv("TEST_MODE"), default=True)

    # Debug: Show parsed config
    print(f"[DEBUG] Configuration:")
    print(f"  - TARGET_GROUP: {target_group}")
    print(f"  - TEST_MODE: {test_mode} (raw env: '{os.getenv('TEST_MODE')}')")
    print(f"  - Available Keys: {len(gemini_api_keys)}")
    print(f"  - MODEL_CALL_TIMEOUT_SECONDS: {model_call_timeout_seconds}")

    topic_filter = os.getenv("TOPIC_FILTER")
    ignored_topics = [
        t.strip() for t in (os.getenv("IGNORED_TOPICS") or "").split(",") if t.strip()
    ]
    if ignored_topics:
        print(f"Ignored topics: {ignored_topics}")

    cutoff_utc = get_cutoff_time()
    timeframe_label = (
        f"{cutoff_utc.astimezone(HK_TZ).strftime('%m/%d %H:%M')} - "
        f"{datetime.now(tz=pytz.UTC).astimezone(HK_TZ).strftime('%m/%d %H:%M')} (Asia/Hong_Kong)"
    )

    update_runtime_state(phase="connect_telegram")
    async with TelegramClient(StringSession(session_string), api_id, api_hash) as client:
        if not await client.is_user_authorized():
            raise RuntimeError("The provided session string is not authorized.")

        update_runtime_state(phase="fetch_topics")
        target = await client.get_entity(target_group)
        topics_fetch_started = time.monotonic()
        topics, total_topic_count = await fetch_topics(client, target)
        log(
            f"Fetched topics count={len(topics)} total_count={total_topic_count} "
            f"elapsed={time.monotonic() - topics_fetch_started:.2f}s"
        )

        if topic_filter:
            topics = [t for t in topics if topic_filter in (t.title or "")]
            if not topics:
                print(f"No topics matched filter '{topic_filter}'.")
                return

        if not topics:
            print("No forum topics found.")
            return

        total_label = f"{total_topic_count}" if total_topic_count else "unknown"
        print(
            f"Found {len(topics)} topics (total: {total_label}). Processing recent messages..."
        )

        summaries_sent = 0
        topics_no_activity: list[str] = []
        topics_no_summary: list[str] = []

        # Track key usage to balance load across topics
        current_key_usage_idx = 0

        for idx, topic in enumerate(topics):
            update_runtime_state(
                phase="topic_loop",
                topic=topic.title or "",
                topic_index=idx + 1,
                topics_total=len(topics),
            )
            if topic.title in ignored_topics:
                print(f"Skipping ignored topic: {topic.title}")
                continue
            if not can_speak_in_topic(topic):
                print(f"Skipping closed topic (no speaking permission): {topic.title}")
                continue

            topic_started = time.monotonic()
            messages, truncated = await fetch_messages_for_topic(
                client, target, topic, cutoff_utc
            )
            log(
                f"Topic fetch done topic='{topic.title}' messages={len(messages)} "
                f"elapsed={time.monotonic() - topic_started:.2f}s"
            )
            if not messages:
                topics_no_activity.append(topic.title)
                continue

            print(f"Topic '{topic.title}': {len(messages)} messages in window")

            summary = ""
            feedback = None
            retried = False

            # Use rotating index for starting key to distribute load
            start_key_idx = current_key_usage_idx

            if len(messages) > CHUNK_SIZE:
                total_chunks = (len(messages) + CHUNK_SIZE - 1) // CHUNK_SIZE
                print(
                    f"  > Large topic ({len(messages)} msgs). Splitting into {total_chunks} chunks of {CHUNK_SIZE}..."
                )
                partial_summaries = []

                local_key_idx = start_key_idx

                for i in range(0, len(messages), CHUNK_SIZE):
                    chunk_num = i // CHUNK_SIZE + 1
                    chunk = messages[i : i + CHUNK_SIZE]

                    # Short delay between chunks (except for first)
                    if i > 0:
                        print(f"  > Waiting {CHUNK_DELAY_SECONDS}s...")
                        await asyncio.sleep(CHUNK_DELAY_SECONDS)

                    print(
                        f"  > Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} messages)..."
                    )
                    chunk_text = format_messages(
                        chunk, truncated=False, timeframe_label=timeframe_label
                    )

                    chunk_summary, chunk_feedback, next_idx = await run_summary_with_retry(
                        gemini_api_keys,
                        local_key_idx,
                        topic.title,
                        chunk_text,
                        timeframe_label,
                        model_call_timeout_seconds=model_call_timeout_seconds,
                    )
                    local_key_idx = next_idx  # Update local index for next chunk

                    if chunk_summary:
                        partial_summaries.append(chunk_summary)
                    else:
                        print(f"  > Chunk {chunk_num}/{total_chunks} failed: {chunk_feedback}")

                if partial_summaries:
                    print(f"  > Waiting {CHUNK_DELAY_SECONDS}s before final summary...")
                    await asyncio.sleep(CHUNK_DELAY_SECONDS)
                    print("  > Generating final summary from partial summaries...")
                    combined_text = "\n\n".join(partial_summaries)
                    
                    # Pass the latest key index to continue rotation
                    summary, feedback, next_idx = await run_summary_with_retry(
                        gemini_api_keys,
                        local_key_idx,
                        topic.title,
                        combined_text,
                        timeframe_label,
                        model_call_timeout_seconds=model_call_timeout_seconds,
                    )
                    # Update global index for next topic
                    current_key_usage_idx = next_idx
                else:
                    print("  > No partial summaries generated.")
            else:
                # Standard processing for smaller topics
                text_data = format_messages(messages, truncated, timeframe_label)
                summary, feedback, next_idx = await run_summary_with_retry(
                    gemini_api_keys,
                    start_key_idx,
                    topic.title,
                    text_data,
                    timeframe_label,
                    model_call_timeout_seconds=model_call_timeout_seconds,
                )
                current_key_usage_idx = next_idx

            if not summary:
                retried = False
                should_retry = (feedback and "prompt_blocked" in feedback) or (
                    feedback and "exception" in feedback
                )
                # Note: With new fail-fast logic, blocked prompts shouldn't be retried blindly,
                # but we keep this fallback specifically for "exception" or if we want to try truncation.
                # If it was blocked, our new logic breaks early, so maybe we shouldn't retry?
                # Let's keep the fallback for robust truncation only if it WASN'T a hard block.
                
                is_hard_block = feedback and "prompt_blocked" in feedback
                
                if (not is_hard_block) and len(messages) > FALLBACK_MESSAGES:
                    fallback_msgs = messages[-FALLBACK_MESSAGES:]
                    fallback_text = format_messages(
                        fallback_msgs, truncated=True, timeframe_label=timeframe_label
                    )
                    print(
                        f"Retrying topic '{topic.title}' with last {FALLBACK_MESSAGES} messages."
                    )
                    summary, feedback, next_idx = await run_summary_with_retry(
                        gemini_api_keys,
                        current_key_usage_idx,
                        topic.title,
                        fallback_text,
                        timeframe_label,
                        model_call_timeout_seconds=model_call_timeout_seconds,
                    )
                    current_key_usage_idx = next_idx
                    if summary:
                        retried = True

            if not summary:
                reason = f"{feedback} (retried)" if retried else feedback
                print(f"Gemini failed topic '{topic.title}'. Reason: {reason}")
                topics_no_summary.append(topic.title)
                continue

            summary = summary.rstrip() + "\n\n#总结"

            final_count = len(messages)
            if retried:
                summary = f"(重试后生成，使用最后 {FALLBACK_MESSAGES} 条消息)\n\n{summary}"
                final_count = min(len(messages), FALLBACK_MESSAGES)

            try:
                await send_summary(client, target, topic, summary, final_count, test_mode)
                destination = "Saved Messages" if test_mode else f"Topic: {topic.title}"
                print(f"Sent summary to {destination}")
                summaries_sent += 1
                update_runtime_state(summaries_sent=summaries_sent)
            except (UserBannedInChannelError, ChatWriteForbiddenError) as exc:
                print(
                    f"Write restricted for topic '{topic.title}': {exc}. Sending to Saved Messages instead."
                )
                fallback_note = (
                    f"[Summary not delivered] Topic: {topic.title}\n"
                    f"Reason: {exc}\n\n{summary}"
                )
                await client.send_message("me", fallback_note)
                topics_no_summary.append(f"{topic.title} (write restricted)")
            except Exception as exc:
                print(f"Failed to send summary for topic '{topic.title}': {exc}")
                topics_no_summary.append(topic.title)

        if summaries_sent == 0 and test_mode:
            notice_lines = [
                f"No summaries sent for timeframe: {timeframe_label}.",
            ]
            if topics_no_activity:
                notice_lines.append("No activity in topics: " + ", ".join(topics_no_activity))
            if topics_no_summary:
                notice_lines.append("Failed to summarize topics: " + ", ".join(topics_no_summary))

            await client.send_message("me", "\n".join(notice_lines))
            print("Sent notice to Saved Messages about missing summaries.")

    update_runtime_state(phase="done")
    dump_runtime_diagnostics("normal_exit")


if __name__ == "__main__":
    install_signal_handlers()
    try:
        asyncio.run(run())
    except BaseException:
        dump_runtime_diagnostics("abnormal_exit")
        raise
