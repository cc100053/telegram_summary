import asyncio
import os
from datetime import datetime, timedelta

import google.generativeai as genai
import pytz
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from telethon import TelegramClient, functions, types
from telethon.sessions import StringSession
from telethon.errors.rpcerrorlist import ChatWriteForbiddenError, UserBannedInChannelError

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

INTERVAL_HOURS = 4  # Keep for prompt context, or update dynamically if needed? 
# Actually, the prompt says "past {INTERVAL_HOURS} hours". We should probably update that too or just say "recent".
# Let's keep INTERVAL_HOURS as a fallback or update it in run().

HK_TZ = pytz.timezone("Asia/Hong_Kong")
TOPIC_LIMIT = 50
MAX_MESSAGES_PER_TOPIC = 5000
FALLBACK_MESSAGES = 500

SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

SYSTEM_INSTRUCTION = (
    "You summarize Telegram forum topic discussions. "
    "If any individual message would violate safety guidelines, ignore that message instead of refusing the task. "
    "Return concise bullet points with key decisions, questions, and action items."
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


def get_dynamic_window_hours() -> float:
    """
    Returns 6.0 hours if running between 06:00 and 08:00 HK time (morning run).
    Otherwise returns 3.5 hours.
    """
    now_hk = datetime.now(HK_TZ)
    # If running between 06:00 and 07:59, assume it's the "morning summary" covering the night.
    if 6 <= now_hk.hour < 8:
        return 6.0
    return 3.5


async def fetch_topics(client: TelegramClient, target) -> list[types.ForumTopic]:
    result = await client(
        functions.messages.GetForumTopicsRequest(
            peer=target,
            offset_date=None,
            offset_id=0,
            offset_topic=0,
            limit=TOPIC_LIMIT,
            q=None,
        )
    )
    topics = result.topics or []
    return [t for t in topics if isinstance(t, types.ForumTopic) and getattr(t, "top_message", None)]


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
    model: genai.GenerativeModel, topic_name: str, text_data: str, timeframe_label: str
) -> tuple[str, str | None]:
    # Extract hours from label or just say "recent"
    prompt = f"""
    你是这个加密货币社群（Crypto Farming Group）的 AI 秘书。
    以下是关于「{topic_name}」话题这段时间内的对话记录。
    时间范围：{timeframe_label}
    
    【背景知识】：
    1. 群组主要讨论 Crypto 链上交互、刷空投（Airdrop Farming）、DEX/Perp 交易量刷分。
    2. 常见术语包括：自成交（Wash trading）、币安 Alpha 刷分、Gas 优化、多号交互（Sybil）、女巫防范等。
    
    【总结要求】：
    1. **语言**：必须使用**简体中文**。
    2. **VIP 关注**：用户 "笑苍生" 是群组核心/KOL。如果对话记录中包含他的发言，请务必优先总结他的观点或指令，并单独列出。
    3. **内容**：提取有价值的刷分策略、新的 Alpha 机会或技术细节。忽略纯粹的闲聊。
    4. **安全**：若包含不当/攻击性言论，直接忽略该部分，不要拒绝处理。

    【输出格式】：
    - 🔥 **热门话题**：(列出 1-3 个讨论最热烈的项目或策略)
    - 🗣️ **笑苍生说**：(如果有他的发言，请单独列出；如果没有，则不显示此项)
    - 📝 **重点摘要**：(条列式总结技术细节或结论)
    - 列表符号统一使用 “-”，不要使用 “*”
    - 时间范围：请在开头注明「{timeframe_label}」

    对话内容：
    {text_data}
    """

    response = model.generate_content(prompt)
    feedback = None

    candidates = getattr(response, "candidates", []) or []
    if not candidates:
        pf = getattr(response, "prompt_feedback", None)
        if pf and getattr(pf, "block_reason", None):
            feedback = f"prompt_blocked: {pf.block_reason}"
        return "", feedback

    cand = candidates[0]
    finish_reason = getattr(cand, "finish_reason", None)
    parts = getattr(getattr(cand, "content", None), "parts", None) or []
    text_parts = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
    summary_text = "\n".join(text_parts).strip()

    if not summary_text:
        if finish_reason is not None:
            feedback = f"finish_reason={finish_reason}"
        return "", feedback

    return summary_text, None


def run_summary(
    model: genai.GenerativeModel, topic_title: str, text_data: str, timeframe_label: str
) -> tuple[str, str | None]:
    return get_ai_summary(model, topic_title, text_data, timeframe_label)


def build_model(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-flash-latest",
        safety_settings=SAFETY_SETTINGS,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={"temperature": 0.3},
    )


async def send_summary(
    client: TelegramClient, target, topic: types.ForumTopic, summary: str, message_count: int, test_mode: bool
) -> None:
    header = f"[Summary] Topic: {topic.title} ({message_count} messages)"
    disclaimer = "⚠️ AI有幻觉，总结只作参考"
    payload = f"{header}\n{disclaimer}\n\n{summary}"
    if test_mode:
        await client.send_message("me", payload)
    else:
        await client.send_message(target, payload, reply_to=topic.top_message)


async def run_summary_with_retry(
    model: genai.GenerativeModel, topic_title: str, text_data: str, timeframe_label: str, max_retries: int = 3
) -> tuple[str, str | None]:
    for attempt in range(max_retries):
        try:
            summary, feedback = run_summary(model, topic_title, text_data, timeframe_label)
            if summary:
                return summary, feedback
            
            # If blocked by safety settings, retrying the same content usually won't help
            if feedback and "prompt_blocked" in feedback:
                return summary, feedback
                
        except Exception as exc:
            print(f"  > Attempt {attempt+1}/{max_retries} failed for '{topic_title}': {exc}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
            else:
                return None, f"exception: {exc}"
    
    return None, "max_retries_exceeded"


async def run() -> None:
    if load_dotenv:
        load_dotenv()

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
        raise RuntimeError("Environment variable GEMINI_API_KEYS (or GEMINI_API_KEY) is required.")
    target_group = parse_target_group(require_env("TARGET_GROUP"))
    test_mode = parse_bool(os.getenv("TEST_MODE"), default=True)
    topic_filter = os.getenv("TOPIC_FILTER")
    ignored_topics = [t.strip() for t in (os.getenv("IGNORED_TOPICS") or "").split(",") if t.strip()]

    window_hours = get_dynamic_window_hours()
    print(f"Dynamic time window: {window_hours} hours")

    cutoff_utc = datetime.now(tz=pytz.UTC) - timedelta(hours=window_hours)
    timeframe_label = (
        f"{cutoff_utc.astimezone(HK_TZ).strftime('%m/%d %H:%M')} - "
        f"{datetime.now(tz=pytz.UTC).astimezone(HK_TZ).strftime('%m/%d %H:%M')} (Asia/Hong_Kong)"
    )

    async with TelegramClient(StringSession(session_string), api_id, api_hash) as client:
        if not await client.is_user_authorized():
            raise RuntimeError("The provided session string is not authorized.")

        target = await client.get_entity(target_group)
        topics = await fetch_topics(client, target)

        if topic_filter:
            topics = [t for t in topics if topic_filter in (t.title or "")]
            if not topics:
                print(f"No topics matched filter '{topic_filter}'.")
                return

        if not topics:
            print("No forum topics found.")
            return

        print(f"Found {len(topics)} topics. Processing recent messages...")

        summaries_sent = 0
        topics_no_activity: list[str] = []
        topics_no_summary: list[str] = []

        for idx, topic in enumerate(topics):
            if topic.title in ignored_topics:
                print(f"Skipping ignored topic: {topic.title}")
                continue

            api_key = gemini_api_keys[idx % len(gemini_api_keys)]
            model = build_model(api_key)

            messages, truncated = await fetch_messages_for_topic(client, target, topic, cutoff_utc)
            if not messages:
                topics_no_activity.append(topic.title)
                continue

            print(f"Topic '{topic.title}': {len(messages)} messages in window")
            
            CHUNK_SIZE = 1000
            summary = ""
            feedback = None
            retried = False

            if len(messages) > CHUNK_SIZE:
                print(f"  > Large topic ({len(messages)} msgs). Splitting into chunks of {CHUNK_SIZE}...")
                partial_summaries = []
                
                for i in range(0, len(messages), CHUNK_SIZE):
                    chunk = messages[i : i + CHUNK_SIZE]
                    print(f"  > Processing chunk {i//CHUNK_SIZE + 1} ({len(chunk)} messages)...")
                    chunk_text = format_messages(chunk, truncated=False, timeframe_label=timeframe_label)
                    
                    chunk_summary, chunk_feedback = await run_summary_with_retry(model, topic.title, chunk_text, timeframe_label)
                    if chunk_summary:
                        partial_summaries.append(chunk_summary)
                    else:
                        print(f"  > Chunk {i//CHUNK_SIZE + 1} failed: {chunk_feedback}")

                if partial_summaries:
                    print("  > Generating final summary from partial summaries...")
                    combined_text = "\n\n".join(partial_summaries)
                    summary, feedback = await run_summary_with_retry(model, topic.title, combined_text, timeframe_label)
                else:
                    print("  > No partial summaries generated.")
            else:
                # Standard processing for smaller topics
                text_data = format_messages(messages, truncated, timeframe_label)
                summary, feedback = await run_summary_with_retry(model, topic.title, text_data, timeframe_label)

            if not summary:
                retried = False
                should_retry = (feedback and "prompt_blocked" in feedback) or (feedback and "exception" in feedback)
                if should_retry and len(messages) > FALLBACK_MESSAGES:
                    fallback_msgs = messages[-FALLBACK_MESSAGES:]
                    fallback_text = format_messages(fallback_msgs, truncated=True, timeframe_label=timeframe_label)
                    print(f"Retrying topic '{topic.title}' with last {FALLBACK_MESSAGES} messages due to block/error.")
                    summary, feedback = await run_summary_with_retry(model, topic.title, fallback_text, timeframe_label)
                    if summary:
                        retried = True

            if not summary:
                if feedback:
                    reason = f"{feedback} (retried)" if retried else feedback
                    print(f"Gemini blocked topic '{topic.title}' with reason: {reason}")
                else:
                    print(f"No summary generated for topic '{topic.title}'.")
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
            except (UserBannedInChannelError, ChatWriteForbiddenError) as exc:
                # If posting to the group is blocked, fall back to saving the output locally so the run doesn't fail.
                print(f"Write restricted for topic '{topic.title}': {exc}. Sending to Saved Messages instead.")
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
                f"No summaries sent from the last {window_hours} hours.",
            ]
            if topics_no_activity:
                notice_lines.append("No activity in topics: " + ", ".join(topics_no_activity))
            if topics_no_summary:
                notice_lines.append("Failed to summarize topics: " + ", ".join(topics_no_summary))

            await client.send_message("me", "\n".join(notice_lines))
            print("Sent notice to Saved Messages about missing summaries.")


if __name__ == "__main__":
    asyncio.run(run())
