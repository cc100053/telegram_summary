import asyncio
import os
from datetime import datetime, timedelta

import google.generativeai as genai
import pytz
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from telethon import TelegramClient, functions, types
from telethon.sessions import StringSession

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

TIME_WINDOW_HOURS = 4
INTERVAL_HOURS = TIME_WINDOW_HOURS
HK_TZ = pytz.timezone("Asia/Hong_Kong")
TOPIC_LIMIT = 50
MAX_MESSAGES_PER_TOPIC = 5000
FALLBACK_MESSAGES = 500
TOPIC_FILTER = os.getenv("TOPIC_FILTER")

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
    prompt = f"""
    你是这个加密货币社群（Crypto Farming Group）的 AI 秘书。
    以下是关于「{topic_name}」话题过去 {INTERVAL_HOURS} 小时内的对话记录。
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


async def send_summary(
    client: TelegramClient, target, topic: types.ForumTopic, summary: str, test_mode: bool
) -> None:
    header = f"[Summary] Topic: {topic.title}"
    payload = f"{header}\n\n{summary}"
    if test_mode:
        await client.send_message("me", payload)
    else:
        await client.send_message(target, payload, reply_to=topic.top_message)


async def run() -> None:
    if load_dotenv:
        load_dotenv()

    api_id = int(require_env("TG_API_ID"))
    api_hash = require_env("TG_API_HASH")
    session_string = require_env("TG_SESSION_STRING")
    gemini_api_key = require_env("GEMINI_API_KEY")
    target_group = parse_target_group(require_env("TARGET_GROUP"))
    test_mode = parse_bool(os.getenv("TEST_MODE"), default=True)

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(
        model_name="gemini-flash-latest",
        safety_settings=SAFETY_SETTINGS,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={"temperature": 0.3},
    )

    cutoff_utc = datetime.now(tz=pytz.UTC) - timedelta(hours=TIME_WINDOW_HOURS)
    timeframe_label = (
        f"{cutoff_utc.astimezone(HK_TZ).strftime('%m/%d %H:%M')} - "
        f"{datetime.now(tz=pytz.UTC).astimezone(HK_TZ).strftime('%m/%d %H:%M')} (Asia/Hong_Kong)"
    )

    async with TelegramClient(StringSession(session_string), api_id, api_hash) as client:
        if not await client.is_user_authorized():
            raise RuntimeError("The provided session string is not authorized.")

        target = await client.get_entity(target_group)
        topics = await fetch_topics(client, target)

        if TOPIC_FILTER:
            topics = [t for t in topics if TOPIC_FILTER in (t.title or "")]
            if not topics:
                print(f"No topics matched filter '{TOPIC_FILTER}'.")
                return

        if not topics:
            print("No forum topics found.")
            return

        print(f"Found {len(topics)} topics. Processing recent messages...")

        summaries_sent = 0
        topics_no_activity: list[str] = []
        topics_no_summary: list[str] = []

        for topic in topics:
            messages, truncated = await fetch_messages_for_topic(client, target, topic, cutoff_utc)
            if not messages:
                topics_no_activity.append(topic.title)
                continue

            print(f"Topic '{topic.title}': {len(messages)} messages in window")
            text_data = format_messages(messages, truncated, timeframe_label)
            retried = False

            try:
                summary, feedback = run_summary(model, topic.title, text_data, timeframe_label)
            except Exception as exc:  # noqa: BLE001
                print(f"Gemini error for topic '{topic.title}': {exc}")
                summary = None
                feedback = "exception"

            if not summary:
                retried = False
                should_retry = (feedback and "prompt_blocked" in feedback) or (feedback == "exception")
                if should_retry and len(messages) > FALLBACK_MESSAGES:
                    fallback_msgs = messages[-FALLBACK_MESSAGES:]
                    fallback_text = format_messages(fallback_msgs, truncated=True, timeframe_label=timeframe_label)
                    print(f"Retrying topic '{topic.title}' with last {FALLBACK_MESSAGES} messages due to block.")
                    try:
                        summary, feedback = run_summary(model, topic.title, fallback_text, timeframe_label)
                        retried = True
                    except Exception as exc:  # noqa: BLE001
                        print(f"Gemini error on retry for topic '{topic.title}': {exc}")

                if not summary:
                    if feedback:
                        reason = f"{feedback} (retried)" if retried else feedback
                        print(f"Gemini blocked topic '{topic.title}' with reason: {reason}")
                    else:
                        print(f"No summary generated for topic '{topic.title}'.")
                    topics_no_summary.append(topic.title)
                    continue

            if retried:
                summary = f"(重试后生成，使用最后 {FALLBACK_MESSAGES} 条消息)\n\n{summary}"

            await send_summary(client, target, topic, summary, test_mode)
            destination = "Saved Messages" if test_mode else f"Topic: {topic.title}"
            print(f"Sent summary to {destination}")
            summaries_sent += 1

        if summaries_sent == 0 and test_mode:
            notice_lines = [
                f"No summaries sent from the last {TIME_WINDOW_HOURS} hours.",
            ]
            if topics_no_activity:
                notice_lines.append("No activity in topics: " + ", ".join(topics_no_activity))
            if topics_no_summary:
                notice_lines.append("Failed to summarize topics: " + ", ".join(topics_no_summary))

            await client.send_message("me", "\n".join(notice_lines))
            print("Sent notice to Saved Messages about missing summaries.")


if __name__ == "__main__":
    asyncio.run(run())
