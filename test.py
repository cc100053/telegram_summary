import asyncio
import os

from telethon import TelegramClient
from telethon.sessions import StringSession

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required.")
    return value


def parse_target_group(raw: str):
    """Allow either a t.me link/username or numeric chat id."""
    stripped = raw.strip()
    if stripped.lstrip("-").isdigit():
        value = int(stripped)
        if value < 0 and not stripped.startswith("-100"):
            value = int(f"-100{abs(value)}")
        return value
    return stripped


async def main() -> None:
    if load_dotenv:
        load_dotenv()

    api_id = int(require_env("TG_API_ID"))
    api_hash = require_env("TG_API_HASH")
    session_string = require_env("TG_SESSION_STRING")
    target_group = parse_target_group(require_env("TEST_TARGET_GROUP"))
    message = os.getenv("TEST_MESSAGE", "test")

    async with TelegramClient(StringSession(session_string), api_id, api_hash) as client:
        if not await client.is_user_authorized():
            raise RuntimeError("The provided session string is not authorized.")

        await client.send_message(target_group, message)
        me = await client.get_me()
        print(f"Sent '{message}' as @{getattr(me, 'username', me.id)} to {target_group}")


if __name__ == "__main__":
    asyncio.run(main())
