"""
Helper script to generate a Telethon StringSession for a user account.
Run locally once and store the printed session string in your secrets.
"""

import asyncio

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.sessions import StringSession


async def main() -> None:
    api_id = int(input("Enter your Telegram API ID: ").strip())
    api_hash = input("Enter your Telegram API Hash: ").strip()
    phone = input("Enter your phone number (with country code): ").strip()

    client = TelegramClient(StringSession(), api_id, api_hash)
    await client.connect()

    if not await client.is_user_authorized():
        await client.send_code_request(phone)
        code = input("Enter the login code you received: ").strip()

        try:
            await client.sign_in(phone=phone, code=code)
        except SessionPasswordNeededError:
            password = input("2FA password required (if enabled, else press Enter): ").strip()
            if not password:
                raise RuntimeError("Account requires 2FA password but none was provided.")
            await client.sign_in(password=password)

    session_string = client.session.save()
    print("\nYour StringSession (keep it secret and secure):")
    print(session_string)

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
