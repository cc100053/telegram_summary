
**Role:** You are a Senior Python Developer.

**Task:** Write the final `main.py` script for a Telegram Userbot.

**Environment & Libraries:**

  * **Libraries:** `telethon`, `google-generativeai`, `pytz`.
  * **Authentication:** Use `StringSession` (loaded from `os.environ['TG_SESSION_STRING']`).
  * **Configuration:** Load `TG_API_ID`, `TG_API_HASH`, `TARGET_GROUP`, `GEMINI_API_KEY`, `TEST_MODE` from environment variables.
  * **Timezone:** `Asia/Tokyo`.

**Key Functionality:**

1.  **Schedule:** The script runs once (for a cron job), processes data from the last **8 hours**, then exits.
2.  **Topics:** The target group has **Forum Topics**. Iterate through them.
3.  **Fetching:** Fetch only text messages.
4.  **Safety:** Set Gemini safety settings to `BLOCK_NONE` (to avoid blocking crypto slang).

**The LLM Prompt Logic (Crucial):**
In the `get_ai_summary` function, you must use a specific prompt structure for Gemini to handle the user's specific context.

  * **Context:** The group is a Chinese Crypto Farming community (discussing Airdrops, DEX/Perp farming, "自成交", "币安alpha").
  * **Language:** Strict **Simplified Chinese (简体中文)** output.
  * **VIP Handling:** The user **"笑苍生"** is a Key Opinion Leader (KOL). The prompt must instruct Gemini to prioritize summarizing his messages if they appear in the text data.
  * **Output Format:**
      * 🔥 **热门话题**
      * 🗣️ **笑苍生说** (Only if he spoke)
      * 📝 **重点摘要**

**Code Structure Requirement:**
Please use the following prompt string inside the `get_ai_summary` function:

```python
    prompt = f"""
    你是这个加密货币社群（Crypto Farming Group）的 AI 秘书。
    以下是关于「{topic_name}」话题过去 {INTERVAL_HOURS} 小时内的对话记录。
    
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

    对话内容：
    {text_data}
    """
```