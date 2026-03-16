import asyncio
import logging
import os
from typing import Optional

import requests

import database as db

logger = logging.getLogger(__name__)

LLM_API_URL = os.getenv("LLM_API_URL", "https://api.x.ai/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "grok-4-1-fast-non-reasoning")
LLM_SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "You are a professional translator. When given text and a target language, "
    "translate it accurately. Do not translate URLs, links, or emojis - pass them through unchanged. "
    "Respond with ONLY the translated text, nothing else.",
)

if not LLM_API_KEY:
    logger.warning("LLM_API_KEY not set - translations will not work")


async def translate_text(
    content: str,
    target_language: str,
    guild_id: Optional[int] = None,
    source_language: Optional[str] = None,
    message_id: Optional[int] = None,
) -> Optional[str]:
    """
    Translate text to target language using LLM API.
    Returns translated text or None if translation fails or API is not configured.

    Also tracks token usage in the database for billing purposes.
    """
    if not LLM_API_KEY:
        logger.warning("LLM_API_KEY not configured, skipping translation")
        return None

    try:
        translation_prompt = (
            f"Translate the following text to {target_language}. "
            f"Do not translate URLs, links, or emojis - pass those through unchanged. "
            f"Respond with ONLY the translated text, nothing else:\n\n{content}"
        )

        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": translation_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }

        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            result = response.json()
            translated = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            # Track token usage if available in response
            usage = result.get("usage", {})
            if usage and guild_id:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                # Track asynchronously in background (don't await to avoid blocking)
                asyncio.create_task(
                    db.track_token_usage(
                        guild_id=guild_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        model=LLM_MODEL,
                        source_language=source_language,
                        target_language=target_language,
                        message_id=message_id,
                    )
                )
                logger.debug(
                    f"Token usage: {total_tokens} total (prompt: {prompt_tokens}, completion: {completion_tokens})"
                )

            if translated:
                logger.debug(
                    f"Translation successful: {content[:50]}... -> {translated[:50]}..."
                )
                return translated

            logger.debug("Translation empty response from API")
            return None

        logger.error(f"LLM API error {response.status_code}: {response.text[:200]}")
        return None

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return None
