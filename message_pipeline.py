import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import discord

import database as db
from language_service import (
    CODE_TO_LANGUAGE,
    LANGUAGE_CODES,
    apply_directional_isolation,
    detect_language,
)
from translation_service import translate_text

logger = logging.getLogger(__name__)


@dataclass
class MessageContext:
    message: discord.Message
    guild_id: int
    base_id: int
    expected_language: Optional[str]
    has_content: bool
    has_attachments: bool
    has_embeds: bool
    is_media_only: bool
    should_forward_original_embeds: bool
    detected_code: Optional[str]
    detected_language: Optional[str]
    is_alt_source: bool
    source_main_id: Optional[int]
    source_main_default_language: Optional[str]
    source_alts: List[dict]
    reply_to_message_id: Optional[int]


@dataclass
class RouteAction:
    destination_channel_id: int
    action_type: str  # "copy" or "translate"
    target_language: Optional[str] = None
    include_mismatch_guard: bool = False


@dataclass
class OutboundMessage:
    destination_channel_id: int
    header_text: str
    body_text: str
    translated_text: Optional[str]
    attachments_data: List[tuple]
    embeds: List[discord.Embed]
    detected_language: Optional[str]
    target_language: Optional[str]
    reply_to_message_id: Optional[int] = None  # Reserved for future DB-backed reply routing.


def should_forward_embeds(message: discord.Message) -> bool:
    """
    Forward original embeds only when there is no URL in content.
    If content has a URL, Discord will auto-unfurl it in destination channels,
    and forwarding embeds as well causes duplicate previews.
    """
    content = (message.content or "").lower()
    has_url = "http://" in content or "https://" in content
    return bool(message.embeds) and not has_url


def is_media_only_message(
    message: discord.Message,
    has_content: bool,
    has_attachments: bool,
    has_embeds: bool,
) -> bool:
    """Classify whether a message should skip translation and be forwarded as media-only."""
    content_lower = message.content.strip().lower() if has_content else ""
    is_media_only = False

    if content_lower.startswith(("http://", "https://")):
        if any(
            content_lower.endswith(ext)
            for ext in (
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".webp",
                ".mp4",
                ".mov",
                ".webm",
            )
        ):
            is_media_only = True
        elif any(
            domain in content_lower
            for domain in (
                "tenor.com/",
                "giphy.com/",
                "imgur.com/",
                "gfycat.com/",
                "i.redd.it/",
            )
        ):
            is_media_only = True

    if not has_content and (has_attachments or has_embeds):
        is_media_only = True

    return is_media_only


def analyze_message(
    message: discord.Message,
    main_to_alts: Dict[int, Dict[int, List[dict]]],
    alt_to_main: Dict[int, dict],
) -> Optional[MessageContext]:
    """Analyze inbound message once and normalize all routing/language state."""
    if message.author.bot:
        return None

    has_content = bool(message.content.strip())
    has_attachments = bool(message.attachments)
    has_embeds = bool(message.embeds)

    if not has_content and not has_attachments and not has_embeds:
        return None

    is_media_only = is_media_only_message(
        message,
        has_content,
        has_attachments,
        has_embeds,
    )
    base_id = (
        message.channel.parent_id
        if isinstance(message.channel, discord.Thread)
        else message.channel.id
    )
    guild_id = message.guild.id

    # expected_language is fetched async in on_message.
    expected_language = None

    detected_code = (
        None
        if is_media_only
        else detect_language(message.content, expected_language=expected_language)
    )
    detected_language = CODE_TO_LANGUAGE.get(detected_code) if detected_code else None

    is_alt_source = base_id in alt_to_main
    source_main_id = alt_to_main[base_id]["main_id"] if is_alt_source else None
    source_main_default_language = None
    source_alts = []

    if is_alt_source and source_main_id is not None:
        source_alts = main_to_alts.get(guild_id, {}).get(source_main_id, [])
    elif not is_alt_source:
        source_alts = main_to_alts.get(guild_id, {}).get(base_id, [])

    logger.debug(
        f"Message analyzed in {message.channel.id}: "
        f"text={has_content}, attachments={len(message.attachments)}, embeds={len(message.embeds)}, "
        f"media_only={is_media_only}, detected={detected_language or 'Unknown'}"
    )

    return MessageContext(
        message=message,
        guild_id=guild_id,
        base_id=base_id,
        expected_language=expected_language,
        has_content=has_content,
        has_attachments=has_attachments,
        has_embeds=has_embeds,
        is_media_only=is_media_only,
        should_forward_original_embeds=should_forward_embeds(message),
        detected_code=detected_code,
        detected_language=detected_language,
        is_alt_source=is_alt_source,
        source_main_id=source_main_id,
        source_main_default_language=source_main_default_language,
        source_alts=source_alts,
        reply_to_message_id=message.reference.message_id if message.reference else None,
    )


def build_route_plan(context: MessageContext) -> List[RouteAction]:
    """Build delivery plan without performing network/translation work."""
    plan: List[RouteAction] = []

    if context.is_alt_source and context.source_main_id is not None:
        if context.is_media_only:
            plan.append(
                RouteAction(destination_channel_id=context.source_main_id, action_type="copy")
            )
            for alt in context.source_alts:
                if alt["id"] != context.base_id:
                    plan.append(RouteAction(destination_channel_id=alt["id"], action_type="copy"))
        else:
            # If message language does not match the alt channel's default language,
            # translate it in-place for that same alt channel too.
            if context.expected_language:
                expected_code = LANGUAGE_CODES.get(context.expected_language.lower())
                if (
                    expected_code
                    and context.detected_code
                    and context.detected_code != expected_code
                ):
                    plan.append(
                        RouteAction(
                            destination_channel_id=context.base_id,
                            action_type="translate",
                            target_language=context.expected_language,
                            include_mismatch_guard=True,
                        )
                    )

            plan.append(
                RouteAction(
                    destination_channel_id=context.source_main_id,
                    action_type="translate",
                    target_language=context.source_main_default_language,
                )
            )
            for alt in context.source_alts:
                if alt["id"] != context.base_id:
                    plan.append(
                        RouteAction(
                            destination_channel_id=alt["id"],
                            action_type="translate",
                            target_language=alt["lang"],
                        )
                    )
        return plan

    if not context.is_media_only and context.expected_language:
        expected_code = LANGUAGE_CODES.get(context.expected_language.lower())
        if expected_code and context.detected_code and context.detected_code != expected_code:
            plan.append(
                RouteAction(
                    destination_channel_id=context.base_id,
                    action_type="translate",
                    target_language=context.expected_language,
                    include_mismatch_guard=True,
                )
            )

    if context.source_alts:
        if context.is_media_only:
            for alt in context.source_alts:
                plan.append(RouteAction(destination_channel_id=alt["id"], action_type="copy"))
        else:
            for alt in context.source_alts:
                plan.append(
                    RouteAction(
                        destination_channel_id=alt["id"],
                        action_type="translate",
                        target_language=alt["lang"],
                    )
                )

    return plan


async def fetch_attachment_payloads(
    attachments: List[discord.Attachment],
) -> List[tuple]:
    """Download attachments once and keep raw bytes for per-destination file creation."""
    payloads: List[tuple] = []
    if not attachments:
        return payloads

    import aiohttp

    async with aiohttp.ClientSession() as session:
        for attachment in attachments:
            try:
                async with session.get(attachment.url) as resp:
                    if resp.status == 200:
                        payloads.append((attachment.filename, await resp.read()))
                    else:
                        logger.warning(
                            f"Failed to download attachment {attachment.filename}: status {resp.status}"
                        )
            except Exception as e:
                logger.warning(f"Failed to download attachment {attachment.filename}: {e}")
    return payloads


async def compose_outbound_messages(
    context: MessageContext,
    plan: List[RouteAction],
) -> List[OutboundMessage]:
    """Compose outbound payloads (copy/translate) that can be sent via one sender function."""
    outbounds: List[OutboundMessage] = []

    attachments_data: List[tuple] = []
    if context.has_attachments:
        attachments_data = await fetch_attachment_payloads(context.message.attachments)

    embeds = list(context.message.embeds) if context.should_forward_original_embeds else []
    body_text = context.message.content or ""

    # For each destination, look up what message to reply to (if this is a reply).
    # This uses bidirectional linking: works whether replying in main or alt channel.
    reply_targets: dict = {}
    if context.reply_to_message_id:
        try:
            linked_messages = await db.get_all_linked_messages(
                message_id=context.reply_to_message_id,
                channel_id=context.base_id,
            )

            # Build a map of channel_id -> message_id for quick lookup.
            for linked_msg in linked_messages:
                reply_targets[linked_msg["channel_id"]] = linked_msg["message_id"]

            if reply_targets:
                logger.debug(
                    f"Found reply targets in {len(reply_targets)} channels for message {context.reply_to_message_id}"
                )
        except Exception as e:
            logger.warning(f"Failed to find reply targets: {e}")

    async def compose_translate(action: RouteAction) -> Optional[OutboundMessage]:
        if not action.target_language:
            return None

        translated = await translate_text(
            body_text,
            action.target_language,
            guild_id=context.guild_id,
            source_language=context.detected_language,
            message_id=context.message.id,
        )
        if not translated:
            return None

        if action.include_mismatch_guard and translated == body_text:
            return None

        translated = apply_directional_isolation(
            translated,
            target_language=action.target_language,
            detected_language=context.detected_language,
        )

        header = (
            f"<@{context.message.author.id}> -- "
            f"{(context.detected_language or 'Unknown')} -> {action.target_language}"
        )
        return OutboundMessage(
            destination_channel_id=action.destination_channel_id,
            header_text=header,
            body_text=body_text,
            translated_text=translated,
            attachments_data=attachments_data,
            embeds=embeds,
            detected_language=context.detected_language,
            target_language=action.target_language,
            reply_to_message_id=reply_targets.get(action.destination_channel_id),
        )

    translate_actions: List[RouteAction] = []
    for action in plan:
        if action.action_type == "copy":
            outbounds.append(
                OutboundMessage(
                    destination_channel_id=action.destination_channel_id,
                    header_text=f"<@{context.message.author.id}>",
                    body_text=body_text,
                    translated_text=None,
                    attachments_data=attachments_data,
                    embeds=embeds,
                    detected_language=context.detected_language,
                    target_language=None,
                    reply_to_message_id=reply_targets.get(action.destination_channel_id),
                )
            )
        elif action.action_type == "translate":
            translate_actions.append(action)

    if translate_actions:
        translated_outbounds = await asyncio.gather(
            *(compose_translate(action) for action in translate_actions)
        )
        outbounds.extend(
            [outbound for outbound in translated_outbounds if outbound is not None]
        )

    return outbounds


async def send_outbound_message(
    guild: discord.Guild,
    outbound: OutboundMessage,
) -> Optional[discord.Message]:
    """Single sender entrypoint for all outbound message variants. Returns the sent message."""
    try:
        channel = guild.get_channel(int(outbound.destination_channel_id))
        if not channel:
            logger.error(f"Channel {outbound.destination_channel_id} not found")
            return None

        files = [
            discord.File(io.BytesIO(data), filename=filename)
            for filename, data in outbound.attachments_data
        ]
        body = (
            outbound.translated_text
            if outbound.translated_text is not None
            else outbound.body_text
        )

        # Prepare reply reference if this is a reply.
        reply_reference = None
        if outbound.reply_to_message_id:
            try:
                reply_msg = await channel.fetch_message(outbound.reply_to_message_id)
                reply_reference = discord.MessageReference(
                    message_id=reply_msg.id,
                    channel_id=channel.id,
                )
                logger.debug(
                    f"Replying to message {outbound.reply_to_message_id} in channel {channel.id}"
                )
            except discord.NotFound:
                logger.warning(
                    f"Reply target message {outbound.reply_to_message_id} not found in channel {channel.id}"
                )
            except Exception as e:
                logger.warning(f"Failed to create reply reference: {e}")

        sent_message = None
        if outbound.translated_text is not None:
            if len(body) > 2000:
                embed = discord.Embed(description=body)
                sent_message = await channel.send(
                    content=outbound.header_text,
                    embed=embed,
                    files=files,
                    reference=reply_reference,
                )
            else:
                sent_message = await channel.send(
                    f"{outbound.header_text}\n{body}",
                    files=files,
                    reference=reply_reference,
                )
        else:
            if body or files:
                if body:
                    sent_message = await channel.send(
                        f"{outbound.header_text}\n{body}",
                        files=files,
                        reference=reply_reference,
                    )
                else:
                    sent_message = await channel.send(
                        outbound.header_text,
                        files=files,
                        reference=reply_reference,
                    )

        for embed in outbound.embeds:
            try:
                await channel.send(embed=embed)
            except Exception as e:
                logger.warning(f"Failed to send embed: {e}")

        logger.info(
            f"Sent outbound to channel {outbound.destination_channel_id} "
            f"(translated={outbound.translated_text is not None}, attachments={len(files)}, embeds={len(outbound.embeds)}, "
            f"reply={outbound.reply_to_message_id is not None})"
        )
        return sent_message
    except Exception as e:
        logger.error(f"Failed to send outbound message: {e}")
        return None
