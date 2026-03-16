import os
import logging
import asyncio
from typing import Optional
import discord
from discord.ext import commands

# Database layer
import database as db
from message_pipeline import (
    analyze_message,
    build_route_plan,
    compose_outbound_messages,
    send_outbound_message,
)
from language_service import CODE_TO_LANGUAGE, detect_language
from command_handlers import register_commands
from reaction_sync import handle_reaction_add

# ========================== CONFIG ==========================
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")          # optional - faster slash command sync
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DATA_DIR = os.getenv("DATA_DIR", "/data")

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# In-memory storage for channel mappings (populated from channel names at startup)
main_to_alts = {}           # guild_id -> {main_id: [{"id": alt_id, "lang": lang_name}, ...]}
alt_to_main = {}            # alt_id -> {"main_id": , "lang": }


async def get_channel_language(guild_id: int, channel_id: int) -> Optional[str]:
    """Get the default language for a channel from database"""
    try:
        return await db.get_channel_default(guild_id, channel_id)
    except Exception as e:
        logger.error(f"Failed to get channel default: {e}")
        return None

def rebuild_alt_maps(guild: discord.Guild):
    main_to_alts[guild.id] = {}
    for ch in guild.text_channels:
        if "-but-in-" not in ch.name:
            continue
        try:
            main_name, lang = ch.name.rsplit("-but-in-", 1)
            main_ch = discord.utils.get(guild.text_channels, name=main_name.strip())
            if main_ch:
                lang = lang.strip().title()
                if main_ch.id not in main_to_alts[guild.id]:
                    main_to_alts[guild.id][main_ch.id] = []
                main_to_alts[guild.id][main_ch.id].append({"id": ch.id, "lang": lang})
                alt_to_main[ch.id] = {"main_id": main_ch.id, "lang": lang}
                logger.debug(f"Mapped alt: {ch.name}")
        except Exception:
            pass

# ======================= BOT =======================
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True  # Required for reaction syncing
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    # Initialize database
    await db.init_database()
    
    logger.info(f"✅ Logged in as {bot.user}")
    for g in bot.guilds:
        rebuild_alt_maps(g)
        logger.info(f"Built alt mappings for {g.name} ({len(alt_to_main)} alts)")
    
    # Slash command sync
    try:
        if GUILD_ID:
            g = discord.Object(id=int(GUILD_ID))
            await bot.tree.sync(guild=g)
            logger.info(f"Commands synced to guild {GUILD_ID}")
        else:
            await bot.tree.sync()
            logger.info("Commands synced globally")
    except Exception as e:
        logger.error(f"Sync failed: {e}")

@bot.event
async def on_message(message: discord.Message):
    context = analyze_message(message, main_to_alts, alt_to_main)
    if not context:
        return

    # Fetch expected language from database
    context.expected_language = await get_channel_language(context.guild_id, context.base_id)
    
    # Re-detect with expected language if needed
    if context.has_content and not context.is_media_only and context.expected_language:
        context.detected_code = detect_language(message.content, expected_language=context.expected_language)
        context.detected_language = CODE_TO_LANGUAGE.get(context.detected_code) if context.detected_code else None
    
    # Fetch source_main_default_language if needed
    if context.is_alt_source and context.source_main_id is not None:
        context.source_main_default_language = await get_channel_language(context.guild_id, context.source_main_id) or "English"

    plan = build_route_plan(context)
    if not plan:
        return

    logger.info(f"Built processing plan for message {message.id}: {len(plan)} actions")

    outbounds = await compose_outbound_messages(context, plan)
    if not outbounds:
        return

    # Send messages and track them in database
    sent_messages = await asyncio.gather(
        *(send_outbound_message(message.guild, outbound) for outbound in outbounds),
        return_exceptions=True
    )
    
    # Track all sent messages in database for reply/reaction syncing
    for outbound, result in zip(outbounds, sent_messages):
        if isinstance(result, discord.Message):
            try:
                await db.track_message(
                    source_message_id=message.id,
                    source_channel_id=context.base_id,
                    dest_channel_id=outbound.destination_channel_id,
                    dest_message_id=result.id,
                    guild_id=context.guild_id,
                    author_id=message.author.id,
                    source_language=context.detected_language,
                    dest_language=outbound.target_language,
                    reply_to_message_id=message.reference.message_id if message.reference else None
                )
            except Exception as e:
                logger.error(f"Failed to track message in database: {e}")


@bot.event
async def on_reaction_add(reaction: discord.Reaction, user: discord.User):
    await handle_reaction_add(bot, reaction, user, logger)


register_commands(bot, rebuild_alt_maps, logger)

bot.run(TOKEN)