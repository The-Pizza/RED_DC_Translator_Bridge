import os
import json
import logging
import asyncio
import io
import requests
import discord
from discord import app_commands
from discord.ext import commands
import fasttext

# ========================== CONFIG ==========================
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")          # optional - faster slash command sync
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DATA_DIR = os.getenv("DATA_DIR", "/data")

# LLM Configuration
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.x.ai/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "grok-4-1-fast-non-reasoning")
LLM_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", 
    "You are a professional translator. When given text and a target language, "
    "translate it accurately. Do not translate URLs, links, or emojis - pass them through unchanged. "
    "Respond with ONLY the translated text, nothing else."
)
DEFAULT_FILE = os.path.join(DATA_DIR, "default_languages.json")

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

if not LLM_API_KEY:
    logger.warning("⚠️  LLM_API_KEY not set - translations will not work")

# Language map (extend as needed)
LANGUAGE_CODES = {
    "english": "en", "arabic": "ar", "spanish": "es", "french": "fr",
    "german": "de", "italian": "it", "portuguese": "pt", "russian": "ru",
    "chinese": "zh", "japanese": "ja", "korean": "ko", "hindi": "hi"
}

# Reverse map: ISO 639-1 code -> Language name (title case)
CODE_TO_LANGUAGE = {v: k.title() for k, v in LANGUAGE_CODES.items()}

# Load fastText language identification model
model_path = os.path.join(DATA_DIR, "lid.176.bin")
if not os.path.exists(model_path):
    logger.error(f"fastText model not found at {model_path}. Please download it.")
    model = None
else:
    try:
        model = fasttext.load_model(model_path)
        logger.info("fastText language model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load fastText model: {e}")
        model = None

# In-memory storage
default_languages = {}      # guild_id_str -> {channel_id_str: lang_name}
main_to_alts = {}           # guild_id -> {main_id: [{{"id": alt_id, "lang": lang_name}}, ...]}
alt_to_main = {}            # alt_id -> {{"main_id": , "lang": }}

def detect_language(text: str, expected_language: str = None) -> str:
    """
    Detect language using fastText. Returns ISO 639-1 code (e.g., 'en', 'es')
    
    If expected_language is provided (e.g., "English", "Arabic"), checks if text
    matches that language with sufficient confidence before falling back to detection.
    This avoids false positives on ambiguous words like "Pizza" and reduces unnecessary translations.
    """
    if not model:
        return None
    
    try:
        # Get predictions (top 5 for better accuracy)
        predictions = model.predict(text, k=5)
        labels = predictions[0]  # list of '__label__xx'
        probs = predictions[1]   # corresponding probabilities
        
        # Strip '__label__' prefix
        codes = [label.replace('__label__', '') for label in labels]
        
        # If expected language provided, check confidence first
        if expected_language:
            expected_code = LANGUAGE_CODES.get(expected_language.lower())
            if expected_code:
                # Check if expected code is in top predictions with >50% confidence
                for code, prob in zip(codes, probs):
                    if code == expected_code and prob > 0.5:
                        logger.debug(f"Language confident match: expected {expected_language} ({prob:.1%})")
                        return expected_code
                # If not confident, fall through to standard detection
        
        # Standard detection: take the top prediction
        detected_code = codes[0]
        detected_lang = CODE_TO_LANGUAGE.get(detected_code, "Unknown")
        if expected_language:
            logger.debug(f"Language mismatch from expected {expected_language}: detected {detected_lang}")
        else:
            logger.debug(f"Language detected: {detected_lang}")
        return detected_code
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return None

def load_defaults():
    global default_languages
    if os.path.exists(DEFAULT_FILE):
        try:
            with open(DEFAULT_FILE, encoding="utf-8") as f:
                default_languages = json.load(f)
            logger.info(f"Loaded defaults for {len(default_languages)} guilds")
        except Exception as e:
            logger.error(f"Failed to load defaults: {e}")

def save_defaults():
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(DEFAULT_FILE, "w", encoding="utf-8") as f:
            json.dump(default_languages, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save defaults: {e}")

def message_to_dict(msg: discord.Message) -> dict:
    return {
        "id": str(msg.id),
        "content": msg.content,
        "author": {"id": str(msg.author.id), "name": msg.author.name, "bot": msg.author.bot},
        "channel_id": str(msg.channel.id),
        "guild_id": str(msg.guild.id) if msg.guild else None,
        "created_at": msg.created_at.isoformat(),
        "attachments": [a.url for a in msg.attachments],
        "embeds": [e.to_dict() for e in msg.embeds],
        "is_thread": isinstance(msg.channel, discord.Thread)
    }

def should_forward_embeds(message: discord.Message) -> bool:
    """
    Forward original embeds only when there is no URL in content.
    If content has a URL, Discord will auto-unfurl it in destination channels,
    and forwarding embeds as well causes duplicate previews.
    """
    content = (message.content or "").lower()
    has_url = "http://" in content or "https://" in content
    return bool(message.embeds) and not has_url

async def translate_text(content: str, target_language: str) -> str:
    """
    Translate text to target language using LLM API.
    Returns translated text or None if translation fails or API is not configured.
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
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": translation_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            translated = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            if translated:
                logger.debug(f"Translation successful: {content[:50]}... → {translated[:50]}...")
                return translated
            else:
                logger.debug(f"Translation empty response from API")
                return None
        else:
            logger.error(f"LLM API error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return None

async def post_translation_to_discord(guild: discord.Guild, channel_id: str, author_id: str, 
                                      detected_language: str, target_language: str, 
                                      translated_text: str, attachments: list = None, embeds: list = None):
    """Post translated message to a Discord channel with optional attachments and embeds"""
    try:
        channel = guild.get_channel(int(channel_id))
        if not channel:
            logger.error(f"Channel {channel_id} not found")
            return
        
        # Create message header
        header = f"<@{author_id}> -- {detected_language} → {target_language}"
        
        # Prepare files from attachments (download and re-upload to preserve them)
        files = []
        if attachments:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for attachment in attachments:
                    try:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                data = await resp.read()
                                files.append(discord.File(io.BytesIO(data), filename=attachment.filename))
                    except Exception as e:
                        logger.warning(f"Failed to download attachment {attachment.filename}: {e}")
        
        # Post as embed if text is long, otherwise as regular message
        if len(translated_text) > 2000:
            embed = discord.Embed(description=translated_text)
            await channel.send(content=header, embed=embed, files=files)
        else:
            await channel.send(f"{header}\n{translated_text}", files=files)
        
        # Send original embeds only when URL auto-unfurl is not available.
        if embeds:
            for embed in embeds:
                try:
                    await channel.send(embed=embed)
                except Exception as e:
                    logger.warning(f"Failed to send embed: {e}")
        
        logger.info(f"Posted translation to channel {channel_id} with {len(files)} attachments and {len(embeds) if embeds else 0} embeds")
    except Exception as e:
        logger.error(f"Failed to post translation: {e}")


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
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    load_defaults()
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
    if message.author.bot:
        return
    
    # Allow messages with attachments/embeds even if content is empty
    has_content = bool(message.content.strip())
    has_attachments = bool(message.attachments)
    has_embeds = bool(message.embeds)
    
    if not has_content and not has_attachments and not has_embeds:
        return

    logger.debug(f"Message in {message.channel.id}: {message.content[:100] if has_content else '[no text]'}... (attachments: {len(message.attachments)}, embeds: {len(message.embeds)})")
    
    # Check if message is only a media URL (image, gif, video)
    content_lower = message.content.strip().lower() if has_content else ""
    is_media_only = False
    
    # Check for direct media file URLs
    if content_lower.startswith(('http://', 'https://')):
        # Direct file extensions
        if any(content_lower.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.mov', '.webm')):
            is_media_only = True
        # Media hosting domains (tenor, giphy, imgur, etc.)
        elif any(domain in content_lower for domain in ('tenor.com/', 'giphy.com/', 'imgur.com/', 'gfycat.com/', 'i.redd.it/')):
            is_media_only = True
    
    # Also treat messages with only attachments/embeds as media-only
    if not has_content and (has_attachments or has_embeds):
        is_media_only = True

    # Detect language once for all operations (skip detection for media-only)
    detected_code = None if is_media_only else detect_language(message.content)
    detected_language = CODE_TO_LANGUAGE.get(detected_code) if detected_code else None

    # Base channel (handles threads)
    base_id = message.channel.parent_id if isinstance(message.channel, discord.Thread) else message.channel.id
    guild_id = message.guild.id

    # 1. ALTERNATE CHANNEL → toMain + other alts
    if base_id in alt_to_main:
        main_id = alt_to_main[base_id]["main_id"]
        main_default = default_languages.get(str(guild_id), {}).get(str(main_id), "English")
        
        if is_media_only:
            logger.info(f"Media-only message from alt channel, posting without translation")
            # Post media to main channel without translation (URL auto-embeds, attachments, embeds)
            tasks = []
            
            async def post_to_main():
                channel = message.guild.get_channel(main_id)
                if channel:
                    # Prepare files from attachments
                    files = []
                    if message.attachments:
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            for attachment in message.attachments:
                                try:
                                    async with session.get(attachment.url) as resp:
                                        if resp.status == 200:
                                            data = await resp.read()
                                            files.append(discord.File(io.BytesIO(data), filename=attachment.filename))
                                except Exception as e:
                                    logger.warning(f"Failed to download attachment: {e}")
                    
                    # Send content with attachments
                    if message.content or files:
                        await channel.send(f"<@{message.author.id}>\n{message.content}" if message.content else f"<@{message.author.id}>", files=files)
                    
                    # Send embeds separately only when URL auto-unfurl is not available
                    if should_forward_embeds(message):
                        for embed in message.embeds:
                            try:
                                await channel.send(embed=embed)
                            except Exception as e:
                                logger.warning(f"Failed to send embed: {e}")
            
            tasks.append(post_to_main())
            
            # Post to other alts without translation
            other_alts = main_to_alts.get(guild_id, {}).get(main_id, [])
            for alt in other_alts:
                if alt["id"] != base_id:
                    async def post_to_alt(alt_data=alt):
                        channel = message.guild.get_channel(alt_data["id"])
                        if channel:
                            # Prepare files from attachments
                            files = []
                            if message.attachments:
                                import aiohttp
                                async with aiohttp.ClientSession() as session:
                                    for attachment in message.attachments:
                                        try:
                                            async with session.get(attachment.url) as resp:
                                                if resp.status == 200:
                                                    data = await resp.read()
                                                    files.append(discord.File(io.BytesIO(data), filename=attachment.filename))
                                        except Exception as e:
                                                logger.warning(f"Failed to download attachment: {e}")
                            
                            # Send content with attachments
                            if message.content or files:
                                await channel.send(f"<@{message.author.id}>\n{message.content}" if message.content else f"<@{message.author.id}>", files=files)
                            
                            # Send embeds separately only when URL auto-unfurl is not available
                            if should_forward_embeds(message):
                                for embed in message.embeds:
                                    try:
                                        await channel.send(embed=embed)
                                    except Exception as e:
                                        logger.warning(f"Failed to send embed: {e}")
                    
                    tasks.append(post_to_alt())
            
            # Run all posts in parallel
            if tasks:
                await asyncio.gather(*tasks)
        else:
            logger.info(f"Message from alt channel: {detected_language} → {main_default}")
            
            # Create translation tasks to run in parallel
            tasks = []
            
            # Translate to main channel
            async def translate_to_main():
                translated = await translate_text(message.content, main_default)
                if translated:
                    forward_embeds = message.embeds if should_forward_embeds(message) else None
                    await post_translation_to_discord(message.guild, str(main_id), str(message.author.id),
                                                    detected_language or "Unknown", main_default, translated,
                                                    attachments=message.attachments, embeds=forward_embeds)
            
            tasks.append(translate_to_main())
            
            # Translate to all other alternate channels
            other_alts = main_to_alts.get(guild_id, {}).get(main_id, [])
            for alt in other_alts:
                if alt["id"] != base_id:  # Don't send back to the source channel
                    async def translate_to_alt(alt_data=alt):  # Capture alt in closure
                        logger.info(f"Message to alt channel {alt_data['id']}: {detected_language} → {alt_data['lang']}")
                        translated = await translate_text(message.content, alt_data["lang"])
                        if translated:
                            forward_embeds = message.embeds if should_forward_embeds(message) else None
                            await post_translation_to_discord(message.guild, str(alt_data["id"]), str(message.author.id),
                                                            detected_language or "Unknown", alt_data["lang"], translated,
                                                            attachments=message.attachments, embeds=forward_embeds)
                    
                    tasks.append(translate_to_alt())
            
            # Run all translations in parallel
            if tasks:
                await asyncio.gather(*tasks)
        
        return

    # 2. DEFAULT LANGUAGE MONITORING (mismatch only)
    # Skip language monitoring for media-only messages
    if not is_media_only:
        guild_defaults = default_languages.get(str(guild_id), {})
        default_lang = guild_defaults.get(str(base_id))
        if default_lang:
            try:
                # Use context-aware detection: check if text is in the default language
                # If confident it matches default language, don't translate (avoid false positives on ambiguous words)
                context_detected_code = detect_language(message.content, expected_language=default_lang)
                expected_code = LANGUAGE_CODES.get(default_lang.lower())
                if expected_code and context_detected_code and context_detected_code != expected_code:
                    context_detected_lang = CODE_TO_LANGUAGE.get(context_detected_code, "Unknown")
                    logger.info(f"Language mismatch in channel: detected {context_detected_lang}, expected {default_lang}")
                    translated = await translate_text(message.content, default_lang)
                    # Only post if translation is different from original (avoid false positives)
                    if translated and translated != message.content:
                        forward_embeds = message.embeds if should_forward_embeds(message) else None
                        await post_translation_to_discord(message.guild, str(base_id), str(message.author.id),
                                                        context_detected_lang, default_lang, translated,
                                                        attachments=message.attachments, embeds=forward_embeds)
                elif expected_code and not context_detected_code:
                    logger.debug("Language detection unavailable/unknown; skipping default-language mismatch translation")
            except Exception as e:
                logger.debug(f"Default language check failed: {e}")

    # 3. MAIN CHANNEL → toChild (one call per alternate)
    alts = main_to_alts.get(guild_id, {}).get(base_id, [])
    if alts:
        if is_media_only:
            logger.info(f"Media-only message from main channel, posting to {len(alts)} alts without translation")
            tasks = []
            for alt in alts:
                async def post_to_alt(alt_data=alt):
                    channel = message.guild.get_channel(alt_data["id"])
                    if channel:
                        # Prepare files from attachments
                        files = []
                        if message.attachments:
                            import aiohttp
                            async with aiohttp.ClientSession() as session:
                                for attachment in message.attachments:
                                    try:
                                        async with session.get(attachment.url) as resp:
                                            if resp.status == 200:
                                                data = await resp.read()
                                                files.append(discord.File(io.BytesIO(data), filename=attachment.filename))
                                    except Exception as e:
                                        logger.warning(f"Failed to download attachment: {e}")
                        
                        # Send content with attachments
                        if message.content or files:
                            await channel.send(f"<@{message.author.id}>\n{message.content}" if message.content else f"<@{message.author.id}>", files=files)
                        
                        # Send embeds separately only when URL auto-unfurl is not available
                        if should_forward_embeds(message):
                            for embed in message.embeds:
                                try:
                                    await channel.send(embed=embed)
                                except Exception as e:
                                    logger.warning(f"Failed to send embed: {e}")
                
                tasks.append(post_to_alt())
            
            # Run all posts in parallel
            if tasks:
                await asyncio.gather(*tasks)
        else:
            # Create translation tasks to run in parallel
            tasks = []
            for alt in alts:
                async def translate_to_alt(alt_data=alt):  # Capture alt in closure
                    logger.info(f"Message from main channel: {detected_language} → {alt_data['lang']}")
                    translated = await translate_text(message.content, alt_data["lang"])
                    if translated:
                        forward_embeds = message.embeds if should_forward_embeds(message) else None
                        await post_translation_to_discord(message.guild, str(alt_data["id"]), str(message.author.id),
                                                        detected_language or "Unknown", alt_data["lang"], translated,
                                                        attachments=message.attachments, embeds=forward_embeds)
                
                tasks.append(translate_to_alt())
            
            # Run all translations in parallel
            if tasks:
                await asyncio.gather(*tasks)

# ======================= COMMANDS =======================
@bot.tree.command(name="channeldefaultlanguage", description="Set default language (channel + threads)")
@app_commands.describe(language="e.g. English, Arabic")
async def cmd_default(interaction: discord.Interaction, language: str):
    if not interaction.guild:
        return await interaction.response.send_message("Server only", ephemeral=True)
    
    base_id = interaction.channel.parent_id if isinstance(interaction.channel, discord.Thread) else interaction.channel.id
    guild_str = str(interaction.guild.id)
    
    if guild_str not in default_languages:
        default_languages[guild_str] = {}
    default_languages[guild_str][str(base_id)] = language.title()
    save_defaults()
    
    await interaction.response.send_message(f"✅ Default language **{language.title()}** set for this channel + threads.")

@bot.tree.command(name="removedefaultlanguage", description="Stop monitoring this channel")
async def cmd_remove(interaction: discord.Interaction):
    if not interaction.guild:
        return await interaction.response.send_message("Server only", ephemeral=True)
    
    base_id = interaction.channel.parent_id if isinstance(interaction.channel, discord.Thread) else interaction.channel.id
    guild_str = str(interaction.guild.id)
    
    if guild_str in default_languages and str(base_id) in default_languages[guild_str]:
        del default_languages[guild_str][str(base_id)]
        save_defaults()
        await interaction.response.send_message("✅ Monitoring removed.")
    else:
        await interaction.response.send_message("No default set here.")

@bot.tree.command(name="createalternatelanguagechannel", description="Create 'Name but in X' channel")
@app_commands.describe(language="e.g. Arabic")
async def cmd_create(interaction: discord.Interaction, language: str):
    if not interaction.guild:
        return await interaction.response.send_message("Server only", ephemeral=True)
    
    main_ch = interaction.channel.parent if isinstance(interaction.channel, discord.Thread) else interaction.channel
    new_name = f"{main_ch.name} but in {language.title()}"
    
    try:
        new_ch = await interaction.guild.create_text_channel(
            new_name,
            category=main_ch.category,
            topic=f"Alternate {language.title()} translations"
        )
        # Set default language for the new alternate channel
        guild_str = str(interaction.guild.id)
        if guild_str not in default_languages:
            default_languages[guild_str] = {}
        default_languages[guild_str][str(new_ch.id)] = language.title()
        save_defaults()
        
        rebuild_alt_maps(interaction.guild)  # immediate update
        await interaction.response.send_message(f"✅ Created **{new_name}** and enabled bridging.")
    except discord.Forbidden:
        await interaction.response.send_message("❌ Missing Manage Channels permission.", ephemeral=True)

bot.run(TOKEN)