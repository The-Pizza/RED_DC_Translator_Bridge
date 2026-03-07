import os
import json
import logging
import requests
import discord
from discord import app_commands
from discord.ext import commands
from lingua import Language, LanguageDetectorBuilder

# ========================== CONFIG ==========================
TOKEN = os.getenv("DISCORD_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
GUILD_ID = os.getenv("GUILD_ID")          # optional - faster slash command sync
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DATA_DIR = os.getenv("DATA_DIR", "/data")
DEFAULT_FILE = os.path.join(DATA_DIR, "default_languages.json")

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Language map (extend as needed)
LANGUAGE_CODES = {
    "english": "en", "arabic": "ar", "spanish": "es", "french": "fr",
    "german": "de", "italian": "it", "portuguese": "pt", "russian": "ru",
    "chinese": "zh", "japanese": "ja", "korean": "ko", "hindi": "hi"
}

# Reverse map: ISO 639-1 code -> Language name (title case)
CODE_TO_LANGUAGE = {v: k.title() for k, v in LANGUAGE_CODES.items()}

# Initialize lingua detector with specific languages for better accuracy on short text
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.ARABIC, Language.SPANISH, Language.FRENCH,
    Language.GERMAN, Language.ITALIAN, Language.PORTUGUESE, Language.RUSSIAN,
    Language.CHINESE, Language.JAPANESE, Language.KOREAN, Language.HINDI
).build()

# In-memory storage
default_languages = {}      # guild_id_str -> {channel_id_str: lang_name}
main_to_alts = {}           # guild_id -> {main_id: [{{"id": alt_id, "lang": lang_name}}, ...]}
alt_to_main = {}            # alt_id -> {{"main_id": , "lang": }}

def detect_language(text: str) -> str:
    """Detect language using lingua. Returns ISO 639-1 code (e.g., 'en', 'es')"""
    try:
        lang = detector.detect_language_of(text)
        if lang:
            logger.debug(f"Language detected: {lang}")
            return lang.iso_code_639_1.name.lower()
        return None
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

def forward(payload: dict):
    if not WEBHOOK_URL:
        return
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=10, verify=False)
        if r.status_code >= 400:
            logger.error(f"Webhook error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.error(f"Webhook failed: {e}")

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
    if message.author.bot or not message.content.strip():
        return

    logger.debug(f"Message in {message.channel.id}: {message.content[:100]}...")

    # Detect language once for all operations
    detected_code = detect_language(message.content)
    detected_language = CODE_TO_LANGUAGE.get(detected_code) if detected_code else None

    # Base channel (handles threads)
    base_id = message.channel.parent_id if isinstance(message.channel, discord.Thread) else message.channel.id
    guild_id = message.guild.id

    # 1. ALTERNATE CHANNEL → toMain + other alts
    if base_id in alt_to_main:
        main_id = alt_to_main[base_id]["main_id"]
        main_default = default_languages.get(str(guild_id), {}).get(str(main_id), "English")
        
        # Forward to main channel
        payload = message_to_dict(message)
        payload["DetectedLanguage"] = detected_language
        payload["TargetLanguage"] = main_default
        payload["AltLanguageChannel"] = {
            "Type": "toMain",
            "TranslateTo": main_default,
            "ChannelId": str(main_id)
        }
        forward(payload)
        logger.info(json.dumps(payload, default=str, ensure_ascii=False))
        
        # Forward to all other alternate channels
        other_alts = main_to_alts.get(guild_id, {}).get(main_id, [])
        for alt in other_alts:
            if alt["id"] != base_id:  # Don't send back to the source channel
                payload = message_to_dict(message)
                payload["DetectedLanguage"] = detected_language
                payload["TargetLanguage"] = alt["lang"]
                payload["AltLanguageChannel"] = {
                    "Type": "toAlt",
                    "TranslateTo": alt["lang"],
                    "ChannelId": str(alt["id"])
                }
                forward(payload)
                logger.info(json.dumps(payload, default=str, ensure_ascii=False))
        
        return

    # 2. DEFAULT LANGUAGE MONITORING (mismatch only)
    guild_defaults = default_languages.get(str(guild_id), {})
    default_lang = guild_defaults.get(str(base_id))
    if default_lang and len(message.content) >= 3:
        try:
            expected_code = LANGUAGE_CODES.get(default_lang.lower())
            if expected_code and detected_code != expected_code:
                payload = message_to_dict(message)
                payload["DetectedLanguage"] = detected_language
                payload["TargetLanguage"] = default_lang
                forward(payload)
                logger.info(json.dumps(payload, default=str, ensure_ascii=False))
        except Exception:
            pass  # short/undetectable text

    # 3. MAIN CHANNEL → toChild (one call per alternate)
    alts = main_to_alts.get(guild_id, {}).get(base_id, [])
    for alt in alts:
        payload = message_to_dict(message)
        payload["DetectedLanguage"] = detected_language
        payload["TargetLanguage"] = alt["lang"]
        payload["AltLanguageChannel"] = {
            "Type": "toChild",
            "TranslateTo": alt["lang"],
            "ChannelId": str(alt["id"])
        }
        forward(payload)
        logger.info(json.dumps(payload, default=str, ensure_ascii=False))

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