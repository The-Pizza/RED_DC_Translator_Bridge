import os
import json
import logging
import asyncio
import io
from dataclasses import dataclass
from typing import List, Optional
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

def detect_language(text: str, expected_language: str = None) -> str:
    """
    Detect language using fastText. Returns ISO 639-1 code (e.g., 'en', 'es')

    If expected_language is provided (e.g., "English", "Arabic"), and that language
    appears in the top predictions with probability reasonably close to the top result,
    prefer the expected language. This reduces false positives on short/ambiguous text
    (for example: "Pizza", or typo-heavy messages).
    """
    if not model:
        return None
    
    try:
        # fastText expects one line at a time - replace newlines with spaces
        text_single_line = text.replace('\n', ' ').replace('\r', ' ').strip()
        
        # Get predictions (top 5 for better accuracy)
        predictions = model.predict(text_single_line, k=5)
        labels = predictions[0]  # list of '__label__xx'
        probs = predictions[1]   # corresponding probabilities
        
        # Strip '__label__' prefix
        codes = [label.replace('__label__', '') for label in labels]
        
        # Prefer expected language when it is reasonably close to the top prediction.
        # This handles ambiguous text where channel context is a stronger signal.
        if expected_language:
            expected_code = LANGUAGE_CODES.get(expected_language.lower())
            if expected_code:
                top_code = codes[0] if len(codes) > 0 else None
                top_prob = float(probs[0]) if len(probs) > 0 else 0.0

                if top_code == expected_code:
                    logger.debug(f"Language match: expected {expected_language} is already top prediction")
                    return expected_code

                for code, prob in zip(codes, probs):
                    prob = float(prob)
                    # Use expected language if it is within a reasonable margin of the top probability.
                    # Ratio threshold keeps behavior stable across both high-confidence and low-confidence inputs.
                    if code == expected_code and top_prob > 0 and (prob / top_prob) >= 0.60:
                        logger.debug(
                            f"Language override by channel context: expected {expected_language} "
                            f"({prob:.1%}) vs top {top_code} ({top_prob:.1%})"
                        )
                        return expected_code
                # If not within margin, fall through to standard detection.
        
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

def should_forward_embeds(message: discord.Message) -> bool:
    """
    Forward original embeds only when there is no URL in content.
    If content has a URL, Discord will auto-unfurl it in destination channels,
    and forwarding embeds as well causes duplicate previews.
    """
    content = (message.content or "").lower()
    has_url = "http://" in content or "https://" in content
    return bool(message.embeds) and not has_url


def is_media_only_message(message: discord.Message, has_content: bool, has_attachments: bool, has_embeds: bool) -> bool:
    """Classify whether a message should skip translation and be forwarded as media-only."""
    content_lower = message.content.strip().lower() if has_content else ""
    is_media_only = False

    if content_lower.startswith(("http://", "https://")):
        if any(content_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov", ".webm")):
            is_media_only = True
        elif any(domain in content_lower for domain in ("tenor.com/", "giphy.com/", "imgur.com/", "gfycat.com/", "i.redd.it/")):
            is_media_only = True

    if not has_content and (has_attachments or has_embeds):
        is_media_only = True

    return is_media_only


def analyze_message(message: discord.Message) -> Optional[MessageContext]:
    """Analyze inbound message once and normalize all routing/language state."""
    if message.author.bot:
        return None

    has_content = bool(message.content.strip())
    has_attachments = bool(message.attachments)
    has_embeds = bool(message.embeds)

    if not has_content and not has_attachments and not has_embeds:
        return None

    is_media_only = is_media_only_message(message, has_content, has_attachments, has_embeds)
    base_id = message.channel.parent_id if isinstance(message.channel, discord.Thread) else message.channel.id
    guild_id = message.guild.id
    expected_language = default_languages.get(str(guild_id), {}).get(str(base_id))

    detected_code = None if is_media_only else detect_language(message.content, expected_language=expected_language)
    detected_language = CODE_TO_LANGUAGE.get(detected_code) if detected_code else None

    is_alt_source = base_id in alt_to_main
    source_main_id = alt_to_main[base_id]["main_id"] if is_alt_source else None
    source_main_default_language = None
    source_alts = []

    if is_alt_source and source_main_id is not None:
        source_main_default_language = default_languages.get(str(guild_id), {}).get(str(source_main_id), "English")
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
    )


def build_route_plan(context: MessageContext) -> List[RouteAction]:
    """Build delivery plan without performing network/translation work."""
    plan: List[RouteAction] = []

    if context.is_alt_source and context.source_main_id is not None:
        if context.is_media_only:
            plan.append(RouteAction(destination_channel_id=context.source_main_id, action_type="copy"))
            for alt in context.source_alts:
                if alt["id"] != context.base_id:
                    plan.append(RouteAction(destination_channel_id=alt["id"], action_type="copy"))
        else:
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


async def fetch_attachment_payloads(attachments: List[discord.Attachment]) -> List[tuple]:
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
                        logger.warning(f"Failed to download attachment {attachment.filename}: status {resp.status}")
            except Exception as e:
                logger.warning(f"Failed to download attachment {attachment.filename}: {e}")
    return payloads


async def compose_outbound_messages(context: MessageContext, plan: List[RouteAction]) -> List[OutboundMessage]:
    """Compose outbound payloads (copy/translate) that can be sent via one sender function."""
    outbounds: List[OutboundMessage] = []

    attachments_data: List[tuple] = []
    if context.has_attachments:
        attachments_data = await fetch_attachment_payloads(context.message.attachments)

    embeds = list(context.message.embeds) if context.should_forward_original_embeds else []
    body_text = context.message.content or ""

    async def compose_translate(action: RouteAction) -> Optional[OutboundMessage]:
        if not action.target_language:
            return None

        translated = await translate_text(body_text, action.target_language)
        if not translated:
            return None

        if action.include_mismatch_guard and translated == body_text:
            return None

        header = f"<@{context.message.author.id}> -- {(context.detected_language or 'Unknown')} → {action.target_language}"
        return OutboundMessage(
            destination_channel_id=action.destination_channel_id,
            header_text=header,
            body_text=body_text,
            translated_text=translated,
            attachments_data=attachments_data,
            embeds=embeds,
            detected_language=context.detected_language,
            target_language=action.target_language,
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
                )
            )
        elif action.action_type == "translate":
            translate_actions.append(action)

    if translate_actions:
        translated_outbounds = await asyncio.gather(*(compose_translate(action) for action in translate_actions))
        outbounds.extend([outbound for outbound in translated_outbounds if outbound is not None])

    return outbounds


async def send_outbound_message(guild: discord.Guild, outbound: OutboundMessage):
    """Single sender entrypoint for all outbound message variants."""
    try:
        channel = guild.get_channel(int(outbound.destination_channel_id))
        if not channel:
            logger.error(f"Channel {outbound.destination_channel_id} not found")
            return

        files = [discord.File(io.BytesIO(data), filename=filename) for filename, data in outbound.attachments_data]
        body = outbound.translated_text if outbound.translated_text is not None else outbound.body_text

        if outbound.translated_text is not None:
            if len(body) > 2000:
                embed = discord.Embed(description=body)
                await channel.send(content=outbound.header_text, embed=embed, files=files)
            else:
                await channel.send(f"{outbound.header_text}\n{body}", files=files)
        else:
            if body or files:
                if body:
                    await channel.send(f"{outbound.header_text}\n{body}", files=files)
                else:
                    await channel.send(outbound.header_text, files=files)

        for embed in outbound.embeds:
            try:
                await channel.send(embed=embed)
            except Exception as e:
                logger.warning(f"Failed to send embed: {e}")

        logger.info(
            f"Sent outbound to channel {outbound.destination_channel_id} "
            f"(translated={outbound.translated_text is not None}, attachments={len(files)}, embeds={len(outbound.embeds)})"
        )
    except Exception as e:
        logger.error(f"Failed to send outbound message: {e}")

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
    context = analyze_message(message)
    if not context:
        return

    plan = build_route_plan(context)
    if not plan:
        return

    logger.info(f"Built processing plan for message {message.id}: {len(plan)} actions")

    outbounds = await compose_outbound_messages(context, plan)
    if not outbounds:
        return

    await asyncio.gather(*(send_outbound_message(message.guild, outbound) for outbound in outbounds))

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