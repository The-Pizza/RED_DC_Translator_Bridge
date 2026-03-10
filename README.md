# RED DC Translator Bridge

A Discord bot that automatically detects and bridges messages across multiple language channels, enabling seamless multilingual conversations within Discord servers.

## Overview

RED DC Translator Bridge is a sophisticated Discord bot that:

- **Detects message languages** using Meta's fastText model with context-aware confidence thresholds
- **Bridges alternate language channels** (e.g., "general-but-in-spanish")
- **Translates messages directly using LLM APIs** with parallel execution for speed (Grok, OpenAI-compatible, etc.)
- **Monitors channel language mismatches** against expected defaults
- **Handles media-only messages** (Tenor, Giphy, Imgur) intelligently to preserve auto-embeds
- **Creates language-specific channels** on demand through slash commands

The bot is designed for environments where teams communicate in multiple languages and need intelligent message routing and translation coordination.

## Docker Deployment

**This code is primarily designed to be run from a Docker image.**

The official Docker image is hosted in the GitHub Container Repository:

```
https://github.com/The-Pizza/RED_DC_Translator_Bridge
```

### Quick Start with Docker

Pull and run the Docker image:

```bash
docker run -d \
  -e DISCORD_TOKEN=your_bot_token \
  -e LLM_API_KEY=your_api_key \
  -e LLM_API_URL=https://api.x.ai/v1/chat/completions \
  -e LLM_MODEL=grok-4-1-fast-non-reasoning \
  -e LOG_LEVEL=INFO \
  -e GUILD_ID=optional_server_id \
  -v translator-data:/data \
  ghcr.io/the-pizza/red-dc-translator-bridge:latest
```

**Important**: The container automatically downloads the 131MB fastText language model (`lid.176.bin`) on first startup. This is stored in the `/data` volume for persistence.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_TOKEN` | Yes | Discord bot token |
| `LLM_API_KEY` | Yes | API key for your LLM provider (Grok, OpenAI, etc.) |
| `LLM_API_URL` | No | LLM API endpoint (default: `https://api.x.ai/v1/chat/completions` for Grok) |
| `LLM_MODEL` | No | Model identifier (default: `grok-4-1-fast-non-reasoning`) |
| `LLM_SYSTEM_PROMPT` | No | Custom system prompt for the translator (default: professional translator prompt) |
| `GUILD_ID` | No | Specific server ID for faster slash command sync (global sync if omitted) |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) - options: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `DATA_DIR` | No | Directory for storing data (default: `/data`) |
| `DATABASE_URL` | No | Database connection string (default: `sqlite+aiosqlite:///data/bot.db` for SQLite) |

### LLM Provider Configuration

#### Grok (Default)
```bash
docker run -d \
  -e DISCORD_TOKEN=your_discord_token \
  -e LLM_API_KEY=your_grok_api_key \
  -e LLM_API_URL=https://api.x.ai/v1/chat/completions \
  -e LLM_MODEL=grok-4-1-fast-non-reasoning \
  -v translator-data:/data \
  ghcr.io/the-pizza/red-dc-translator-bridge:latest
```

#### OpenAI
```bash
docker run -d \
  -e DISCORD_TOKEN=your_discord_token \
  -e LLM_API_KEY=your_openai_api_key \
  -e LLM_API_URL=https://api.openai.com/v1/chat/completions \
  -e LLM_MODEL=gpt-4-turbo \
  -v translator-data:/data \
  ghcr.io/the-pizza/red-dc-translator-bridge:latest
```

#### Compatible Alternatives
Any OpenAI-compatible API endpoint works (Azure OpenAI, local LM Studio, Ollama, etc.). Just set:
- `LLM_API_URL` to your endpoint
- `LLM_API_KEY` to your authentication key
- `LLM_MODEL` to the appropriate model name

## Local Development

### Prerequisites
- Python 3.10+
- Build tools (`build-essential`, `g++` for fastText compilation)
- Discord bot token
- LLM API credentials (Grok, OpenAI, etc.)

### Setup

```bash
# Clone repository
git clone https://github.com/The-Pizza/RED_DC_Translator_Bridge.git
cd RED_DC_Translator_Bridge

# Install build dependencies (Debian/Ubuntu)
sudo apt-get update && sudo apt-get install -y build-essential g++

# Install dependencies
pip install -r requirements.txt

# Download fastText language model (131MB)
mkdir -p data
wget -O data/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Set environment variables
export DISCORD_TOKEN=your_discord_token
export LLM_API_KEY=your_api_key
export LLM_API_URL=https://api.x.ai/v1/chat/completions
export LLM_MODEL=grok-4-1-fast-non-reasoning
export LOG_LEVEL=INFO

# Run bot
python bot.py
```

## Commands

### `/channeldefaultlanguage`
Set the default language for a channel. When messages in a different language are posted, they'll be automatically translated to the channel's default language.

**Usage:** `/channeldefaultlanguage language:English`

### `/removedefaultlanguage`
Stop monitoring a channel for language mismatches.

**Usage:** `/removedefaultlanguage`

### `/createalternatelanguagechannel`
Create a new alternate language channel in the format `channel-name-but-in-{language}`. Messages in the main channel will automatically be translated to this alternate channel, and vice versa.

**Usage:** `/createalternatelanguagechannel language:Spanish`

### `/stats`
Display bot usage statistics including message counts, translations, configured channels, and most popular languages. Results are shown only to you (ephemeral message).

**Usage:** `/stats`

## How It Works

### Language Detection

The bot uses Meta's **fastText** pre-trained model (`lid.176.bin`) to detect language with context-aware confidence thresholds:
- When a channel has an expected default language, the bot requires **50% confidence** for a different language detection
- Supports 176 languages with high accuracy even on short text fragments
- Model is lightweight (131MB) and extremely fast

### Message Routing & Translation

1. **Message Detection**: When a message is posted, the bot detects its language using fastText
2. **Media Handling**: Messages containing only media URLs (Tenor, Giphy, Imgur, etc.) are cross-posted without translation to preserve auto-embeds
3. **Routing Logic**:
   - **Alternate channels**: Messages from "channel-but-in-spanish" are translated to the main channel and all other alternate language channels
   - **Default language monitoring**: If a channel has a default language set and a message in a different language is posted, it gets translated to the channel's default language
   - **Main channels**: Messages are translated and forwarded to all alternate language channels
4. **Parallel Translation**: When translating to multiple channels, all LLM API calls execute concurrently using `asyncio.gather()` for optimal performance
5. **LLM Translation**: The detected text is sent to your configured LLM (Grok, OpenAI, etc.) for high-quality translation
6. **Posting**: The translated message is posted with a header showing the original author and language pair
7. **Message Tracking**: All messages are tracked in the database with unique IDs for reply routing and reaction syncing

### Reaction Syncing

When a user adds a reaction to a message in any bridged channel, the bot automatically:
- Detects the reaction on the tracked message
- Looks up all related messages (translations in other channels)
- Adds the same reaction to all linked messages asynchronously
- Tracks the reaction sync in the database for statistics

This keeps reactions synchronized across all language channels, maintaining engagement context regardless of which language channel users are viewing.

### Supported Languages

English, Arabic, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Hindi

(Can be extended by modifying the `LANGUAGE_CODES` dictionary in `bot.py`)

## System Requirements

- 256MB+ RAM (plus 131MB for fastText model)
- Stable internet connection
- Discord server with required bot permissions:
  - **Manage Channels** - for creating alternate language channels
  - **Read Messages/View Channels** - for monitoring messages
  - **Send Messages** - for posting translations
  - **Add Reactions** - for syncing reactions across bridged messages
  - **Read Message History** - for fetching messages when syncing reactions
  - **Embed Links** - for sending rich embeds
- LLM API access (free tiers available for many providers)
- Build tools for compiling fastText (if installing from source)

## Discord Bot Setup

When creating your Discord bot in the [Discord Developer Portal](https://discord.com/developers/applications):

1. Enable the following **Privileged Gateway Intents**:
   - **Message Content Intent** - required to read message text
   - *(Reactions Intent is not privileged, enabled by default)*

2. Bot permissions (for invite link):
   - Manage Channels
   - Read Messages/View Channels  
   - Send Messages
   - Add Reactions
   - Read Message History
   - Embed Links
   - Use Slash Commands

## Configuration Storage

The bot uses **SQLAlchemy** with database persistence for storing:
- Channel default language settings
- Message tracking for reply routing and reaction syncing
- Usage statistics

### Database Backends

**SQLite (Default)**:
- Stores data in `/data/bot.db`
- Zero configuration required
- Perfect for single-instance deployments
- Database file persists in the `/data` volume

**PostgreSQL (Optional)**:
- For larger deployments or multi-shard setups
- Set `DATABASE_URL` environment variable:
  ```bash
  DATABASE_URL=postgresql+asyncpg://user:password@postgres-host:5432/botdb
  ```
- Requires PostgreSQL server (not included in container)

The bot automatically creates all required tables on first startup. No manual database setup needed.

## Requirements

See [requirements.txt](requirements.txt):

- `discord.py>=2.3.0` - Discord bot library
- `requests` - HTTP client for LLM API calls
- `aiohttp` - Async HTTP client
- `fasttext` - Meta's language detection engine (requires numpy<2 for compatibility)
- `numpy<2` - Numerical computing library (pinned for fastText runtime compatibility)
- `sqlalchemy>=2.0.0` - Database ORM with async support
- `aiosqlite>=0.19.0` - Async SQLite driver
- `asyncpg>=0.29.0` - Async PostgreSQL driver
- `alembic>=1.13.0` - Database migrations (future use)

## Performance

- **Parallel Translation**: When routing messages to multiple channels, all translations execute concurrently for minimal latency
- **Language Detection**: fastText inference is extremely fast (<1ms per message)
- **Media Handling**: Media-only URLs bypass translation entirely to preserve Discord auto-embeds

## License

See LICENSE.md

## Repository

Source code and Docker image: https://github.com/The-Pizza/RED_DC_Translator_Bridge
