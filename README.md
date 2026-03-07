# RED DC Translator Bridge

A Discord bot that automatically detects and bridges messages across multiple language channels, enabling seamless multilingual conversations within Discord servers.

## Overview

RED DC Translator Bridge is a sophisticated Discord bot that:

- **Detects message languages** using linguistic analysis
- **Bridges alternate language channels** (e.g., "general-but-in-spanish")
- **Monitors channel language mismatches** against expected defaults
- **Forwards messages with translation metadata** to configured webhook endpoints
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
  -e WEBHOOK_URL=https://your-webhook-endpoint.com \
  -e LOG_LEVEL=INFO \
  -e GUILD_ID=optional_server_id \
  -v translator-data:/data \
  ghcr.io/the-pizza/red-dc-translator-bridge:latest
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DISCORD_TOKEN` | Yes | Discord bot token |
| `WEBHOOK_URL` | No | Webhook endpoint for message forwarding |
| `GUILD_ID` | No | Specific server ID for faster slash command sync (global sync if omitted) |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) - options: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `DATA_DIR` | No | Directory for storing configuration (default: `/data`) |

## Local Development

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Running Locally

```bash
export DISCORD_TOKEN=your_bot_token
export WEBHOOK_URL=https://your-webhook-endpoint.com
python bot.py
```

## How It Works

### Channel Bridging

The bot recognizes channels with the naming pattern `name-but-in-language`:

- **Main Channel**: `general`
- **Alternate Channel**: `general-but-in-spanish`

Messages are automatically routed between these channels with language metadata attached.

### Commands

#### `/channeldefaultlanguage <language>`
Set the expected default language for a channel. The bot will forward any messages in other languages to the configured webhook.

**Example:**
```
/channeldefaultlanguage English
```

#### `/removedefaultlanguage`
Stop monitoring a channel for language mismatches.

#### `/createalternatelanguagechannel <language>`
Create a new alternate language channel (automatically named `current-channel-name but in <language>`).

**Example:**
```
/createalternatelanguagechannel Spanish
```

### Message Flow

1. **Detection**: Message language is automatically detected
2. **Routing**:
   - If in an alternate channel → forward to main channel + other alternates
   - If in main channel → forward to all alternate channels
   - If language mismatches channel default → forward to webhook
3. **Webhook Payload**: Message with language metadata is sent to configured endpoint

### Supported Languages

English, Arabic, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Hindi

(Can be extended by modifying the `LANGUAGE_CODES` dictionary in `bot.py`)

## Configuration Storage

Default language settings are persisted to `/data/default_languages.json` (or `DATA_DIR` if configured).

## Requirements

See [requirements.txt](requirements.txt):

- `discord.py>=2.3.0` - Discord bot library
- `requests` - HTTP client for webhooks
- `lingua-language-detector` - Language detection engine

## License

See LICENSE.md

## Repository

Source code and Docker image: https://github.com/The-Pizza/RED_DC_Translator_Bridge
