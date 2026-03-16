import discord
from discord import app_commands

import database as db


def register_commands(bot, rebuild_alt_maps, logger):
    @bot.tree.command(
        name="channeldefaultlanguage",
        description="Set default language (channel + threads)",
    )
    @app_commands.describe(language="e.g. English, Arabic")
    async def cmd_default(interaction: discord.Interaction, language: str):
        if not interaction.guild:
            return await interaction.response.send_message("Server only", ephemeral=True)

        base_id = (
            interaction.channel.parent_id
            if isinstance(interaction.channel, discord.Thread)
            else interaction.channel.id
        )

        try:
            await db.set_channel_default(interaction.guild.id, base_id, language.title())
            await interaction.response.send_message(
                f"✅ Default language **{language.title()}** set for this channel + threads.",
                ephemeral=True,
            )
        except Exception as e:
            logger.error(f"Failed to set channel default: {e}")
            await interaction.response.send_message(
                f"❌ Failed to set default language: {str(e)}",
                ephemeral=True,
            )

    @bot.tree.command(
        name="removedefaultlanguage",
        description="Stop monitoring this channel",
    )
    async def cmd_remove(interaction: discord.Interaction):
        if not interaction.guild:
            return await interaction.response.send_message("Server only", ephemeral=True)

        base_id = (
            interaction.channel.parent_id
            if isinstance(interaction.channel, discord.Thread)
            else interaction.channel.id
        )

        try:
            removed = await db.remove_channel_default(interaction.guild.id, base_id)
            if removed:
                await interaction.response.send_message("✅ Monitoring removed.", ephemeral=True)
            else:
                await interaction.response.send_message("No default set here.", ephemeral=True)
        except Exception as e:
            logger.error(f"Failed to remove channel default: {e}")
            await interaction.response.send_message(
                f"❌ Failed to remove default: {str(e)}",
                ephemeral=True,
            )

    @bot.tree.command(
        name="createalternatelanguagechannel",
        description="Create 'Name but in X' channel",
    )
    @app_commands.describe(language="e.g. Arabic")
    async def cmd_create(interaction: discord.Interaction, language: str):
        if not interaction.guild:
            return await interaction.response.send_message("Server only", ephemeral=True)

        main_ch = (
            interaction.channel.parent
            if isinstance(interaction.channel, discord.Thread)
            else interaction.channel
        )
        new_name = f"{main_ch.name}-but-in-{language.title()}"

        try:
            new_ch = await interaction.guild.create_text_channel(
                new_name,
                category=main_ch.category,
                topic=f"Alternate {language.title()} translations",
            )
            await db.set_channel_default(interaction.guild.id, new_ch.id, language.title())

            rebuild_alt_maps(interaction.guild)
            await interaction.response.send_message(
                f"✅ Created **{new_name}** and enabled bridging.",
                ephemeral=True,
            )
        except discord.Forbidden:
            await interaction.response.send_message(
                "❌ Missing Manage Channels permission.",
                ephemeral=True,
            )
        except Exception as e:
            logger.error(f"Failed to create alternate channel: {e}")
            await interaction.response.send_message(
                f"❌ Failed to create channel: {str(e)}",
                ephemeral=True,
            )

    @bot.tree.command(name="stats", description="Show bot statistics")
    async def cmd_stats(interaction: discord.Interaction):
        """Display bot usage statistics from the database."""
        try:
            await interaction.response.defer(ephemeral=True)

            stats = await db.get_statistics()

            guild_id = interaction.guild.id if interaction.guild else None
            token_stats = await db.get_current_month_token_usage(guild_id=guild_id)

            embed = discord.Embed(
                title="📊 Translation Bot Statistics",
                color=discord.Color.blue(),
                timestamp=discord.utils.utcnow(),
            )

            embed.add_field(
                name="💬 Messages",
                value=(
                    f"**Unique messages:** {stats['unique_messages']:,}\n"
                    f"**Total tracked:** {stats['total_messages_tracked']:,}\n"
                    f"**Translated:** {stats['translated_messages']:,}\n"
                    f"**With replies:** {stats['reply_messages']:,}"
                ),
                inline=False,
            )

            if token_stats["total_tokens"] > 0:
                embed.add_field(
                    name="🤖 AI Token Usage (Current Month)",
                    value=(
                        f"**Total tokens:** {token_stats['total_tokens']:,}\n"
                        f"**Prompt tokens:** {token_stats['total_prompt_tokens']:,}\n"
                        f"**Completion tokens:** {token_stats['total_completion_tokens']:,}\n"
                        f"**Translations:** {token_stats['translation_count']:,}"
                    ),
                    inline=False,
                )

            if stats.get("total_reactions_synced", 0) > 0:
                embed.add_field(
                    name="😀 Reactions",
                    value=(
                        f"**Total synced:** {stats['total_reactions_synced']:,}\n"
                        f"**Unique emojis:** {stats['unique_emojis_synced']:,}\n"
                        f"**Users:** {stats['users_with_synced_reactions']:,}"
                    ),
                    inline=False,
                )

            embed.add_field(
                name="⚙️ Configuration",
                value=(
                    f"**Guilds configured:** {stats['guilds_configured']}\n"
                    f"**Channels monitored:** {stats['channels_configured']}"
                ),
                inline=False,
            )

            if stats["top_languages"]:
                lang_list = "\n".join(
                    [f"**{lang}:** {count}" for lang, count in stats["top_languages"]]
                )
                embed.add_field(name="🌐 Top Languages", value=lang_list, inline=False)

            embed.set_footer(text=f"Requested by {interaction.user.name}")

            await interaction.followup.send(embed=embed, ephemeral=True)

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            await interaction.followup.send(
                f"❌ Failed to retrieve statistics: {str(e)}",
                ephemeral=True,
            )
