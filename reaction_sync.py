import asyncio

import discord

import database as db


async def handle_reaction_add(bot, reaction: discord.Reaction, user: discord.User, logger):
    """
    Sync reactions across bridged/translated messages.
    When a user adds a reaction to a message, add the same reaction to all linked messages.
    """
    if user.bot:
        return

    message = reaction.message

    if not message.guild:
        return

    try:
        linked_messages = await db.get_all_linked_messages(message.id, message.channel.id)

        if not linked_messages:
            return

        logger.info(
            f"Syncing reaction {reaction.emoji} from message {message.id} to {len(linked_messages)} linked messages"
        )

        emoji_str = str(reaction.emoji)

        async def add_reaction_to_message(linked_msg: dict):
            try:
                channel = bot.get_channel(linked_msg["channel_id"])
                if not channel:
                    logger.warning(
                        f"Channel {linked_msg['channel_id']} not found for reaction sync"
                    )
                    return False

                target_message = await channel.fetch_message(linked_msg["message_id"])
                if not target_message:
                    logger.warning(
                        f"Message {linked_msg['message_id']} not found in channel {linked_msg['channel_id']}"
                    )
                    return False

                await target_message.add_reaction(reaction.emoji)

                await db.track_reaction_sync(
                    source_message_id=message.id,
                    source_channel_id=message.channel.id,
                    emoji=emoji_str,
                    user_id=user.id,
                    synced_to_message_id=linked_msg["message_id"],
                    synced_to_channel_id=linked_msg["channel_id"],
                    guild_id=linked_msg["guild_id"],
                )

                return True
            except discord.Forbidden:
                logger.warning(
                    f"Missing permissions to add reaction in channel {linked_msg['channel_id']}"
                )
                return False
            except discord.NotFound:
                logger.warning("Message or channel not found for reaction sync")
                return False
            except Exception as e:
                logger.error(f"Failed to sync reaction: {e}")
                return False

        results = await asyncio.gather(
            *(add_reaction_to_message(linked_msg) for linked_msg in linked_messages),
            return_exceptions=True,
        )

        success_count = sum(1 for r in results if r is True)
        logger.info(
            f"Successfully synced reaction to {success_count}/{len(linked_messages)} messages"
        )

    except Exception as e:
        logger.error(f"Error in reaction sync handler: {e}")
