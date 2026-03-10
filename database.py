"""
Database layer using SQLAlchemy with support for both SQLite and PostgreSQL.
Handles channel defaults, message tracking for replies and reaction syncing.
"""
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from sqlalchemy import BigInteger, String, DateTime, Text, Index, select, delete, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)

# ========================== CONFIGURATION ==========================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:////data/bot.db"  # Default to SQLite in /data volume
)

# For PostgreSQL, use: postgresql+asyncpg://user:pass@host:5432/dbname
# SQLAlchemy automatically handles schema differences between SQLite and PostgreSQL


# ========================== MODELS ==========================
class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class ChannelDefault(Base):
    """Stores default language for each channel"""
    __tablename__ = "channel_defaults"
    
    guild_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    channel_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    language: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ChannelDefault(guild={self.guild_id}, channel={self.channel_id}, lang={self.language})>"


class MessageMapping(Base):
    """
    Tracks messages sent across bridged channels.
    Each row represents one message in one destination channel.
    Multiple rows with same source_message_id = message sent to multiple channels.
    """
    __tablename__ = "message_mappings"
    
    # Primary key: combination of source message and destination channel
    source_message_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    dest_channel_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    
    # Message IDs
    dest_message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source_channel_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    # Author info (for display in bridged messages)
    author_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    # Language/translation info
    source_language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    dest_language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Reply tracking - if this message was a reply, store the original reply target
    reply_to_message_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_dest_message', 'dest_message_id'),
        Index('idx_source_channel', 'source_channel_id'),
        Index('idx_guild', 'guild_id'),
        Index('idx_reply_to', 'reply_to_message_id'),
    )
    
    def __repr__(self):
        return f"<MessageMapping(src={self.source_message_id}, dest_ch={self.dest_channel_id}, dest_msg={self.dest_message_id})>"


class ReactionSync(Base):
    """
    Tracks reactions synced across bridged messages.
    Records when a reaction on one message is copied to its translations.
    """
    __tablename__ = "reaction_syncs"
    
    # Use Integer for autoincrement compatibility with SQLite
    # BigInteger autoincrement can have issues with SQLite
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Original reaction details
    source_message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source_channel_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    # The emoji that was added (stored as string - can be unicode or custom emoji ID)
    emoji: Mapped[str] = mapped_column(String(200), nullable=False)
    
    # User who added the reaction
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    # Where the reaction was synced to
    synced_to_message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    synced_to_channel_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    # When the sync happened
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Indexes for queries
    __table_args__ = (
        Index('idx_source_msg_emoji', 'source_message_id', 'emoji'),
        Index('idx_guild_reactions', 'guild_id'),
        Index('idx_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ReactionSync(msg={self.source_message_id}, emoji={self.emoji}, to={self.synced_to_message_id})>"


class TokenUsage(Base):
    """
    Tracks AI token usage for billing and analytics.
    Records prompt tokens, completion tokens, and total tokens for each translation.
    """
    __tablename__ = "token_usage"
    
    # Use Integer for autoincrement compatibility with SQLite
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Guild and message context
    guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    message_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    
    # Token counts
    prompt_tokens: Mapped[int] = mapped_column(nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(nullable=False, default=0)
    
    # Model and language info
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    source_language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    target_language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Indexes for queries
    __table_args__ = (
        Index('idx_guild_tokens', 'guild_id'),
        Index('idx_created_at_tokens', 'created_at'),
    )
    
    def __repr__(self):
        return f"<TokenUsage(id={self.id}, total={self.total_tokens}, created={self.created_at})>"


# ========================== DATABASE ENGINE ==========================
engine = None
async_session_maker = None


async def init_database():
    """
    Initialize database engine and create all tables.
    Call this once on bot startup.
    """
    global engine, async_session_maker
    
    logger.info(f"Initializing database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    
    # Create async engine
    # echo=False for production, set to True for SQL debugging
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,  # Verify connections before using
    )
    
    # Create session factory
    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Create all tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("✅ Database initialized and tables created")


async def close_database():
    """Close database connections. Call on bot shutdown."""
    global engine
    if engine:
        await engine.dispose()
        logger.info("Database connections closed")


@asynccontextmanager
async def get_session():
    """
    Context manager for database sessions.
    Usage:
        async with get_session() as session:
            result = await session.execute(query)
    """
    if async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    session = async_session_maker()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ========================== CHANNEL DEFAULTS ==========================
async def set_channel_default(guild_id: int, channel_id: int, language: str):
    """Set or update default language for a channel"""
    async with get_session() as session:
        # Try to get existing record
        stmt = select(ChannelDefault).where(
            ChannelDefault.guild_id == guild_id,
            ChannelDefault.channel_id == channel_id
        )
        result = await session.execute(stmt)
        channel_default = result.scalar_one_or_none()
        
        if channel_default:
            # Update existing
            channel_default.language = language
            channel_default.updated_at = datetime.utcnow()
        else:
            # Create new
            channel_default = ChannelDefault(
                guild_id=guild_id,
                channel_id=channel_id,
                language=language
            )
            session.add(channel_default)
        
    logger.debug(f"Set default language for channel {channel_id}: {language}")


async def get_channel_default(guild_id: int, channel_id: int) -> Optional[str]:
    """Get default language for a channel"""
    async with get_session() as session:
        stmt = select(ChannelDefault.language).where(
            ChannelDefault.guild_id == guild_id,
            ChannelDefault.channel_id == channel_id
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def remove_channel_default(guild_id: int, channel_id: int) -> bool:
    """Remove default language for a channel. Returns True if removed, False if not found."""
    async with get_session() as session:
        stmt = delete(ChannelDefault).where(
            ChannelDefault.guild_id == guild_id,
            ChannelDefault.channel_id == channel_id
        )
        result = await session.execute(stmt)
        return result.rowcount > 0


async def get_all_channel_defaults(guild_id: int) -> Dict[int, str]:
    """Get all channel defaults for a guild as {channel_id: language}"""
    async with get_session() as session:
        stmt = select(ChannelDefault).where(ChannelDefault.guild_id == guild_id)
        result = await session.execute(stmt)
        channels = result.scalars().all()
        return {ch.channel_id: ch.language for ch in channels}


# ========================== MESSAGE TRACKING ==========================
async def track_message(
    source_message_id: int,
    source_channel_id: int,
    dest_channel_id: int,
    dest_message_id: int,
    guild_id: int,
    author_id: int,
    source_language: Optional[str] = None,
    dest_language: Optional[str] = None,
    reply_to_message_id: Optional[int] = None
):
    """
    Track a message that was sent/bridged to another channel.
    This allows us to link reactions and replies across channels.
    """
    async with get_session() as session:
        mapping = MessageMapping(
            source_message_id=source_message_id,
            source_channel_id=source_channel_id,
            dest_channel_id=dest_channel_id,
            dest_message_id=dest_message_id,
            guild_id=guild_id,
            author_id=author_id,
            source_language=source_language,
            dest_language=dest_language,
            reply_to_message_id=reply_to_message_id
        )
        session.add(mapping)
    
    logger.debug(f"Tracked message: {source_message_id} -> {dest_message_id} in channel {dest_channel_id}")


async def get_message_destinations(source_message_id: int) -> List[MessageMapping]:
    """
    Get all destinations where a source message was sent.
    Useful for syncing reactions across all copies of a message.
    """
    async with get_session() as session:
        stmt = select(MessageMapping).where(
            MessageMapping.source_message_id == source_message_id
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def find_linked_message(dest_message_id: int) -> Optional[MessageMapping]:
    """
    Find the message mapping for a destination message.
    Useful when a reaction is added to a bridged message - we need to find the source.
    """
    async with get_session() as session:
        stmt = select(MessageMapping).where(
            MessageMapping.dest_message_id == dest_message_id
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def find_reply_destinations(reply_to_message_id: int, dest_channel_id: int) -> Optional[int]:
    """
    When user replies to a bridged message, find what message ID to reply to in the destination channel.
    
    Args:
        reply_to_message_id: The message being replied to (in source channel)
        dest_channel_id: The channel where we want to send the reply
        
    Returns:
        The message ID to reply to in the destination channel, or None if not found
    """
    async with get_session() as session:
        # Find where the original message was sent to this destination channel
        stmt = select(MessageMapping.dest_message_id).where(
            MessageMapping.source_message_id == reply_to_message_id,
            MessageMapping.dest_channel_id == dest_channel_id
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


# ========================== REACTION SYNCING ==========================
async def track_reaction_sync(
    source_message_id: int,
    source_channel_id: int,
    emoji: str,
    user_id: int,
    synced_to_message_id: int,
    synced_to_channel_id: int,
    guild_id: int
):
    """
    Record that a reaction was synced from one message to another.
    
    Args:
        source_message_id: The original message that got the reaction
        source_channel_id: The channel of the original message
        emoji: The emoji (as string - unicode or custom emoji ID)
        user_id: User who added the reaction
        synced_to_message_id: The message we synced the reaction to
        synced_to_channel_id: The channel we synced to
        guild_id: The guild ID
    """
    async with get_session() as session:
        sync = ReactionSync(
            source_message_id=source_message_id,
            source_channel_id=source_channel_id,
            emoji=emoji,
            user_id=user_id,
            synced_to_message_id=synced_to_message_id,
            synced_to_channel_id=synced_to_channel_id,
            guild_id=guild_id
        )
        session.add(sync)
    
    logger.debug(f"Tracked reaction sync: {emoji} from {source_message_id} to {synced_to_message_id}")


async def get_all_linked_messages(message_id: int, channel_id: int) -> List[Dict]:
    """
    Get all messages linked to a given message (both as source and as destination).
    This includes the original message and all its translations.
    
    Args:
        message_id: The message ID to find links for
        channel_id: The channel ID of the message
        
    Returns:
        List of dicts with 'message_id' and 'channel_id' keys for all linked messages
    """
    async with get_session() as session:
        linked_messages = []
        
        # Case 1: This message is a source - find all destinations
        stmt = select(MessageMapping).where(
            MessageMapping.source_message_id == message_id
        )
        result = await session.execute(stmt)
        mappings = result.scalars().all()
        
        if mappings:
            # This was a source message, add all destinations
            for mapping in mappings:
                linked_messages.append({
                    'message_id': mapping.dest_message_id,
                    'channel_id': mapping.dest_channel_id,
                    'guild_id': mapping.guild_id
                })
            return linked_messages
        
        # Case 2: This message is a destination - find the source and all other destinations
        stmt = select(MessageMapping).where(
            MessageMapping.dest_message_id == message_id,
            MessageMapping.dest_channel_id == channel_id
        )
        result = await session.execute(stmt)
        mapping = result.scalar_one_or_none()
        
        if mapping:
            # Found the source, now get all destinations including the source
            source_msg_id = mapping.source_message_id
            
            # Get all destinations of this source
            stmt = select(MessageMapping).where(
                MessageMapping.source_message_id == source_msg_id
            )
            result = await session.execute(stmt)
            all_mappings = result.scalars().all()
            
            # Add the source message itself
            if all_mappings:
                linked_messages.append({
                    'message_id': source_msg_id,
                    'channel_id': all_mappings[0].source_channel_id,
                    'guild_id': all_mappings[0].guild_id
                })
            
            # Add all destinations except the one we started with
            for m in all_mappings:
                if m.dest_message_id != message_id or m.dest_channel_id != channel_id:
                    linked_messages.append({
                        'message_id': m.dest_message_id,
                        'channel_id': m.dest_channel_id,
                        'guild_id': m.guild_id
                    })
        
        return linked_messages


# ========================== STATISTICS ==========================
async def get_statistics() -> Dict:
    """Get statistics about bot usage"""
    async with get_session() as session:
        stats = {}
        
        # Count total tracked messages
        stmt = select(func.count()).select_from(MessageMapping)
        result = await session.execute(stmt)
        stats['total_messages_tracked'] = result.scalar()
        
        # Count unique source messages (actual messages sent by users)
        stmt = select(func.count(func.distinct(MessageMapping.source_message_id)))
        result = await session.execute(stmt)
        stats['unique_messages'] = result.scalar()
        
        # Count guilds with configured channels
        stmt = select(func.count(func.distinct(ChannelDefault.guild_id)))
        result = await session.execute(stmt)
        stats['guilds_configured'] = result.scalar()
        
        # Count total configured channels
        stmt = select(func.count()).select_from(ChannelDefault)
        result = await session.execute(stmt)
        stats['channels_configured'] = result.scalar()
        
        # Most popular languages
        stmt = select(
            ChannelDefault.language,
            func.count(ChannelDefault.channel_id).label('count')
        ).group_by(ChannelDefault.language).order_by(func.count(ChannelDefault.channel_id).desc()).limit(5)
        result = await session.execute(stmt)
        stats['top_languages'] = [(row.language, row.count) for row in result]
        
        # Count messages with translations
        stmt = select(func.count()).select_from(MessageMapping).where(
            MessageMapping.dest_language.isnot(None)
        )
        result = await session.execute(stmt)
        stats['translated_messages'] = result.scalar()
        
        # Count reply tracking
        stmt = select(func.count()).select_from(MessageMapping).where(
            MessageMapping.reply_to_message_id.isnot(None)
        )
        result = await session.execute(stmt)
        stats['reply_messages'] = result.scalar()
        
        # Count total reaction syncs
        stmt = select(func.count()).select_from(ReactionSync)
        result = await session.execute(stmt)
        stats['total_reactions_synced'] = result.scalar()
        
        # Count unique reactions (distinct emoji types synced)
        stmt = select(func.count(func.distinct(ReactionSync.emoji)))
        result = await session.execute(stmt)
        stats['unique_emojis_synced'] = result.scalar()
        
        # Count unique users who had reactions synced
        stmt = select(func.count(func.distinct(ReactionSync.user_id)))
        result = await session.execute(stmt)
        stats['users_with_synced_reactions'] = result.scalar()
        
        return stats


# ========================== TOKEN TRACKING ==========================
async def track_token_usage(
    guild_id: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    model: Optional[str] = None,
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    message_id: Optional[int] = None
):
    """
    Track AI token usage for a translation.
    
    Args:
        guild_id: The guild where the translation occurred
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
        model: The AI model used (e.g., "grok-4-1-fast-non-reasoning")
        source_language: Source language of the translation
        target_language: Target language of the translation
        message_id: Discord message ID (optional)
    """
    async with get_session() as session:
        usage = TokenUsage(
            guild_id=guild_id,
            message_id=message_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            source_language=source_language,
            target_language=target_language
        )
        session.add(usage)
    
    logger.debug(f"Tracked token usage: {total_tokens} tokens (prompt: {prompt_tokens}, completion: {completion_tokens})")


async def get_token_usage_for_period(
    guild_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict:
    """
    Get token usage statistics for a specific period.
    
    Args:
        guild_id: Filter by guild (None for all guilds)
        start_date: Start of period (None for beginning of time)
        end_date: End of period (None for now)
        
    Returns:
        Dict with token usage statistics
    """
    async with get_session() as session:
        # Build the query
        stmt = select(
            func.sum(TokenUsage.prompt_tokens).label('total_prompt_tokens'),
            func.sum(TokenUsage.completion_tokens).label('total_completion_tokens'),
            func.sum(TokenUsage.total_tokens).label('total_tokens'),
            func.count(TokenUsage.id).label('translation_count')
        )
        
        # Add filters
        if guild_id is not None:
            stmt = stmt.where(TokenUsage.guild_id == guild_id)
        if start_date is not None:
            stmt = stmt.where(TokenUsage.created_at >= start_date)
        if end_date is not None:
            stmt = stmt.where(TokenUsage.created_at <= end_date)
        
        result = await session.execute(stmt)
        row = result.first()
        
        return {
            'total_prompt_tokens': row.total_prompt_tokens or 0,
            'total_completion_tokens': row.total_completion_tokens or 0,
            'total_tokens': row.total_tokens or 0,
            'translation_count': row.translation_count or 0
        }


async def get_current_month_token_usage(guild_id: Optional[int] = None) -> Dict:
    """
    Get token usage for the current billing period (current month).
    
    Args:
        guild_id: Filter by guild (None for all guilds)
        
    Returns:
        Dict with current month's token usage statistics
    """
    from datetime import datetime
    
    now = datetime.utcnow()
    # Start of current month
    start_of_month = datetime(now.year, now.month, 1)
    
    return await get_token_usage_for_period(
        guild_id=guild_id,
        start_date=start_of_month,
        end_date=None
    )


async def cleanup_old_messages(days: int = 30):
    """
    Clean up old message mappings to prevent database bloat.
    Discord API only allows accessing messages from last ~14 days anyway.
    
    Args:
        days: Delete message mappings older than this many days
    """
    from datetime import timedelta
    
    async with get_session() as session:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        stmt = delete(MessageMapping).where(MessageMapping.created_at < cutoff_date)
        result = await session.execute(stmt)
        deleted = result.rowcount
        
    if deleted > 0:
        logger.info(f"Cleaned up {deleted} message mappings older than {days} days")
    
    return deleted
