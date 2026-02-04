"""
Database configuration and connection management.
Handles database connections, sessions, and connection pooling.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool

from src.common.config import get_settings
from src.common.models import Base
from loguru import logger


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self):
        self.settings = get_settings()
        self.async_engine = None
        self.sync_engine = None
        self.async_session_factory = None
        self.sync_session_factory = None
    
    def initialize(self):
        """Initialize database engines and session factories"""
        logger.info("Initializing database connections")
        
        # Async engine for main application
        self.async_engine = create_async_engine(
            self.settings.database.url,
            poolclass=QueuePool,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            pool_pre_ping=self.settings.database.pool_pre_ping,
            echo=self.settings.database.echo,
        )
        
        # Sync engine for migrations and scripts
        sync_url = self.settings.database.url.replace('+asyncpg', '')
        self.sync_engine = create_engine(
            sync_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        
        # Session factories
        self.async_session_factory = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        self.sync_session_factory = sessionmaker(
            self.sync_engine,
            class_=Session,
            expire_on_commit=False,
        )
        
        logger.info("Database connections initialized")
    
    async def create_tables(self):
        """Create all database tables"""
        logger.info("Creating database tables")
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")
    
    async def drop_tables(self):
        """Drop all database tables"""
        logger.warning("Dropping all database tables")
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("Database tables dropped")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    def get_sync_session(self) -> Session:
        """Get sync database session (for scripts)"""
        return self.sync_session_factory()
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self):
        """Close all database connections"""
        logger.info("Closing database connections")
        if self.async_engine:
            await self.async_engine.dispose()
        if self.sync_engine:
            self.sync_engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.initialize()
    return _db_manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection for FastAPI"""
    db_manager = get_db_manager()
    async with db_manager.get_session() as session:
        yield session
