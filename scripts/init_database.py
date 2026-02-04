#!/usr/bin/env python3
"""Initialize database schema"""
from src.common.database import get_db_manager
from src.common.models import create_tables
from loguru import logger


def main():
    logger.info("Initializing database")
    db_manager = get_db_manager()
    create_tables(db_manager.sync_engine)
    logger.info("Database initialized successfully")


if __name__ == "__main__":
    main()
