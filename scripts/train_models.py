#!/usr/bin/env python3
"""Train ML models"""
from loguru import logger
from src.prediction.hybrid_model import HybridPredictor


def main():
    logger.info("Training ML models")
    predictor = HybridPredictor()
    # Training logic
    logger.info("Training completed")


if __name__ == "__main__":
    main()
