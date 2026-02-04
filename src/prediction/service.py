"""
Prediction service that orchestrates ARIMA, LSTM, and Hybrid models.
Handles model training, prediction, and performance tracking.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.prediction.hybrid_model import HybridPredictor
from src.prediction.lstm_model import LSTMPredictionModel
from src.prediction.arima_model import ARIMAPredictor
from src.common.database import get_db_session
from src.common.models import (
    Function, Invocation, InvocationPrediction, MLModel
)
from src.common.config import get_settings


class PredictionService:
    """
    High-level service for managing predictions across all functions.
    Handles model lifecycle, predictions, and performance tracking.
    """
    
    def __init__(self):
        self.config = get_settings().prediction
        self.models: Dict[int, HybridPredictor] = {}  # function_id -> model
        self.running = False
        
    async def start(self):
        """Start prediction service"""
        self.running = True
        logger.info("Starting prediction service")
        
        # Load existing models
        await self._load_models()
        
        # Start prediction loop
        asyncio.create_task(self._prediction_loop())
        asyncio.create_task(self._retraining_loop())
        
    async def stop(self):
        """Stop prediction service"""
        self.running = False
        logger.info("Stopping prediction service")
    
    async def _prediction_loop(self):
        """Main prediction loop - runs every update_interval"""
        while self.running:
            try:
                await self._generate_predictions()
                await asyncio.sleep(self.config.update_interval)
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(60)
    
    async def _retraining_loop(self):
        """Retraining loop - checks model performance and retrains if needed"""
        while self.running:
            try:
                await self._check_and_retrain()
                await asyncio.sleep(self.config.retrain_interval)
            except Exception as e:
                logger.error(f"Error in retraining loop: {e}")
                await asyncio.sleep(300)
    
    async def _generate_predictions(self):
        """Generate predictions for all active functions"""
        async with get_db_session() as session:
            # Get all active functions
            result = await session.execute(
                select(Function).where(Function.status == "active")
            )
            functions = result.scalars().all()
            
            for function in functions:
                try:
                    await self._predict_for_function(session, function)
                except Exception as e:
                    logger.error(f"Error predicting for {function.name}: {e}")
    
    async def _predict_for_function(
        self,
        session: AsyncSession,
        function: Function
    ):
        """Generate predictions for a single function"""
        
        # Get or create model
        if function.id not in self.models:
            await self._initialize_model(session, function)
        
        model = self.models[function.id]
        
        # Get historical data
        history = await self._get_historical_data(session, function.id)
        
        if len(history) < self.config.min_data_points:
            logger.debug(f"Insufficient data for {function.name}")
            return
        
        # Generate predictions
        steps_ahead = self.config.prediction_horizon // self.config.update_interval
        predictions = model.predict(steps_ahead)
        
        # Store predictions
        await self._store_predictions(
            session,
            function.id,
            predictions
        )
        
        logger.debug(
            f"Generated {steps_ahead} predictions for {function.name}, "
            f"mean: {predictions.mean():.2f}"
        )
    
    async def _initialize_model(
        self,
        session: AsyncSession,
        function: Function
    ):
        """Initialize and train model for a function"""
        logger.info(f"Initializing model for {function.name}")
        
        # Create hybrid model
        model = HybridPredictor(
            arima_weight=self.config.arima_weight,
            lstm_weight=self.config.lstm_weight,
        )
        
        # Get training data
        history = await self._get_historical_data(
            session,
            function.id,
            window=self.config.history_window * 2  # Larger window for training
        )
        
        if len(history) >= self.config.min_data_points:
            # Train model
            timeseries = np.array(history)
            model.fit(timeseries)
            
            # Store model
            self.models[function.id] = model
            
            # Save model metadata
            await self._save_model_metadata(session, function.id, model)
        else:
            logger.warning(f"Insufficient data to train model for {function.name}")
    
    async def _get_historical_data(
        self,
        session: AsyncSession,
        function_id: int,
        window: int = None
    ) -> List[float]:
        """Get historical invocation counts"""
        
        if window is None:
            window = self.config.history_window
        
        # Calculate time window
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=window)
        
        # Query invocations grouped by time intervals
        result = await session.execute(
            select(Invocation).where(
                and_(
                    Invocation.function_id == function_id,
                    Invocation.timestamp >= start_time,
                    Invocation.timestamp <= end_time
                )
            ).order_by(Invocation.timestamp)
        )
        invocations = result.scalars().all()
        
        # Aggregate into time series (per minute)
        time_series = []
        interval = 60  # 1 minute intervals
        
        current_time = start_time
        count = 0
        
        for inv in invocations:
            if inv.timestamp >= current_time + timedelta(seconds=interval):
                time_series.append(count)
                count = 0
                current_time += timedelta(seconds=interval)
            count += 1
        
        time_series.append(count)
        
        return time_series
    
    async def _store_predictions(
        self,
        session: AsyncSession,
        function_id: int,
        predictions: np.ndarray
    ):
        """Store predictions in database"""
        
        prediction_time = datetime.utcnow()
        
        for i, pred in enumerate(predictions):
            window_start = prediction_time + timedelta(
                seconds=i * self.config.update_interval
            )
            window_end = window_start + timedelta(
                seconds=self.config.update_interval
            )
            
            prediction = InvocationPrediction(
                function_id=function_id,
                predicted_at=prediction_time,
                prediction_window_start=window_start,
                prediction_window_end=window_end,
                predicted_invocations=float(pred),
                model_type=self.config.model_type,
                model_version="1.0.0",
            )
            
            session.add(prediction)
        
        await session.commit()
    
    async def _check_and_retrain(self):
        """Check model performance and retrain if necessary"""
        async with get_db_session() as session:
            for function_id, model in self.models.items():
                try:
                    # Calculate recent MAPE
                    mape = await self._calculate_recent_mape(session, function_id)
                    
                    if mape is not None and mape > self.config.retrain_threshold * 100:
                        logger.warning(
                            f"Model performance degraded for function {function_id}, "
                            f"MAPE: {mape:.2f}%. Retraining..."
                        )
                        
                        # Get function
                        result = await session.execute(
                            select(Function).where(Function.id == function_id)
                        )
                        function = result.scalar_one()
                        
                        # Retrain
                        await self._initialize_model(session, function)
                
                except Exception as e:
                    logger.error(f"Error checking/retraining model for function {function_id}: {e}")
    
    async def _calculate_recent_mape(
        self,
        session: AsyncSession,
        function_id: int,
        hours: int = 24
    ) -> Optional[float]:
        """Calculate MAPE for recent predictions"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Get predictions with actuals
        result = await session.execute(
            select(InvocationPrediction).where(
                and_(
                    InvocationPrediction.function_id == function_id,
                    InvocationPrediction.predicted_at >= cutoff_time,
                    InvocationPrediction.actual_invocations.isnot(None)
                )
            )
        )
        predictions = result.scalars().all()
        
        if not predictions:
            return None
        
        # Calculate MAPE
        errors = []
        for pred in predictions:
            if pred.actual_invocations > 0:
                error = abs(
                    pred.predicted_invocations - pred.actual_invocations
                ) / pred.actual_invocations
                errors.append(error)
        
        if not errors:
            return None
        
        mape = np.mean(errors) * 100
        return mape
    
    async def _save_model_metadata(
        self,
        session: AsyncSession,
        function_id: int,
        model: HybridPredictor
    ):
        """Save model metadata to database"""
        
        model_info = model.get_model_info()
        
        ml_model = MLModel(
            name=f"hybrid_predictor_func_{function_id}",
            version="1.0.0",
            model_type="hybrid",
            model_path=f"/data/models/function_{function_id}/",
            config=model_info,
            trained_at=datetime.utcnow(),
            is_active=True,
            is_global_model=False,
        )
        
        session.add(ml_model)
        await session.commit()
    
    async def _load_models(self):
        """Load existing trained models from disk/database"""
        async with get_db_session() as session:
            result = await session.execute(
                select(MLModel).where(
                    and_(
                        MLModel.model_type == "hybrid",
                        MLModel.is_active == True
                    )
                )
            )
            models = result.scalars().all()
            
            logger.info(f"Loading {len(models)} existing models")
            
            # TODO: Load actual model weights from disk
            # For now, models will be initialized on first prediction
    
    async def get_prediction(
        self,
        function_id: int,
        horizon_seconds: int = 300
    ) -> Optional[Dict]:
        """Get current prediction for a function"""
        
        async with get_db_session() as session:
            now = datetime.utcnow()
            future = now + timedelta(seconds=horizon_seconds)
            
            result = await session.execute(
                select(InvocationPrediction).where(
                    and_(
                        InvocationPrediction.function_id == function_id,
                        InvocationPrediction.prediction_window_start >= now,
                        InvocationPrediction.prediction_window_end <= future
                    )
                ).order_by(InvocationPrediction.prediction_window_start)
            )
            predictions = result.scalars().all()
            
            if not predictions:
                return None
            
            return {
                'function_id': function_id,
                'predictions': [
                    {
                        'window_start': p.prediction_window_start.isoformat(),
                        'window_end': p.prediction_window_end.isoformat(),
                        'predicted_invocations': p.predicted_invocations,
                    }
                    for p in predictions
                ],
                'total_predicted': sum(p.predicted_invocations for p in predictions),
            }
