"""
Hybrid prediction model combining ARIMA and LSTM.
Leverages strengths of both statistical and deep learning approaches.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger

from src.prediction.lstm_model import LSTMPredictionModel
from src.prediction.arima_model import ARIMAPredictor, analyze_seasonality
from src.common.config import get_settings


class HybridPredictor:
    """
    Ensemble predictor combining ARIMA and LSTM models.
    ARIMA captures linear trends and seasonality.
    LSTM captures complex non-linear patterns.
    """
    
    def __init__(
        self,
        arima_weight: float = 0.3,
        lstm_weight: float = 0.7,
        config: Optional[dict] = None,
    ):
        """
        Initialize hybrid predictor.
        
        Args:
            arima_weight: Weight for ARIMA predictions (0-1)
            lstm_weight: Weight for LSTM predictions (0-1)
            config: Configuration dictionary
        """
        if abs((arima_weight + lstm_weight) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        self.arima_weight = arima_weight
        self.lstm_weight = lstm_weight
        self.config = config or get_settings().prediction
        
        # Initialize models
        self.arima_model = ARIMAPredictor(auto_tune=True)
        self.lstm_model = LSTMPredictionModel(self.config)
        
        # Performance tracking
        self.arima_performance = []
        self.lstm_performance = []
        self.hybrid_performance = []
        
        logger.info(
            f"Hybrid predictor initialized - "
            f"ARIMA: {arima_weight:.2f}, LSTM: {lstm_weight:.2f}"
        )
    
    def fit(
        self,
        timeseries: np.ndarray,
        features: Optional[np.ndarray] = None,
    ):
        """
        Fit both ARIMA and LSTM models.
        
        Args:
            timeseries: Historical invocation counts
            features: Additional features for LSTM (optional)
        """
        logger.info("Fitting hybrid model...")
        
        # Analyze seasonality to help ARIMA
        seasonality = analyze_seasonality(timeseries)
        
        if seasonality['has_seasonality']:
            logger.info("Seasonality detected, using SARIMA")
            self.arima_model.seasonal_order = (1, 1, 1, 24)
        
        # Fit ARIMA on raw time series
        self.arima_model.fit(timeseries)
        
        # Fit LSTM on detrended residuals + features
        if features is None:
            features = self._extract_features(timeseries)
        
        # Get ARIMA residuals as additional feature for LSTM
        arima_residuals = self.arima_model.calculate_residuals()
        
        # Combine residuals with other features
        lstm_features = np.column_stack([
            timeseries[len(timeseries) - len(arima_residuals):],
            arima_residuals,
            features[len(features) - len(arima_residuals):]
        ])
        
        self.lstm_model.fit(lstm_features, timeseries[-len(arima_residuals):])
        
        logger.info("Hybrid model training completed")
    
    def predict(
        self,
        steps_ahead: int = 1,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            steps_ahead: Number of future timesteps to predict
            features: Future features for LSTM (if available)
        
        Returns:
            Array of predictions
        """
        # Get ARIMA predictions
        arima_predictions = self.arima_model.predict(steps_ahead)
        
        # Get LSTM predictions
        if features is None:
            # If no features provided, use historical patterns
            features = self._forecast_features(steps_ahead)
        
        lstm_predictions = self.lstm_model.predict(features, steps_ahead)
        
        # Weighted ensemble
        hybrid_predictions = (
            self.arima_weight * arima_predictions +
            self.lstm_weight * lstm_predictions
        )
        
        # Ensure non-negative
        hybrid_predictions = np.maximum(hybrid_predictions, 0)
        
        logger.debug(
            f"Predictions - ARIMA: {arima_predictions.mean():.2f}, "
            f"LSTM: {lstm_predictions.mean():.2f}, "
            f"Hybrid: {hybrid_predictions.mean():.2f}"
        )
        
        return hybrid_predictions
    
    def predict_with_confidence(
        self,
        steps_ahead: int = 1,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        # Get predictions from both models
        predictions = self.predict(steps_ahead)
        
        # Get ARIMA confidence intervals
        arima_pred, arima_lower, arima_upper = \
            self.arima_model.predict_with_confidence(steps_ahead, alpha)
        
        # For LSTM, use prediction variance as proxy for confidence
        # (can be improved with Monte Carlo dropout or ensemble)
        lstm_pred = self.lstm_model.predict(None, steps_ahead)
        lstm_std = np.std(lstm_pred) * 1.96  # 95% confidence
        
        lstm_lower = lstm_pred - lstm_std
        lstm_upper = lstm_pred + lstm_std
        
        # Combine confidence intervals
        lower_bound = (
            self.arima_weight * arima_lower +
            self.lstm_weight * lstm_lower
        )
        upper_bound = (
            self.arima_weight * arima_upper +
            self.lstm_weight * lstm_upper
        )
        
        return predictions, lower_bound, upper_bound
    
    def adapt_weights(
        self,
        recent_predictions: Dict[str, np.ndarray],
        recent_actual: np.ndarray,
    ):
        """
        Dynamically adapt model weights based on recent performance.
        
        Args:
            recent_predictions: Dict with 'arima' and 'lstm' predictions
            recent_actual: Actual values
        """
        # Calculate MAPE for each model
        arima_mape = self._calculate_mape(
            recent_actual,
            recent_predictions['arima']
        )
        lstm_mape = self._calculate_mape(
            recent_actual,
            recent_predictions['lstm']
        )
        
        # Track performance
        self.arima_performance.append(arima_mape)
        self.lstm_performance.append(lstm_mape)
        
        # Inverse weighting based on error
        if arima_mape + lstm_mape > 0:
            self.arima_weight = lstm_mape / (arima_mape + lstm_mape)
            self.lstm_weight = arima_mape / (arima_mape + lstm_mape)
        
        logger.info(
            f"Weights adapted - ARIMA: {self.arima_weight:.3f} "
            f"(MAPE: {arima_mape:.2f}%), "
            f"LSTM: {self.lstm_weight:.3f} (MAPE: {lstm_mape:.2f}%)"
        )
    
    def should_retrain(
        self,
        recent_mape: float,
        threshold: float = 20.0,
    ) -> bool:
        """
        Determine if model should be retrained based on performance.
        
        Args:
            recent_mape: Recent mean absolute percentage error
            threshold: MAPE threshold for retraining
        
        Returns:
            True if retraining recommended
        """
        if recent_mape > threshold:
            logger.warning(
                f"Model performance degraded (MAPE: {recent_mape:.2f}% > {threshold}%). "
                f"Retraining recommended."
            )
            return True
        
        return False
    
    def _extract_features(self, timeseries: np.ndarray) -> np.ndarray:
        """Extract time-based features from series"""
        features = []
        
        for i in range(len(timeseries)):
            # Create synthetic timestamp
            ts = datetime.now() - timedelta(minutes=len(timeseries) - i)
            
            features.append([
                ts.hour,
                ts.weekday(),
                int(ts.weekday() >= 5),  # is_weekend
                ts.day,
                ts.month,
            ])
        
        return np.array(features)
    
    def _forecast_features(self, steps_ahead: int) -> np.ndarray:
        """Forecast future features"""
        features = []
        
        for i in range(steps_ahead):
            ts = datetime.now() + timedelta(minutes=i)
            
            features.append([
                ts.hour,
                ts.weekday(),
                int(ts.weekday() >= 5),
                ts.day,
                ts.month,
            ])
        
        return np.array(features)
    
    @staticmethod
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    def get_model_info(self) -> dict:
        """Get information about both models"""
        return {
            'arima_order': self.arima_model.order,
            'arima_aic': self.arima_model.fitted_model.aic if self.arima_model.fitted_model else None,
            'lstm_architecture': {
                'input_size': self.lstm_model.model.input_size,
                'hidden_size': self.lstm_model.model.hidden_size,
                'num_layers': self.lstm_model.model.num_layers,
            },
            'weights': {
                'arima': self.arima_weight,
                'lstm': self.lstm_weight,
            },
            'recent_performance': {
                'arima_mape': np.mean(self.arima_performance[-10:]) if self.arima_performance else None,
                'lstm_mape': np.mean(self.lstm_performance[-10:]) if self.lstm_performance else None,
            }
        }
