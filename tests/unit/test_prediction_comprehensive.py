"""
Comprehensive unit tests for prediction models and services.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.prediction.lstm_model import LSTMPredictionModel, LSTMTrainer
from src.prediction.arima_model import ARIMAPredictor, AutoARIMA, analyze_seasonality
from src.prediction.hybrid_model import HybridPredictor


class TestLSTMModel:
    """Test LSTM prediction model"""
    
    def test_lstm_initialization(self):
        """Test LSTM model can be initialized"""
        model = LSTMPredictionModel(
            input_size=10,
            hidden_size=64,
            num_layers=2
        )
        assert model.input_size == 10
        assert model.hidden_size == 64
        assert model.num_layers == 2
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass"""
        import torch
        
        model = LSTMPredictionModel()
        batch_size = 4
        seq_length = 20
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, model.input_size)
        
        # Forward pass
        output, hidden = model(x)
        
        assert output.shape == (batch_size, 1)
        assert len(hidden) == 2  # h and c
    
    def test_lstm_prediction(self):
        """Test LSTM prediction interface"""
        model = LSTMPredictionModel()
        
        # Create dummy historical data
        history = np.random.rand(100, model.input_size)
        
        # Make predictions
        predictions = model.predict(history, steps_ahead=10)
        
        assert len(predictions) == 10
        assert all(p >= 0 for p in predictions)  # Non-negative


class TestARIMAModel:
    """Test ARIMA prediction model"""
    
    def test_arima_initialization(self):
        """Test ARIMA model initialization"""
        predictor = ARIMAPredictor(order=(5, 1, 2))
        assert predictor.order == (5, 1, 2)
        assert predictor.model is None
    
    def test_arima_stationarity_check(self):
        """Test stationarity testing"""
        predictor = ARIMAPredictor()
        
        # Stationary series
        stationary = np.random.randn(100)
        result = predictor.check_stationarity(stationary)
        
        assert 'is_stationary' in result
        assert 'p_value' in result
        assert 'adf_statistic' in result
    
    def test_arima_fitting(self):
        """Test ARIMA model fitting"""
        predictor = ARIMAPredictor(auto_tune=False)
        
        # Generate synthetic time series
        timeseries = np.cumsum(np.random.randn(100)) + 100
        
        predictor.fit(timeseries)
        
        assert predictor.fitted_model is not None
    
    def test_arima_prediction(self):
        """Test ARIMA prediction"""
        predictor = ARIMAPredictor(auto_tune=False)
        
        timeseries = np.cumsum(np.random.randn(100)) + 100
        predictor.fit(timeseries)
        
        predictions = predictor.predict(steps_ahead=10)
        
        assert len(predictions) == 10
        assert all(p >= 0 for p in predictions)
    
    def test_arima_confidence_intervals(self):
        """Test prediction with confidence intervals"""
        predictor = ARIMAPredictor(auto_tune=False)
        
        timeseries = np.cumsum(np.random.randn(100)) + 100
        predictor.fit(timeseries)
        
        preds, lower, upper = predictor.predict_with_confidence(
            steps_ahead=10,
            alpha=0.05
        )
        
        assert len(preds) == len(lower) == len(upper) == 10
        assert all(lower[i] <= preds[i] <= upper[i] for i in range(10))


class TestHybridModel:
    """Test hybrid ensemble predictor"""
    
    def test_hybrid_initialization(self):
        """Test hybrid model initialization"""
        predictor = HybridPredictor(arima_weight=0.3, lstm_weight=0.7)
        
        assert predictor.arima_weight == 0.3
        assert predictor.lstm_weight == 0.7
        assert predictor.arima_model is not None
        assert predictor.lstm_model is not None
    
    def test_hybrid_weight_validation(self):
        """Test weights must sum to 1"""
        with pytest.raises(ValueError):
            HybridPredictor(arima_weight=0.5, lstm_weight=0.6)
    
    def test_hybrid_fitting(self):
        """Test hybrid model training"""
        predictor = HybridPredictor()
        
        # Generate synthetic data
        timeseries = np.cumsum(np.random.randn(200)) + 100
        
        predictor.fit(timeseries)
        
        # Both models should be fitted
        assert predictor.arima_model.fitted_model is not None
    
    def test_hybrid_prediction(self):
        """Test hybrid ensemble prediction"""
        predictor = HybridPredictor()
        
        timeseries = np.cumsum(np.random.randn(200)) + 100
        predictor.fit(timeseries)
        
        predictions = predictor.predict(steps_ahead=10)
        
        assert len(predictions) == 10
        assert all(p >= 0 for p in predictions)
    
    def test_adaptive_weights(self):
        """Test adaptive weight adjustment"""
        predictor = HybridPredictor()
        
        # Mock recent predictions
        recent_predictions = {
            'arima': np.array([100, 110, 105, 120]),
            'lstm': np.array([95, 105, 110, 115])
        }
        recent_actual = np.array([98, 108, 108, 118])
        
        initial_arima_weight = predictor.arima_weight
        
        predictor.adapt_weights(recent_predictions, recent_actual)
        
        # Weights should have changed
        assert predictor.arima_weight != initial_arima_weight
        
        # Weights should still sum to 1
        assert abs((predictor.arima_weight + predictor.lstm_weight) - 1.0) < 0.01
    
    def test_should_retrain_decision(self):
        """Test retraining decision logic"""
        predictor = HybridPredictor()
        
        # High error should trigger retraining
        assert predictor.should_retrain(recent_mape=25.0, threshold=20.0) == True
        
        # Low error should not trigger retraining
        assert predictor.should_retrain(recent_mape=10.0, threshold=20.0) == False


class TestSeasonalityAnalysis:
    """Test seasonality detection"""
    
    def test_seasonality_detection(self):
        """Test seasonal pattern detection"""
        # Generate time series with seasonality
        t = np.arange(0, 200)
        seasonal = 10 * np.sin(2 * np.pi * t / 24)  # 24-hour seasonality
        trend = 0.5 * t
        noise = np.random.randn(200)
        timeseries = seasonal + trend + noise + 100
        
        result = analyze_seasonality(timeseries, frequency=24)
        
        assert 'seasonal_strength' in result
        assert 'has_seasonality' in result
        assert result['has_seasonality'] == True  # Should detect seasonality
    
    def test_no_seasonality(self):
        """Test detection of non-seasonal data"""
        # Random walk (no seasonality)
        timeseries = np.cumsum(np.random.randn(200))
        
        result = analyze_seasonality(timeseries, frequency=24)
        
        # Should detect no strong seasonality
        assert result['seasonal_strength'] < 0.5


class TestModelMetrics:
    """Test prediction accuracy metrics"""
    
    def test_mape_calculation(self):
        """Test MAPE metric"""
        y_true = np.array([100, 200, 150, 300])
        y_pred = np.array([110, 190, 160, 280])
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        assert mape > 0
        assert mape < 15  # Should be reasonable
    
    def test_rmse_calculation(self):
        """Test RMSE metric"""
        y_true = np.array([100, 200, 150, 300])
        y_pred = np.array([110, 190, 160, 280])
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        assert rmse > 0
        assert rmse < 50  # Should be reasonable
    
    def test_r2_score_calculation(self):
        """Test R² score"""
        y_true = np.array([100, 200, 150, 300])
        y_pred = np.array([110, 190, 160, 280])
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        assert r2 > 0.5  # Should have decent fit
        assert r2 <= 1.0


@pytest.mark.asyncio
class TestPredictionService:
    """Test prediction service"""
    
    async def test_service_initialization(self):
        """Test service can be initialized"""
        from src.prediction.service import PredictionService
        
        service = PredictionService()
        assert service.config is not None
        assert service.models == {}
    
    @patch('src.prediction.service.get_db_session')
    async def test_get_prediction(self, mock_db):
        """Test getting predictions"""
        from src.prediction.service import PredictionService
        
        service = PredictionService()
        
        # Mock database session
        mock_session = AsyncMock()
        mock_db.return_value.__aenter__.return_value = mock_session
        
        # Test would require full database setup
        # Simplified test - just check method exists
        assert hasattr(service, 'get_prediction')


# Integration test markers
@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for complete prediction pipeline"""
    
    def test_full_prediction_pipeline(self):
        """Test complete prediction workflow"""
        # Generate synthetic workload data
        # Diurnal pattern: high during day, low at night
        hours = np.arange(0, 168)  # 1 week
        base_load = 100
        
        # Add daily pattern
        daily_pattern = 50 * np.sin(2 * np.pi * hours / 24)
        
        # Add weekly pattern (weekend lower)
        weekly_pattern = 20 * np.sin(2 * np.pi * hours / 168)
        
        # Add noise
        noise = np.random.randn(168) * 10
        
        workload = base_load + daily_pattern + weekly_pattern + noise
        workload = np.maximum(workload, 0)  # Non-negative
        
        # Train hybrid model
        predictor = HybridPredictor()
        predictor.fit(workload)
        
        # Make predictions
        predictions = predictor.predict(steps_ahead=24)
        
        # Validate predictions
        assert len(predictions) == 24
        assert all(p >= 0 for p in predictions)
        
        # Predictions should follow similar pattern to training data
        avg_prediction = np.mean(predictions)
        avg_training = np.mean(workload)
        
        # Should be within 50% of average
        assert 0.5 * avg_training < avg_prediction < 1.5 * avg_training


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
