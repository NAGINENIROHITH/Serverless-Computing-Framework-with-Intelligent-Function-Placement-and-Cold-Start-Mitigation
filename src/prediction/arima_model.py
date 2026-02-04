"""
ARIMA-based prediction model for serverless function invocations.
Implements statistical time series forecasting for trend and seasonality.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from typing import Tuple, Optional, List
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ARIMAPredictor:
    """
    ARIMA model for time series prediction.
    Automatically tunes parameters using AIC/BIC criteria.
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 2),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto_tune: bool = True,
    ):
        """
        Initialize ARIMA predictor.
        
        Args:
            order: (p, d, q) order for ARIMA
            seasonal_order: (P, D, Q, s) for seasonal ARIMA
            auto_tune: Whether to automatically tune parameters
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_tune = auto_tune
        self.model = None
        self.fitted_model = None
        self.is_seasonal = seasonal_order is not None
    
    def check_stationarity(self, timeseries: np.ndarray) -> dict:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Returns:
            Dictionary with test statistics
        """
        result = adfuller(timeseries, autolag='AIC')
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'critical_values': result[4],
        }
    
    def find_optimal_order(
        self,
        timeseries: np.ndarray,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
    ) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order using grid search and AIC.
        
        Args:
            timeseries: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
        
        Returns:
            Optimal (p, d, q) order
        """
        logger.info("Finding optimal ARIMA order...")
        
        best_aic = np.inf
        best_order = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(timeseries, order=(p, d, q))
                        fitted = model.fit()
                        aic = fitted.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            logger.debug(f"New best order: {best_order}, AIC: {aic:.2f}")
                    
                    except Exception:
                        continue
        
        logger.info(f"Optimal ARIMA order: {best_order}, AIC: {best_aic:.2f}")
        return best_order
    
    def fit(self, timeseries: np.ndarray):
        """
        Fit ARIMA model to time series data.
        
        Args:
            timeseries: Historical invocation data
        """
        logger.info("Fitting ARIMA model...")
        
        # Check stationarity
        stationarity = self.check_stationarity(timeseries)
        logger.info(f"Stationarity test - p-value: {stationarity['p_value']:.4f}")
        
        # Auto-tune order if enabled
        if self.auto_tune:
            self.order = self.find_optimal_order(timeseries)
        
        # Fit model
        try:
            if self.is_seasonal:
                self.model = SARIMAX(
                    timeseries,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                self.model = ARIMA(timeseries, order=self.order)
            
            self.fitted_model = self.model.fit(disp=False)
            
            logger.info(f"ARIMA model fitted successfully")
            logger.info(f"AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
        
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
    
    def predict(self, steps_ahead: int = 1) -> np.ndarray:
        """
        Make predictions for future timesteps.
        
        Args:
            steps_ahead: Number of future steps to predict
        
        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Forecast
        forecast = self.fitted_model.forecast(steps=steps_ahead)
        
        # Ensure non-negative predictions (invocations can't be negative)
        forecast = np.maximum(forecast, 0)
        
        return forecast.values if hasattr(forecast, 'values') else forecast
    
    def predict_with_confidence(
        self,
        steps_ahead: int = 1,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            steps_ahead: Number of future steps to predict
            alpha: Significance level for confidence intervals (default 95%)
        
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Forecast with confidence intervals
        forecast_result = self.fitted_model.get_forecast(steps=steps_ahead)
        
        predictions = forecast_result.predicted_mean.values
        conf_int = forecast_result.conf_int(alpha=alpha).values
        
        lower_bound = conf_int[:, 0]
        upper_bound = conf_int[:, 1]
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        lower_bound = np.maximum(lower_bound, 0)
        upper_bound = np.maximum(upper_bound, 0)
        
        return predictions, lower_bound, upper_bound
    
    def calculate_residuals(self) -> np.ndarray:
        """Get model residuals for diagnostics"""
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.fitted_model.resid
    
    def get_model_summary(self) -> str:
        """Get detailed model summary"""
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return str(self.fitted_model.summary())


class AutoARIMA:
    """
    Automatic ARIMA model selection and fitting.
    Wrapper around pmdarima for easy auto-tuning.
    """
    
    def __init__(
        self,
        seasonal: bool = True,
        m: int = 24,  # Seasonality period (e.g., 24 for hourly data)
        stepwise: bool = True,
        suppress_warnings: bool = True,
    ):
        """
        Initialize Auto-ARIMA.
        
        Args:
            seasonal: Whether to fit seasonal model
            m: Seasonality period
            stepwise: Use stepwise algorithm (faster)
            suppress_warnings: Suppress fit warnings
        """
        self.seasonal = seasonal
        self.m = m
        self.stepwise = stepwise
        self.suppress_warnings = suppress_warnings
        self.model = None
    
    def fit(self, timeseries: np.ndarray):
        """
        Automatically fit best ARIMA model.
        
        Args:
            timeseries: Historical time series data
        """
        try:
            from pmdarima import auto_arima
            
            logger.info("Auto-fitting ARIMA model...")
            
            self.model = auto_arima(
                timeseries,
                seasonal=self.seasonal,
                m=self.m,
                stepwise=self.stepwise,
                suppress_warnings=self.suppress_warnings,
                error_action='ignore',
                trace=False,
                n_jobs=-1,  # Use all CPUs
            )
            
            logger.info(f"Best model: {self.model.order} with AIC: {self.model.aic():.2f}")
        
        except ImportError:
            logger.warning("pmdarima not installed, falling back to manual ARIMA")
            # Fallback to manual ARIMA
            predictor = ARIMAPredictor(auto_tune=True)
            predictor.fit(timeseries)
            self.model = predictor.fitted_model
    
    def predict(self, steps_ahead: int = 1) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = self.model.predict(n_periods=steps_ahead)
        return np.maximum(predictions, 0)  # Ensure non-negative


def analyze_seasonality(
    timeseries: np.ndarray,
    frequency: int = 24,
) -> dict:
    """
    Analyze seasonality in time series.
    
    Args:
        timeseries: Time series data
        frequency: Expected seasonal frequency
    
    Returns:
        Dictionary with seasonality analysis
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(
        timeseries,
        model='additive',
        period=frequency,
        extrapolate_trend='freq',
    )
    
    # Calculate strength of seasonality
    seasonal_strength = 1 - (
        np.var(decomposition.resid) / 
        np.var(decomposition.seasonal + decomposition.resid)
    )
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'seasonal_strength': seasonal_strength,
        'has_seasonality': seasonal_strength > 0.3,
    }
