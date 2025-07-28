"""
Time Series Forecasting Suite - Contribution #25
Advanced time series analysis and forecasting capabilities.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


@dataclass
class ForecastResult:
    """Results from time series forecasting."""
    predictions: np.ndarray
    confidence_intervals: np.ndarray
    model: BaseEstimator
    metrics: Dict[str, float]


class TimeSeriesFeatureExtractor:
    """Extract features from time series data."""
    
    def __init__(self, lag_features: int = 5, window_features: bool = True):
        self.lag_features = lag_features
        self.window_features = window_features
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, ts_data: pd.Series) -> pd.DataFrame:
        """Extract time series features."""
        features = pd.DataFrame(index=ts_data.index)
        
        # Lag features
        for lag in range(1, self.lag_features + 1):
            features[f'lag_{lag}'] = ts_data.shift(lag)
        
        # Rolling window features
        if self.window_features:
            for window in [3, 7, 14, 30]:
                features[f'rolling_mean_{window}'] = ts_data.rolling(window).mean()
                features[f'rolling_std_{window}'] = ts_data.rolling(window).std()
                features[f'rolling_min_{window}'] = ts_data.rolling(window).min()
                features[f'rolling_max_{window}'] = ts_data.rolling(window).max()
        
        # Time-based features
        if isinstance(ts_data.index, pd.DatetimeIndex):
            features['hour'] = ts_data.index.hour
            features['day_of_week'] = ts_data.index.dayofweek
            features['month'] = ts_data.index.month
            features['quarter'] = ts_data.index.quarter
            features['year'] = ts_data.index.year
        
        # Difference features
        features['diff_1'] = ts_data.diff(1)
        features['diff_7'] = ts_data.diff(7)
        
        return features.dropna()


class ARIMAForecaster:
    """ARIMA-based time series forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, ts_data: pd.Series):
        """Fit ARIMA model."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(ts_data, order=self.order)
            self.fitted_model = self.model.fit()
            self.logger.info(f"ARIMA{self.order} model fitted successfully")
        except ImportError:
            self.logger.error("statsmodels not available for ARIMA")
            raise
    
    def forecast(self, steps: int = 10) -> ForecastResult:
        """Generate forecasts."""
        if not self.fitted_model:
            raise ValueError("Model not fitted")
        
        forecast = self.fitted_model.forecast(steps=steps)
        conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
        
        metrics = {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic
        }
        
        return ForecastResult(
            predictions=forecast.values,
            confidence_intervals=conf_int.values,
            model=self.fitted_model,
            metrics=metrics
        )


class MLForecaster:
    """Machine learning-based time series forecasting."""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.logger = logging.getLogger(__name__)
    
    def _create_model(self):
        """Create ML model."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, ts_data: pd.Series):
        """Fit ML model."""
        # Extract features
        features = self.feature_extractor.extract_features(ts_data)
        
        # Create target (next value prediction)
        target = ts_data.shift(-1).dropna()
        
        # Align features and target
        common_index = features.index.intersection(target.index)
        X = features.loc[common_index]
        y = target.loc[common_index]
        
        # Fit model
        self.model = self._create_model()
        self.model.fit(X, y)
        
        self.logger.info(f"{self.model_type} forecaster fitted successfully")
    
    def forecast(self, ts_data: pd.Series, steps: int = 10) -> ForecastResult:
        """Generate forecasts."""
        if not self.model:
            raise ValueError("Model not fitted")
        
        predictions = []
        current_data = ts_data.copy()
        
        for step in range(steps):
            # Extract features for current data
            features = self.feature_extractor.extract_features(current_data)
            
            # Get last available feature vector
            if len(features) > 0:
                last_features = features.iloc[-1:].fillna(0)
                pred = self.model.predict(last_features)[0]
                predictions.append(pred)
                
                # Update current data with prediction
                new_index = current_data.index[-1] + pd.Timedelta(days=1)
                current_data.loc[new_index] = pred
            else:
                predictions.append(current_data.iloc[-1])
        
        # Simple confidence intervals (placeholder)
        predictions_array = np.array(predictions)
        std_dev = np.std(predictions_array)
        conf_int = np.column_stack([
            predictions_array - 1.96 * std_dev,
            predictions_array + 1.96 * std_dev
        ])
        
        return ForecastResult(
            predictions=predictions_array,
            confidence_intervals=conf_int,
            model=self.model,
            metrics={}
        )


class TimeSeriesValidator:
    """Validate time series forecasting models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def time_series_split_validation(
        self, 
        ts_data: pd.Series, 
        forecaster, 
        n_splits: int = 5,
        test_size: int = 30
    ) -> Dict[str, List[float]]:
        """Perform time series cross-validation."""
        
        data_length = len(ts_data)
        metrics = {'mae': [], 'mse': [], 'rmse': [], 'mape': []}
        
        for i in range(n_splits):
            # Calculate split points
            test_end = data_length - i * test_size
            test_start = test_end - test_size
            train_end = test_start
            
            if train_end < test_size:
                break
            
            # Split data
            train_data = ts_data.iloc[:train_end]
            test_data = ts_data.iloc[test_start:test_end]
            
            # Fit and forecast
            forecaster.fit(train_data)
            
            if hasattr(forecaster, 'forecast'):
                result = forecaster.forecast(len(test_data))
                predictions = result.predictions
            else:
                predictions = forecaster.predict(test_data.index)
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, predictions)
            mse = mean_squared_error(test_data, predictions)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
            
            metrics['mae'].append(mae)
            metrics['mse'].append(mse)
            metrics['rmse'].append(rmse)
            metrics['mape'].append(mape)
        
        return metrics


class SeasonalityDetector:
    """Detect seasonality in time series."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_seasonality(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Detect seasonal patterns."""
        seasonality_info = {
            'has_trend': False,
            'has_seasonal': False,
            'seasonal_periods': [],
            'strength_of_trend': 0.0,
            'strength_of_seasonality': 0.0
        }
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                ts_data.dropna(), 
                model='additive', 
                period=min(len(ts_data) // 2, 365)
            )
            
            # Analyze trend
            trend_variance = np.var(decomposition.trend.dropna())
            data_variance = np.var(ts_data.dropna())
            seasonality_info['strength_of_trend'] = trend_variance / data_variance
            seasonality_info['has_trend'] = seasonality_info['strength_of_trend'] > 0.1
            
            # Analyze seasonality
            seasonal_variance = np.var(decomposition.seasonal.dropna())
            seasonality_info['strength_of_seasonality'] = seasonal_variance / data_variance
            seasonality_info['has_seasonal'] = seasonality_info['strength_of_seasonality'] > 0.1
            
        except ImportError:
            self.logger.warning("statsmodels not available for seasonality detection")
        except Exception as e:
            self.logger.error(f"Error in seasonality detection: {e}")
        
        return seasonality_info


class TimeSeriesForecastingSuite:
    """Complete time series forecasting suite."""
    
    def __init__(self):
        self.forecasters = {}
        self.validator = TimeSeriesValidator()
        self.seasonality_detector = SeasonalityDetector()
        self.logger = logging.getLogger(__name__)
    
    def add_forecaster(self, name: str, forecaster):
        """Add a forecaster to the suite."""
        self.forecasters[name] = forecaster
    
    def auto_forecast(self, ts_data: pd.Series, forecast_horizon: int = 30) -> Dict[str, ForecastResult]:
        """Automatically select and apply best forecasting method."""
        
        # Detect seasonality
        seasonality_info = self.seasonality_detector.detect_seasonality(ts_data)
        self.logger.info(f"Seasonality analysis: {seasonality_info}")
        
        # Initialize forecasters
        forecasters = {
            'ml_rf': MLForecaster('random_forest'),
            'ml_linear': MLForecaster('linear')
        }
        
        # Add ARIMA if statsmodels available
        try:
            forecasters['arima'] = ARIMAForecaster()
        except ImportError:
            pass
        
        results = {}
        
        # Train and evaluate each forecaster
        for name, forecaster in forecasters.items():
            try:
                forecaster.fit(ts_data)
                
                if name == 'arima':
                    result = forecaster.forecast(forecast_horizon)
                else:
                    result = forecaster.forecast(ts_data, forecast_horizon)
                
                results[name] = result
                self.logger.info(f"Forecaster {name} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Error with forecaster {name}: {e}")
        
        return results
    
    def compare_forecasters(self, ts_data: pd.Series) -> pd.DataFrame:
        """Compare different forecasting methods."""
        
        forecasters = {
            'ml_rf': MLForecaster('random_forest'),
            'ml_linear': MLForecaster('linear')
        }
        
        comparison_results = []
        
        for name, forecaster in forecasters.items():
            try:
                metrics = self.validator.time_series_split_validation(ts_data, forecaster)
                
                comparison_results.append({
                    'method': name,
                    'mean_mae': np.mean(metrics['mae']),
                    'mean_mse': np.mean(metrics['mse']),
                    'mean_rmse': np.mean(metrics['rmse']),
                    'mean_mape': np.mean(metrics['mape']),
                    'std_mae': np.std(metrics['mae'])
                })
                
            except Exception as e:
                self.logger.error(f"Error comparing {name}: {e}")
        
        return pd.DataFrame(comparison_results).sort_values('mean_mae')


if __name__ == "__main__":
    # Example usage
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    ts_data = pd.Series(
        np.sin(np.arange(365) * 2 * np.pi / 30) + 
        np.random.normal(0, 0.1, 365) + 
        np.arange(365) * 0.01,
        index=dates
    )
    
    # Initialize forecasting suite
    suite = TimeSeriesForecastingSuite()
    
    # Auto forecast
    results = suite.auto_forecast(ts_data, forecast_horizon=30)
    print(f"Generated forecasts with {len(results)} methods")
    
    # Compare methods
    comparison = suite.compare_forecasters(ts_data)
    print("Method comparison:")
    print(comparison)
    
    print("Time Series Forecasting Suite implemented successfully! 🚀")
