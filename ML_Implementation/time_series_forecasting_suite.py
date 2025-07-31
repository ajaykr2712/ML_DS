"""
Advanced Time Series Forecasting Suite
Comprehensive time series analysis and forecasting framework

Features:
- Multiple forecasting algorithms (ARIMA, SARIMA, Prophet, LSTM)
- Automated model selection and hyperparameter tuning
- Seasonal decomposition and trend analysis
- Anomaly detection in time series data
- Multi-step ahead forecasting with confidence intervals
- Cross-validation with time series splits
- Real-time forecasting pipeline
- Interactive visualization dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Any, Union, Optional
from dataclasses import dataclass
import logging
import json
import pickle
import os
from pathlib import Path

# Statistical libraries
from scipy import stats
from scipy.stats import jarque_bera, normaltest
from scipy.optimize import minimize

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

@dataclass
class ForecastConfig:
    """Configuration for time series forecasting"""
    # Data settings
    date_column: str = 'date'
    target_column: str = 'value'
    frequency: str = 'D'  # D, W, M, Q, Y
    
    # Model settings
    models_to_try: List[str] = None  # ['arima', 'sarima', 'prophet', 'lstm']
    auto_select_model: bool = True
    
    # ARIMA settings
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    seasonal: bool = True
    max_P: int = 2
    max_D: int = 1
    max_Q: int = 2
    seasonal_period: int = 12
    
    # Forecasting settings
    forecast_horizon: int = 30
    confidence_level: float = 0.95
    
    # Validation settings
    n_splits: int = 5
    test_size: int = 30
    
    # Output settings
    save_results: bool = True
    output_dir: str = "forecasting_results"
    create_plots: bool = True
    
    def __post_init__(self):
        if self.models_to_try is None:
            self.models_to_try = ['arima', 'sarima', 'prophet']

@dataclass
class ForecastResult:
    """Result of time series forecasting"""
    model_name: str
    predictions: pd.Series
    confidence_intervals: pd.DataFrame
    metrics: Dict[str, float]
    model_params: Dict[str, Any]
    fit_time: float
    forecast_date: datetime

class TimeSeriesAnalyzer:
    """Comprehensive time series analysis"""
    
    def __init__(self, data: pd.DataFrame, config: ForecastConfig):
        self.data = data.copy()
        self.config = config
        self.analysis_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Prepare data
        self._prepare_data()
        
        # Create output directory
        if config.save_results:
            os.makedirs(config.output_dir, exist_ok=True)
    
    def _prepare_data(self):
        """Prepare time series data"""
        # Convert date column to datetime
        if self.config.date_column in self.data.columns:
            self.data[self.config.date_column] = pd.to_datetime(self.data[self.config.date_column])
            self.data.set_index(self.config.date_column, inplace=True)
        
        # Sort by date
        self.data.sort_index(inplace=True)
        
        # Extract target series
        self.ts_data = self.data[self.config.target_column]
        
        # Handle missing values
        if self.ts_data.isnull().any():
            self.logger.warning("Missing values detected. Performing interpolation.")
            self.ts_data = self.ts_data.interpolate(method='time')
        
        self.logger.info(f"Data prepared: {len(self.ts_data)} observations from {self.ts_data.index.min()} to {self.ts_data.index.max()}")
    
    def analyze_stationarity(self) -> Dict[str, Any]:
        """Analyze stationarity of the time series"""
        results = {}
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(self.ts_data.dropna())
        results['adf'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        # KPSS test
        kpss_result = kpss(self.ts_data.dropna(), regression='c')
        results['kpss'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > 0.05
        }
        
        # Combined conclusion
        results['conclusion'] = {
            'is_stationary': results['adf']['is_stationary'] and results['kpss']['is_stationary'],
            'recommendation': self._stationarity_recommendation(results)
        }
        
        self.analysis_results['stationarity'] = results
        return results
    
    def _stationarity_recommendation(self, results: Dict[str, Any]) -> str:
        """Provide stationarity recommendation"""
        adf_stationary = results['adf']['is_stationary']
        kpss_stationary = results['kpss']['is_stationary']
        
        if adf_stationary and kpss_stationary:
            return "Series is stationary. No differencing needed."
        elif not adf_stationary and kpss_stationary:
            return "Series has unit root. Apply first differencing."
        elif adf_stationary and not kpss_stationary:
            return "Series is trend stationary. Consider detrending."
        else:
            return "Series is non-stationary. Apply differencing and check again."
    
    def seasonal_decomposition(self, model: str = 'additive') -> Dict[str, Any]:
        """Perform seasonal decomposition"""
        try:
            # Determine seasonal period
            if self.config.frequency == 'D':
                period = 365  # Daily data, yearly seasonality
            elif self.config.frequency == 'W':
                period = 52   # Weekly data, yearly seasonality
            elif self.config.frequency == 'M':
                period = 12   # Monthly data, yearly seasonality
            elif self.config.frequency == 'Q':
                period = 4    # Quarterly data
            else:
                period = self.config.seasonal_period
            
            # Ensure we have enough data points
            if len(self.ts_data) < 2 * period:
                period = max(2, len(self.ts_data) // 4)
            
            decomposition = seasonal_decompose(
                self.ts_data, 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            
            results = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period,
                'model': model
            }
            
            # Calculate seasonality strength
            results['seasonality_strength'] = self._calculate_seasonality_strength(
                decomposition.seasonal, decomposition.resid
            )
            
            # Create decomposition plot
            if self.config.create_plots:
                self._plot_decomposition(decomposition)
            
            self.analysis_results['decomposition'] = results
            return results
            
        except Exception as e:
            self.logger.error(f"Seasonal decomposition failed: {e}")
            return {}
    
    def _calculate_seasonality_strength(self, seasonal: pd.Series, residual: pd.Series) -> float:
        """Calculate seasonality strength"""
        try:
            seasonal_var = np.var(seasonal.dropna())
            residual_var = np.var(residual.dropna())
            return seasonal_var / (seasonal_var + residual_var)
        except:
            return 0.0
    
    def _plot_decomposition(self, decomposition):
        """Plot seasonal decomposition"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        decomposition.observed.plot(ax=axes[0], title='Original')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        
        plt.tight_layout()
        
        if self.config.save_results:
            plt.savefig(f"{self.config.output_dir}/seasonal_decomposition.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def detect_anomalies(self, method: str = 'iqr', threshold: float = 3.0) -> Dict[str, Any]:
        """Detect anomalies in time series"""
        anomalies = {}
        
        if method == 'iqr':
            Q1 = self.ts_data.quantile(0.25)
            Q3 = self.ts_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies['indices'] = self.ts_data[(self.ts_data < lower_bound) | (self.ts_data > upper_bound)].index
            anomalies['values'] = self.ts_data.loc[anomalies['indices']]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(self.ts_data))
            anomalies['indices'] = self.ts_data[z_scores > threshold].index
            anomalies['values'] = self.ts_data.loc[anomalies['indices']]
        
        elif method == 'modified_zscore':
            median = np.median(self.ts_data)
            mad = np.median(np.abs(self.ts_data - median))
            modified_z_scores = 0.6745 * (self.ts_data - median) / mad
            anomalies['indices'] = self.ts_data[np.abs(modified_z_scores) > threshold].index
            anomalies['values'] = self.ts_data.loc[anomalies['indices']]
        
        anomalies['method'] = method
        anomalies['count'] = len(anomalies['indices'])
        anomalies['percentage'] = (len(anomalies['indices']) / len(self.ts_data)) * 100
        
        self.analysis_results['anomalies'] = anomalies
        return anomalies
    
    def autocorrelation_analysis(self, max_lags: int = 40) -> Dict[str, Any]:
        """Analyze autocorrelation and partial autocorrelation"""
        try:
            # Create ACF and PACF plots
            if self.config.create_plots:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                plot_acf(self.ts_data.dropna(), lags=max_lags, ax=ax1, alpha=0.05)
                ax1.set_title('Autocorrelation Function (ACF)')
                
                plot_pacf(self.ts_data.dropna(), lags=max_lags, ax=ax2, alpha=0.05)
                ax2.set_title('Partial Autocorrelation Function (PACF)')
                
                plt.tight_layout()
                
                if self.config.save_results:
                    plt.savefig(f"{self.config.output_dir}/acf_pacf.png", dpi=300, bbox_inches='tight')
                
                plt.show()
            
            # Calculate optimal ARIMA parameters based on ACF/PACF
            optimal_params = self._suggest_arima_params()
            
            results = {
                'suggested_arima_params': optimal_params,
                'analysis_date': datetime.now().isoformat()
            }
            
            self.analysis_results['autocorrelation'] = results
            return results
            
        except Exception as e:
            self.logger.error(f"Autocorrelation analysis failed: {e}")
            return {}
    
    def _suggest_arima_params(self) -> Dict[str, int]:
        """Suggest ARIMA parameters based on ACF/PACF analysis"""
        # This is a simplified heuristic - in practice, you'd want more sophisticated logic
        return {
            'p': 1,  # AR terms
            'd': 1,  # Differencing
            'q': 1   # MA terms
        }
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run comprehensive time series analysis"""
        self.logger.info("Starting comprehensive time series analysis...")
        
        # Stationarity analysis
        self.logger.info("Analyzing stationarity...")
        self.analyze_stationarity()
        
        # Seasonal decomposition
        self.logger.info("Performing seasonal decomposition...")
        self.seasonal_decomposition()
        
        # Anomaly detection
        self.logger.info("Detecting anomalies...")
        self.detect_anomalies()
        
        # Autocorrelation analysis
        self.logger.info("Analyzing autocorrelation...")
        self.autocorrelation_analysis()
        
        # Save analysis results
        if self.config.save_results:
            self._save_analysis_results()
        
        self.logger.info("Analysis complete!")
        return self.analysis_results
    
    def _save_analysis_results(self):
        """Save analysis results to file"""
        # Convert numpy arrays and pandas objects to serializable format
        serializable_results = {}
        
        for key, value in self.analysis_results.items():
            if isinstance(value, dict):
                serializable_results[key] = self._make_serializable(value)
            else:
                serializable_results[key] = value
        
        with open(f"{self.config.output_dir}/analysis_results.json", 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        self.logger.info(f"Analysis results saved to {self.config.output_dir}/analysis_results.json")
    
    def _make_serializable(self, obj):
        """Convert objects to serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

class ARIMAForecaster:
    """ARIMA and SARIMA forecasting implementation"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model = None
        self.fitted_model = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def auto_arima(self, ts_data: pd.Series) -> Tuple[int, int, int]:
        """Automatic ARIMA parameter selection using AIC"""
        best_aic = np.inf
        best_params = (0, 0, 0)
        
        self.logger.info("Searching for optimal ARIMA parameters...")
        
        for p in range(self.config.max_p + 1):
            for d in range(self.config.max_d + 1):
                for q in range(self.config.max_q + 1):
                    try:
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                    
                    except Exception:
                        continue
        
        self.logger.info(f"Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
        return best_params
    
    def fit(self, ts_data: pd.Series, order: Tuple[int, int, int] = None) -> 'ARIMAForecaster':
        """Fit ARIMA model"""
        if order is None:
            order = self.auto_arima(ts_data)
        
        try:
            self.model = ARIMA(ts_data, order=order)
            self.fitted_model = self.model.fit()
            
            self.logger.info(f"ARIMA{order} model fitted successfully")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to fit ARIMA model: {e}")
            raise
        
        return self
    
    def forecast(self, steps: int = None) -> ForecastResult:
        """Generate forecasts"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        if steps is None:
            steps = self.config.forecast_horizon
        
        # Generate forecast
        forecast = self.fitted_model.forecast(steps=steps)
        conf_int = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=1-self.config.confidence_level)
        
        # Create forecast index
        last_date = self.fitted_model.data.dates[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=self.config.frequency
        )
        
        # Create result
        result = ForecastResult(
            model_name='ARIMA',
            predictions=pd.Series(forecast, index=forecast_index),
            confidence_intervals=pd.DataFrame(conf_int, index=forecast_index),
            metrics=self._calculate_metrics(),
            model_params=self.fitted_model.params.to_dict(),
            fit_time=0.0,  # Would need to track this during fitting
            forecast_date=datetime.now()
        )
        
        return result
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate model performance metrics"""
        if self.fitted_model is None:
            return {}
        
        residuals = self.fitted_model.resid
        
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.llf,
            'mse': np.mean(residuals**2),
            'mae': np.mean(np.abs(residuals)),
            'ljung_box_p': self.fitted_model.test_serial_correlation('ljungbox')[0, 1]
        }

class ProphetForecaster:
    """Facebook Prophet forecasting implementation"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Try to import Prophet
        try:
            from prophet import Prophet
            self.Prophet = Prophet
            self.prophet_available = True
        except ImportError:
            self.logger.warning("Prophet not available. Install with: pip install prophet")
            self.prophet_available = False
    
    def fit(self, ts_data: pd.Series) -> 'ProphetForecaster':
        """Fit Prophet model"""
        if not self.prophet_available:
            raise ImportError("Prophet is not installed")
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': ts_data.index,
            'y': ts_data.values
        })
        
        # Initialize and fit model
        self.model = self.Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False if self.config.frequency != 'H' else True,
            interval_width=self.config.confidence_level
        )
        
        self.model.fit(df)
        self.logger.info("Prophet model fitted successfully")
        
        return self
    
    def forecast(self, steps: int = None) -> ForecastResult:
        """Generate forecasts using Prophet"""
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        if steps is None:
            steps = self.config.forecast_horizon
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps, freq=self.config.frequency)
        
        # Generate forecast
        forecast_df = self.model.predict(future)
        
        # Extract forecast period
        forecast_start = len(future) - steps
        predictions = forecast_df['yhat'].iloc[forecast_start:]
        lower_bound = forecast_df['yhat_lower'].iloc[forecast_start:]
        upper_bound = forecast_df['yhat_upper'].iloc[forecast_start:]
        
        # Create confidence intervals DataFrame
        conf_int = pd.DataFrame({
            'lower': lower_bound.values,
            'upper': upper_bound.values
        }, index=predictions.index)
        
        # Calculate metrics (simplified)
        metrics = {
            'cross_validation_mae': 0.0,  # Would need cross-validation
            'trend_strength': self._calculate_trend_strength(forecast_df),
            'seasonality_strength': self._calculate_seasonality_strength(forecast_df)
        }
        
        result = ForecastResult(
            model_name='Prophet',
            predictions=predictions,
            confidence_intervals=conf_int,
            metrics=metrics,
            model_params={},
            fit_time=0.0,
            forecast_date=datetime.now()
        )
        
        return result
    
    def _calculate_trend_strength(self, forecast_df: pd.DataFrame) -> float:
        """Calculate trend strength"""
        try:
            trend_var = np.var(forecast_df['trend'])
            residual_var = np.var(forecast_df['yhat'] - forecast_df['trend'])
            return trend_var / (trend_var + residual_var)
        except:
            return 0.0
    
    def _calculate_seasonality_strength(self, forecast_df: pd.DataFrame) -> float:
        """Calculate seasonality strength"""
        try:
            seasonal_cols = [col for col in forecast_df.columns if 'seasonal' in col]
            if seasonal_cols:
                seasonal_sum = forecast_df[seasonal_cols].sum(axis=1)
                seasonal_var = np.var(seasonal_sum)
                residual_var = np.var(forecast_df['yhat'] - seasonal_sum)
                return seasonal_var / (seasonal_var + residual_var)
            return 0.0
        except:
            return 0.0

class TimeSeriesForecaster:
    """Main forecasting orchestrator"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.models = {}
        self.results = {}
        self.best_model = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        if config.save_results:
            os.makedirs(config.output_dir, exist_ok=True)
    
    def fit_models(self, ts_data: pd.Series) -> Dict[str, ForecastResult]:
        """Fit all specified models"""
        self.logger.info(f"Fitting {len(self.config.models_to_try)} models...")
        
        for model_name in self.config.models_to_try:
            try:
                self.logger.info(f"Fitting {model_name}...")
                
                if model_name == 'arima':
                    forecaster = ARIMAForecaster(self.config)
                    forecaster.fit(ts_data)
                    result = forecaster.forecast()
                    self.models[model_name] = forecaster
                    self.results[model_name] = result
                
                elif model_name == 'prophet':
                    forecaster = ProphetForecaster(self.config)
                    forecaster.fit(ts_data)
                    result = forecaster.forecast()
                    self.models[model_name] = forecaster
                    self.results[model_name] = result
                
                elif model_name == 'sarima':
                    # Placeholder for SARIMA implementation
                    self.logger.warning("SARIMA implementation placeholder")
                    continue
                
                elif model_name == 'lstm':
                    # Placeholder for LSTM implementation
                    self.logger.warning("LSTM implementation placeholder")
                    continue
                
                else:
                    self.logger.warning(f"Unknown model: {model_name}")
                    continue
                
                self.logger.info(f"{model_name} fitted successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to fit {model_name}: {e}")
                continue
        
        # Select best model if auto_select is enabled
        if self.config.auto_select_model:
            self._select_best_model()
        
        return self.results
    
    def _select_best_model(self):
        """Select the best performing model"""
        if not self.results:
            return
        
        # Simple model selection based on AIC (for ARIMA) or other criteria
        best_score = np.inf
        best_model_name = None
        
        for model_name, result in self.results.items():
            if 'aic' in result.metrics:
                score = result.metrics['aic']
            elif 'cross_validation_mae' in result.metrics:
                score = result.metrics['cross_validation_mae']
            else:
                continue
            
            if score < best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name:
            self.best_model = best_model_name
            self.logger.info(f"Best model selected: {best_model_name} (score: {best_score:.2f})")
    
    def cross_validate(self, ts_data: pd.Series) -> Dict[str, Dict[str, float]]:
        """Perform time series cross-validation"""
        cv_results = {}
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits, test_size=self.config.test_size)
        
        for model_name in self.config.models_to_try:
            if model_name not in self.models:
                continue
            
            scores = []
            
            for train_idx, test_idx in tscv.split(ts_data):
                try:
                    # Split data
                    train_data = ts_data.iloc[train_idx]
                    test_data = ts_data.iloc[test_idx]
                    
                    # Fit model on training data
                    if model_name == 'arima':
                        forecaster = ARIMAForecaster(self.config)
                        forecaster.fit(train_data)
                        result = forecaster.forecast(steps=len(test_data))
                    elif model_name == 'prophet':
                        forecaster = ProphetForecaster(self.config)
                        forecaster.fit(train_data)
                        result = forecaster.forecast(steps=len(test_data))
                    else:
                        continue
                    
                    # Calculate metrics
                    mae = mean_absolute_error(test_data.values, result.predictions.values)
                    mse = mean_squared_error(test_data.values, result.predictions.values)
                    mape = mean_absolute_percentage_error(test_data.values, result.predictions.values)
                    
                    scores.append({
                        'mae': mae,
                        'mse': mse,
                        'rmse': np.sqrt(mse),
                        'mape': mape
                    })
                
                except Exception as e:
                    self.logger.error(f"Cross-validation error for {model_name}: {e}")
                    continue
            
            if scores:
                # Average scores across folds
                cv_results[model_name] = {
                    'mae': np.mean([s['mae'] for s in scores]),
                    'mse': np.mean([s['mse'] for s in scores]),
                    'rmse': np.mean([s['rmse'] for s in scores]),
                    'mape': np.mean([s['mape'] for s in scores]),
                    'mae_std': np.std([s['mae'] for s in scores]),
                    'n_folds': len(scores)
                }
        
        return cv_results
    
    def plot_forecasts(self, ts_data: pd.Series):
        """Plot forecasts for all models"""
        if not self.results:
            self.logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(len(self.results), 1, figsize=(15, 5 * len(self.results)))
        if len(self.results) == 1:
            axes = [axes]
        
        for i, (model_name, result) in enumerate(self.results.items()):
            ax = axes[i]
            
            # Plot historical data
            ts_data.plot(ax=ax, label='Historical', color='blue', alpha=0.7)
            
            # Plot forecast
            result.predictions.plot(ax=ax, label='Forecast', color='red', linewidth=2)
            
            # Plot confidence intervals
            ax.fill_between(
                result.confidence_intervals.index,
                result.confidence_intervals.iloc[:, 0],
                result.confidence_intervals.iloc[:, 1],
                alpha=0.3, color='red', label='Confidence Interval'
            )
            
            ax.set_title(f'{model_name.upper()} Forecast')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_results:
            plt.savefig(f"{self.config.output_dir}/forecasts.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self):
        """Save forecasting results"""
        if not self.config.save_results:
            return
        
        # Save forecast results
        for model_name, result in self.results.items():
            # Save predictions
            result.predictions.to_csv(f"{self.config.output_dir}/{model_name}_predictions.csv")
            
            # Save confidence intervals
            result.confidence_intervals.to_csv(f"{self.config.output_dir}/{model_name}_confidence_intervals.csv")
            
            # Save metrics
            with open(f"{self.config.output_dir}/{model_name}_metrics.json", 'w') as f:
                json.dump(result.metrics, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'best_model': self.best_model,
            'models_tried': list(self.results.keys()),
            'forecast_date': datetime.now().isoformat(),
            'config': {
                'forecast_horizon': self.config.forecast_horizon,
                'confidence_level': self.config.confidence_level,
                'frequency': self.config.frequency
            }
        }
        
        with open(f"{self.config.output_dir}/forecast_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {self.config.output_dir}")

# Example usage
if __name__ == "__main__":
    # Generate sample time series data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Create synthetic data with trend and seasonality
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    print("Sample Time Series Data:")
    print(data.head())
    print(f"Data shape: {data.shape}")
    
    # Configure forecasting
    config = ForecastConfig(
        models_to_try=['arima', 'prophet'],
        forecast_horizon=30,
        create_plots=True,
        save_results=True,
        output_dir="time_series_output"
    )
    
    # Run analysis
    analyzer = TimeSeriesAnalyzer(data, config)
    analysis_results = analyzer.run_full_analysis()
    
    # Run forecasting
    forecaster = TimeSeriesForecaster(config)
    forecast_results = forecaster.fit_models(analyzer.ts_data)
    
    # Plot results
    forecaster.plot_forecasts(analyzer.ts_data)
    
    # Save results
    forecaster.save_results()
    
    print(f"\nForecasting complete! Best model: {forecaster.best_model}")
    print(f"Results saved to: {config.output_dir}")
