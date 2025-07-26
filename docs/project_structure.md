# Data Science Project Structure and Standards

## Project Organization

```
ML_DS/
├── README.md                 # Project overview and setup instructions
├── requirements.txt          # Python dependencies
├── setup.py                 # Package installation configuration
├── pyproject.toml           # Modern Python project configuration
├── .gitignore               # Git ignore patterns
├── .github/                 # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yml           # Continuous integration
│       ├── deploy.yml       # Deployment pipeline
│       └── tests.yml        # Test automation
├── docs/                    # Documentation
│   ├── api_reference.md     # API documentation
│   ├── user_guide.md        # User guide
│   └── development_guide.md # Development setup
├── data/                    # Data storage (excluded from git)
│   ├── raw/                 # Original, immutable data
│   ├── interim/             # Intermediate transformed data
│   ├── processed/           # Final, canonical datasets
│   └── external/            # External data sources
├── notebooks/               # Jupyter notebooks
│   ├── exploratory/         # Exploratory data analysis
│   ├── experiments/         # Model experiments
│   └── reports/             # Analysis reports
├── src/                     # Source code
│   ├── __init__.py
│   ├── data/                # Data processing modules
│   │   ├── __init__.py
│   │   ├── ingestion.py     # Data ingestion
│   │   ├── validation.py    # Data validation
│   │   └── preprocessing.py # Data preprocessing
│   ├── features/            # Feature engineering
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   └── selection.py
│   ├── models/              # Model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py    # Base model class
│   │   ├── sklearn_models.py
│   │   └── deep_learning.py
│   ├── evaluation/          # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py       # Custom metrics
│   │   └── validation.py    # Cross-validation
│   ├── deployment/          # Deployment utilities
│   │   ├── __init__.py
│   │   ├── api.py           # REST API
│   │   └── batch_inference.py
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── config.py        # Configuration management
│       ├── logging.py       # Logging setup
│       └── io.py            # I/O utilities
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test data and fixtures
├── configs/                 # Configuration files
│   ├── model_config.yaml    # Model configurations
│   ├── data_config.yaml     # Data pipeline configurations
│   └── deployment_config.yaml
├── scripts/                 # Executable scripts
│   ├── train_model.py       # Training script
│   ├── evaluate_model.py    # Evaluation script
│   └── deploy_model.py      # Deployment script
├── models/                  # Trained model artifacts
│   ├── checkpoints/         # Model checkpoints
│   └── production/          # Production models
└── reports/                 # Generated reports
    ├── figures/             # Generated graphics
    └── performance/         # Performance reports
```

## Coding Standards

### Python Style Guide

1. **PEP 8 Compliance**: Follow Python Enhancement Proposal 8
2. **Line Length**: Maximum 88 characters (Black formatter default)
3. **Imports**: Group imports (standard library, third-party, local)
4. **Naming Conventions**:
   - Variables and functions: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_CASE`
   - Private methods: `_leading_underscore`

### Code Quality Tools

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Documentation Standards

```python
def train_model(X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> BaseModel:
    """
    Train a machine learning model with given data and configuration.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        config: Training configuration dictionary containing:
            - model_type: Type of model to train
            - hyperparameters: Model hyperparameters
            - validation_split: Fraction for validation
    
    Returns:
        Trained model instance
    
    Raises:
        ValueError: If input data is invalid
        ConfigurationError: If config is malformed
    
    Example:
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> config = {"model_type": "random_forest", "n_estimators": 100}
        >>> model = train_model(X, y, config)
    """
    # Implementation here
    pass
```

## Configuration Management

### YAML Configuration Example

```yaml
# configs/model_config.yaml
model:
  type: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    random_state: 42

training:
  validation_split: 0.2
  cross_validation:
    folds: 5
    shuffle: true
  early_stopping:
    patience: 10
    monitor: "val_accuracy"

data:
  features:
    - feature1
    - feature2
    - feature3
  target: "target_column"
  preprocessing:
    scaling: "standard"
    handle_missing: "median"
```

### Python Configuration Management

```python
# src/utils/config.py
import yaml
from typing import Dict, Any
from pathlib import Path

class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
```

## Testing Strategy

### Test Structure

```python
# tests/unit/test_models.py
import pytest
import numpy as np
from src.models.sklearn_models import RandomForestModel

class TestRandomForestModel:
    """Test suite for RandomForestModel."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return RandomForestModel(n_estimators=10, random_state=42)
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.n_estimators == 10
        assert model.random_state == 42
        assert not model.is_fitted
    
    def test_model_training(self, model, sample_data):
        """Test model training functionality."""
        X, y = sample_data
        model.fit(X, y)
        
        assert model.is_fitted
        assert hasattr(model, 'feature_importance_')
    
    def test_model_prediction(self, model, sample_data):
        """Test model prediction functionality."""
        X, y = sample_data
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 2)
        assert np.all((predictions >= 0) & (predictions <= 1))
    
    def test_invalid_input(self, model):
        """Test handling of invalid input."""
        with pytest.raises(ValueError):
            model.fit([], [])
        
        with pytest.raises(ValueError):
            model.predict(np.random.randn(5, 3))  # Wrong number of features
```

### Integration Tests

```python
# tests/integration/test_pipeline.py
import pytest
from src.data.preprocessing import DataPreprocessor
from src.models.sklearn_models import RandomForestModel
from src.evaluation.metrics import ModelEvaluator

def test_full_pipeline():
    """Test complete ML pipeline integration."""
    # Load test data
    X, y = load_test_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Train model
    model = RandomForestModel()
    model.fit(X_processed, y)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_processed, y)
    
    # Assertions
    assert metrics['accuracy'] > 0.5
    assert metrics['precision'] > 0.5
    assert metrics['recall'] > 0.5
```

## Logging Standards

```python
# src/utils/logging.py
import logging
import sys
from pathlib import Path

def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    format_string: str = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
```

## Data Management

### Data Validation

```python
# src/data/validation.py
import pandas as pd
from typing import List, Dict, Any
import logging

class DataValidator:
    """Data validation utilities."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.logger = logging.getLogger(__name__)
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame against schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        required_cols = self.schema.get('required_columns', [])
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing columns: {missing_cols}")
        
        # Check data types
        for col, expected_type in self.schema.get('column_types', {}).items():
            if col in df.columns:
                if df[col].dtype != expected_type:
                    results['warnings'].append(
                        f"Column {col} has type {df[col].dtype}, "
                        f"expected {expected_type}"
                    )
        
        # Check value ranges
        for col, range_config in self.schema.get('value_ranges', {}).items():
            if col in df.columns:
                min_val, max_val = range_config['min'], range_config['max']
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if not out_of_range.empty:
                    results['warnings'].append(
                        f"Column {col} has {len(out_of_range)} "
                        f"values outside range [{min_val}, {max_val}]"
                    )
        
        return results
```

This structure provides a solid foundation for any data science project, ensuring consistency, maintainability, and scalability.
