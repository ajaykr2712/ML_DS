"""
Advanced Data Pipeline System
Implementing Contributions #1, #21, #22, #23, #24
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class DataSource:
    """Configuration for data sources."""
    name: str
    source_type: str  # 'file', 'database', 'api', 'stream'
    connection_params: Dict[str, Any]
    schema: Optional[Dict[str, str]] = None


class DataValidator:
    """Advanced data validation system."""
    
    def __init__(self):
        self.validation_rules = {}
        self.logger = logging.getLogger(__name__)
    
    def add_rule(self, column: str, rule_type: str, params: Dict[str, Any]):
        """Add validation rule for a column."""
        if column not in self.validation_rules:
            self.validation_rules[column] = []
        self.validation_rules[column].append({
            'type': rule_type,
            'params': params
        })
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate entire dataframe and return validation results."""
        validation_results = {}
        
        for column, rules in self.validation_rules.items():
            if column not in df.columns:
                validation_results[column] = ['Column missing']
                continue
                
            column_errors = []
            for rule in rules:
                errors = self._apply_rule(df[column], rule)
                column_errors.extend(errors)
            
            if column_errors:
                validation_results[column] = column_errors
        
        return validation_results
    
    def _apply_rule(self, series: pd.Series, rule: Dict[str, Any]) -> List[str]:
        """Apply single validation rule to a series."""
        errors = []
        rule_type = rule['type']
        params = rule['params']
        
        if rule_type == 'not_null':
            null_count = series.isnull().sum()
            if null_count > 0:
                errors.append(f'{null_count} null values found')
        
        elif rule_type == 'range':
            min_val, max_val = params['min'], params['max']
            out_of_range = ((series < min_val) | (series > max_val)).sum()
            if out_of_range > 0:
                errors.append(f'{out_of_range} values out of range [{min_val}, {max_val}]')
        
        elif rule_type == 'unique':
            duplicate_count = series.duplicated().sum()
            if duplicate_count > 0:
                errors.append(f'{duplicate_count} duplicate values found')
        
        elif rule_type == 'regex':
            pattern = params['pattern']
            invalid_count = (~series.str.match(pattern)).sum()
            if invalid_count > 0:
                errors.append(f'{invalid_count} values do not match pattern {pattern}')
        
        return errors


class DataTransformer(BaseEstimator, TransformerMixin):
    """Advanced data transformation pipeline."""
    
    def __init__(self, transformations: List[Dict[str, Any]]):
        self.transformations = transformations
        self.fitted_transformers = {}
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit all transformations to the data."""
        current_data = X.copy()
        
        for i, transformation in enumerate(self.transformations):
            transformer = self._create_transformer(transformation)
            transformer.fit(current_data)
            self.fitted_transformers[i] = transformer
            current_data = transformer.transform(current_data)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all fitted transformations to the data."""
        result = X.copy()
        
        for i in range(len(self.transformations)):
            transformer = self.fitted_transformers[i]
            result = transformer.transform(result)
        
        return result
    
    def _create_transformer(self, config: Dict[str, Any]):
        """Create transformer based on configuration."""
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        
        transform_type = config['type']
        
        if transform_type == 'scale':
            return StandardScaler()
        elif transform_type == 'impute':
            strategy = config.get('strategy', 'mean')
            return SimpleImputer(strategy=strategy)
        elif transform_type == 'encode_categorical':
            return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")


class DataPipeline:
    """Main data pipeline orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_sources = []
        self.validator = DataValidator()
        self.transformer = None
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def add_data_source(self, source: DataSource):
        """Add a data source to the pipeline."""
        self.data_sources.append(source)
    
    def setup_validation(self, validation_config: Dict[str, Any]):
        """Setup data validation rules."""
        for column, rules in validation_config.items():
            for rule in rules:
                self.validator.add_rule(column, rule['type'], rule['params'])
    
    def setup_transformation(self, transformation_config: List[Dict[str, Any]]):
        """Setup data transformation pipeline."""
        self.transformer = DataTransformer(transformation_config)
    
    async def load_data(self, source: DataSource) -> pd.DataFrame:
        """Load data from a single source asynchronously."""
        self.logger.info(f"Loading data from source: {source.name}")
        
        if source.source_type == 'file':
            return await self._load_file_data(source)
        elif source.source_type == 'database':
            return await self._load_database_data(source)
        elif source.source_type == 'api':
            return await self._load_api_data(source)
        else:
            raise ValueError(f"Unsupported source type: {source.source_type}")
    
    async def _load_file_data(self, source: DataSource) -> pd.DataFrame:
        """Load data from file source."""
        file_path = source.connection_params['path']
        file_type = source.connection_params.get('type', 'csv')
        
        loop = asyncio.get_event_loop()
        
        if file_type == 'csv':
            return await loop.run_in_executor(
                self.executor, 
                lambda: pd.read_csv(file_path)
            )
        elif file_type == 'parquet':
            return await loop.run_in_executor(
                self.executor,
                lambda: pd.read_parquet(file_path)
            )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    async def _load_database_data(self, source: DataSource) -> pd.DataFrame:
        """Load data from database source."""
        # Placeholder for database loading
        # In real implementation, would use sqlalchemy or similar
        self.logger.info(f"Loading from database: {source.connection_params}")
        return pd.DataFrame()
    
    async def _load_api_data(self, source: DataSource) -> pd.DataFrame:
        """Load data from API source."""
        # Placeholder for API loading
        # In real implementation, would use aiohttp or similar
        self.logger.info(f"Loading from API: {source.connection_params}")
        return pd.DataFrame()
    
    async def process_data(self) -> pd.DataFrame:
        """Process all data sources through the complete pipeline."""
        # Load all data sources concurrently
        load_tasks = [self.load_data(source) for source in self.data_sources]
        dataframes = await asyncio.gather(*load_tasks)
        
        # Combine dataframes
        if len(dataframes) == 1:
            combined_data = dataframes[0]
        else:
            combined_data = pd.concat(dataframes, ignore_index=True)
        
        self.logger.info(f"Combined data shape: {combined_data.shape}")
        
        # Validate data
        validation_results = self.validator.validate_dataframe(combined_data)
        if validation_results:
            self.logger.warning(f"Validation issues found: {validation_results}")
        
        # Transform data
        if self.transformer:
            if not hasattr(self.transformer, 'fitted_transformers'):
                self.transformer.fit(combined_data)
            transformed_data = self.transformer.transform(combined_data)
            self.logger.info(f"Transformed data shape: {transformed_data.shape}")
            return transformed_data
        
        return combined_data
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline."""
        return {
            'num_sources': len(self.data_sources),
            'validation_rules': len(self.validator.validation_rules),
            'has_transformer': self.transformer is not None
        }


class StreamingDataPipeline(DataPipeline):
    """Streaming version of data pipeline for real-time data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.stream_buffer_size = config.get('buffer_size', 1000)
        self.processing_interval = config.get('interval', 1.0)
    
    async def start_streaming(self, callback_func):
        """Start streaming data processing."""
        self.logger.info("Starting streaming data pipeline")
        
        while True:
            try:
                # Process batch of streaming data
                batch_data = await self.process_stream_batch()
                if not batch_data.empty:
                    await callback_func(batch_data)
                
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                self.logger.error(f"Error in streaming pipeline: {e}")
                await asyncio.sleep(self.processing_interval)
    
    async def process_stream_batch(self) -> pd.DataFrame:
        """Process a batch of streaming data."""
        # Placeholder for streaming data processing
        # In real implementation, would integrate with Kafka, Kinesis, etc.
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    config = {
        'name': 'example_pipeline',
        'buffer_size': 1000
    }
    
    pipeline = DataPipeline(config)
    
    # Add data source
    source = DataSource(
        name='example_csv',
        source_type='file',
        connection_params={'path': 'data.csv', 'type': 'csv'}
    )
    pipeline.add_data_source(source)
    
    print("Data pipeline created successfully!")
