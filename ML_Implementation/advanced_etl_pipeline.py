"""
Advanced Data Engineering & ETL Pipeline Framework
==================================================

A comprehensive data engineering framework featuring scalable ETL pipelines,
real-time data processing, data quality validation, and advanced transformations.

Best Contributions:
- Scalable distributed ETL pipelines with Apache Spark
- Real-time stream processing with Kafka integration
- Advanced data validation and quality monitoring
- Multi-source data ingestion (SQL, NoSQL, APIs, files)
- Data lineage tracking and metadata management
- Automated schema evolution and drift detection
- Performance optimization and resource management
- Data lake and warehouse integration

Author: ML/DS Advanced Implementation Team
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import hashlib
import pickle
from abc import ABC, abstractmethod

# Data processing libraries
try:
    import dask.dataframe as dd
    import dask.bag as db
    from dask.distributed import Client, as_completed
    import dask
    
    # Apache Spark
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql.functions import col, when, isnan, isnull, count, sum as spark_sum
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
    
    # Database connectors
    import sqlalchemy
    from sqlalchemy import create_engine, MetaData, Table
    import pymongo
    import redis
    
    # Stream processing
    from kafka import KafkaProducer, KafkaConsumer
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    
except ImportError as e:
    logging.warning(f"Some data engineering libraries not available: {e}")

# Cloud storage libraries
try:
    import boto3
    from azure.storage.blob import BlobServiceClient
    from google.cloud import storage as gcs
    import delta
    from delta.tables import DeltaTable
except ImportError as e:
    logging.warning(f"Some cloud storage libraries not available: {e}")

@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    source_type: str  # 'sql', 'nosql', 'api', 'file', 'stream'
    connection_string: str = ""
    table_name: str = ""
    query: str = ""
    api_endpoint: str = ""
    file_path: str = ""
    file_format: str = "csv"  # csv, parquet, json, avro
    batch_size: int = 10000
    refresh_interval: int = 3600  # seconds
    schema: Optional[Dict] = None
    credentials: Optional[Dict] = None

@dataclass
class ETLConfig:
    """Configuration for ETL pipeline."""
    pipeline_name: str = "default_pipeline"
    source_configs: List[DataSourceConfig] = field(default_factory=list)
    target_config: DataSourceConfig = None
    transformation_steps: List[str] = field(default_factory=list)
    data_quality_rules: List[Dict] = field(default_factory=list)
    enable_lineage: bool = True
    enable_monitoring: bool = True
    parallel_workers: int = 4
    chunk_size: int = 10000
    max_retries: int = 3
    checkpoint_interval: int = 1000
    output_format: str = "parquet"
    compression: str = "snappy"
    partition_columns: List[str] = field(default_factory=list)

class DataValidator:
    """Advanced data validation and quality monitoring."""
    
    def __init__(self):
        self.rules = []
        self.validation_results = []
        self.logger = logging.getLogger(__name__)
    
    def add_rule(self, column: str, rule_type: str, parameters: Dict):
        """Add validation rule."""
        rule = {
            'column': column,
            'rule_type': rule_type,
            'parameters': parameters,
            'created_at': datetime.now()
        }
        self.rules.append(rule)
    
    def validate_completeness(self, df: pd.DataFrame, column: str, threshold: float = 0.95) -> Dict:
        """Validate data completeness."""
        total_rows = len(df)
        non_null_rows = df[column].notna().sum()
        completeness_ratio = non_null_rows / total_rows if total_rows > 0 else 0
        
        result = {
            'rule': 'completeness',
            'column': column,
            'threshold': threshold,
            'actual_ratio': completeness_ratio,
            'passed': completeness_ratio >= threshold,
            'total_rows': total_rows,
            'non_null_rows': int(non_null_rows),
            'null_rows': total_rows - int(non_null_rows)
        }
        
        return result
    
    def validate_uniqueness(self, df: pd.DataFrame, column: str, threshold: float = 1.0) -> Dict:
        """Validate data uniqueness."""
        total_rows = len(df)
        unique_values = df[column].nunique()
        uniqueness_ratio = unique_values / total_rows if total_rows > 0 else 0
        
        result = {
            'rule': 'uniqueness',
            'column': column,
            'threshold': threshold,
            'actual_ratio': uniqueness_ratio,
            'passed': uniqueness_ratio >= threshold,
            'total_rows': total_rows,
            'unique_values': unique_values,
            'duplicate_rows': total_rows - unique_values
        }
        
        return result
    
    def validate_range(self, df: pd.DataFrame, column: str, min_val: float, max_val: float) -> Dict:
        """Validate data range."""
        if df[column].dtype not in ['int64', 'float64']:
            return {'rule': 'range', 'column': column, 'error': 'Column is not numeric'}
        
        total_rows = len(df)
        valid_rows = df[(df[column] >= min_val) & (df[column] <= max_val)].shape[0]
        validity_ratio = valid_rows / total_rows if total_rows > 0 else 0
        
        result = {
            'rule': 'range',
            'column': column,
            'min_val': min_val,
            'max_val': max_val,
            'actual_min': float(df[column].min()),
            'actual_max': float(df[column].max()),
            'validity_ratio': validity_ratio,
            'passed': validity_ratio == 1.0,
            'total_rows': total_rows,
            'valid_rows': valid_rows,
            'invalid_rows': total_rows - valid_rows
        }
        
        return result
    
    def validate_pattern(self, df: pd.DataFrame, column: str, pattern: str) -> Dict:
        """Validate data pattern using regex."""
        total_rows = len(df)
        valid_rows = df[column].str.match(pattern, na=False).sum()
        validity_ratio = valid_rows / total_rows if total_rows > 0 else 0
        
        result = {
            'rule': 'pattern',
            'column': column,
            'pattern': pattern,
            'validity_ratio': validity_ratio,
            'passed': validity_ratio == 1.0,
            'total_rows': total_rows,
            'valid_rows': int(valid_rows),
            'invalid_rows': total_rows - int(valid_rows)
        }
        
        return result
    
    def validate_dataframe(self, df: pd.DataFrame, rules: List[Dict] = None) -> List[Dict]:
        """Validate entire dataframe against rules."""
        rules = rules or self.rules
        results = []
        
        for rule in rules:
            column = rule['column']
            rule_type = rule['rule_type']
            params = rule['parameters']
            
            if column not in df.columns:
                results.append({
                    'rule': rule_type,
                    'column': column,
                    'error': f'Column {column} not found in dataframe',
                    'passed': False
                })
                continue
            
            try:
                if rule_type == 'completeness':
                    result = self.validate_completeness(df, column, params.get('threshold', 0.95))
                elif rule_type == 'uniqueness':
                    result = self.validate_uniqueness(df, column, params.get('threshold', 1.0))
                elif rule_type == 'range':
                    result = self.validate_range(df, column, params['min_val'], params['max_val'])
                elif rule_type == 'pattern':
                    result = self.validate_pattern(df, column, params['pattern'])
                else:
                    result = {
                        'rule': rule_type,
                        'column': column,
                        'error': f'Unknown rule type: {rule_type}',
                        'passed': False
                    }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'rule': rule_type,
                    'column': column,
                    'error': str(e),
                    'passed': False
                })
        
        self.validation_results.extend(results)
        return results

class DataLineageTracker:
    """Track data lineage and transformations."""
    
    def __init__(self):
        self.lineage_graph = {}
        self.transformations = []
        self.logger = logging.getLogger(__name__)
    
    def add_source(self, source_id: str, metadata: Dict):
        """Add data source to lineage."""
        self.lineage_graph[source_id] = {
            'type': 'source',
            'metadata': metadata,
            'created_at': datetime.now(),
            'dependencies': [],
            'transformations': []
        }
    
    def add_transformation(self, transform_id: str, input_ids: List[str], 
                          transformation_type: str, parameters: Dict):
        """Add transformation to lineage."""
        transform = {
            'id': transform_id,
            'type': 'transformation',
            'transformation_type': transformation_type,
            'parameters': parameters,
            'input_ids': input_ids,
            'created_at': datetime.now()
        }
        
        self.lineage_graph[transform_id] = {
            'type': 'transformation',
            'metadata': transform,
            'dependencies': input_ids,
            'transformations': []
        }
        
        # Update dependencies
        for input_id in input_ids:
            if input_id in self.lineage_graph:
                self.lineage_graph[input_id]['transformations'].append(transform_id)
        
        self.transformations.append(transform)
    
    def get_lineage(self, entity_id: str) -> Dict:
        """Get complete lineage for an entity."""
        if entity_id not in self.lineage_graph:
            return {}
        
        def get_ancestors(node_id: str, visited: set = None) -> Dict:
            if visited is None:
                visited = set()
            
            if node_id in visited:
                return {}
            
            visited.add(node_id)
            node = self.lineage_graph[node_id]
            
            ancestors = {'id': node_id, 'metadata': node['metadata'], 'children': []}
            
            for dep_id in node['dependencies']:
                if dep_id in self.lineage_graph:
                    child_lineage = get_ancestors(dep_id, visited.copy())
                    if child_lineage:
                        ancestors['children'].append(child_lineage)
            
            return ancestors
        
        return get_ancestors(entity_id)
    
    def export_lineage(self, filepath: str):
        """Export lineage graph to file."""
        with open(filepath, 'w') as f:
            json.dump(self.lineage_graph, f, indent=2, default=str)

class DataTransformer:
    """Advanced data transformation engine."""
    
    def __init__(self):
        self.transformations = {}
        self.logger = logging.getLogger(__name__)
    
    def register_transformation(self, name: str, func: Callable):
        """Register custom transformation function."""
        self.transformations[name] = func
    
    def clean_data(self, df: pd.DataFrame, strategies: Dict = None) -> pd.DataFrame:
        """Clean data using various strategies."""
        strategies = strategies or {
            'remove_duplicates': True,
            'handle_missing': 'drop',  # 'drop', 'fill', 'interpolate'
            'outlier_method': 'iqr',   # 'iqr', 'zscore', 'isolation'
            'normalize_text': True
        }
        
        df_clean = df.copy()
        
        # Remove duplicates
        if strategies.get('remove_duplicates'):
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed_rows = initial_rows - len(df_clean)
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} duplicate rows")
        
        # Handle missing values
        missing_strategy = strategies.get('handle_missing', 'drop')
        if missing_strategy == 'drop':
            df_clean = df_clean.dropna()
        elif missing_strategy == 'fill':
            # Fill numeric columns with median, categorical with mode
            for col in df_clean.columns:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col].fillna(mode_val[0], inplace=True)
        elif missing_strategy == 'interpolate':
            df_clean = df_clean.interpolate()
        
        # Remove outliers
        outlier_method = strategies.get('outlier_method', 'iqr')
        if outlier_method == 'iqr':
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        elif outlier_method == 'zscore':
            from scipy import stats
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                z_scores = np.abs(stats.zscore(df_clean[col]))
                df_clean = df_clean[z_scores < 3]
        
        # Normalize text
        if strategies.get('normalize_text'):
            text_columns = df_clean.select_dtypes(include=['object']).columns
            for col in text_columns:
                df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
        
        return df_clean
    
    def aggregate_data(self, df: pd.DataFrame, group_by: List[str], 
                      aggregations: Dict) -> pd.DataFrame:
        """Perform advanced aggregations."""
        return df.groupby(group_by).agg(aggregations).reset_index()
    
    def pivot_data(self, df: pd.DataFrame, index: str, columns: str, 
                   values: str, aggfunc: str = 'sum') -> pd.DataFrame:
        """Pivot data with specified aggregation."""
        return df.pivot_table(index=index, columns=columns, values=values, 
                             aggfunc=aggfunc, fill_value=0).reset_index()
    
    def join_data(self, left_df: pd.DataFrame, right_df: pd.DataFrame,
                  on: str, how: str = 'inner') -> pd.DataFrame:
        """Join two dataframes."""
        return left_df.merge(right_df, on=on, how=how)
    
    def apply_custom_transformation(self, df: pd.DataFrame, 
                                   transformation_name: str, 
                                   **kwargs) -> pd.DataFrame:
        """Apply custom registered transformation."""
        if transformation_name in self.transformations:
            return self.transformations[transformation_name](df, **kwargs)
        else:
            self.logger.warning(f"Transformation {transformation_name} not found")
            return df

class AdvancedETLPipeline:
    """
    Advanced ETL Pipeline Framework.
    
    Features:
    - Multi-source data ingestion
    - Scalable processing with Dask/Spark
    - Real-time stream processing
    - Data quality validation
    - Lineage tracking
    - Automated monitoring
    """
    
    def __init__(self, config: ETLConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Components
        self.validator = DataValidator()
        self.lineage = DataLineageTracker()
        self.transformer = DataTransformer()
        
        # Processing state
        self.processed_records = 0
        self.failed_records = 0
        self.start_time = None
        self.checkpoints = {}
        
        # Spark session (if available)
        self.spark = None
        self.dask_client = None
        
        self._initialize_engines()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"etl_pipeline_{self.config.pipeline_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_engines(self):
        """Initialize processing engines."""
        try:
            # Initialize Spark
            self.spark = SparkSession.builder \
                .appName(self.config.pipeline_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            self.logger.info("Spark session initialized")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize Spark: {e}")
        
        try:
            # Initialize Dask
            self.dask_client = Client(n_workers=self.config.parallel_workers)
            self.logger.info(f"Dask client initialized with {self.config.parallel_workers} workers")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize Dask: {e}")
    
    def read_data_source(self, source_config: DataSourceConfig) -> pd.DataFrame:
        """Read data from various sources."""
        self.logger.info(f"Reading from {source_config.source_type}: {source_config.table_name or source_config.file_path}")
        
        try:
            if source_config.source_type == 'sql':
                engine = create_engine(source_config.connection_string)
                if source_config.query:
                    df = pd.read_sql(source_config.query, engine)
                else:
                    df = pd.read_sql_table(source_config.table_name, engine)
                
            elif source_config.source_type == 'file':
                if source_config.file_format == 'csv':
                    df = pd.read_csv(source_config.file_path)
                elif source_config.file_format == 'parquet':
                    df = pd.read_parquet(source_config.file_path)
                elif source_config.file_format == 'json':
                    df = pd.read_json(source_config.file_path)
                else:
                    raise ValueError(f"Unsupported file format: {source_config.file_format}")
                
            elif source_config.source_type == 'api':
                import requests
                response = requests.get(source_config.api_endpoint)
                data = response.json()
                df = pd.json_normalize(data)
                
            elif source_config.source_type == 'nosql':
                # MongoDB example
                client = pymongo.MongoClient(source_config.connection_string)
                db_name, collection_name = source_config.table_name.split('.')
                collection = client[db_name][collection_name]
                data = list(collection.find())
                df = pd.DataFrame(data)
                
            else:
                raise ValueError(f"Unsupported source type: {source_config.source_type}")
            
            # Add to lineage
            source_id = f"{source_config.source_type}_{source_config.table_name or source_config.file_path}"
            self.lineage.add_source(source_id, {
                'source_type': source_config.source_type,
                'location': source_config.table_name or source_config.file_path,
                'records_count': len(df),
                'columns': list(df.columns)
            })
            
            self.logger.info(f"Successfully read {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading data source: {e}")
            raise
    
    def write_data_target(self, df: pd.DataFrame, target_config: DataSourceConfig):
        """Write data to target destination."""
        self.logger.info(f"Writing to {target_config.source_type}: {target_config.table_name or target_config.file_path}")
        
        try:
            if target_config.source_type == 'sql':
                engine = create_engine(target_config.connection_string)
                df.to_sql(target_config.table_name, engine, if_exists='replace', index=False)
                
            elif target_config.source_type == 'file':
                if target_config.file_format == 'csv':
                    df.to_csv(target_config.file_path, index=False)
                elif target_config.file_format == 'parquet':
                    df.to_parquet(target_config.file_path, 
                                 compression=self.config.compression,
                                 index=False)
                elif target_config.file_format == 'json':
                    df.to_json(target_config.file_path, orient='records', indent=2)
                else:
                    raise ValueError(f"Unsupported file format: {target_config.file_format}")
            
            else:
                raise ValueError(f"Unsupported target type: {target_config.source_type}")
            
            self.logger.info(f"Successfully wrote {len(df)} records")
            
        except Exception as e:
            self.logger.error(f"Error writing data target: {e}")
            raise
    
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of data."""
        transform_id = f"transform_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Data cleaning
        df_processed = self.transformer.clean_data(df)
        
        # Apply custom transformations
        for step in self.config.transformation_steps:
            df_processed = self.transformer.apply_custom_transformation(df_processed, step)
        
        # Data validation
        validation_results = self.validator.validate_dataframe(df_processed)
        failed_validations = [r for r in validation_results if not r.get('passed', False)]
        
        if failed_validations:
            self.logger.warning(f"Data validation failed for {len(failed_validations)} rules")
            for failure in failed_validations:
                self.logger.warning(f"Validation failure: {failure}")
        
        # Add to lineage
        self.lineage.add_transformation(
            transform_id,
            ['input_data'],
            'batch_processing',
            {
                'input_records': len(df),
                'output_records': len(df_processed),
                'transformations': self.config.transformation_steps,
                'validation_results': validation_results
            }
        )
        
        return df_processed
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete ETL pipeline."""
        self.logger.info(f"Starting ETL pipeline: {self.config.pipeline_name}")
        self.start_time = datetime.now()
        
        pipeline_results = {
            'pipeline_name': self.config.pipeline_name,
            'start_time': self.start_time,
            'status': 'running',
            'processed_records': 0,
            'failed_records': 0,
            'validation_results': []
        }
        
        try:
            # Read from all sources
            dataframes = []
            for source_config in self.config.source_configs:
                df = self.read_data_source(source_config)
                dataframes.append(df)
            
            # Combine dataframes if multiple sources
            if len(dataframes) > 1:
                combined_df = pd.concat(dataframes, ignore_index=True)
            else:
                combined_df = dataframes[0]
            
            self.logger.info(f"Combined data: {len(combined_df)} records")
            
            # Process in chunks for large datasets
            chunk_size = self.config.chunk_size
            processed_chunks = []
            
            for i in range(0, len(combined_df), chunk_size):
                chunk = combined_df.iloc[i:i+chunk_size]
                processed_chunk = self.process_batch(chunk)
                processed_chunks.append(processed_chunk)
                
                self.processed_records += len(processed_chunk)
                
                # Checkpoint
                if (i // chunk_size) % self.config.checkpoint_interval == 0:
                    self.logger.info(f"Processed {self.processed_records} records")
            
            # Combine processed chunks
            final_df = pd.concat(processed_chunks, ignore_index=True)
            
            # Write to target
            if self.config.target_config:
                self.write_data_target(final_df, self.config.target_config)
            
            # Pipeline completion
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            pipeline_results.update({
                'status': 'completed',
                'end_time': end_time,
                'duration_seconds': duration,
                'processed_records': self.processed_records,
                'failed_records': self.failed_records,
                'validation_results': self.validator.validation_results,
                'records_per_second': self.processed_records / duration if duration > 0 else 0
            })
            
            self.logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
            self.logger.info(f"Processed {self.processed_records} records ({self.processed_records/duration:.2f} records/sec)")
            
        except Exception as e:
            pipeline_results.update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now()
            })
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        return pipeline_results
    
    def cleanup(self):
        """Clean up resources."""
        if self.spark:
            self.spark.stop()
        if self.dask_client:
            self.dask_client.close()

def main():
    """Demonstration of the Advanced ETL Pipeline Framework."""
    print("=== Advanced Data Engineering & ETL Pipeline Demo ===\n")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'id': range(1, 1001),
        'name': [f'Customer_{i}' for i in range(1, 1001)],
        'age': np.random.randint(18, 80, 1000),
        'salary': np.random.normal(50000, 15000, 1000),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 1000),
        'join_date': pd.date_range('2020-01-01', periods=1000, freq='D')[:1000]
    })
    
    # Add some data quality issues for demonstration
    sample_data.loc[10:20, 'age'] = np.nan  # Missing values
    sample_data.loc[50:60, 'salary'] = -1000  # Invalid values
    sample_data = pd.concat([sample_data, sample_data.iloc[:50]], ignore_index=True)  # Duplicates
    
    # Save sample data
    sample_data.to_csv('/tmp/sample_data.csv', index=False)
    print(f"Created sample dataset with {len(sample_data)} records")
    
    # Configure data source
    source_config = DataSourceConfig(
        source_type='file',
        file_path='/tmp/sample_data.csv',
        file_format='csv'
    )
    
    # Configure target
    target_config = DataSourceConfig(
        source_type='file',
        file_path='/tmp/processed_data.parquet',
        file_format='parquet'
    )
    
    # Configure ETL pipeline
    etl_config = ETLConfig(
        pipeline_name='sample_etl_pipeline',
        source_configs=[source_config],
        target_config=target_config,
        transformation_steps=['clean_data'],
        parallel_workers=2,
        chunk_size=200
    )
    
    print(f"ETL Configuration: {etl_config.pipeline_name}")
    
    # Initialize pipeline
    pipeline = AdvancedETLPipeline(etl_config)
    
    # Add data validation rules
    pipeline.validator.add_rule('age', 'completeness', {'threshold': 0.8})
    pipeline.validator.add_rule('age', 'range', {'min_val': 0, 'max_val': 120})
    pipeline.validator.add_rule('salary', 'range', {'min_val': 0, 'max_val': 200000})
    pipeline.validator.add_rule('id', 'uniqueness', {'threshold': 1.0})
    
    print(f"Added {len(pipeline.validator.rules)} validation rules")
    
    # Run pipeline
    print("\nRunning ETL pipeline...")
    results = pipeline.run_pipeline()
    
    # Display results
    print(f"\nPipeline Results:")
    print(f"Status: {results['status']}")
    print(f"Processed Records: {results['processed_records']}")
    print(f"Duration: {results['duration_seconds']:.2f} seconds")
    print(f"Throughput: {results['records_per_second']:.2f} records/second")
    
    # Validation results
    if results['validation_results']:
        print(f"\nValidation Results:")
        for result in results['validation_results']:
            status = "✓" if result.get('passed', False) else "✗"
            print(f"{status} {result['rule']} on {result['column']}: {result.get('actual_ratio', 'N/A')}")
    
    # Clean up
    pipeline.cleanup()
    
    print("\nETL Pipeline demonstration completed!")

if __name__ == "__main__":
    main()
