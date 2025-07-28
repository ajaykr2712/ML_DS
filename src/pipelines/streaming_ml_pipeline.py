"""
Real-time Streaming ML Pipeline
Implements real-time machine learning pipelines for streaming data processing.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from abc import ABC, abstractmethod
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """Configuration for streaming pipeline."""
    buffer_size: int = 1000
    batch_size: int = 32
    processing_interval: float = 1.0  # seconds
    max_latency: float = 0.1  # seconds
    enable_backpressure: bool = True
    enable_metrics: bool = True

@dataclass
class DataPoint:
    """Individual data point in the stream."""
    id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class ProcessingResult:
    """Result of processing a data point."""
    id: str
    input_data: DataPoint
    prediction: Any
    confidence: float
    processing_time: float
    timestamp: datetime
    model_version: str

class StreamBuffer:
    """Thread-safe buffer for streaming data."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self._dropped_count = 0
    
    def put(self, item: DataPoint) -> bool:
        """Add item to buffer. Returns True if successful, False if dropped."""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self._dropped_count += 1
                return False
            
            self.buffer.append(item)
            return True
    
    def get_batch(self, batch_size: int) -> List[DataPoint]:
        """Get a batch of items from buffer."""
        with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.popleft())
            return batch
    
    def size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)
    
    def dropped_count(self) -> int:
        """Get number of dropped items."""
        return self._dropped_count
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()

class StreamProcessor(ABC):
    """Abstract base class for stream processors."""
    
    @abstractmethod
    async def process(self, data_point: DataPoint) -> ProcessingResult:
        """Process a single data point."""
        pass
    
    @abstractmethod
    async def process_batch(self, batch: List[DataPoint]) -> List[ProcessingResult]:
        """Process a batch of data points."""
        pass

class MLModelProcessor(StreamProcessor):
    """Stream processor that applies ML model predictions."""
    
    def __init__(self, model: Any, preprocessor: Optional[Callable] = None,
                 model_version: str = "1.0"):
        self.model = model
        self.preprocessor = preprocessor
        self.model_version = model_version
    
    async def process(self, data_point: DataPoint) -> ProcessingResult:
        """Process single data point."""
        start_time = time.time()
        
        try:
            # Preprocess data
            if self.preprocessor:
                processed_data = self.preprocessor(data_point.data)
            else:
                processed_data = data_point.data
            
            # Make prediction
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict([processed_data])[0]
                confidence = getattr(self.model, 'predict_proba', lambda x: [0.5])([processed_data])[0].max()
            else:
                # Assume callable model
                result = self.model(processed_data)
                prediction = result if not isinstance(result, tuple) else result[0]
                confidence = result[1] if isinstance(result, tuple) and len(result) > 1 else 0.5
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                id=data_point.id,
                input_data=data_point,
                prediction=prediction,
                confidence=float(confidence),
                processing_time=processing_time,
                timestamp=datetime.now(),
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Error processing data point {data_point.id}: {e}")
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                id=data_point.id,
                input_data=data_point,
                prediction=None,
                confidence=0.0,
                processing_time=processing_time,
                timestamp=datetime.now(),
                model_version=self.model_version
            )
    
    async def process_batch(self, batch: List[DataPoint]) -> List[ProcessingResult]:
        """Process batch of data points."""
        if not batch:
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # Preprocess batch
            if self.preprocessor:
                processed_batch = [self.preprocessor(dp.data) for dp in batch]
            else:
                processed_batch = [dp.data for dp in batch]
            
            # Batch prediction
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(processed_batch)
                confidences = getattr(self.model, 'predict_proba', 
                                    lambda x: [[0.5] * len(x)])(processed_batch)
                if hasattr(confidences[0], 'max'):
                    confidences = [conf.max() for conf in confidences]
                else:
                    confidences = [0.5] * len(predictions)
            else:
                # Process individually for non-batch models
                predictions = []
                confidences = []
                for data in processed_batch:
                    result = self.model(data)
                    pred = result if not isinstance(result, tuple) else result[0]
                    conf = result[1] if isinstance(result, tuple) and len(result) > 1 else 0.5
                    predictions.append(pred)
                    confidences.append(conf)
            
            processing_time = time.time() - start_time
            timestamp = datetime.now()
            
            for i, data_point in enumerate(batch):
                results.append(ProcessingResult(
                    id=data_point.id,
                    input_data=data_point,
                    prediction=predictions[i],
                    confidence=float(confidences[i]),
                    processing_time=processing_time / len(batch),
                    timestamp=timestamp,
                    model_version=self.model_version
                ))
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return error results for all items in batch
            processing_time = time.time() - start_time
            timestamp = datetime.now()
            
            for data_point in batch:
                results.append(ProcessingResult(
                    id=data_point.id,
                    input_data=data_point,
                    prediction=None,
                    confidence=0.0,
                    processing_time=processing_time / len(batch),
                    timestamp=timestamp,
                    model_version=self.model_version
                ))
        
        return results

class StreamMetrics:
    """Metrics collector for streaming pipeline."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)
        self.error_count = 0
        self.total_processed = 0
        self.last_throughput_time = time.time()
        self.last_throughput_count = 0
        self.lock = threading.RLock()
    
    def record_processing_time(self, processing_time: float):
        """Record processing time for a data point."""
        with self.lock:
            self.processing_times.append(processing_time)
            self.total_processed += 1
    
    def record_error(self):
        """Record an error."""
        with self.lock:
            self.error_count += 1
    
    def update_throughput(self, count: int):
        """Update throughput metrics."""
        with self.lock:
            current_time = time.time()
            time_diff = current_time - self.last_throughput_time
            
            if time_diff >= 1.0:  # Update every second
                throughput = (count - self.last_throughput_count) / time_diff
                self.throughput_history.append(throughput)
                self.last_throughput_time = current_time
                self.last_throughput_count = count
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        with self.lock:
            if not self.processing_times:
                return {
                    'avg_processing_time': 0.0,
                    'p95_processing_time': 0.0,
                    'p99_processing_time': 0.0,
                    'throughput': 0.0,
                    'error_rate': 0.0,
                    'total_processed': self.total_processed
                }
            
            sorted_times = sorted(self.processing_times)
            n = len(sorted_times)
            
            avg_processing_time = sum(sorted_times) / n
            p95_processing_time = sorted_times[int(n * 0.95)] if n > 0 else 0.0
            p99_processing_time = sorted_times[int(n * 0.99)] if n > 0 else 0.0
            
            current_throughput = sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0.0
            error_rate = self.error_count / max(self.total_processed, 1)
            
            return {
                'avg_processing_time': avg_processing_time,
                'p95_processing_time': p95_processing_time,
                'p99_processing_time': p99_processing_time,
                'throughput': current_throughput,
                'error_rate': error_rate,
                'total_processed': self.total_processed
            }

class StreamingPipeline:
    """Main streaming ML pipeline orchestrator."""
    
    def __init__(self, processor: StreamProcessor, config: StreamingConfig):
        self.processor = processor
        self.config = config
        self.buffer = StreamBuffer(config.buffer_size)
        self.metrics = StreamMetrics() if config.enable_metrics else None
        self.running = False
        self.processing_task = None
        self.output_handlers = []
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def add_output_handler(self, handler: Callable[[ProcessingResult], None]):
        """Add output handler for processing results."""
        self.output_handlers.append(handler)
    
    async def put(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put data into the streaming pipeline."""
        if not self.running:
            return False
        
        data_point = DataPoint(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            data=data,
            metadata=metadata
        )
        
        success = self.buffer.put(data_point)
        if not success and self.config.enable_backpressure:
            logger.warning("Buffer full, dropping data point")
        
        return success
    
    async def start(self):
        """Start the streaming pipeline."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting streaming pipeline")
        
        # Start processing loop
        self.processing_task = asyncio.create_task(self._processing_loop())
    
    async def stop(self):
        """Stop the streaming pipeline."""
        if not self.running:
            return
        
        logger.info("Stopping streaming pipeline")
        self.running = False
        
        if self.processing_task:
            await self.processing_task
        
        self.executor.shutdown(wait=True)
    
    async def _processing_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                # Get batch from buffer
                batch = self.buffer.get_batch(self.config.batch_size)
                
                if batch:
                    start_time = time.time()
                    
                    # Process batch
                    results = await self.processor.process_batch(batch)
                    
                    # Handle results
                    for result in results:
                        # Update metrics
                        if self.metrics:
                            self.metrics.record_processing_time(result.processing_time)
                            if result.prediction is None:
                                self.metrics.record_error()
                        
                        # Send to output handlers
                        for handler in self.output_handlers:
                            try:
                                await asyncio.get_event_loop().run_in_executor(
                                    self.executor, handler, result
                                )
                            except Exception as e:
                                logger.error(f"Error in output handler: {e}")
                    
                    # Update throughput metrics
                    if self.metrics:
                        self.metrics.update_throughput(len(results))
                    
                    # Check latency constraint
                    total_time = time.time() - start_time
                    if total_time > self.config.max_latency:
                        logger.warning(f"Processing latency {total_time:.3f}s exceeds limit {self.config.max_latency:.3f}s")
                
                # Sleep to control processing interval
                await asyncio.sleep(self.config.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)  # Error backoff
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get pipeline metrics."""
        if not self.metrics:
            return None
        
        pipeline_metrics = {
            'buffer_size': self.buffer.size(),
            'buffer_dropped': self.buffer.dropped_count(),
            'buffer_utilization': self.buffer.size() / self.config.buffer_size
        }
        
        processing_metrics = self.metrics.get_metrics()
        pipeline_metrics.update(processing_metrics)
        
        return pipeline_metrics

class AnomalyDetectionProcessor(StreamProcessor):
    """Stream processor for real-time anomaly detection."""
    
    def __init__(self, window_size: int = 100, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=window_size)
    
    async def process(self, data_point: DataPoint) -> ProcessingResult:
        """Detect anomalies in streaming data."""
        start_time = time.time()
        
        try:
            # Extract numeric features
            features = self._extract_features(data_point.data)
            
            # Calculate anomaly score
            if len(self.history) < 10:  # Need minimum history
                anomaly_score = 0.0
                is_anomaly = False
            else:
                anomaly_score = self._calculate_anomaly_score(features)
                is_anomaly = anomaly_score > self.threshold
            
            # Update history
            self.history.append(features)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                id=data_point.id,
                input_data=data_point,
                prediction={
                    'is_anomaly': is_anomaly,
                    'anomaly_score': anomaly_score
                },
                confidence=min(anomaly_score / self.threshold, 1.0),
                processing_time=processing_time,
                timestamp=datetime.now(),
                model_version="anomaly_detector_1.0"
            )
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return ProcessingResult(
                id=data_point.id,
                input_data=data_point,
                prediction={'is_anomaly': False, 'anomaly_score': 0.0},
                confidence=0.0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now(),
                model_version="anomaly_detector_1.0"
            )
    
    async def process_batch(self, batch: List[DataPoint]) -> List[ProcessingResult]:
        """Process batch for anomaly detection."""
        results = []
        for data_point in batch:
            result = await self.process(data_point)
            results.append(result)
        return results
    
    def _extract_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract numeric features from data."""
        features = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                features.extend([float(x) for x in value])
        return features
    
    def _calculate_anomaly_score(self, features: List[float]) -> float:
        """Calculate anomaly score using z-score method."""
        if not features or not self.history:
            return 0.0
        
        # Calculate mean and std from history
        feature_history = list(self.history)
        if not feature_history:
            return 0.0
        
        # Assume all feature vectors have same length
        feature_dim = len(features)
        if not all(len(f) == feature_dim for f in feature_history):
            return 0.0
        
        anomaly_scores = []
        for i in range(feature_dim):
            historical_values = [f[i] for f in feature_history]
            mean_val = sum(historical_values) / len(historical_values)
            var_val = sum((x - mean_val) ** 2 for x in historical_values) / len(historical_values)
            std_val = var_val ** 0.5
            
            if std_val > 0:
                z_score = abs(features[i] - mean_val) / std_val
                anomaly_scores.append(z_score)
        
        return max(anomaly_scores) if anomaly_scores else 0.0

# Factory functions and utilities
def create_simple_pipeline(model: Any, config: Optional[StreamingConfig] = None) -> StreamingPipeline:
    """Create a simple streaming pipeline with an ML model."""
    if config is None:
        config = StreamingConfig()
    
    processor = MLModelProcessor(model)
    return StreamingPipeline(processor, config)

def create_anomaly_pipeline(config: Optional[StreamingConfig] = None) -> StreamingPipeline:
    """Create an anomaly detection pipeline."""
    if config is None:
        config = StreamingConfig()
    
    processor = AnomalyDetectionProcessor()
    return StreamingPipeline(processor, config)

async def demo_streaming_pipeline():
    """Demonstrate streaming pipeline capabilities."""
    # Create a simple mock model
    class MockModel:
        def predict(self, X):
            return [sum(x.values()) if isinstance(x, dict) else sum(x) for x in X]
        
        def predict_proba(self, X):
            return [[0.3, 0.7] for _ in X]
    
    # Create pipeline
    model = MockModel()
    config = StreamingConfig(batch_size=5, processing_interval=0.5)
    pipeline = create_simple_pipeline(model, config)
    
    # Add output handler
    def print_result(result: ProcessingResult):
        print(f"Processed {result.id}: prediction={result.prediction}, confidence={result.confidence:.2f}")
    
    pipeline.add_output_handler(print_result)
    
    # Start pipeline
    await pipeline.start()
    
    # Send some data
    for i in range(20):
        data = {'feature1': i, 'feature2': i * 2, 'feature3': i ** 0.5}
        await pipeline.put(data)
        await asyncio.sleep(0.1)
    
    # Wait for processing
    await asyncio.sleep(3)
    
    # Print metrics
    metrics = pipeline.get_metrics()
    if metrics:
        print(f"Pipeline metrics: {metrics}")
    
    # Stop pipeline
    await pipeline.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_streaming_pipeline())
