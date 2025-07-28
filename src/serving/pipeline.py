"""
Real-time Model Serving Pipeline
High-performance model serving infrastructure with A/B testing and monitoring.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import queue
import logging
from contextlib import asynccontextmanager

@dataclass
class PredictionRequest:
    """Request object for model predictions."""
    request_id: str
    data: Any
    model_version: str = "latest"
    timestamp: float = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PredictionResponse:
    """Response object for model predictions."""
    request_id: str
    prediction: Any
    confidence: float
    model_version: str
    latency_ms: float
    timestamp: float
    metadata: Dict[str, Any] = None

class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    @abstractmethod
    async def predict(self, data: Any) -> Any:
        """Make a prediction."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass

class TorchModelWrapper(ModelWrapper):
    """Wrapper for PyTorch models."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    async def predict(self, data: Any) -> Any:
        """Make prediction with PyTorch model."""
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif isinstance(data, list):
                data = torch.tensor(data).float()
            
            data = data.to(self.device)
            prediction = self.model(data)
            
            # Convert back to numpy for serialization
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu().numpy()
            
            return prediction
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get PyTorch model information."""
        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "type": "pytorch",
            "device": self.device,
            "num_parameters": num_params,
            "model_class": self.model.__class__.__name__
        }

class RequestBatcher:
    """Batches requests for efficient processing."""
    
    def __init__(self, 
                 max_batch_size: int = 32,
                 max_wait_time: float = 0.01,
                 max_queue_size: int = 1000):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.response_futures = {}
        self.is_running = False
        self.batch_thread = None
    
    def start(self, prediction_func: Callable):
        """Start the batching service."""
        self.is_running = True
        self.batch_thread = threading.Thread(
            target=self._batch_processing_loop,
            args=(prediction_func,)
        )
        self.batch_thread.start()
    
    def stop(self):
        """Stop the batching service."""
        self.is_running = False
        if self.batch_thread:
            self.batch_thread.join()
    
    async def add_request(self, request: PredictionRequest) -> PredictionResponse:
        """Add a request to the batch queue."""
        future = asyncio.Future()
        self.response_futures[request.request_id] = future
        
        try:
            self.request_queue.put(request, timeout=1.0)
            return await future
        except queue.Full:
            del self.response_futures[request.request_id]
            raise Exception("Request queue is full")
    
    def _batch_processing_loop(self, prediction_func: Callable):
        """Main batching loop."""
        while self.is_running:
            batch_requests = []
            start_time = time.time()
            
            # Collect requests for batch
            while (len(batch_requests) < self.max_batch_size and
                   time.time() - start_time < self.max_wait_time):
                try:
                    request = self.request_queue.get(timeout=0.001)
                    batch_requests.append(request)
                except queue.Empty:
                    if batch_requests:
                        break
                    continue
            
            if batch_requests:
                self._process_batch(batch_requests, prediction_func)
    
    def _process_batch(self, batch_requests: List[PredictionRequest], prediction_func: Callable):
        """Process a batch of requests."""
        try:
            # Prepare batch data
            batch_data = [req.data for req in batch_requests]
            
            # Make batch prediction
            batch_predictions = prediction_func(batch_data)
            
            # Send responses
            for request, prediction in zip(batch_requests, batch_predictions):
                if request.request_id in self.response_futures:
                    response = PredictionResponse(
                        request_id=request.request_id,
                        prediction=prediction,
                        confidence=0.9,  # Placeholder
                        model_version=request.model_version,
                        latency_ms=(time.time() - request.timestamp) * 1000,
                        timestamp=time.time()
                    )
                    
                    future = self.response_futures.pop(request.request_id)
                    if not future.done():
                        future.set_result(response)
        
        except Exception as e:
            # Handle batch processing errors
            for request in batch_requests:
                if request.request_id in self.response_futures:
                    future = self.response_futures.pop(request.request_id)
                    if not future.done():
                        future.set_exception(e)

class ABTestManager:
    """A/B testing manager for model variants."""
    
    def __init__(self):
        self.models = {}
        self.traffic_splits = {}
        self.experiment_configs = {}
    
    def add_model_variant(self, 
                         variant_name: str, 
                         model: ModelWrapper,
                         traffic_percentage: float = 0.0):
        """Add a model variant for A/B testing."""
        self.models[variant_name] = model
        self.traffic_splits[variant_name] = traffic_percentage
    
    def configure_experiment(self, 
                           experiment_name: str,
                           variants: Dict[str, float],
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None):
        """Configure an A/B test experiment."""
        # Normalize traffic splits
        total_traffic = sum(variants.values())
        normalized_variants = {
            variant: percentage / total_traffic 
            for variant, percentage in variants.items()
        }
        
        self.experiment_configs[experiment_name] = {
            "variants": normalized_variants,
            "start_time": start_time or time.time(),
            "end_time": end_time,
            "active": True
        }
    
    def select_model_variant(self, request: PredictionRequest) -> str:
        """Select model variant based on A/B testing rules."""
        # Use request ID hash for consistent routing
        hash_value = hash(request.request_id) % 100
        
        cumulative_percentage = 0
        for variant, percentage in self.traffic_splits.items():
            cumulative_percentage += percentage * 100
            if hash_value < cumulative_percentage:
                return variant
        
        # Default to first available model
        return list(self.models.keys())[0] if self.models else None

class PerformanceMonitor:
    """Monitors model serving performance."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = {
            "latencies": [],
            "throughput": [],
            "error_count": 0,
            "total_requests": 0
        }
        self.lock = threading.Lock()
    
    def record_request(self, response: PredictionResponse, error: bool = False):
        """Record request metrics."""
        with self.lock:
            self.metrics["total_requests"] += 1
            
            if error:
                self.metrics["error_count"] += 1
            else:
                self.metrics["latencies"].append(response.latency_ms)
                
                # Keep only recent latencies
                if len(self.metrics["latencies"]) > self.window_size:
                    self.metrics["latencies"] = self.metrics["latencies"][-self.window_size:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            if not self.metrics["latencies"]:
                return {
                    "avg_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "p99_latency_ms": 0,
                    "error_rate": 0,
                    "total_requests": self.metrics["total_requests"]
                }
            
            latencies = self.metrics["latencies"]
            sorted_latencies = sorted(latencies)
            
            return {
                "avg_latency_ms": np.mean(latencies),
                "p95_latency_ms": np.percentile(sorted_latencies, 95),
                "p99_latency_ms": np.percentile(sorted_latencies, 99),
                "error_rate": self.metrics["error_count"] / self.metrics["total_requests"],
                "total_requests": self.metrics["total_requests"],
                "current_qps": len(latencies) / (self.window_size / 1000) if latencies else 0
            }

class ModelServingPipeline:
    """Main model serving pipeline with all features."""
    
    def __init__(self, 
                 default_model: ModelWrapper,
                 enable_batching: bool = True,
                 batch_config: Dict[str, Any] = None):
        
        self.default_model = default_model
        self.ab_test_manager = ABTestManager()
        self.performance_monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Add default model
        self.ab_test_manager.add_model_variant("default", default_model, 100.0)
        
        # Initialize batching
        self.enable_batching = enable_batching
        if enable_batching:
            batch_config = batch_config or {}
            self.batcher = RequestBatcher(
                max_batch_size=batch_config.get("max_batch_size", 32),
                max_wait_time=batch_config.get("max_wait_time", 0.01),
                max_queue_size=batch_config.get("max_queue_size", 1000)
            )
            self.batcher.start(self._batch_predict)
    
    def add_model_variant(self, variant_name: str, model: ModelWrapper, traffic_percentage: float = 0.0):
        """Add a new model variant."""
        self.ab_test_manager.add_model_variant(variant_name, model, traffic_percentage)
    
    def start_ab_test(self, experiment_name: str, variants: Dict[str, float]):
        """Start an A/B test."""
        self.ab_test_manager.configure_experiment(experiment_name, variants)
        self.logger.info(f"Started A/B test: {experiment_name} with variants: {variants}")
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make a prediction through the serving pipeline."""
        start_time = time.time()
        
        try:
            if self.enable_batching:
                response = await self.batcher.add_request(request)
            else:
                # Select model variant
                variant_name = self.ab_test_manager.select_model_variant(request)
                model = self.ab_test_manager.models[variant_name]
                
                # Make prediction
                prediction = await model.predict(request.data)
                
                response = PredictionResponse(
                    request_id=request.request_id,
                    prediction=prediction,
                    confidence=0.9,  # Placeholder
                    model_version=variant_name,
                    latency_ms=(time.time() - start_time) * 1000,
                    timestamp=time.time()
                )
            
            # Record metrics
            self.performance_monitor.record_request(response, error=False)
            return response
            
        except Exception as e:
            # Record error
            self.performance_monitor.record_request(None, error=True)
            self.logger.error(f"Prediction error for request {request.request_id}: {e}")
            raise
    
    def _batch_predict(self, batch_data: List[Any]) -> List[Any]:
        """Batch prediction function."""
        # For simplicity, use default model for batching
        model = self.ab_test_manager.models["default"]
        
        # Stack batch data
        if isinstance(batch_data[0], np.ndarray):
            stacked_data = np.stack(batch_data)
        elif isinstance(batch_data[0], list):
            stacked_data = np.array(batch_data)
        else:
            stacked_data = batch_data
        
        # Make batch prediction
        batch_predictions = asyncio.run(model.predict(stacked_data))
        
        # Split back into individual predictions
        if isinstance(batch_predictions, np.ndarray):
            return [batch_predictions[i] for i in range(len(batch_data))]
        else:
            return batch_predictions
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the serving pipeline."""
        metrics = self.performance_monitor.get_metrics()
        return {
            "status": "healthy" if metrics["error_rate"] < 0.01 else "degraded",
            "metrics": metrics,
            "available_models": list(self.ab_test_manager.models.keys()),
            "active_experiments": list(self.ab_test_manager.experiment_configs.keys())
        }
    
    def shutdown(self):
        """Shutdown the serving pipeline."""
        if self.enable_batching and hasattr(self, 'batcher'):
            self.batcher.stop()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    async def main():
        print("Testing Real-time Model Serving Pipeline...")
        
        # Initialize model and wrapper
        model = SimpleModel()
        model_wrapper = TorchModelWrapper(model)
        
        # Create serving pipeline
        pipeline = ModelServingPipeline(
            default_model=model_wrapper,
            enable_batching=True,
            batch_config={"max_batch_size": 16, "max_wait_time": 0.01}
        )
        
        # Add a variant model for A/B testing
        variant_model = SimpleModel()
        variant_wrapper = TorchModelWrapper(variant_model)
        pipeline.add_model_variant("variant_1", variant_wrapper, 20.0)
        
        # Start A/B test
        pipeline.start_ab_test("test_experiment", {"default": 80.0, "variant_1": 20.0})
        
        print("Pipeline initialized. Making test predictions...")
        
        # Make test predictions
        tasks = []
        for i in range(10):
            request = PredictionRequest(
                request_id=f"req_{i}",
                data=np.random.randn(10).astype(np.float32)
            )
            tasks.append(pipeline.predict(request))
        
        # Execute all predictions
        responses = await asyncio.gather(*tasks)
        
        print(f"Completed {len(responses)} predictions")
        for response in responses[:3]:  # Show first 3
            print(f"Request {response.request_id}: "
                  f"Model {response.model_version}, "
                  f"Latency {response.latency_ms:.2f}ms")
        
        # Get health status
        health = pipeline.get_health_status()
        print(f"\nHealth Status: {health['status']}")
        print(f"Metrics: {health['metrics']}")
        
        # Shutdown
        pipeline.shutdown()
        print("\nReal-time Model Serving Pipeline implemented successfully! 🚀")
    
    # Run the example
    asyncio.run(main())
