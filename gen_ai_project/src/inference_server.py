"""
Real-time model inference server with streaming capabilities.
Supports multiple model formats and provides efficient batch processing.
"""

import asyncio
import json
import logging
from typing import List, Optional, Union
from datetime import datetime
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from pydantic import BaseModel
import redis
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    model_name: str
    version: str = "latest"
    inputs: Union[List[float], List[List[float]]]
    batch_size: Optional[int] = 1
    timeout: Optional[float] = 30.0

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float]
    model_name: str
    version: str
    inference_time: float
    timestamp: str

class ModelCache:
    """Intelligent model caching with LRU eviction."""
    
    def __init__(self, max_models: int = 5):
        self.max_models = max_models
        self.models = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    def get_model(self, model_name: str, version: str):
        """Get model from cache or load if not present."""
        key = f"{model_name}:{version}"
        
        with self.lock:
            if key in self.models:
                self.access_times[key] = datetime.now()
                return self.models[key]
        
        # Load model (placeholder - implement actual loading logic)
        model = self._load_model(model_name, version)
        
        with self.lock:
            # Evict least recently used model if cache is full
            if len(self.models) >= self.max_models:
                lru_key = min(self.access_times.keys(), key=self.access_times.get)
                del self.models[lru_key]
                del self.access_times[lru_key]
                logger.info(f"Evicted model {lru_key} from cache")
            
            self.models[key] = model
            self.access_times[key] = datetime.now()
            logger.info(f"Loaded model {key} into cache")
        
        return model
    
    def _load_model(self, model_name: str, version: str):
        """Load model from storage (placeholder implementation)."""
        # In real implementation, this would load from S3, MLflow, etc.
        logger.info(f"Loading model {model_name}:{version}")
        
        # Simulate model loading
        if model_name == "classifier":
            # Return a dummy model for demonstration
            class DummyModel:
                def predict(self, X):
                    return np.random.random(len(X))
            return DummyModel()
        
        raise ValueError(f"Unknown model: {model_name}")

class BatchProcessor:
    """Batch processing for efficient model inference."""
    
    def __init__(self, batch_timeout: float = 0.1, max_batch_size: int = 32):
        self.batch_timeout = batch_timeout
        self.max_batch_size = max_batch_size
        self.request_queue = queue.Queue()
        self.response_futures = {}
        self.processing = False
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def start_processing(self):
        """Start the batch processing loop."""
        if not self.processing:
            self.processing = True
            self.executor.submit(self._process_batches)
    
    def stop_processing(self):
        """Stop the batch processing loop."""
        self.processing = False
    
    async def add_request(self, request: PredictionRequest, model) -> PredictionResponse:
        """Add a request to the batch queue."""
        future = asyncio.Future()
        request_id = id(request)
        
        self.response_futures[request_id] = future
        self.request_queue.put((request_id, request, model))
        
        return await future
    
    def _process_batches(self):
        """Process requests in batches."""
        while self.processing:
            batch = []
            batch_models = []
            
            # Collect requests for batch
            start_time = datetime.now()
            while (len(batch) < self.max_batch_size and 
                   (datetime.now() - start_time).total_seconds() < self.batch_timeout):
                try:
                    request_id, request, model = self.request_queue.get(timeout=0.01)
                    batch.append((request_id, request))
                    batch_models.append(model)
                except queue.Empty:
                    if batch:  # Process partial batch if timeout reached
                        break
                    continue
            
            if batch:
                self._process_batch(batch, batch_models)
    
    def _process_batch(self, batch: List, models: List):
        """Process a batch of requests."""
        try:
            # Group requests by model for efficient batching
            model_groups = {}
            for (request_id, request), model in zip(batch, models):
                model_key = f"{request.model_name}:{request.version}"
                if model_key not in model_groups:
                    model_groups[model_key] = {"model": model, "requests": []}
                model_groups[model_key]["requests"].append((request_id, request))
            
            # Process each model group
            for model_key, group in model_groups.items():
                self._process_model_group(group["model"], group["requests"])
                
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            # Send error to all requests in batch
            for request_id, _ in batch:
                if request_id in self.response_futures:
                    future = self.response_futures.pop(request_id)
                    if not future.done():
                        future.set_exception(e)
    
    def _process_model_group(self, model, requests: List):
        """Process requests for a single model."""
        start_time = datetime.now()
        
        # Collect all inputs
        all_inputs = []
        for request_id, request in requests:
            all_inputs.extend(request.inputs)
        
        # Make batch prediction
        batch_predictions = model.predict(np.array(all_inputs))
        
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Distribute results back to individual requests
        pred_index = 0
        for request_id, request in requests:
            num_samples = len(request.inputs)
            predictions = batch_predictions[pred_index:pred_index + num_samples].tolist()
            pred_index += num_samples
            
            response = PredictionResponse(
                predictions=predictions,
                model_name=request.model_name,
                version=request.version,
                inference_time=inference_time,
                timestamp=datetime.now().isoformat()
            )
            
            if request_id in self.response_futures:
                future = self.response_futures.pop(request_id)
                if not future.done():
                    future.set_result(response)

class StreamingInference:
    """Streaming inference for real-time predictions."""
    
    def __init__(self, model_cache: ModelCache):
        self.model_cache = model_cache
        self.active_connections = {}
    
    async def handle_connection(self, websocket: WebSocket, model_name: str, version: str):
        """Handle a WebSocket connection for streaming inference."""
        await websocket.accept()
        connection_id = id(websocket)
        self.active_connections[connection_id] = websocket
        
        try:
            model = self.model_cache.get_model(model_name, version)
            
            while True:
                # Receive data from client
                data = await websocket.receive_text()
                request_data = json.loads(data)
                
                # Process prediction
                inputs = np.array(request_data["inputs"])
                start_time = datetime.now()
                predictions = model.predict(inputs)
                inference_time = (datetime.now() - start_time).total_seconds()
                
                # Send response
                response = {
                    "predictions": predictions.tolist(),
                    "inference_time": inference_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send_text(json.dumps(response))
                
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

# Initialize components
app = FastAPI(title="Real-time Inference Server", version="1.0.0")
model_cache = ModelCache(max_models=10)
batch_processor = BatchProcessor()
streaming_inference = StreamingInference(model_cache)

# Redis for caching predictions (optional)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - caching disabled")

@app.on_event("startup")
async def startup_event():
    """Start background processing on server startup."""
    batch_processor.start_processing()
    logger.info("Inference server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of background processes."""
    batch_processor.stop_processing()
    logger.info("Inference server stopped")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make predictions with optional batching."""
    try:
        # Get model from cache
        model = model_cache.get_model(request.model_name, request.version)
        
        # Check cache for similar requests
        cache_key = None
        if REDIS_AVAILABLE:
            cache_key = f"pred:{request.model_name}:{hash(str(request.inputs))}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info("Returning cached prediction")
                return PredictionResponse.parse_raw(cached_result)
        
        # Use batch processor for efficiency
        response = await batch_processor.add_request(request, model)
        
        # Cache the result
        if REDIS_AVAILABLE and cache_key:
            background_tasks.add_task(
                redis_client.setex, 
                cache_key, 
                300,  # 5 minutes TTL
                response.json()
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/stream/{model_name}/{version}")
async def websocket_endpoint(websocket: WebSocket, model_name: str, version: str):
    """WebSocket endpoint for streaming inference."""
    await streaming_inference.handle_connection(websocket, model_name, version)

@app.get("/models")
async def list_models():
    """List available models in cache."""
    with model_cache.lock:
        models = list(model_cache.models.keys())
    
    return {"cached_models": models}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cached_models": len(model_cache.models),
        "active_connections": len(streaming_inference.active_connections)
    }

@app.get("/metrics")
async def get_metrics():
    """Get server metrics."""
    return {
        "cache_size": len(model_cache.models),
        "queue_size": batch_processor.request_queue.qsize(),
        "active_websockets": len(streaming_inference.active_connections),
        "redis_available": REDIS_AVAILABLE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
