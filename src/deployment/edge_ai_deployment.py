"""
Edge AI Deployment Module
Implements edge computing solutions for ML model deployment with optimization and compression.
"""

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class EdgeDeviceSpec:
    """Specifications for edge deployment target."""
    device_type: str  # 'mobile', 'iot', 'embedded', 'edge_server'
    memory_mb: int
    compute_units: float
    power_budget_watts: float
    latency_requirement_ms: float
    throughput_requirement: float

class ModelOptimizer(ABC):
    """Abstract base class for model optimization techniques."""
    
    @abstractmethod
    def optimize(self, model: Any, target_spec: EdgeDeviceSpec) -> Any:
        """Optimize model for target device specifications."""
        pass

class QuantizationOptimizer(ModelOptimizer):
    """Implements various quantization techniques for model compression."""
    
    def __init__(self, quantization_type: str = "dynamic"):
        self.quantization_type = quantization_type
    
    def optimize(self, model: torch.nn.Module, target_spec: EdgeDeviceSpec) -> torch.nn.Module:
        """Apply quantization to reduce model size and improve inference speed."""
        if self.quantization_type == "dynamic":
            return torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
        elif self.quantization_type == "static":
            return self._apply_static_quantization(model, target_spec)
        elif self.quantization_type == "qat":
            return self._apply_qat(model, target_spec)
        else:
            raise ValueError(f"Unknown quantization type: {self.quantization_type}")
    
    def _apply_static_quantization(self, model: torch.nn.Module, target_spec: EdgeDeviceSpec) -> torch.nn.Module:
        """Apply post-training static quantization."""
        model.eval()
        model_fp32 = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
        model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(model_fp32, inplace=True)
        # Calibration would happen here with representative data
        model_int8 = torch.quantization.convert(model_fp32, inplace=False)
        return model_int8
    
    def _apply_qat(self, model: torch.nn.Module, target_spec: EdgeDeviceSpec) -> torch.nn.Module:
        """Apply quantization-aware training."""
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare_qat(model, inplace=True)
        # QAT training would happen here
        model.eval()
        return torch.quantization.convert(model, inplace=True)

class PruningOptimizer(ModelOptimizer):
    """Implements neural network pruning for model compression."""
    
    def __init__(self, sparsity: float = 0.5, structured: bool = False):
        self.sparsity = sparsity
        self.structured = structured
    
    def optimize(self, model: torch.nn.Module, target_spec: EdgeDeviceSpec) -> torch.nn.Module:
        """Apply pruning to reduce model parameters."""
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        if self.structured:
            prune.global_structured(
                parameters_to_prune,
                pruning_method=prune.L1StructuredPruning,
                amount=self.sparsity,
                dim=0
            )
        else:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.sparsity,
            )
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model

class KnowledgeDistillationOptimizer(ModelOptimizer):
    """Implements knowledge distillation for model compression."""
    
    def __init__(self, teacher_model: torch.nn.Module, temperature: float = 4.0):
        self.teacher_model = teacher_model
        self.temperature = temperature
    
    def optimize(self, student_model: torch.nn.Module, target_spec: EdgeDeviceSpec) -> torch.nn.Module:
        """Train student model using knowledge distillation."""
        # This is a simplified version - full implementation would include training loop
        logger.info(f"Knowledge distillation with temperature {self.temperature}")
        return student_model

class EdgeModelConverter:
    """Converts models to edge-optimized formats."""
    
    def __init__(self):
        self.supported_formats = ['onnx', 'tflite', 'tensorrt', 'openvino']
    
    def convert_to_onnx(self, model: torch.nn.Module, input_shape: Tuple[int, ...], 
                       output_path: str) -> str:
        """Convert PyTorch model to ONNX format."""
        dummy_input = torch.randn(1, *input_shape)
        model.eval()
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        # Optimize ONNX model
        self._optimize_onnx_model(output_path)
        return output_path
    
    def _optimize_onnx_model(self, model_path: str):
        """Optimize ONNX model for edge deployment."""
        # Basic ONNX optimization - could be enhanced with onnxoptimizer if available
        model = onnx.load(model_path)
        # Perform basic optimizations
        onnx.checker.check_model(model)
        onnx.save(model, model_path)

class EdgeInferenceEngine:
    """High-performance inference engine for edge devices."""
    
    def __init__(self, model_path: str, device_spec: EdgeDeviceSpec):
        self.model_path = model_path
        self.device_spec = device_spec
        self.session = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the inference engine based on model format."""
        if self.model_path.endswith('.onnx'):
            providers = self._get_optimal_providers()
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")
    
    def _get_optimal_providers(self) -> List[str]:
        """Select optimal execution providers based on device specs."""
        if self.device_spec.device_type == 'mobile':
            return ['CPUExecutionProvider']
        elif self.device_spec.device_type == 'edge_server':
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        if self.session is None:
            raise RuntimeError("Inference engine not initialized")
        
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        start_time = time.time()
        result = self.session.run([output_name], {input_name: input_data})
        inference_time = time.time() - start_time
        
        # Check latency constraint
        if inference_time * 1000 > self.device_spec.latency_requirement_ms:
            logger.warning(f"Inference time ({inference_time*1000:.2f}ms) exceeds requirement")
        
        return result[0]
    
    def benchmark(self, input_shape: Tuple[int, ...], num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.predict(dummy_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            self.predict(dummy_input)
        total_time = time.time() - start_time
        
        avg_latency = (total_time / num_iterations) * 1000  # ms
        throughput = num_iterations / total_time  # inferences/sec
        
        return {
            'avg_latency_ms': avg_latency,
            'throughput_ips': throughput,
            'meets_latency_req': avg_latency <= self.device_spec.latency_requirement_ms,
            'meets_throughput_req': throughput >= self.device_spec.throughput_requirement
        }

class EdgeDeploymentManager:
    """Main class for managing edge AI deployments."""
    
    def __init__(self):
        self.optimizers = {
            'quantization': QuantizationOptimizer(),
            'pruning': PruningOptimizer(),
            'distillation': KnowledgeDistillationOptimizer
        }
        self.converter = EdgeModelConverter()
    
    def optimize_for_edge(self, model: torch.nn.Module, target_spec: EdgeDeviceSpec,
                         optimization_strategy: List[str]) -> torch.nn.Module:
        """Optimize model for edge deployment using specified strategies."""
        optimized_model = model
        
        for strategy in optimization_strategy:
            if strategy in self.optimizers:
                if strategy == 'distillation':
                    # Would need teacher model for distillation
                    continue
                optimized_model = self.optimizers[strategy].optimize(optimized_model, target_spec)
                logger.info(f"Applied {strategy} optimization")
        
        return optimized_model
    
    def deploy_to_edge(self, model: torch.nn.Module, target_spec: EdgeDeviceSpec,
                      input_shape: Tuple[int, ...], model_name: str) -> EdgeInferenceEngine:
        """Complete pipeline for edge deployment."""
        # Optimize model
        optimization_strategies = self._select_optimization_strategies(target_spec)
        optimized_model = self.optimize_for_edge(model, target_spec, optimization_strategies)
        
        # Convert to edge format
        output_path = f"models/{model_name}_edge.onnx"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        onnx_path = self.converter.convert_to_onnx(optimized_model, input_shape, output_path)
        
        # Create inference engine
        engine = EdgeInferenceEngine(onnx_path, target_spec)
        
        # Validate deployment
        self._validate_deployment(engine, input_shape, target_spec)
        
        return engine
    
    def _select_optimization_strategies(self, target_spec: EdgeDeviceSpec) -> List[str]:
        """Select optimization strategies based on device constraints."""
        strategies = []
        
        if target_spec.memory_mb < 100:
            strategies.extend(['quantization', 'pruning'])
        elif target_spec.memory_mb < 500:
            strategies.append('quantization')
        
        if target_spec.latency_requirement_ms < 50:
            strategies.append('pruning')
        
        return strategies
    
    def _validate_deployment(self, engine: EdgeInferenceEngine, input_shape: Tuple[int, ...],
                           target_spec: EdgeDeviceSpec):
        """Validate that deployment meets requirements."""
        benchmark_results = engine.benchmark(input_shape)
        
        if not benchmark_results['meets_latency_req']:
            raise RuntimeError(f"Latency requirement not met: {benchmark_results['avg_latency_ms']:.2f}ms")
        
        if not benchmark_results['meets_throughput_req']:
            raise RuntimeError(f"Throughput requirement not met: {benchmark_results['throughput_ips']:.2f} ips")
        
        logger.info("Edge deployment validation successful")
        logger.info(f"Latency: {benchmark_results['avg_latency_ms']:.2f}ms")
        logger.info(f"Throughput: {benchmark_results['throughput_ips']:.2f} inferences/sec")

# Example usage and demonstration
def demo_edge_deployment():
    """Demonstrate edge AI deployment capabilities."""
    # Define edge device specifications
    mobile_device = EdgeDeviceSpec(
        device_type='mobile',
        memory_mb=50,
        compute_units=1.0,
        power_budget_watts=2.0,
        latency_requirement_ms=100,
        throughput_requirement=10.0
    )
    
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 8 * 8, 10)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Deploy to edge
    model = SimpleModel()
    deployment_manager = EdgeDeploymentManager()
    
    try:
        engine = deployment_manager.deploy_to_edge(
            model=model,
            target_spec=mobile_device,
            input_shape=(3, 32, 32),
            model_name="simple_classifier"
        )
        
        # Test inference
        test_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
        prediction = engine.predict(test_input)
        logger.info(f"Prediction shape: {prediction.shape}")
        
    except Exception as e:
        logger.error(f"Edge deployment failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_edge_deployment()
