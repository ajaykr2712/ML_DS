"""
GPU Acceleration Utilities
Optimized GPU computation utilities for ML workloads
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from dataclasses import dataclass
import gc
from contextlib import contextmanager

@dataclass
class GPUConfig:
    """GPU configuration settings"""
    device_id: int = 0
    memory_fraction: float = 0.9
    allow_growth: bool = True
    mixed_precision: bool = True
    benchmark_mode: bool = True

class GPUMemoryManager:
    """GPU memory management utilities"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.device = torch.device(f'cuda:{config.device_id}' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configure GPU settings
        self._configure_gpu()
    
    def _configure_gpu(self):
        """Configure GPU settings for optimal performance"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using CPU")
            return
        
        # Set device
        torch.cuda.set_device(self.config.device_id)
        
        # Enable benchmark mode for consistent input sizes
        if self.config.benchmark_mode:
            torch.backends.cudnn.benchmark = True
        
        # Set memory fraction if specified
        if self.config.memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(
                self.config.memory_fraction, 
                self.config.device_id
            )
        
        self.logger.info(f"GPU configured: {torch.cuda.get_device_name(self.device)}")
        self.logger.info(f"Memory allocated: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")
    
    @contextmanager
    def gpu_memory_context(self):
        """Context manager for GPU memory tracking"""
        if not torch.cuda.is_available():
            yield
            return
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        
        # Record initial memory
        initial_memory = torch.cuda.memory_allocated(self.device)
        
        try:
            yield
        finally:
            # Record final memory and cleanup
            final_memory = torch.cuda.memory_allocated(self.device)
            memory_used = (final_memory - initial_memory) / 1e9
            
            self.logger.info(f"GPU memory used: {memory_used:.2f} GB")
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        cached = torch.cuda.memory_reserved(self.device) / 1e9
        total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': total,
            'utilization_percent': (allocated / total) * 100
        }
    
    def clear_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("GPU memory cache cleared")

class CUDATensorProcessor:
    """Optimized tensor operations for CUDA"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def batch_matrix_multiply(self, 
                            matrices: List[torch.Tensor], 
                            batch_size: int = 32) -> List[torch.Tensor]:
        """
        Efficient batch matrix multiplication on GPU
        
        Args:
            matrices: List of tensor pairs to multiply
            batch_size: Batch size for processing
            
        Returns:
            List of multiplication results
        """
        results = []
        
        for i in range(0, len(matrices), batch_size):
            batch = matrices[i:i + batch_size]
            
            # Stack tensors for batch processing
            if len(batch[0]) == 2:  # Matrix multiplication
                batch_a = torch.stack([pair[0] for pair in batch])
                batch_b = torch.stack([pair[1] for pair in batch])
                
                # Move to GPU and compute
                batch_a = batch_a.to(self.device)
                batch_b = batch_b.to(self.device)
                
                batch_result = torch.bmm(batch_a, batch_b)
                
                # Move back to CPU and unstack
                for result in batch_result:
                    results.append(result.cpu())
        
        return results
    
    def parallel_reduce(self, 
                       tensor: torch.Tensor, 
                       operation: str = 'sum',
                       dim: Optional[int] = None) -> torch.Tensor:
        """
        Parallel reduction operations on GPU
        
        Args:
            tensor: Input tensor
            operation: Reduction operation ('sum', 'mean', 'max', 'min')
            dim: Dimension to reduce along
            
        Returns:
            Reduced tensor
        """
        tensor = tensor.to(self.device)
        
        if operation == 'sum':
            result = torch.sum(tensor, dim=dim)
        elif operation == 'mean':
            result = torch.mean(tensor, dim=dim)
        elif operation == 'max':
            result = torch.max(tensor, dim=dim)[0] if dim is not None else torch.max(tensor)
        elif operation == 'min':
            result = torch.min(tensor, dim=dim)[0] if dim is not None else torch.min(tensor)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return result.cpu()
    
    def fast_einsum(self, equation: str, *tensors) -> torch.Tensor:
        """
        Fast Einstein summation on GPU
        
        Args:
            equation: Einstein summation equation
            tensors: Input tensors
            
        Returns:
            Result tensor
        """
        gpu_tensors = [t.to(self.device) for t in tensors]
        result = torch.einsum(equation, *gpu_tensors)
        return result.cpu()

class GPUModelOptimizer:
    """GPU-specific model optimization utilities"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for GPU inference
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        # Move to GPU
        model = model.to(self.device)
        
        # Set to evaluation mode
        model.eval()
        
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
            self.logger.info("Model compiled with torch.compile")
        
        # Enable tensor core usage for mixed precision
        with torch.cuda.amp.autocast():
            # Warmup run
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)
        
        self.logger.info("Model optimized for GPU inference")
        return model
    
    def enable_tensor_cores(self, model: nn.Module) -> nn.Module:
        """
        Enable Tensor Core usage for model
        
        Args:
            model: PyTorch model
            
        Returns:
            Model with Tensor Core optimizations
        """
        # Replace operations that benefit from Tensor Cores
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Ensure weights are properly sized for Tensor Cores
                in_features = module.in_features
                out_features = module.out_features
                
                # Pad to multiples of 8 for optimal Tensor Core usage
                if in_features % 8 != 0 or out_features % 8 != 0:
                    new_in = ((in_features + 7) // 8) * 8
                    new_out = ((out_features + 7) // 8) * 8
                    
                    new_linear = nn.Linear(new_in, new_out)
                    
                    # Copy weights
                    with torch.no_grad():
                        new_linear.weight[:out_features, :in_features] = module.weight
                        if module.bias is not None:
                            new_linear.bias[:out_features] = module.bias
                    
                    # Replace module
                    parent = model
                    name_parts = name.split('.')
                    for part in name_parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, name_parts[-1], new_linear)
        
        return model

class GPUBenchmark:
    """GPU performance benchmarking utilities"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def benchmark_matrix_multiply(self, 
                                sizes: List[Tuple[int, int, int]],
                                num_iterations: int = 100) -> Dict[Tuple[int, int, int], float]:
        """
        Benchmark matrix multiplication performance
        
        Args:
            sizes: List of (M, N, K) dimensions to test
            num_iterations: Number of iterations per size
            
        Returns:
            Dictionary mapping sizes to average time in milliseconds
        """
        results = {}
        
        for m, n, k in sizes:
            times = []
            
            # Generate test matrices
            a = torch.randn(m, k).to(self.device)
            b = torch.randn(k, n).to(self.device)
            
            # Warmup
            for _ in range(10):
                torch.mm(a, b)
            
            torch.cuda.synchronize()
            
            # Benchmark
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                torch.mm(a, b)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            avg_time = np.mean(times)
            results[(m, n, k)] = avg_time
            
            self.logger.info(f"Matrix multiply ({m}x{k}) x ({k}x{n}): {avg_time:.3f} ms")
        
        return results
    
    def benchmark_model_inference(self, 
                                 model: nn.Module,
                                 input_shapes: List[Tuple[int, ...]],
                                 num_iterations: int = 100) -> Dict[Tuple[int, ...], Dict[str, float]]:
        """
        Benchmark model inference performance
        
        Args:
            model: PyTorch model
            input_shapes: List of input shapes to test
            num_iterations: Number of iterations per shape
            
        Returns:
            Dictionary mapping shapes to performance metrics
        """
        model = model.to(self.device)
        model.eval()
        
        results = {}
        
        with torch.no_grad():
            for shape in input_shapes:
                times = []
                memory_usage = []
                
                # Generate test input
                test_input = torch.randn(*shape).to(self.device)
                
                # Warmup
                for _ in range(10):
                    _ = model(test_input)
                
                torch.cuda.synchronize()
                
                # Benchmark
                for _ in range(num_iterations):
                    # Clear cache
                    torch.cuda.empty_cache()
                    
                    # Measure memory before
                    mem_before = torch.cuda.memory_allocated(self.device)
                    
                    # Time inference
                    start_time = time.perf_counter()
                    model(test_input)
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    
                    # Measure memory after
                    mem_after = torch.cuda.memory_allocated(self.device)
                    
                    times.append((end_time - start_time) * 1000)  # ms
                    memory_usage.append((mem_after - mem_before) / 1e6)  # MB
                
                results[shape] = {
                    'avg_time_ms': np.mean(times),
                    'std_time_ms': np.std(times),
                    'avg_memory_mb': np.mean(memory_usage),
                    'throughput_samples_per_sec': 1000 / np.mean(times)
                }
                
                self.logger.info(f"Input shape {shape}: "
                               f"{np.mean(times):.3f}±{np.std(times):.3f} ms, "
                               f"{np.mean(memory_usage):.1f} MB")
        
        return results

class CUDAKernelLauncher:
    """Custom CUDA kernel operations"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def fused_linear_relu(self, input_tensor: torch.Tensor, weight: torch.Tensor, 
                         bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fused linear + ReLU operation for better performance
        
        Args:
            input_tensor: Input tensor
            weight: Weight matrix
            bias: Optional bias vector
            
        Returns:
            Output after linear transformation and ReLU
        """
        # Move tensors to GPU
        input_tensor = input_tensor.to(self.device)
        weight = weight.to(self.device)
        if bias is not None:
            bias = bias.to(self.device)
        
        # Fused operation using torch.nn.functional
        output = torch.nn.functional.linear(input_tensor, weight, bias)
        output = torch.nn.functional.relu(output, inplace=True)
        
        return output
    
    def optimized_attention(self, 
                          query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          scale: Optional[float] = None) -> torch.Tensor:
        """
        Memory-efficient attention computation
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            scale: Scaling factor
            
        Returns:
            Attention output
        """
        # Move to GPU
        query = query.to(self.device)
        key = key.to(self.device)
        value = value.to(self.device)
        
        # Calculate scale
        if scale is None:
            scale = 1.0 / (query.size(-1) ** 0.5)
        
        # Use flash attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, scale=scale
            )
        else:
            # Standard attention implementation
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, value)
        
        return output

# Example usage and testing
if __name__ == "__main__":
    # Setup GPU configuration
    config = GPUConfig(
        device_id=0,
        memory_fraction=0.8,
        mixed_precision=True,
        benchmark_mode=True
    )
    
    # Initialize GPU manager
    gpu_manager = GPUMemoryManager(config)
    
    # Test memory context
    with gpu_manager.gpu_memory_context():
        # Create large tensors
        large_tensor = torch.randn(1000, 1000).cuda()
        result = torch.mm(large_tensor, large_tensor.t())
        
        print(f"Memory stats: {gpu_manager.get_memory_stats()}")
    
    # Benchmark matrix multiplication
    if torch.cuda.is_available():
        device = torch.device('cuda')
        benchmark = GPUBenchmark(device)
        
        sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
        results = benchmark.benchmark_matrix_multiply(sizes, num_iterations=50)
        
        print("Matrix multiplication benchmark results:")
        for size, time_ms in results.items():
            print(f"Size {size}: {time_ms:.3f} ms")
    
    print("GPU acceleration utilities test completed")
