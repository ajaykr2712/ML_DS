"""
Comprehensive evaluation utilities for generative AI models.
Includes metrics for text generation, image generation, and model performance.
"""

import json
import logging
import warnings
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from tqdm.auto import tqdm

# Text evaluation metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    warnings.warn("rouge_score not available. Install with: pip install rouge-score")

try:
    from sacrebleu import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    warnings.warn("sacrebleu not available. Install with: pip install sacrebleu")

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    warnings.warn("bert_score not available. Install with: pip install bert-score")

# Image evaluation metrics (for future use)
try:
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Install with: pip install scipy")


class TextGenerationEvaluator:
    """Comprehensive evaluator for text generation models."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        # Initialize scorers
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
        
        if BLEU_AVAILABLE:
            self.bleu_scorer = BLEU()
    
    def compute_perplexity(self, 
                          model: nn.Module, 
                          dataloader: DataLoader,
                          device: torch.device = None) -> float:
        """
        Compute perplexity on a dataset.
        
        Args:
            model: Language model
            dataloader: DataLoader with tokenized text
            device: Device to run evaluation on
        
        Returns:
            Perplexity score
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing perplexity"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Shift for causal language modeling
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                mask = attention_mask[:, 1:]
                
                # Forward pass
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction='none'
                )
                
                # Apply mask and sum
                masked_loss = loss * mask.reshape(-1)
                total_loss += masked_loss.sum().item()
                total_tokens += mask.sum().item()
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def compute_bleu_score(self, 
                          predictions: List[str], 
                          references: List[str]) -> Dict[str, float]:
        """
        Compute BLEU score for generated text.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
        
        Returns:
            Dictionary with BLEU scores
        """
        if not BLEU_AVAILABLE:
            self.logger.warning("BLEU scorer not available")
            return {}
        
        # Tokenize texts
        pred_tokens = [text.split() for text in predictions]
        ref_tokens = [[text.split()] for text in references]
        
        # Compute BLEU scores
        bleu_scores = {}
        for n in [1, 2, 3, 4]:
            try:
                bleu = self.bleu_scorer.corpus_score(pred_tokens, ref_tokens, max_order=n)
                bleu_scores[f'bleu_{n}'] = bleu.score
            except Exception as e:
                self.logger.warning(f"Error computing BLEU-{n}: {e}")
                bleu_scores[f'bleu_{n}'] = 0.0
        
        return bleu_scores
    
    def compute_rouge_score(self, 
                           predictions: List[str], 
                           references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores for generated text.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
        
        Returns:
            Dictionary with ROUGE scores
        """
        if not ROUGE_AVAILABLE:
            self.logger.warning("ROUGE scorer not available")
            return {}
        
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for metric, score in scores.items():
                rouge_scores[f'{metric}_f1'].append(score.fmeasure)
                rouge_scores[f'{metric}_precision'].append(score.precision)
                rouge_scores[f'{metric}_recall'].append(score.recall)
        
        # Average scores
        avg_scores = {}
        for metric, values in rouge_scores.items():
            avg_scores[metric] = np.mean(values)
        
        return avg_scores
    
    def compute_bert_score(self, 
                          predictions: List[str], 
                          references: List[str],
                          model_type: str = "distilbert-base-uncased") -> Dict[str, float]:
        """
        Compute BERTScore for generated text.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            model_type: BERT model to use for scoring
        
        Returns:
            Dictionary with BERTScore metrics
        """
        if not BERT_SCORE_AVAILABLE:
            self.logger.warning("BERTScore not available")
            return {}
        
        try:
            P, R, F1 = bert_score(predictions, references, model_type=model_type)
            
            return {
                'bert_score_precision': P.mean().item(),
                'bert_score_recall': R.mean().item(),
                'bert_score_f1': F1.mean().item()
            }
        except Exception as e:
            self.logger.error(f"Error computing BERTScore: {e}")
            return {}
    
    def compute_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute diversity metrics for generated texts.
        
        Args:
            texts: List of generated texts
        
        Returns:
            Dictionary with diversity metrics
        """
        if not texts:
            return {}
        
        # Tokenize texts
        all_tokens = []
        all_bigrams = []
        all_trigrams = []
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text.lower())
            all_tokens.extend(tokens)
            
            # N-grams
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens)-2)]
            
            all_bigrams.extend(bigrams)
            all_trigrams.extend(trigrams)
        
        # Compute diversity
        metrics = {}
        
        # Unique token ratio
        if all_tokens:
            metrics['unique_tokens'] = len(set(all_tokens)) / len(all_tokens)
        
        # Unique n-gram ratios
        if all_bigrams:
            metrics['unique_bigrams'] = len(set(all_bigrams)) / len(all_bigrams)
        
        if all_trigrams:
            metrics['unique_trigrams'] = len(set(all_trigrams)) / len(all_trigrams)
        
        # Average text length
        lengths = [len(self.tokenizer.tokenize(text)) for text in texts]
        metrics['avg_length'] = np.mean(lengths)
        metrics['std_length'] = np.std(lengths)
        
        return metrics
    
    def compute_repetition_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute repetition metrics for generated texts.
        
        Args:
            texts: List of generated texts
        
        Returns:
            Dictionary with repetition metrics
        """
        metrics = {}
        repetition_scores = []
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text.lower())
            
            if len(tokens) < 4:
                continue
            
            # Count 4-gram repetitions
            fourgrams = [tuple(tokens[i:i+4]) for i in range(len(tokens)-3)]
            unique_fourgrams = set(fourgrams)
            
            if fourgrams:
                repetition_rate = 1 - (len(unique_fourgrams) / len(fourgrams))
                repetition_scores.append(repetition_rate)
        
        if repetition_scores:
            metrics['repetition_rate'] = np.mean(repetition_scores)
        
        return metrics
    
    def evaluate_generation(self, 
                           model: nn.Module,
                           prompts: List[str],
                           references: Optional[List[str]] = None,
                           max_new_tokens: int = 100,
                           temperature: float = 0.8,
                           top_k: int = 50,
                           top_p: float = 0.9,
                           num_samples: int = 1,
                           device: torch.device = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of text generation.
        
        Args:
            model: Language model
            prompts: List of input prompts
            references: Optional reference texts for comparison
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            num_samples: Number of samples per prompt
            device: Device to run on
        
        Returns:
            Dictionary with evaluation metrics
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        all_generated_texts = []
        all_prompts = []
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Generating text"):
                for _ in range(num_samples):
                    # Tokenize prompt
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512
                    ).to(device)
                    
                    # Generate text
                    if hasattr(model, 'generate'):
                        # Use model's generate method
                        generated = model.generate(
                            inputs['input_ids'],
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=True
                        )
                        
                        # Decode generated text
                        generated_text = self.tokenizer.decode(
                            generated[0, inputs['input_ids'].shape[1]:], 
                            skip_special_tokens=True
                        )
                    else:
                        # Fallback generation (simplified)
                        generated_text = self._simple_generate(
                            model, inputs['input_ids'], max_new_tokens, temperature
                        )
                    
                    all_generated_texts.append(generated_text)
                    all_prompts.append(prompt)
        
        # Compute metrics
        metrics = {}
        
        # Diversity metrics
        diversity_metrics = self.compute_diversity_metrics(all_generated_texts)
        metrics.update(diversity_metrics)
        
        # Repetition metrics
        repetition_metrics = self.compute_repetition_metrics(all_generated_texts)
        metrics.update(repetition_metrics)
        
        # Reference-based metrics (if references provided)
        if references is not None:
            if len(references) == len(prompts):
                # Extend references to match number of samples
                extended_refs = []
                for ref in references:
                    extended_refs.extend([ref] * num_samples)
                references = extended_refs
            
            if len(references) == len(all_generated_texts):
                # BLEU scores
                bleu_scores = self.compute_bleu_score(all_generated_texts, references)
                metrics.update(bleu_scores)
                
                # ROUGE scores
                rouge_scores = self.compute_rouge_score(all_generated_texts, references)
                metrics.update(rouge_scores)
                
                # BERTScore
                bert_scores = self.compute_bert_score(all_generated_texts, references)
                metrics.update(bert_scores)
        
        # Add generated examples
        metrics['examples'] = [
            {
                'prompt': prompt,
                'generated': generated,
                'reference': ref if references else None
            }
            for prompt, generated, ref in zip(
                all_prompts[:10],  # First 10 examples
                all_generated_texts[:10],
                (references[:10] if references else [None] * 10)
            )
        ]
        
        return metrics
    
    def _simple_generate(self, 
                        model: nn.Module, 
                        input_ids: torch.Tensor, 
                        max_new_tokens: int,
                        temperature: float) -> str:
        """Simple generation fallback."""
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            outputs = model(generated)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            generated[0, input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text


class ModelPerformanceEvaluator:
    """Evaluator for model performance metrics (speed, memory, etc.)."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def benchmark_inference_speed(self, 
                                 model: nn.Module,
                                 input_shape: Tuple[int, ...],
                                 num_runs: int = 100,
                                 warmup_runs: int = 10,
                                 device: torch.device = None) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensors
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            device: Device to run on
        
        Returns:
            Dictionary with timing metrics
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randint(0, 1000, input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'throughput_samples_per_sec': input_shape[0] / np.mean(times)
        }
    
    def measure_memory_usage(self, 
                           model: nn.Module,
                           input_shape: Tuple[int, ...],
                           device: torch.device = None) -> Dict[str, float]:
        """
        Measure model memory usage.
        
        Args:
            model: Model to measure
            input_shape: Shape of input tensors
            device: Device to run on
        
        Returns:
            Dictionary with memory metrics
        """
        if device is None:
            device = next(model.parameters()).device
        
        if device.type != 'cuda':
            self.logger.warning("Memory measurement only available for CUDA devices")
            return {}
        
        model.eval()
        dummy_input = torch.randint(0, 1000, input_shape).to(device)
        
        # Measure memory before forward pass
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated(device)
        
        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated(device)
        
        # Measure peak memory
        peak_memory = torch.cuda.max_memory_allocated(device)
        
        return {
            'model_memory_mb': (memory_after - memory_before) / (1024**2),
            'peak_memory_mb': peak_memory / (1024**2),
            'memory_efficiency': (memory_after - memory_before) / peak_memory
        }
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }


class EvaluationReporter:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def generate_report(self, 
                       metrics: Dict[str, Any], 
                       model_name: str,
                       config: Dict[str, Any] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Evaluation metrics
            model_name: Name of the evaluated model
            config: Model configuration
        
        Returns:
            Path to the generated report
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"{model_name}_evaluation_{timestamp}.md"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluation Report: {model_name}\n\n")
            f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model configuration
            if config:
                f.write("## Model Configuration\n\n")
                f.write("```yaml\n")
                import yaml
                yaml.dump(config, f, default_flow_style=False)
                f.write("```\n\n")
            
            # Performance metrics
            f.write("## Performance Metrics\n\n")
            
            # Text generation metrics
            if any(key.startswith(('bleu', 'rouge', 'bert_score')) for key in metrics.keys()):
                f.write("### Text Generation Quality\n\n")
                for key, value in metrics.items():
                    if key.startswith(('bleu', 'rouge', 'bert_score')):
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n")
                f.write("\n")
            
            # Diversity and repetition
            if any(key in metrics for key in ['unique_tokens', 'repetition_rate']):
                f.write("### Diversity and Repetition\n\n")
                for key in ['unique_tokens', 'unique_bigrams', 'unique_trigrams', 'repetition_rate']:
                    if key in metrics:
                        f.write(f"- **{key.replace('_', ' ').title()}**: {metrics[key]:.4f}\n")
                f.write("\n")
            
            # Model performance
            if any(key.endswith('_time') for key in metrics.keys()):
                f.write("### Model Performance\n\n")
                for key, value in metrics.items():
                    if key.endswith(('_time', '_memory_mb', '_parameters')):
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                f.write("\n")
            
            # Examples
            if 'examples' in metrics:
                f.write("## Generation Examples\n\n")
                for i, example in enumerate(metrics['examples'][:5], 1):
                    f.write(f"### Example {i}\n\n")
                    f.write(f"**Prompt:** {example['prompt']}\n\n")
                    f.write(f"**Generated:** {example['generated']}\n\n")
                    if example.get('reference'):
                        f.write(f"**Reference:** {example['reference']}\n\n")
                    f.write("---\n\n")
        
        # Save metrics as JSON
        json_path = self.output_dir / f"{model_name}_metrics_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Remove examples for JSON (too verbose)
            json_metrics = {k: v for k, v in metrics.items() if k != 'examples'}
            json.dump(json_metrics, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation report saved to: {report_path}")
        return str(report_path)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage would require actual models and data
    print("Evaluation utilities loaded successfully!")
    print("Available evaluators:")
    print("- TextGenerationEvaluator: For evaluating text generation quality")
    print("- ModelPerformanceEvaluator: For benchmarking model performance")
    print("- EvaluationReporter: For generating comprehensive reports")
    
    # Check availability of optional dependencies
    print("\nOptional dependencies:")
    print(f"- ROUGE scorer: {ROUGE_AVAILABLE}")
    print(f"- BLEU scorer: {BLEU_AVAILABLE}")
    print(f"- BERTScore: {BERT_SCORE_AVAILABLE}")
    print(f"- SciPy: {SCIPY_AVAILABLE}")
