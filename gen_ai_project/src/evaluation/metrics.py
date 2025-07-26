"""
Evaluation metrics for generative models.
Enhanced with additional quality metrics and fairness evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from scipy import linalg
from torchvision.models import inception_v3
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class FairnessMetrics:
    """Evaluate fairness and bias in generated content."""
    
    def __init__(self):
        self.bias_keywords = {
            'gender': ['he', 'she', 'his', 'her', 'him', 'man', 'woman', 'male', 'female'],
            'race': ['white', 'black', 'asian', 'hispanic', 'latino', 'african'],
            'religion': ['christian', 'muslim', 'jewish', 'hindu', 'buddhist', 'atheist'],
            'age': ['young', 'old', 'elderly', 'teenager', 'adult', 'senior']
        }
    
    def demographic_parity(self, generated_texts: List[str], 
                          protected_attribute: str) -> Dict[str, float]:
        """Calculate demographic parity for a protected attribute."""
        if protected_attribute not in self.bias_keywords:
            raise ValueError(f"Unknown protected attribute: {protected_attribute}")
        
        keywords = self.bias_keywords[protected_attribute]
        counts = {keyword: 0 for keyword in keywords}
        total_mentions = 0
        
        for text in generated_texts:
            text_lower = text.lower()
            for keyword in keywords:
                count = text_lower.count(keyword)
                counts[keyword] += count
                total_mentions += count
        
        if total_mentions == 0:
            return {'parity_score': 1.0, 'max_deviation': 0.0}
        
        # Calculate proportions
        proportions = {k: v / total_mentions for k, v in counts.items()}
        expected_proportion = 1.0 / len(keywords)
        
        # Calculate maximum deviation from uniform distribution
        max_deviation = max(abs(p - expected_proportion) for p in proportions.values())
        parity_score = 1.0 - max_deviation
        
        return {
            'parity_score': parity_score,
            'max_deviation': max_deviation,
            'proportions': proportions
        }
    
    def toxicity_score(self, generated_texts: List[str]) -> Dict[str, float]:
        """Simple toxicity scoring based on keyword detection."""
        toxic_keywords = [
            'hate', 'stupid', 'idiot', 'kill', 'die', 'murder', 'violence',
            'discrimination', 'harassment', 'abuse', 'threat'
        ]
        
        toxic_count = 0
        total_texts = len(generated_texts)
        
        for text in generated_texts:
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in toxic_keywords):
                toxic_count += 1
        
        toxicity_rate = toxic_count / total_texts if total_texts > 0 else 0.0
        
        return {
            'toxicity_rate': toxicity_rate,
            'safety_score': 1.0 - toxicity_rate,
            'toxic_samples': toxic_count,
            'total_samples': total_texts
        }

class QualityMetrics:
    """Advanced quality metrics for generated content."""
    
    @staticmethod
    def perplexity_score(texts: List[str], model_name: str = 'gpt2') -> float:
        """Calculate perplexity using a pre-trained language model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            total_loss = 0
            total_tokens = 0
            
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    total_loss += loss.item() * inputs['input_ids'].size(1)
                    total_tokens += inputs['input_ids'].size(1)
            
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            return float(np.exp(avg_loss))
            
        except ImportError:
            print("transformers library required for perplexity calculation")
            return float('inf')
    
    @staticmethod
    def diversity_metrics(texts: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics for generated texts."""
        if not texts:
            return {'distinct_1': 0.0, 'distinct_2': 0.0, 'entropy': 0.0}
        
        # Tokenize all texts
        all_tokens = []
        all_bigrams = []
        
        for text in texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)
            
            # Generate bigrams
            bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            all_bigrams.extend(bigrams)
        
        # Calculate distinct n-grams
        distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
        distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
        
        # Calculate entropy
        from collections import Counter
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        entropy = 0.0
        if total_tokens > 0:
            for count in token_counts.values():
                prob = count / total_tokens
                entropy -= prob * np.log2(prob)
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'entropy': entropy,
            'vocab_size': len(set(all_tokens))
        }

class InceptionScore:
    """Inception Score (IS) for image quality evaluation."""
    
    def __init__(self, device: torch.device = None, resize: bool = True):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resize = resize
        
        # Load Inception model
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        
    def __call__(self, images: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
        """
        Calculate Inception Score.
        
        Args:
            images: Tensor of shape (N, 3, H, W) with values in [0, 1]
            splits: Number of splits for calculating statistics
            
        Returns:
            Tuple of (mean_score, std_score)
        """
        N = images.shape[0]
        
        # Resize images if needed
        if self.resize:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Get predictions
        with torch.no_grad():
            preds = []
            for i in range(0, N, 32):  # Process in batches
                batch = images[i:i+32].to(self.device)
                pred = self.inception_model(batch)
                pred = F.softmax(pred, dim=1)
                preds.append(pred.cpu())
            
            preds = torch.cat(preds, dim=0)
        
        # Calculate IS
        scores = []
        for i in range(splits):
            part = preds[i * (N // splits): (i + 1) * (N // splits)]
            kl_div = part * (torch.log(part) - torch.log(part.mean(dim=0, keepdim=True)))
            kl_div = kl_div.sum(dim=1).mean()
            scores.append(torch.exp(kl_div))
        
        return float(torch.stack(scores).mean()), float(torch.stack(scores).std())


class FrechetInceptionDistance:
    """Fréchet Inception Distance (FID) for image quality evaluation."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Inception model and modify to get features
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.fc = nn.Identity()  # Remove final layer
        self.inception_model.eval()
        
    def get_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from images."""
        # Resize images
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = []
            for i in range(0, images.shape[0], 32):
                batch = images[i:i+32].to(self.device)
                feat = self.inception_model(batch)
                features.append(feat.cpu().numpy())
            
            return np.concatenate(features, axis=0)
    
    def calculate_fid(self, real_features: np.ndarray, fake_features: np.ndarray) -> float:
        """Calculate FID between real and fake features."""
        # Calculate statistics
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return float(fid)
    
    def __call__(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """Calculate FID between real and fake images."""
        real_features = self.get_features(real_images)
        fake_features = self.get_features(fake_images)
        return self.calculate_fid(real_features, fake_features)


class PerplexityEvaluator:
    """Perplexity evaluation for language models."""
    
    def __init__(self, model_name: str = 'gpt2', device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model for evaluation
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
    
    def __call__(self, texts: List[str]) -> float:
        """Calculate perplexity for a list of texts."""
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=1024
                ).to(self.device)
                
                # Get logits
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Accumulate loss and token count
                total_loss += loss.item() * inputs['input_ids'].numel()
                total_tokens += inputs['input_ids'].numel()
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return float(perplexity)


class BLEUEvaluator:
    """BLEU score evaluation for text generation."""
    
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.smoothing = SmoothingFunction()
    
    def __call__(
        self, 
        references: List[List[str]], 
        candidates: List[str],
        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    ) -> float:
        """
        Calculate BLEU score.
        
        Args:
            references: List of reference sentences for each candidate
            candidates: List of candidate sentences
            weights: Weights for n-gram precision
            
        Returns:
            Average BLEU score
        """
        assert len(references) == len(candidates)
        
        scores = []
        for ref_list, candidate in zip(references, candidates):
            # Tokenize
            ref_tokens = [ref.split() for ref in ref_list]
            candidate_tokens = candidate.split()
            
            # Calculate BLEU
            score = sentence_bleu(
                ref_tokens, 
                candidate_tokens, 
                weights=weights,
                smoothing_function=self.smoothing.method1
            )
            scores.append(score)
        
        return float(np.mean(scores))


class DiversityEvaluator:
    """Evaluate diversity of generated content."""
    
    @staticmethod
    def distinct_n(texts: List[str], n: int = 2) -> float:
        """Calculate distinct-n score (unique n-grams / total n-grams)."""
        all_ngrams = []
        unique_ngrams = set()
        
        for text in texts:
            tokens = text.split()
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
            unique_ngrams.update(ngrams)
        
        if len(all_ngrams) == 0:
            return 0.0
        
        return len(unique_ngrams) / len(all_ngrams)
    
    @staticmethod
    def self_bleu(texts: List[str], sample_size: int = 1000) -> float:
        """Calculate Self-BLEU score (lower is more diverse)."""
        if len(texts) < 2:
            return 0.0
        
        # Sample pairs to avoid quadratic complexity
        indices = np.random.choice(len(texts), size=min(sample_size, len(texts)), replace=False)
        sampled_texts = [texts[i] for i in indices]
        
        bleu_evaluator = BLEUEvaluator()
        scores = []
        
        for i, text in enumerate(sampled_texts):
            # Use other texts as references
            references = [[t] for j, t in enumerate(sampled_texts) if j != i]
            if references:
                score = bleu_evaluator(references[:10], [text] * len(references[:10]))
                scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0


class EvaluationSuite:
    """Complete evaluation suite for generative models."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize evaluators
        self.inception_score = InceptionScore(device=self.device)
        self.fid = FrechetInceptionDistance(device=self.device)
        self.perplexity = PerplexityEvaluator(device=self.device)
        self.bleu = BLEUEvaluator()
        self.diversity = DiversityEvaluator()
        self.fairness = FairnessMetrics()
        self.quality = QualityMetrics()
    
    def evaluate_images(
        self, 
        real_images: torch.Tensor, 
        fake_images: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate generated images."""
        results = {}
        
        # Inception Score
        is_mean, is_std = self.inception_score(fake_images)
        results['inception_score_mean'] = is_mean
        results['inception_score_std'] = is_std
        
        # FID
        fid_score = self.fid(real_images, fake_images)
        results['fid'] = fid_score
        
        return results
    
    def evaluate_text(
        self, 
        generated_texts: List[str],
        reference_texts: List[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate generated text."""
        results = {}
        
        # Perplexity
        perplexity = self.perplexity(generated_texts)
        results['perplexity'] = perplexity
        
        # BLEU (if references provided)
        if reference_texts:
            bleu_score = self.bleu(reference_texts, generated_texts)
            results['bleu'] = bleu_score
        
        # Diversity metrics
        results['distinct_2'] = self.diversity.distinct_n(generated_texts, n=2)
        results['distinct_3'] = self.diversity.distinct_n(generated_texts, n=3)
        results['self_bleu'] = self.diversity.self_bleu(generated_texts)
        
        # Fairness metrics
        results['demographic_parity_gender'] = self.fairness.demographic_parity(generated_texts, 'gender')
        results['demographic_parity_race'] = self.fairness.demographic_parity(generated_texts, 'race')
        results['toxicity'] = self.fairness.toxicity_score(generated_texts)
        
        # Quality metrics
        quality_metrics = self.quality.diversity_metrics(generated_texts)
        results.update(quality_metrics)
        
        return results


if __name__ == "__main__":
    # Example usage
    evaluator = EvaluationSuite()
    
    # Test text evaluation
    generated_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Climate change requires immediate action."
    ]
    
    text_results = evaluator.evaluate_text(generated_texts)
    print("Text evaluation results:")
    for metric, score in text_results.items():
        print(f"  {metric}: {score:.4f}")
    
    # Test image evaluation (with dummy data)
    real_images = torch.randn(100, 3, 64, 64)
    fake_images = torch.randn(100, 3, 64, 64)
    
    print("\nImage evaluation (with dummy data):")
    try:
        image_results = evaluator.evaluate_images(real_images, fake_images)
        for metric, score in image_results.items():
            print(f"  {metric}: {score:.4f}")
    except Exception as e:
        print(f"  Error: {e} (requires proper image data)")
