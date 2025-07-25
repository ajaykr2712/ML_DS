"""
Advanced A/B Testing Framework for ML Models
===========================================

This module implements a comprehensive A/B testing framework for ML models with:
- Statistical significance testing
- Multi-armed bandit algorithms
- Bayesian inference
- Sequential testing
- Effect size estimation

Author: ML Experimentation Team
Date: July 2025
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt

@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    alpha: float = 0.05  # Significance level
    power: float = 0.8   # Statistical power
    minimum_effect_size: float = 0.05  # Minimum detectable effect
    traffic_allocation: float = 0.5    # Proportion for treatment group
    sequential_testing: bool = True
    bayesian_inference: bool = False

class ABTestAnalyzer:
    """Statistical analyzer for A/B tests."""
    
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_sample_size(self, baseline_rate: float) -> int:
        """Calculate required sample size for A/B test."""
        effect_size = self.config.minimum_effect_size
        alpha = self.config.alpha
        power = self.config.power
        
        # Cohen's d for effect size
        pooled_std = np.sqrt(baseline_rate * (1 - baseline_rate))
        d = effect_size / pooled_std
        
        # Sample size calculation using normal approximation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / d) ** 2
        return int(np.ceil(n))
    
    def analyze_results(self, control_data: np.ndarray, 
                       treatment_data: np.ndarray) -> Dict:
        """Analyze A/B test results."""
        results = {
            'timestamp': datetime.now(),
            'sample_sizes': {
                'control': len(control_data),
                'treatment': len(treatment_data)
            }
        }
        
        # Basic statistics
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        
        results['means'] = {
            'control': control_mean,
            'treatment': treatment_mean,
            'difference': treatment_mean - control_mean,
            'lift': (treatment_mean - control_mean) / control_mean * 100
        }
        
        # Statistical significance testing
        if self.config.bayesian_inference:
            results['statistical_test'] = self._bayesian_analysis(control_data, treatment_data)
        else:
            results['statistical_test'] = self._frequentist_analysis(control_data, treatment_data)
        
        # Effect size
        results['effect_size'] = self._calculate_effect_size(control_data, treatment_data)
        
        # Confidence intervals
        results['confidence_intervals'] = self._calculate_confidence_intervals(
            control_data, treatment_data
        )
        
        return results
    
    def _frequentist_analysis(self, control: np.ndarray, treatment: np.ndarray) -> Dict:
        """Perform frequentist statistical analysis."""
        # Welch's t-test (unequal variances)
        statistic, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        
        significant = p_value < self.config.alpha
        
        return {
            'test_type': 'welch_ttest',
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'alpha': self.config.alpha
        }
    
    def _bayesian_analysis(self, control: np.ndarray, treatment: np.ndarray) -> Dict:
        """Perform Bayesian analysis."""
        # Simple Bayesian approach using normal priors
        control_mean, control_std = np.mean(control), np.std(control, ddof=1)
        treatment_mean, treatment_std = np.mean(treatment), np.std(treatment, ddof=1)
        
        # Posterior distributions (assuming normal likelihood)
        control_posterior_var = control_std**2 / len(control)
        treatment_posterior_var = treatment_std**2 / len(treatment)
        
        # Probability that treatment > control
        diff_mean = treatment_mean - control_mean
        diff_var = control_posterior_var + treatment_posterior_var
        
        prob_treatment_better = 1 - stats.norm.cdf(0, diff_mean, np.sqrt(diff_var))
        
        return {
            'test_type': 'bayesian',
            'prob_treatment_better': prob_treatment_better,
            'credible_interval_95': stats.norm.interval(
                0.95, diff_mean, np.sqrt(diff_var)
            ),
            'posterior_mean_diff': diff_mean,
            'posterior_std_diff': np.sqrt(diff_var)
        }
    
    def _calculate_effect_size(self, control: np.ndarray, treatment: np.ndarray) -> Dict:
        """Calculate various effect size measures."""
        control_mean, control_std = np.mean(control), np.std(control, ddof=1)
        treatment_mean, treatment_std = np.mean(treatment), np.std(treatment, ddof=1)
        
        # Cohen's d
        pooled_std = np.sqrt(((len(control) - 1) * control_std**2 + 
                             (len(treatment) - 1) * treatment_std**2) / 
                            (len(control) + len(treatment) - 2))
        cohens_d = (treatment_mean - control_mean) / pooled_std
        
        # Glass's delta
        glass_delta = (treatment_mean - control_mean) / control_std
        
        return {
            'cohens_d': cohens_d,
            'glass_delta': glass_delta,
            'interpretation': self._interpret_effect_size(abs(cohens_d))
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "Small effect"
        elif effect_size < 0.5:
            return "Medium effect"
        elif effect_size < 0.8:
            return "Large effect"
        else:
            return "Very large effect"
    
    def _calculate_confidence_intervals(self, control: np.ndarray, 
                                      treatment: np.ndarray) -> Dict:
        """Calculate confidence intervals for means and difference."""
        alpha = self.config.alpha
        
        # Control group CI
        control_ci = stats.t.interval(
            1 - alpha, len(control) - 1,
            loc=np.mean(control),
            scale=stats.sem(control)
        )
        
        # Treatment group CI
        treatment_ci = stats.t.interval(
            1 - alpha, len(treatment) - 1,
            loc=np.mean(treatment),
            scale=stats.sem(treatment)
        )
        
        # Difference CI
        diff_mean = np.mean(treatment) - np.mean(control)
        diff_se = np.sqrt(stats.sem(control)**2 + stats.sem(treatment)**2)
        df = len(control) + len(treatment) - 2
        
        diff_ci = stats.t.interval(1 - alpha, df, loc=diff_mean, scale=diff_se)
        
        return {
            'control': control_ci,
            'treatment': treatment_ci,
            'difference': diff_ci,
            'confidence_level': 1 - alpha
        }

class MultiArmedBandit:
    """Multi-armed bandit for dynamic traffic allocation."""
    
    def __init__(self, n_arms: int, algorithm: str = "thompson_sampling"):
        self.n_arms = n_arms
        self.algorithm = algorithm
        self.rewards = [[] for _ in range(n_arms)]
        self.arm_counts = np.zeros(n_arms)
        self.total_rewards = np.zeros(n_arms)
        
    def select_arm(self) -> int:
        """Select arm based on bandit algorithm."""
        if self.algorithm == "epsilon_greedy":
            return self._epsilon_greedy()
        elif self.algorithm == "ucb":
            return self._upper_confidence_bound()
        elif self.algorithm == "thompson_sampling":
            return self._thompson_sampling()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def update_reward(self, arm: int, reward: float):
        """Update reward for selected arm."""
        self.rewards[arm].append(reward)
        self.arm_counts[arm] += 1
        self.total_rewards[arm] += reward
    
    def _epsilon_greedy(self, epsilon: float = 0.1) -> int:
        """Epsilon-greedy arm selection."""
        if np.random.random() < epsilon or np.all(self.arm_counts == 0):
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.total_rewards / np.maximum(self.arm_counts, 1))
    
    def _upper_confidence_bound(self) -> int:
        """Upper Confidence Bound arm selection."""
        if np.any(self.arm_counts == 0):
            return np.argmin(self.arm_counts)
        
        total_count = np.sum(self.arm_counts)
        ucb_values = []
        
        for arm in range(self.n_arms):
            mean_reward = self.total_rewards[arm] / self.arm_counts[arm]
            confidence = np.sqrt(2 * np.log(total_count) / self.arm_counts[arm])
            ucb_values.append(mean_reward + confidence)
        
        return np.argmax(ucb_values)
    
    def _thompson_sampling(self) -> int:
        """Thompson Sampling for Bernoulli bandits."""
        samples = []
        
        for arm in range(self.n_arms):
            if self.arm_counts[arm] == 0:
                # Prior: Beta(1, 1)
                sample = np.random.beta(1, 1)
            else:
                # Posterior: Beta(successes + 1, failures + 1)
                successes = self.total_rewards[arm]
                failures = self.arm_counts[arm] - successes
                sample = np.random.beta(successes + 1, failures + 1)
            
            samples.append(sample)
        
        return np.argmax(samples)
    
    def get_statistics(self) -> Dict:
        """Get bandit statistics."""
        return {
            'arm_counts': self.arm_counts.tolist(),
            'total_rewards': self.total_rewards.tolist(),
            'average_rewards': (self.total_rewards / np.maximum(self.arm_counts, 1)).tolist(),
            'total_trials': int(np.sum(self.arm_counts))
        }

# Example usage
if __name__ == "__main__":
    print("Advanced A/B Testing Framework Demo")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    control_data = np.random.normal(0.2, 0.1, 1000)  # 20% baseline conversion
    treatment_data = np.random.normal(0.25, 0.1, 1000)  # 25% treatment conversion
    
    # Test 1: Traditional A/B Test Analysis
    print("\n1. Traditional A/B Test Analysis:")
    config = ABTestConfig(alpha=0.05, power=0.8, minimum_effect_size=0.03)
    analyzer = ABTestAnalyzer(config)
    
    required_sample_size = analyzer.calculate_sample_size(0.2)
    print(f"Required sample size per group: {required_sample_size}")
    
    results = analyzer.analyze_results(control_data, treatment_data)
    print(f"Control mean: {results['means']['control']:.4f}")
    print(f"Treatment mean: {results['means']['treatment']:.4f}")
    print(f"Lift: {results['means']['lift']:.2f}%")
    print(f"P-value: {results['statistical_test']['p_value']:.6f}")
    print(f"Significant: {results['statistical_test']['significant']}")
    print(f"Effect size (Cohen's d): {results['effect_size']['cohens_d']:.4f}")
    
    # Test 2: Bayesian A/B Test
    print("\n2. Bayesian A/B Test Analysis:")
    bayesian_config = ABTestConfig(bayesian_inference=True)
    bayesian_analyzer = ABTestAnalyzer(bayesian_config)
    
    bayesian_results = bayesian_analyzer.analyze_results(control_data, treatment_data)
    bayesian_test = bayesian_results['statistical_test']
    print(f"Probability treatment is better: {bayesian_test['prob_treatment_better']:.4f}")
    print(f"95% Credible Interval: {bayesian_test['credible_interval_95']}")
    
    # Test 3: Multi-Armed Bandit
    print("\n3. Multi-Armed Bandit Simulation:")
    bandit = MultiArmedBandit(n_arms=3, algorithm="thompson_sampling")
    
    # Simulate different conversion rates for each arm
    true_rates = [0.2, 0.25, 0.22]
    
    for trial in range(1000):
        arm = bandit.select_arm()
        reward = np.random.random() < true_rates[arm]
        bandit.update_reward(arm, reward)
    
    stats_result = bandit.get_statistics()
    print("Bandit Statistics:")
    for i, (count, reward) in enumerate(zip(stats_result['arm_counts'], 
                                           stats_result['average_rewards'])):
        print(f"  Arm {i}: {count} trials, {reward:.4f} avg reward (true: {true_rates[i]:.3f})")
    
    print("\nA/B Testing framework demo completed!")
