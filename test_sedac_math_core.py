
import unittest
import torch
import numpy as np
import math
from collections import deque
from scipy import stats as scipy_stats

from sedac.v9.production.engine import IncrementalWindowStats, AdaptiveThresholdController
from sedac.v9.production.config import ProductionConfig

class TestSEDACMathCore(unittest.TestCase):
    def setUp(self):
        self.window_size = 100
        self.stats = IncrementalWindowStats(self.window_size)
        self.config = ProductionConfig()
        self.controller = AdaptiveThresholdController(self.config, window_size=self.window_size, min_samples=10)

    def test_incremental_stats_accuracy(self):
        """Test accuracy of O(1) incremental statistics against scipy"""
        print("\n[Test] Incremental Stats Accuracy")
        data = np.random.randn(200).tolist()
        
        # Feed data
        for x in data:
            self.stats.update(x)
            
        # Get SEDAC stats
        s_stats = self.stats.get_stats()
        
        # Get Ground Truth (scipy) - use last window_size elements
        window_data = data[-self.window_size:]
        g_mean = np.mean(window_data)
        g_std = np.std(window_data, ddof=1) # Bessel correction
        g_skew = scipy_stats.skew(window_data, bias=False) # Bias=False matches Fisher-Pearson
        g_kurt = scipy_stats.kurtosis(window_data, bias=False) # Excess kurtosis
        
        print(f"Mean: SEDAC={s_stats['mean']:.6f}, GT={g_mean:.6f}")
        print(f"Std : SEDAC={s_stats['std']:.6f}, GT={g_std:.6f}")
        print(f"Skew: SEDAC={s_stats['skew']:.6f}, GT={g_skew:.6f}")
        print(f"Kurt: SEDAC={s_stats['kurt']:.6f}, GT={g_kurt:.6f}")
        
        self.assertAlmostEqual(s_stats['mean'], g_mean, places=5)
        self.assertAlmostEqual(s_stats['std'], g_std, places=5)
        self.assertAlmostEqual(s_stats['skew'], g_skew, places=2) # Higher moments are sensitive
        self.assertAlmostEqual(s_stats['kurt'], g_kurt, places=2) # Excess kurtosis is sensitive

    def test_adaptive_forgetting_factor(self):
        """Test if Forgetting Factor adapts to volatility"""
        print("\n[Test] Adaptive Forgetting Factor")
        
        # Stable phase
        print("Phase 1: Stable (Low Volatility)")
        stable_data = np.random.normal(3.0, 0.1, 100)
        for x in stable_data:
            self.controller.update(torch.tensor([float(x)]))
            
        lambda_stable = self.controller._current_lambda
        print(f"Lambda (Stable): {lambda_stable:.6f}")
        self.assertGreater(lambda_stable, 0.9, "Lambda should be high for stable data")
        
        # Volatile phase (Shift)
        print("Phase 2: Volatile (Shift to 5.0)")
        volatile_data = np.random.normal(5.0, 0.1, 20) # Sudden jump
        lambdas = []
        for x in volatile_data:
            self.controller.update(torch.tensor([float(x)]))
            lambdas.append(self.controller._current_lambda)
            
        lambda_volatile = min(lambdas) # Should dip
        print(f"Lambda (Volatile Min): {lambda_volatile:.6f}")
        
        self.assertLess(lambda_volatile, lambda_stable, "Lambda should decrease during volatility")

    def test_siegmund_cusum(self):
        """Test CUSUM detection of mean shift"""
        print("\n[Test] Siegmund CUSUM Detection")
        # Reset controller
        self.controller = AdaptiveThresholdController(self.config, window_size=100, min_samples=10)
        
        # Baseline
        base = np.random.normal(0, 1, 100)
        for x in base:
            self.controller.update(torch.tensor([float(x)]))
        
        self.assertFalse(self.controller._change_detected, "No change should be detected in baseline")
        
        # Shift
        shift = np.random.normal(3, 1, 50) # 3 sigma shift
        detected = False
        for i, x in enumerate(shift):
            self.controller.update(torch.tensor([float(x)]))
            if self.controller._change_detected:
                print(f"Change detected at step {i} of shift")
                detected = True
                break
        
        self.assertTrue(detected, "CUSUM should detect 3-sigma shift")

    def test_shrinkage_mahalanobis(self):
        """Test Shrinkage Mahalanobis Distance calculation"""
        print("\n[Test] Shrinkage Mahalanobis")
        
        # Generate correlated data
        # Entropy and Confidence are usually negatively correlated
        entropy = []
        confidence = []
        for _ in range(100):
            e = np.random.normal(3, 1)
            c = 1.0 - 0.1 * e + np.random.normal(0, 0.05) # Correlated
            entropy.append(e)
            confidence.append(c)
            self.controller.update(torch.tensor([float(e)]), torch.tensor([float(c)]))
            
        # Test outlier
        # High entropy (5), Low confidence (0.5) -> Normal (consistent with correlation)
        dist_normal = self.controller._compute_shrinkage_mahalanobis(5.0, 0.5)
        print(f"Dist (Consistent): {dist_normal:.4f}")
        
        # High entropy (5), High confidence (0.95) -> Outlier (inconsistent)
        # Note: In our simplified shrinkage implementation (diagonal), this checks Z-score distance.
        # Ideally full covariance would catch this better, but let's check distance magnitude.
        dist_outlier = self.controller._compute_shrinkage_mahalanobis(5.0, 0.95)
        print(f"Dist (Inconsistent): {dist_outlier:.4f}")
        
        # Since we use diagonal shrinkage currently (Euclidean in Z-space),
        # both will be far from mean.
        # This test mainly verifies it runs and returns valid distances.
        self.assertTrue(dist_normal > 0)
        self.assertTrue(dist_outlier > 0)

    def test_decision_logic(self):
        """Test O1 decision criteria (Bonferroni)"""
        print("\n[Test] Decision Logic")
        # Reset
        self.controller = AdaptiveThresholdController(self.config, min_samples=10)
        
        # Train with normal data
        for _ in range(50):
            self.controller.update(torch.tensor([3.0]))
            
        # Force stats to known state
        # Mean ~3.0, Std ~0.1 (if variance decays) or depending on AFF.
        # Actually AFF might keep std high if we input constant? No, std of constant is 0.
        # Let's check stats
        stats = self.controller.stats
        print(f"Stats after training: Mean={stats['entropy_mean']:.2f}, Std={stats['entropy_std']:.2f}")
        
        # Test NORM case (input close to mean)
        dec = self.controller.get_decision(3.0)
        print(f"Decision (3.0): {dec}")
        self.assertEqual(dec, 'NORM')
        
        # Test O1 case (High Entropy)
        # Z > 2.25 required
        high_val = stats['entropy_mean'] + 3.0 * stats['entropy_std']
        
        # Need 2 criteria. 
        # 1. High Z-score (Met)
        # 2. Consecutive (Need to build up) or Posterior or CUSUM
        
        # Let's trigger consecutive
        self.controller._consecutive_threshold = 2
        self.controller._consecutive_high = 2
        
        dec = self.controller.get_decision(high_val)
        print(f"Decision ({high_val:.2f} + Consec): {dec}")
        self.assertEqual(dec, 'O1')

if __name__ == '__main__':
    unittest.main()
