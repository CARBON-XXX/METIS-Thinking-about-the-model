"""
METIS Sliding Window Statistics
Numerically stable online statistics

Recomputed from buffer on demand, ensuring precision for higher-order moments.
O(N) for N=window_size, but guarantees precision for skewness/kurtosis.
"""
import collections
import math
from typing import Dict


class SlidingWindowStats:
    """
    Sliding window statistics.
    
    Computes: mean, std (Bessel), skewness (Fisher-Pearson), kurtosis (Fisher excess)
    All using unbiased estimators.
    """
    
    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.buffer = collections.deque(maxlen=window_size)
    
    @property
    def n(self) -> int:
        return len(self.buffer)
    
    def update(self, x: float) -> None:
        self.buffer.append(x)
    
    def get_stats(self) -> Dict[str, float]:
        """
        Compute complete statistics.
        
        Returns:
            mean, std (Bessel), skew (Fisher g1 unbiased), kurt (Fisher excess unbiased), n
        """
        n = len(self.buffer)
        if n < 2:
            return {"mean": 0.0, "std": 0.1, "skew": 0.0, "kurt": 0.0, "n": n}
        
        data = list(self.buffer)
        
        # Mean
        mean = sum(data) / n
        
        # Variance (Bessel correction: n-1)
        m2 = sum((x - mean) ** 2 for x in data)
        var = m2 / (n - 1)
        std = math.sqrt(var) if var > 1e-10 else 0.01
        
        # Population std for moment standardization
        std_pop = math.sqrt(m2 / n) if m2 > 0 else 0.01
        
        # Skewness (Fisher-Pearson, bias-corrected)
        skew = 0.0
        if n >= 3 and std_pop > 1e-6:
            m3 = sum((x - mean) ** 3 for x in data) / n
            g1 = m3 / (std_pop ** 3)
            skew = g1 * math.sqrt(n * (n - 1)) / (n - 2)
        
        # Kurtosis (Fisher excess, bias-corrected)
        kurt = 0.0
        if n >= 4 and std_pop > 1e-6:
            m4 = sum((x - mean) ** 4 for x in data) / n
            g2 = (m4 / (std_pop ** 4)) - 3.0
            kurt = ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))
        
        return {"mean": mean, "std": std, "skew": skew, "kurt": kurt, "n": n}
    
    @property
    def mean(self) -> float:
        if not self.buffer:
            return 0.0
        return sum(self.buffer) / len(self.buffer)
    
    @property
    def std(self) -> float:
        if len(self.buffer) < 2:
            return 0.1
        m = self.mean
        var = sum((x - m) ** 2 for x in self.buffer) / (len(self.buffer) - 1)
        return math.sqrt(var) if var > 1e-10 else 0.01
