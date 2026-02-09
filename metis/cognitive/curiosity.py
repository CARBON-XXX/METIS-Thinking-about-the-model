"""
METIS Curiosity Driver
The engine of autonomous self-evolution

AGI records all tasks that trigger high-entropy thresholds at runtime -> knowledge gaps.
During idle time (Dreaming Phase), performs targeted learning on these high-entropy samples.

Closed loop: detect confusion -> record -> targeted learning -> eliminate confusion

This is the infrastructure for AGI autonomous self-evolution (Self-Supervised Learning).
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

from ..core.types import KnowledgeGap


# ── Constants ──
GAP_Z_THRESHOLD_DEFAULT = 1.0       # Default z-score threshold
MAX_GAPS_DEFAULT = 1000             # Default max gap records

# Gap detection thresholds (based on gap_z_threshold)
PEAK_Z_MULTIPLIER = 2.0             # Peak multiplier
MEAN_Z_MULTIPLIER = 0.5             # Mean multiplier
HIGH_Z_RATIO_THRESHOLD = 0.3        # High-z ratio threshold

# Gap categorization thresholds
CAT_COMPLETE_UNKNOWN_Z = 3.0        # Complete unknown
CAT_SUSTAINED_RATIO = 0.5           # Sustained confusion ratio
CAT_SPIKE_Z = 2.0                   # Local spike


class CuriosityDriver:
    """
    Curiosity Driver.
    
    Responsibilities:
    1. Real-time observation of semantic entropy
    2. Identify knowledge gaps (high-entropy regions)
    3. Persistent storage for Dreaming Phase
    4. Provide targeted training data for learning
    """

    def __init__(
        self,
        gap_z_threshold: float = GAP_Z_THRESHOLD_DEFAULT,
        storage_path: Optional[str] = None,
        max_gaps: int = MAX_GAPS_DEFAULT,
    ):
        # z-score threshold: peak or mean z-score exceeding this -> knowledge gap
        self._gap_z_threshold = gap_z_threshold
        self._storage_path = Path(storage_path) if storage_path else None
        self._max_gaps = max_gaps
        
        self._gaps: List[KnowledgeGap] = []
        self._current_query = ""
        self._current_context = ""
        self._session_entropies: List[float] = []
        self._session_z_scores: List[float] = []
        
        if self._storage_path and self._storage_path.exists():
            self._load()
    
    def start_session(self, query: str, context: str = "") -> None:
        """Start new session"""
        self._current_query = query
        self._current_context = context
        self._session_entropies = []
        self._session_z_scores = []
    
    def observe(self, semantic_entropy: float, z_score: float = 0.0) -> None:
        """Observe one token's semantic entropy and z-score"""
        self._session_entropies.append(semantic_entropy)
        self._session_z_scores.append(z_score)
    
    def end_session(self) -> Optional[KnowledgeGap]:
        """
        End session, determine if a knowledge gap was detected based on z-score distribution.
        
        Detection logic (relative distribution, not absolute values):
        - z-score peak > threshold -> significant entropy spike exists
        - z-score mean > threshold * 0.5 -> overall elevated
        - high z-score ratio > 30% -> sustained confusion
        
        Returns:
            KnowledgeGap if detected, else None
        """
        if not self._session_entropies:
            return None
        
        mean_e = sum(self._session_entropies) / len(self._session_entropies)
        peak_e = max(self._session_entropies)
        
        # z-score based judgment (adaptive, similar to Bonferroni multiple testing)
        # Requires >= 2/3 criteria to classify as gap, avoids single spike false trigger
        if self._session_z_scores:
            peak_z = max(self._session_z_scores)
            mean_z = sum(self._session_z_scores) / len(self._session_z_scores)
            high_z_ratio = sum(1 for z in self._session_z_scores if z > self._gap_z_threshold) / len(self._session_z_scores)
            
            criteria = 0
            if peak_z > self._gap_z_threshold * PEAK_Z_MULTIPLIER:  # A. Extreme spike exists
                criteria += 1
            if mean_z > self._gap_z_threshold * MEAN_Z_MULTIPLIER:   # B. Overall elevated
                criteria += 1
            if high_z_ratio > HIGH_Z_RATIO_THRESHOLD:                     # C. >30% tokens have high z
                criteria += 1
            is_gap = criteria >= 2
        else:
            is_gap = False
        
        if is_gap:
            gap = KnowledgeGap(
                query=self._current_query,
                context=self._current_context,
                entropy_peak=peak_e,
                entropy_mean=mean_e,
                category=self._categorize_z(self._session_z_scores),
                timestamp=datetime.now().isoformat(),
            )
            self._add_gap(gap)
            return gap
        
        return None
    
    def record_se_gap(
        self,
        query: str,
        semantic_entropy: float,
        n_clusters: int,
        n_samples: int,
    ) -> KnowledgeGap:
        """
        SE feedback loop: record gap when Kuhn et al. SE verification finds genuine uncertainty.
        
        Implements System 2 -> CuriosityDriver information backflow:
        High SE uncertainty -> record as knowledge gap -> for Dreaming Phase targeted learning
        
        Args:
            query: User question
            semantic_entropy: SE value (bits)
            n_clusters: Number of semantic equivalence classes
            n_samples: Total samples
        """
        gap = KnowledgeGap(
            query=query,
            context=f"SE={semantic_entropy:.2f}, clusters={n_clusters}/{n_samples}",
            entropy_peak=semantic_entropy,
            entropy_mean=semantic_entropy,
            category="se_verified_uncertainty",
            timestamp=datetime.now().isoformat(),
        )
        self._add_gap(gap)
        return gap

    def get_unresolved_gaps(self) -> List[KnowledgeGap]:
        """Get all unresolved knowledge gaps"""
        return [g for g in self._gaps if not g.resolved]
    
    def get_training_data(self) -> List[Dict]:
        """Get training data for Dreaming Phase targeted learning"""
        return [
            {
                "query": g.query,
                "context": g.context,
                "entropy_peak": g.entropy_peak,
                "entropy_mean": g.entropy_mean,
                "category": g.category,
            }
            for g in self._gaps
            if not g.resolved
        ]
    
    def mark_resolved(self, query: str) -> None:
        """Mark gap as resolved through learning"""
        for g in self._gaps:
            if g.query == query:
                g.resolved = True
        self._save()
    
    @property
    def gap_count(self) -> int:
        return len([g for g in self._gaps if not g.resolved])
    
    def _categorize_z(self, z_scores: List[float]) -> str:
        """Categorize knowledge gap based on z-score distribution"""
        if not z_scores:
            return "mild_uncertainty"
        peak_z = max(z_scores)
        mean_z = sum(z_scores) / len(z_scores)
        high_ratio = sum(1 for z in z_scores if z > 1.0) / len(z_scores)
        
        if peak_z > 3.0:
            return "complete_unknown"    # Extreme outlier: completely unknown
        elif high_ratio > 0.5:
            return "sustained_confusion" # Sustained high-z: overall confusion
        elif peak_z > 2.0:
            return "spike_uncertainty"   # Spike: localized confusion
        return "mild_uncertainty"
    
    def _add_gap(self, gap: KnowledgeGap) -> None:
        self._gaps.append(gap)
        if len(self._gaps) > self._max_gaps:
            self._gaps = self._gaps[-self._max_gaps:]
        self._save()
    
    def _save(self) -> None:
        if not self._storage_path:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "query": g.query, "context": g.context,
                "entropy_peak": g.entropy_peak, "entropy_mean": g.entropy_mean,
                "category": g.category, "timestamp": g.timestamp,
                "resolved": g.resolved,
            }
            for g in self._gaps
        ]
        with open(self._storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load(self) -> None:
        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._gaps = [
                KnowledgeGap(
                    query=d["query"], context=d.get("context", ""),
                    entropy_peak=d["entropy_peak"], entropy_mean=d["entropy_mean"],
                    category=d.get("category", "unknown"),
                    timestamp=d.get("timestamp", ""),
                    resolved=d.get("resolved", False),
                )
                for d in data
            ]
        except Exception:
            self._gaps = []
