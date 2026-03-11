"""
METIS Dreaming Daemon — Autonomous Knowledge Evolution

Phase 18: Background worker that monitors GPU utilization, detects idle periods,
loads unresolved knowledge gaps from CuriosityDriver storage, formats them as
GRPO training data, and launches headless training via subprocess.

Closed loop:
  Runtime inference → CuriosityDriver records knowledge gaps →
  Daemon detects GPU idle → formats JSONL → launches GRPO training →
  marks gaps resolved → model improves → fewer gaps next time

Usage:
    python -m metis.daemon --gap-path data/knowledge_gaps.json \\
        --base-model experiment_output_dpo_balanced/metis_dpo_cognitive \\
        --output-dir experiment_output_dreams
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from metis.sentinel import DegradationSentinel

logger = logging.getLogger("metis.daemon")

# ── Critical gap categories worth training on ──
CRITICAL_CATEGORIES = {"complete_unknown", "sustained_confusion", "se_verified_uncertainty"}
# Entropy peak threshold: even non-critical categories qualify if entropy is extreme
ENTROPY_PEAK_THRESHOLD = 2.0
# Lock file to prevent concurrent daemon instances
LOCK_FILE = "/tmp/metis_daemon.lock"


class DreamingDaemon:
    """Autonomous knowledge evolution — GPU idle → train on knowledge gaps."""

    def __init__(
        self,
        gap_storage_path: Path,
        training_script: str = "tools/run_dream_training.py",
        base_model: str = "experiment_output_dpo_balanced/metis_dpo_cognitive",
        output_dir: str = "experiment_output_dreams",
        check_interval_minutes: int = 30,
        gpu_idle_threshold: float = 10.0,
        gpu_idle_duration_seconds: int = 120,
        min_critical_gaps: int = 5,
        max_gaps_per_batch: int = 50,
        sentinel: Optional["DegradationSentinel"] = None,
        evolutionary: bool = False,
        training_mode: str = "grpo",
        golden_dataset_path: Optional[str] = None,
        blend_ratio: float = 0.2,
        anchor_model: Optional[str] = None,
    ):
        """
        Args:
            gap_storage_path: Path to CuriosityDriver JSON storage file.
            training_script: Path to headless training script.
            base_model: Base model path for GRPO training.
            output_dir: Output directory for training artifacts.
            check_interval_minutes: How often to check for idle GPU (minutes).
            gpu_idle_threshold: GPU utilization % below which is "idle".
            gpu_idle_duration_seconds: GPU must be idle for this long before training.
            min_critical_gaps: Minimum unresolved critical gaps to trigger training.
            max_gaps_per_batch: Maximum gaps per training batch.
            sentinel: DegradationSentinel instance for post-training regression
                gate. If None, training results are accepted unconditionally
                (DANGEROUS for unattended operation).
            evolutionary: If True AND sentinel passes, promote the merged
                model as the new base for the next training cycle. This
                enables compounding self-improvement but requires the
                sentinel to prevent mode collapse.
            training_mode: Training strategy. 'grpo' = subprocess GRPO (default).
                'egts' = in-process EGTS + Counterfactual + CPT/DPO dual-track
                via night_training.py pipeline.
            golden_dataset_path: Path to golden anchor JSONL file containing
                high-quality exemplars. Mixed into every training batch to
                prevent catastrophic forgetting (Experience Replay).
                Format: one {"prompt": "..."} per line.
            blend_ratio: Fraction of the final dataset that comes from gap
                prompts.  Default 0.2 means 20% gaps, 80% golden anchors.
                Formula: n_golden = n_gaps * (1 - blend_ratio) / blend_ratio
            anchor_model: Path or HuggingFace ID of the pristine pre-trained
                base model used as the KL-divergence anchor.  REQUIRED when
                evolutionary=True (Phase 23.5 rigid constraint).
        """
        # ── Phase 23.5: Evolutionary Anchor Binding ──
        # If continuous evolution is active, the anchor model is MANDATORY
        # to prevent catastrophic compounding drift across training cycles.
        if evolutionary and anchor_model is None:
            raise RuntimeError(
                "Evolutionary mode active: --anchor-model is strictly required "
                "to prevent catastrophic compounding drift. Provide the pristine "
                "pre-trained base model path (e.g. Qwen/Qwen2.5-7B-Instruct)."
            )
        if training_mode not in ("grpo", "egts"):
            raise ValueError(f"training_mode must be 'grpo' or 'egts', got '{training_mode}'")
        self._training_mode = training_mode
        self._gap_path = Path(gap_storage_path)
        self._training_script = training_script
        self._base_model = base_model
        self._output_dir = output_dir
        self._check_interval = check_interval_minutes * 60  # seconds
        self._gpu_idle_threshold = gpu_idle_threshold
        self._gpu_idle_duration = gpu_idle_duration_seconds
        self._min_gaps = min_critical_gaps
        self._max_gaps = max_gaps_per_batch
        self._stop_event = threading.Event()
        self._current_proc: Optional[subprocess.Popen] = None
        self._training_count = 0
        self._sentinel = sentinel
        self._evolutionary = evolutionary
        self._rollback_count = 0
        self._golden_dataset_path: Optional[Path] = (
            Path(golden_dataset_path) if golden_dataset_path else None
        )
        self._blend_ratio = max(0.01, min(1.0, blend_ratio))
        self._anchor_model = anchor_model

    def run(self) -> None:
        """Main daemon loop: sleep → check GPU → load gaps → train → mark resolved."""
        logger.info("=" * 60)
        logger.info("  METIS Dreaming Daemon started")
        logger.info(f"  Gap storage: {self._gap_path}")
        logger.info(f"  Check interval: {self._check_interval // 60} min")
        logger.info(f"  GPU idle threshold: {self._gpu_idle_threshold}%")
        logger.info("=" * 60)

        # Acquire lock
        if not self._acquire_lock():
            logger.error("Another daemon instance is running. Exiting.")
            return

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            while not self._stop_event.is_set():
                try:
                    self._cycle()
                except Exception as e:
                    logger.error(f"Daemon cycle error: {e}", exc_info=True)

                # Sleep in small increments so we can respond to stop quickly
                for _ in range(self._check_interval):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
        finally:
            self._release_lock()
            logger.info("Dreaming Daemon stopped.")

    def stop(self) -> None:
        """Graceful shutdown via threading.Event."""
        logger.info("Stop requested.")
        self._stop_event.set()
        if self._current_proc is not None:
            logger.info("Terminating active training process...")
            self._current_proc.terminate()

    # ── Internal: single daemon cycle ──

    def _cycle(self) -> None:
        """One check-and-maybe-train cycle."""
        # 1. Check GPU idle
        if not self._check_gpu_idle():
            logger.debug("GPU not idle, skipping cycle.")
            return

        # 2. Load unresolved gaps
        gaps = self._load_unresolved_gaps()
        if not gaps:
            logger.debug("No unresolved gaps found.")
            return

        # 3. Filter critical gaps
        critical = self._filter_critical(gaps)
        if len(critical) < self._min_gaps:
            logger.info(
                f"Only {len(critical)} critical gaps "
                f"(need {self._min_gaps}), skipping."
            )
            return

        # 4. Format training dataset
        batch = critical[:self._max_gaps]
        dataset_path = self._format_jsonl_dataset(batch)

        # 5. Launch training (mode-dependent)
        if self._training_mode == "egts":
            logger.info(
                f"Launching EGTS night training: {len(batch)} gaps"
            )
            success = self._run_egts_training(batch)
        else:
            logger.info(
                f"Launching GRPO dream training: {len(batch)} gaps → {dataset_path}"
            )
            proc = self._launch_training(dataset_path)
            if proc is None:
                return
            # 6. Monitor and wait
            success = self._monitor_and_wait(proc)

        if not success:
            logger.warning("Dream training failed. Gaps NOT marked resolved.")
            return

        # 7. Post-training sentinel gate
        if self._sentinel is not None:
            verdict = self._run_sentinel_gate()
            if not verdict:
                logger.warning(
                    "Sentinel REJECTED new weights. "
                    "Gaps NOT marked resolved — will retry next cycle."
                )
                return

        # 8. Mark resolved on success
        self._mark_resolved(batch)
        self._training_count += 1
        logger.info(
            f"Dream training #{self._training_count} complete. "
            f"Resolved {len(batch)} gaps."
        )

        # Cleanup temp dataset
        try:
            os.unlink(dataset_path)
        except OSError:
            pass

    # ── GPU idle detection ──

    def _check_gpu_idle(self) -> bool:
        """Check if GPU is idle. Primary: pynvml. Fallback: nvidia-smi."""
        try:
            return self._check_gpu_idle_pynvml()
        except Exception:
            return self._check_gpu_idle_smi()

    def _check_gpu_idle_pynvml(self) -> bool:
        """Use pynvml for accurate GPU utilization reading."""
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # Check utilization over the idle duration
            start = time.monotonic()
            while time.monotonic() - start < self._gpu_idle_duration:
                if self._stop_event.is_set():
                    return False
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                if util.gpu > self._gpu_idle_threshold:
                    return False
                time.sleep(5)
            return True
        finally:
            pynvml.nvmlShutdown()

    def _check_gpu_idle_smi(self) -> bool:
        """Fallback: parse nvidia-smi output."""
        start = time.monotonic()
        while time.monotonic() - start < self._gpu_idle_duration:
            if self._stop_event.is_set():
                return False
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=10,
                )
                util = float(result.stdout.strip().split("\n")[0])
                if util > self._gpu_idle_threshold:
                    return False
            except Exception as e:
                logger.debug(f"nvidia-smi failed: {e}")
                return False
            time.sleep(5)
        return True

    # ── Gap loading and filtering ──

    def _load_unresolved_gaps(self) -> List[Dict[str, Any]]:
        """Read CuriosityDriver JSON storage, filter resolved=False."""
        if not self._gap_path.exists():
            return []
        try:
            with open(self._gap_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [g for g in data if not g.get("resolved", False)]
        except Exception as e:
            logger.warning(f"Failed to load gaps: {e}")
            return []

    def _filter_critical(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only critical gaps worth training on.

        Critical = category in {complete_unknown, sustained_confusion,
        se_verified_uncertainty} OR entropy_peak > threshold.
        """
        critical = []
        for g in gaps:
            cat = g.get("category", "")
            peak = g.get("entropy_peak", 0.0)
            if cat in CRITICAL_CATEGORIES or peak > ENTROPY_PEAK_THRESHOLD:
                critical.append(g)
        # Sort by entropy_peak descending (worst gaps first)
        critical.sort(key=lambda g: g.get("entropy_peak", 0.0), reverse=True)
        return critical

    # ── Dataset formatting ──

    def _format_jsonl_dataset(self, gaps: List[Dict[str, Any]]) -> Path:
        """Write training prompts as JSONL for run_dream_training.py.

        Phase 23 — Dynamic Data Blending (Experience Replay):
        If a golden anchor dataset is configured, gap prompts are blended
        with randomly-sampled golden prompts at the configured ratio.
        This prevents catastrophic forgetting by ensuring the model
        revisits high-quality exemplars every training cycle.

        Mathematical constraint:
            n_gaps     = len(gaps)
            n_golden   = n_gaps * (1 - blend_ratio) / blend_ratio
            final_size = n_gaps + n_golden
            gap_fraction ≈ blend_ratio  (e.g. 0.2 → 20% gaps, 80% golden)

        Format matches TRL GRPOTrainer expectation:
            {"prompt": "...", "_meta": {"category": ..., "entropy_peak": ..., "source": "gap"|"golden"}}
        """
        # ── Build gap records ──
        records: List[Dict[str, Any]] = []
        for g in gaps:
            records.append({
                "prompt": g["query"],
                "_meta": {
                    "category": g.get("category", "unknown"),
                    "entropy_peak": g.get("entropy_peak", 0.0),
                    "entropy_mean": g.get("entropy_mean", 0.0),
                    "context": g.get("context", ""),
                    "timestamp": g.get("timestamp", ""),
                    "source": "gap",
                },
            })

        # ── Blend golden anchor data (Experience Replay) ──
        golden_records = self._load_golden_samples(len(gaps))
        if golden_records:
            records.extend(golden_records)
            random.shuffle(records)
            logger.info(
                f"Blended dataset: {len(gaps)} gaps + {len(golden_records)} golden "
                f"= {len(records)} total (ratio={self._blend_ratio:.0%} gap)"
            )
        else:
            logger.info(f"No golden dataset — pure gap training: {len(gaps)} prompts")

        # ── Write JSONL ──
        fd, path = tempfile.mkstemp(
            suffix=".jsonl", prefix="metis_dream_", dir="/tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(records)} prompts to {path}")
        return Path(path)

    def _load_golden_samples(self, n_gaps: int) -> List[Dict[str, Any]]:
        """Load and sample golden anchor prompts for experience replay.

        Returns:
            List of JSONL-ready records, or empty list if no golden dataset.
        """
        if self._golden_dataset_path is None or not self._golden_dataset_path.exists():
            return []

        # Load all golden prompts
        all_golden: List[str] = []
        try:
            with open(self._golden_dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    prompt = obj.get("prompt", "")
                    if prompt:
                        all_golden.append(prompt)
        except Exception as e:
            logger.warning(f"Failed to load golden dataset: {e}")
            return []

        if not all_golden:
            return []

        # n_golden = n_gaps * (1 - blend_ratio) / blend_ratio
        if self._blend_ratio >= 1.0:
            return []  # 100% gaps, no golden needed
        n_golden = int(n_gaps * (1.0 - self._blend_ratio) / self._blend_ratio)
        n_golden = max(1, n_golden)

        # Sample with replacement if pool is smaller than required
        if len(all_golden) >= n_golden:
            sampled = random.sample(all_golden, n_golden)
        else:
            sampled = random.choices(all_golden, k=n_golden)

        return [
            {
                "prompt": p,
                "_meta": {"source": "golden"},
            }
            for p in sampled
        ]

    # ── Training subprocess ──

    def _launch_training(self, dataset_path: Path) -> Optional[subprocess.Popen]:
        """Launch headless training via subprocess."""
        cmd = [
            sys.executable, self._training_script,
            "--dataset", str(dataset_path),
            "--base-model", self._base_model,
            "--output-dir", self._output_dir,
        ]
        # Phase 23: pass KL-anchor to training subprocess
        if self._anchor_model:
            cmd.extend(["--anchor-model", self._anchor_model])
        logger.info(f"Command: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._current_proc = proc
            return proc
        except Exception as e:
            logger.error(f"Failed to launch training: {e}")
            return None

    def _monitor_and_wait(self, proc: subprocess.Popen) -> bool:
        """Poll process, stream logs, return success."""
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    logger.info(f"[DREAM] {line}")
                if self._stop_event.is_set():
                    proc.terminate()
                    return False
            proc.wait()
            self._current_proc = None
            success = proc.returncode == 0
            logger.info(
                f"Training exited with code {proc.returncode}"
            )
            return success
        except Exception as e:
            logger.error(f"Training monitoring error: {e}")
            self._current_proc = None
            return False

    # ── EGTS in-process training ──

    def _run_egts_training(self, gaps: List[Dict[str, Any]]) -> bool:
        """Run EGTS night training in-process via night_training.py pipeline.

        Uses Entropy-Guided Tree Search + Counterfactual Simulation +
        CPT/DPO dual-track for deeper knowledge acquisition.
        """
        try:
            from metis.pipeline.night_training import run_night_training
            from metis.pipeline.config import ExperimentConfig

            # Write gaps to temp JSON for night_training to consume
            import tempfile
            fd, gap_path = tempfile.mkstemp(
                suffix=".json", prefix="metis_egts_gaps_"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(gaps, f, ensure_ascii=False, indent=2)

            config = ExperimentConfig(model_name=self._base_model)
            run_night_training(
                config,
                knowledge_gap_path=gap_path,
                output_dir=self._output_dir,
            )

            # Cleanup temp file
            try:
                os.unlink(gap_path)
            except OSError:
                pass

            logger.info("EGTS night training completed successfully.")
            return True
        except Exception as e:
            logger.error(f"EGTS training failed: {e}", exc_info=True)
            return False

    # ── Sentinel gate ──

    def _run_sentinel_gate(self) -> bool:
        """Run the degradation sentinel on newly trained model.

        Returns True if the model passes (or no sentinel configured).
        Returns False if rollback was triggered.
        """
        if self._sentinel is None:
            return True

        # Check emergency brake first
        if self._sentinel.should_halt():
            logger.critical(
                "EMERGENCY BRAKE ENGAGED — too many consecutive regressions. "
                "Daemon will stop after this cycle."
            )
            self._stop_event.set()
            return False

        merged_path = os.path.join(self._output_dir, "merged")
        if not os.path.exists(merged_path):
            logger.warning(
                f"No merged model at {merged_path} — skipping sentinel."
            )
            return True

        verdict = self._sentinel.evaluate(merged_path)

        if verdict.passed:
            logger.info(
                f"Sentinel PASS: {verdict.reason} "
                f"(acc={verdict.overall_accuracy*100:.1f}%)"
            )
            # Evolutionary mode: promote merged model as new base
            if self._evolutionary:
                promoted = self._sentinel.promote(
                    self._output_dir, self._base_model
                )
                if promoted:
                    logger.info(
                        f"Evolutionary promotion: {merged_path} → "
                        f"{self._base_model}"
                    )
            return True
        else:
            logger.warning(
                f"Sentinel FAIL: {verdict.reason} "
                f"(acc={verdict.overall_accuracy*100:.1f}%)"
            )
            archive = self._sentinel.rollback(self._output_dir)
            self._rollback_count += 1
            logger.warning(
                f"Rollback #{self._rollback_count}: "
                f"weights archived to {archive}"
            )
            return False

    # ── Gap resolution ──

    def _mark_resolved(self, gaps: List[Dict[str, Any]]) -> None:
        """Update CuriosityDriver storage JSON: set resolved=True for trained gaps."""
        if not self._gap_path.exists():
            return
        try:
            with open(self._gap_path, "r", encoding="utf-8") as f:
                all_gaps = json.load(f)

            resolved_queries = {g["query"] for g in gaps}
            for g in all_gaps:
                if g["query"] in resolved_queries:
                    g["resolved"] = True

            with open(self._gap_path, "w", encoding="utf-8") as f:
                json.dump(all_gaps, f, ensure_ascii=False, indent=2)
            logger.info(f"Marked {len(resolved_queries)} gaps as resolved.")
        except Exception as e:
            logger.warning(f"Failed to mark gaps resolved: {e}")

    # ── Lock file management ──

    def _acquire_lock(self) -> bool:
        """Acquire a lock file to prevent concurrent daemon instances."""
        try:
            if os.path.exists(LOCK_FILE):
                # Check if PID in lock file is still alive
                with open(LOCK_FILE, "r") as f:
                    pid = int(f.read().strip())
                try:
                    os.kill(pid, 0)  # Check if process exists
                    return False  # Process alive → can't acquire
                except OSError:
                    pass  # Process dead → stale lock, take over
            with open(LOCK_FILE, "w") as f:
                f.write(str(os.getpid()))
            return True
        except Exception as e:
            logger.warning(f"Lock acquisition error: {e}")
            return False

    def _release_lock(self) -> None:
        """Release the lock file."""
        try:
            if os.path.exists(LOCK_FILE):
                os.unlink(LOCK_FILE)
        except OSError:
            pass

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info(f"Received signal {signum}, stopping...")
        self.stop()


def main() -> None:
    """CLI entry point for the Dreaming Daemon."""
    parser = argparse.ArgumentParser(
        description="METIS Dreaming Daemon — Autonomous Knowledge Evolution"
    )
    parser.add_argument(
        "--gap-path", required=True,
        help="Path to CuriosityDriver JSON storage file",
    )
    parser.add_argument(
        "--base-model",
        default="experiment_output_dpo_balanced/metis_dpo_cognitive",
        help="Base model path for training",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_output_dreams",
        help="Output directory for training artifacts",
    )
    parser.add_argument(
        "--training-script",
        default="tools/run_dream_training.py",
        help="Path to headless training script",
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Check interval in minutes (default: 30)",
    )
    parser.add_argument(
        "--gpu-threshold", type=float, default=10.0,
        help="GPU utilization %% below which is idle (default: 10)",
    )
    parser.add_argument(
        "--min-gaps", type=int, default=5,
        help="Minimum critical gaps to trigger training (default: 5)",
    )
    parser.add_argument(
        "--max-gaps", type=int, default=50,
        help="Maximum gaps per training batch (default: 50)",
    )
    parser.add_argument(
        "--training-mode", choices=["grpo", "egts"], default="grpo",
        help="Training strategy: 'grpo' (subprocess GRPO) or 'egts' (EGTS night training)",
    )
    parser.add_argument(
        "--golden-dataset", default=None,
        help="Path to golden anchor JSONL for experience replay (Phase 23)",
    )
    parser.add_argument(
        "--blend-ratio", type=float, default=0.2,
        help="Fraction of gaps in blended dataset (default: 0.2 = 20%% gaps, 80%% golden)",
    )
    parser.add_argument(
        "--anchor-model", type=str, default=None,
        help="Pristine pre-trained model for KL-divergence anchor (Phase 23). "
             "REQUIRED when --evolutionary is set.",
    )
    parser.add_argument(
        "--evolutionary", action="store_true", default=False,
        help="Enable compounding self-improvement: promote sentinel-approved "
             "models as new base. Requires --anchor-model (Phase 23.5).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    daemon = DreamingDaemon(
        gap_storage_path=Path(args.gap_path),
        training_script=args.training_script,
        base_model=args.base_model,
        output_dir=args.output_dir,
        check_interval_minutes=args.interval,
        gpu_idle_threshold=args.gpu_threshold,
        min_critical_gaps=args.min_gaps,
        max_gaps_per_batch=args.max_gaps,
        training_mode=args.training_mode,
        golden_dataset_path=args.golden_dataset,
        blend_ratio=args.blend_ratio,
        anchor_model=args.anchor_model,
        evolutionary=args.evolutionary,
    )
    daemon.run()


if __name__ == "__main__":
    main()
