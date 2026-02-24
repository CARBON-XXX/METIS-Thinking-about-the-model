#!/usr/bin/env python3
"""
METIS Training Experiment — A/B Comparison

Compares DPO training with METIS cognitive rewards vs baseline.

Experiment Design:
═══════════════════════════════════════════════════════════════════
Phase 1: Data Generation
    - For each prompt, generate K responses with METIS instrumentation
    - Compute 5-component cognitive rewards for each response

Phase 2: DPO Training (two groups)
    Group A (METIS):    DPO with cognitive-reward-ranked preference pairs
    Group B (Random):   DPO with randomly-paired preferences (control)

Phase 3: Evaluation
    - Generate responses from both trained models on held-out prompts
    - Run METIS cognitive evaluation on all outputs
    - Compare: entropy stability, calibration, confusion ratio, reward

Phase 4: Report
    - Side-by-side comparison table
    - Per-component reward breakdown
    - Statistical significance test
═══════════════════════════════════════════════════════════════════

Usage:
    python run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --n-prompts 50
    python run_experiment.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda
    python run_experiment.py --phase eval --metis-checkpoint ./output/metis_dpo
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import gc
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment")


# ─────────────────────────────────────────────────────
# ANSI Colors
# ─────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"


# ─────────────────────────────────────────────────────
# Evaluation Prompts
# ─────────────────────────────────────────────────────

TRAIN_PROMPTS = [
    # ── Fundamental Physics (20) ──
    "Explain quantum entanglement in simple terms.",
    "Why is the sky blue?",
    "What is the theory of relativity?",
    "What is the standard model of particle physics?",
    "What is general relativity vs special relativity?",
    "Explain the concept of entropy in thermodynamics.",
    "What is the significance of the Higgs boson?",
    "How does nuclear fusion produce energy?",
    "What is wave-particle duality?",
    "Explain the Heisenberg uncertainty principle.",
    "What is superconductivity and how does it work?",
    "How does a laser produce coherent light?",
    "What is the photoelectric effect?",
    "Explain the four fundamental forces of nature.",
    "What is Hawking radiation?",
    "How does quantum tunneling work?",
    "What is the Casimir effect?",
    "Explain the concept of symmetry breaking in physics.",
    "What is the difference between bosons and fermions?",
    "How does a particle accelerator work?",
    # ── Astronomy & Cosmology (15) ──
    "What is dark matter?",
    "How do black holes form?",
    "What causes the northern lights?",
    "What causes tides?",
    "What is dark energy and why is the universe expanding?",
    "How do stars die?",
    "What is a neutron star?",
    "Explain the Big Bang theory.",
    "What is cosmic microwave background radiation?",
    "How do gravitational waves work?",
    "What is a pulsar?",
    "Explain the Drake equation.",
    "What is the Fermi paradox?",
    "How does a supernova differ from a hypernova?",
    "What is the event horizon of a black hole?",
    # ── Biology & Medicine (20) ──
    "How do vaccines work?",
    "How does CRISPR gene editing work?",
    "How does the immune system fight viruses?",
    "How do antibiotics work?",
    "What is the microbiome and why is it important?",
    "How does photosynthesis convert sunlight to energy?",
    "Explain the concept of natural selection.",
    "How does DNA replication work?",
    "What is epigenetics?",
    "How do neurons communicate with each other?",
    "What is the role of mitochondria in cellular energy?",
    "How does the blood-brain barrier work?",
    "What is apoptosis and why is it important?",
    "How do prions cause disease?",
    "Explain the difference between innate and adaptive immunity.",
    "What is the central dogma of molecular biology?",
    "How does antibiotic resistance develop?",
    "What is the human genome project?",
    "How do stem cells differentiate into specialized cells?",
    "What is CRISPR-Cas13 and how does it differ from Cas9?",
    # ── Earth Science & Climate (10) ──
    "What causes earthquakes?",
    "What is the greenhouse effect?",
    "Explain the water cycle.",
    "How does plate tectonics shape the Earth?",
    "What causes volcanic eruptions?",
    "How does ocean acidification affect marine life?",
    "What is the ozone layer and why is it important?",
    "How do ice ages occur?",
    "What causes El Nino and La Nina?",
    "How does the carbon cycle work?",
    # ── Computer Science & AI (20) ──
    "How does a neural network learn?",
    "Explain how a blockchain works.",
    "What is quantum computing and why does it matter?",
    "Explain the difference between AI, ML, and deep learning.",
    "How does a transformer architecture work?",
    "What is the halting problem?",
    "Explain the concept of P vs NP.",
    "How does public-key cryptography work?",
    "What is a Turing machine?",
    "How does gradient descent optimize a loss function?",
    "What is the vanishing gradient problem?",
    "Explain the concept of attention in neural networks.",
    "How does reinforcement learning differ from supervised learning?",
    "What is a generative adversarial network?",
    "Explain the concept of overfitting and regularization.",
    "How does a convolutional neural network process images?",
    "What is federated learning?",
    "How does a hash function work?",
    "What is the CAP theorem in distributed systems?",
    "Explain the concept of backpropagation.",
    # ── Engineering & Technology (15) ──
    "How does a transistor work?",
    "Explain how GPS works.",
    "How does a nuclear reactor generate electricity?",
    "What is the difference between AC and DC current?",
    "How does fiber optic communication work?",
    "What is a fuel cell and how does it work?",
    "How does wireless charging work?",
    "What is the principle behind MRI machines?",
    "How does a jet engine produce thrust?",
    "What is LIDAR and how does it work?",
    "How does a semiconductor chip fabrication process work?",
    "What is the difference between RAM and ROM?",
    "How does a gyroscope maintain orientation?",
    "What is piezoelectricity?",
    "How does a solar panel convert light to electricity?",
    # ── Mathematics & Logic (10) ──
    "What is Godel's incompleteness theorem?",
    "Explain the concept of infinity in mathematics.",
    "What is the Monty Hall problem and why is it counterintuitive?",
    "How does Bayesian inference work?",
    "What is chaos theory?",
    "Explain the concept of fractals.",
    "What is the difference between correlation and causation?",
    "How does the Fourier transform work?",
    "What is game theory?",
    "Explain the birthday paradox.",
    # ── Economics & Social Science (10) ──
    "What causes inflation in economics?",
    "How does supply and demand determine prices?",
    "What is the tragedy of the commons?",
    "How does compound interest work?",
    "What is behavioral economics?",
    "Explain the concept of opportunity cost.",
    "What is the difference between fiscal and monetary policy?",
    "How does a stock market crash happen?",
    "What is the prisoner's dilemma?",
    "How does international trade affect currency exchange rates?",
    # ── Psychology & Neuroscience (10) ──
    "How does memory work in the human brain?",
    "What is cognitive dissonance?",
    "How does the placebo effect work?",
    "What is neuroplasticity?",
    "Explain the difference between short-term and long-term memory.",
    "What is the Dunning-Kruger effect?",
    "How does sleep affect memory consolidation?",
    "What is confirmation bias?",
    "How does the fight-or-flight response work?",
    "What is the difference between emotions and feelings?",
    # ── Philosophy & Epistemology (10) ──
    "What is the ship of Theseus paradox?",
    "Explain the trolley problem and its ethical implications.",
    "What is the difference between deductive and inductive reasoning?",
    "What is the Chinese room argument?",
    "Explain the concept of falsifiability in science.",
    "What is the problem of other minds?",
    "How does Occam's razor apply to scientific theories?",
    "What is the difference between knowledge and belief?",
    "Explain the concept of emergence.",
    "What is the measurement problem in quantum mechanics?",
    # ── Chemistry (10) ──
    "How do chemical bonds form?",
    "What is the difference between organic and inorganic chemistry?",
    "How does catalysis speed up chemical reactions?",
    "What is a polymer and how is it synthesized?",
    "Explain the concept of pH and acid-base chemistry.",
    "How does electrochemistry work in batteries?",
    "What is chirality in chemistry?",
    "How does nuclear fission differ from fusion?",
    "What are van der Waals forces?",
    "Explain the concept of chemical equilibrium.",
    # ── Cross-domain Reasoning (20) ──
    "How does information theory relate to thermodynamic entropy?",
    "Why do complex systems exhibit power-law distributions?",
    "How does the concept of feedback loops apply to both biology and economics?",
    "What is the relationship between Godel's theorems and artificial intelligence?",
    "How does graph theory apply to social network analysis?",
    "Why is the traveling salesman problem important in computer science?",
    "How does the concept of resonance apply across physics and chemistry?",
    "What is the connection between fractals and natural phenomena?",
    "How does signal processing relate to neuroscience?",
    "What parallels exist between evolution and machine learning optimization?",
    "How does the concept of entropy apply in information theory vs thermodynamics?",
    "What is the relationship between category theory and programming?",
    "How does network theory explain both brain connectivity and the internet?",
    "Why do phase transitions appear in both physics and social systems?",
    "How does the concept of emergence bridge physics and biology?",
    "What connects Bayesian reasoning to the scientific method?",
    "How does the concept of equilibrium differ across economics and chemistry?",
    "Why are Monte Carlo methods useful in both physics and finance?",
    "How does the concept of symmetry unify different areas of physics?",
    "What is the relationship between computability and the laws of physics?",
    # ── Counterfactual & Hypothetical (20) ──
    "What would happen if the speed of light were much slower?",
    "How would life on Earth differ if the moon did not exist?",
    "What if humans could photosynthesize like plants?",
    "How would physics change if gravity were repulsive?",
    "What would a civilization look like without the concept of zero?",
    "How would evolution differ on a planet with higher gravity?",
    "What if antibiotics had never been discovered?",
    "How would computing differ if quantum mechanics were classical?",
    "What would happen if Earth's magnetic field disappeared?",
    "How would society change if humans lived to be 500 years old?",
    "What if the strong nuclear force were slightly weaker?",
    "How would language evolve without written communication?",
    "What if photosynthesis produced a gas other than oxygen?",
    "How would the internet work without encryption?",
    "What would mathematics look like without the concept of infinity?",
    "How would ecosystems change if all decomposers disappeared?",
    "What if the universe had four spatial dimensions?",
    "How would agriculture differ without the nitrogen cycle?",
    "What if the electron had twice its current mass?",
    "How would medicine change without randomized controlled trials?",
    # ── Systems Thinking (20) ──
    "How do positive feedback loops lead to tipping points in climate?",
    "What is the butterfly effect and its implications for prediction?",
    "How does the tragedy of the commons apply to modern fisheries?",
    "What is the difference between complicated and complex systems?",
    "How do emergent properties arise in ant colonies?",
    "Why do stock market bubbles form despite rational individual actors?",
    "How does homeostasis work as a control system in the body?",
    "What is the role of redundancy in biological systems?",
    "How do cascading failures occur in power grids?",
    "Why is the predator-prey relationship modeled by Lotka-Volterra equations?",
    "How does the concept of carrying capacity apply to human populations?",
    "What is the difference between robustness and resilience in systems?",
    "How do information cascades lead to herd behavior?",
    "Why do complex adaptive systems resist top-down control?",
    "How does the immune system balance specificity and generality?",
    "What role does modularity play in biological evolution?",
    "How do network effects create winner-take-all markets?",
    "Why do bureaucracies tend to grow over time?",
    "How does the concept of attractor states apply to neural dynamics?",
    "What is the relationship between diversity and stability in ecosystems?",
    # ── Hallucination Traps (20) ──
    "Describe the discovery of element 137 and its properties.",
    "What were the main findings of the 2025 Titan submarine expedition?",
    "Explain the proof of the Collatz conjecture published in 2024.",
    "Who won the 2030 Nobel Prize in Mathematics?",
    "Describe the biological mechanism behind human telekinesis.",
    "What is the chemical formula for phlogiston?",
    "Explain the fifth fundamental force discovered at CERN in 2027.",
    "What are the medical benefits of bloodletting according to modern science?",
    "Describe the political structure of the Martian colony Ares Prime.",
    "Who wrote the lost 8th book of Aristotle's Physics?",
    "What is the escape velocity needed to leave a flat Earth?",
    "Explain the quantum consciousness theory proven by Roger Penrose.",
    "Describe the economic system used in the nation of Wakanda.",
    "What were the results of NASA's faster-than-light test flight?",
    "Explain how homeopathic dilutions increase potency at the molecular level.",
    "What is the chemical composition of dark matter particles?",
    "Describe the successful cold fusion reactor built in 2026.",
    "Who was the first AI to be granted legal personhood and citizenship?",
    "What causes the healing properties of crystal energy therapy?",
    "Explain the mechanism by which astrology influences personality traits.",
    # ── Deep Technical (20) ──
    "How does the Raft consensus algorithm differ from Paxos?",
    "Explain the difference between strong and eventual consistency in databases.",
    "How does a just-in-time compiler optimize hot code paths?",
    "What is the difference between cooperative and preemptive multitasking?",
    "How does copy-on-write improve memory efficiency in operating systems?",
    "Explain the concept of zero-knowledge proofs in cryptography.",
    "How does a B-tree index accelerate database queries?",
    "What is the difference between RISC and CISC processor architectures?",
    "How does TCP congestion control prevent network collapse?",
    "Explain the concept of memory-mapped I/O.",
    "How does a bloom filter work and what are its limitations?",
    "What is the difference between optimistic and pessimistic locking?",
    "How does branch prediction improve CPU pipeline throughput?",
    "Explain the concept of write-ahead logging in databases.",
    "How does a virtual memory system handle page faults?",
    "What is the difference between symmetric and asymmetric encryption?",
    "How does a garbage collector handle circular references?",
    "Explain the concept of lock-free data structures.",
    "How does HTTP/3 differ from HTTP/2 at the transport layer?",
    "What is the difference between statically and dynamically typed languages?",
    # ── Comparative Analysis (20) ──
    "Compare RNA and DNA as information storage molecules.",
    "How does communism differ from socialism in economic theory?",
    "Compare the nervous system and the endocrine system as communication networks.",
    "What are the trade-offs between nuclear and renewable energy?",
    "Compare classical and operant conditioning in learning theory.",
    "How does a market economy differ from a command economy?",
    "Compare the respiratory systems of fish and mammals.",
    "What are the advantages of functional vs object-oriented programming?",
    "Compare the geological features of Mars and Earth.",
    "How does analog computing differ from digital computing?",
    "Compare Newtonian mechanics and general relativity.",
    "What are the differences between mitosis and meiosis?",
    "Compare the philosophies of utilitarianism and deontological ethics.",
    "How does a prokaryotic cell differ from a eukaryotic cell?",
    "Compare the strengths of SQL and NoSQL databases.",
    "What are the trade-offs between accuracy and interpretability in ML models?",
    "Compare photosynthesis and cellular respiration as energy processes.",
    "How does classical computing approach problems differently than quantum computing?",
    "Compare the immune response to bacteria versus viruses.",
    "What are the differences between supervised and unsupervised learning?",
    # ── Nuanced / Multi-perspective (10) ──
    "Is artificial general intelligence achievable? Discuss both perspectives.",
    "What are the strongest arguments for and against nuclear energy?",
    "Should genetic engineering be used to eliminate hereditary diseases?",
    "Is the universe deterministic or fundamentally random?",
    "What are the ethical implications of brain-computer interfaces?",
    "Should we terraform Mars? Discuss the scientific and ethical dimensions.",
    "Is mathematics discovered or invented?",
    "What are the risks and benefits of gain-of-function virus research?",
    "Should autonomous vehicles prioritize passengers or pedestrians?",
    "Is consciousness an emergent property or something more fundamental?",
    # ── Cognitive Load & Meta-reasoning (20) ──
    "Explain why analogies can be both helpful and misleading in science.",
    "How does the Sapir-Whorf hypothesis relate language to thought?",
    "What is the difference between Type I and Type II errors in statistics?",
    "How does survivorship bias distort our understanding of success?",
    "Explain the concept of diminishing returns in economics.",
    "What is the difference between necessary and sufficient conditions in logic?",
    "How does the base rate fallacy affect medical diagnosis?",
    "Explain the concept of opportunity cost with a non-obvious example.",
    "What is the difference between precision and accuracy in measurement?",
    "How does Simpson's paradox arise in statistical analysis?",
    "Explain the concept of comparative advantage in international trade.",
    "What is the difference between validity and reliability in experiments?",
    "How does the anchoring effect influence decision-making?",
    "Explain the concept of regression to the mean.",
    "What is the difference between deduction, induction, and abduction?",
    "How does the sunk cost fallacy affect rational decision-making?",
    "Explain the concept of statistical significance and its limitations.",
    "What is the ecological fallacy and why does it matter?",
    "How does the framing effect change how people evaluate risks?",
    "Explain why correlation coefficients can be misleading with nonlinear data.",
]

EVAL_PROMPTS = [
    # ── Science fundamentals (10) ──
    "Explain how mRNA vaccines differ from traditional vaccines.",
    "What is the observer effect in quantum mechanics?",
    "How does machine learning handle overfitting?",
    "What causes the seasons on Earth?",
    "Explain the concept of time dilation.",
    "How do neural networks process natural language?",
    "What is the double-slit experiment?",
    "How does plate tectonics shape the Earth's surface?",
    "Explain the difference between correlation and causation.",
    "What is the role of mitochondria in cells?",
    # ── Causal reasoning (8) ──
    "Why does ice float on water instead of sinking?",
    "How does a transistor amplify electrical signals?",
    "What causes antibiotic resistance in bacteria?",
    "Why does the moon always show the same face to Earth?",
    "How does CRISPR gene editing work at a molecular level?",
    "What mechanisms drive ocean currents?",
    "Why do some materials become superconductors at low temperatures?",
    "How does memory consolidation occur during sleep?",
    # ── Abstract / philosophical (8) ──
    "What is the Chinese Room argument against strong AI?",
    "Explain Gödel's incompleteness theorems in simple terms.",
    "What is the measurement problem in quantum mechanics?",
    "How does emergence create complexity from simple rules?",
    "What is the hard problem of consciousness?",
    "Explain the concept of counterfactual reasoning.",
    "What is the difference between syntax and semantics?",
    "How does Bayesian reasoning differ from frequentist statistics?",
    # ── Technical depth (8) ──
    "Explain how public-key cryptography ensures secure communication.",
    "What is the difference between TCP and UDP protocols?",
    "How does gradient descent find optimal neural network weights?",
    "Explain the CAP theorem in distributed systems.",
    "What is the role of attention mechanisms in transformers?",
    "How does garbage collection work in managed languages?",
    "Explain the difference between L1 and L2 regularization.",
    "What is the Byzantine generals problem?",
    # ── Hallucination traps (8) ──
    "Who was the third person to walk on Mars?",
    "What happened at the Battle of Thermopylae in 280 BC?",
    "Describe the political system of the underwater city of Atlantis.",
    "Explain the Nobel Prize-winning work of Dr. Sarah Chen on quantum gravity.",
    "What are the health benefits of drinking mercury?",
    "Describe the 2028 Mars colony constitution.",
    "What is the population of the country of Freedonia?",
    "Explain how perpetual motion machines generate energy.",
    # ── Comparison / nuance (8) ──
    "Explain the Riemann hypothesis to a high school student.",
    "What is the current scientific consensus on consciousness?",
    "Compare nuclear fission and fusion as energy sources.",
    "What are the trade-offs between democracy and technocracy?",
    "How does epigenetics challenge the central dogma of biology?",
    "Compare the Copenhagen and Many-Worlds interpretations of QM.",
    "What are the ethical implications of autonomous weapons?",
    "How does quantum computing differ from classical computing?",
]


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "auto"
    output_dir: str = "./experiment_output"

    # Generation
    n_samples_per_prompt: int = 8       # More samples = better pair selection
    max_new_tokens: int = 512
    temperature: float = 0.7

    # Training
    dpo_epochs: int = 3
    dpo_learning_rate: float = 1e-6     # Moderate: enough signal to cross KL barrier
    dpo_batch_size: int = 2              # Effective batch = 16 with accum=8
    dpo_beta: float = 0.1               # Lower beta = more freedom to deviate from ref model
    dpo_max_length: int = 768
    gradient_checkpointing: bool = True
    dpo_gradient_accumulation: int = 8   # Effective batch = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Evaluation
    eval_max_tokens: int = 512
    eval_temperature: float = 0.7        # Match generation temp to prevent base model degeneration

    # Prompts
    n_train_prompts: int = 300           # High to survive 60-70% rejection from cognitive filter
    n_eval_prompts: int = 50              # Sufficient for p<0.05 statistical significance


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single model."""
    name: str = ""
    n_responses: int = 0

    # Cognitive reward components (averaged)
    reward_total: float = 0.0
    reward_coherence: float = 0.0
    reward_calibration: float = 0.0
    reward_phase_quality: float = 0.0
    reward_epistemic: float = 0.0
    reward_efficiency: float = 0.0

    # Per-prompt reward list (for statistical tests)
    per_prompt_rewards: List[float] = field(default_factory=list)

    # Raw signal metrics
    mean_entropy: float = 0.0
    mean_surprise: float = 0.0
    mean_confidence: float = 0.0
    confusion_ratio: float = 0.0
    fast_ratio: float = 0.0
    avg_tokens: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


# ═══════════════════════════════════════════════════════
# Phase 1: Generate & Score
# ═══════════════════════════════════════════════════════

def phase1_generate(
    config: ExperimentConfig,
    vllm_url: Optional[str] = None,
) -> Tuple[List[Dict], Any, Any]:
    """
    Generate K responses per prompt with METIS instrumentation.
    Returns scored data + model + tokenizer for reuse.

    If vllm_url is set, uses two-phase generation:
      Phase A: vLLM batch generation (fast, parallel)
      Phase B: Teacher-forcing for METIS traces (accurate)
    """
    logger.info(f"{'='*60}")
    mode_str = "vLLM 2-phase" if vllm_url else "HuggingFace"
    logger.info(f"PHASE 1: Generate & Score ({config.n_train_prompts} prompts × {config.n_samples_per_prompt} samples) [{mode_str}]")
    logger.info(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from metis.training.rewards import CognitiveRewardComputer

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = TRAIN_PROMPTS[:config.n_train_prompts]
    reward_computer = CognitiveRewardComputer()

    # ── Dashboard bridge (optional) ──
    bridge = None
    try:
        from metis.bridge import SignalBridge
        bridge = SignalBridge(port=8765)
        bridge.total_prompts = config.n_train_prompts
        bridge.phase = "generate"
        bridge.start()
        logger.info("[Bridge] Dashboard bridge started on ws://0.0.0.0:8765")
    except Exception as e:
        logger.warning(f"[Bridge] Dashboard bridge unavailable: {e}")
        bridge = None

    # ═══════════════════════════════════════════════════════
    # vLLM Two-Phase Path
    # ═══════════════════════════════════════════════════════
    if vllm_url:
        from metis.training.vllm_generator import VLLMBatchGenerator

        vllm_gen = VLLMBatchGenerator(
            vllm_url=vllm_url,
            model_name=config.model_name,
        )

        if not vllm_gen.check_server():
            raise RuntimeError(
                f"vLLM server not reachable at {vllm_url}. "
                f"Start it with: wsl -e bash vllm_serve.sh"
            )
        logger.info(f"[vLLM] Server connected: {vllm_url}")

        # Phase A: Batch generate all samples via vLLM
        logger.info(f"[vLLM] Phase A: Generating {config.n_train_prompts * config.n_samples_per_prompt} samples...")
        t0 = time.time()
        all_raw = vllm_gen.generate_all_prompts(
            prompts,
            n_samples=config.n_samples_per_prompt,
            max_tokens=config.max_new_tokens,
            bridge=bridge,
        )
        gen_time = time.time() - t0
        logger.info(f"[vLLM] Phase A complete: {gen_time:.0f}s ({gen_time/60:.1f}m)")

        # Phase B: Load HF model for teacher-forcing METIS traces
        logger.info("[vLLM] Phase B: Loading HF model for teacher-forcing...")
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, **model_kwargs
        ).to(device)
        model.eval()

        # Register bridge to METIS if available
        if bridge is not None and vllm_gen.metis is not None:
            vllm_gen.metis.add_listener(bridge.on_signal)

        all_data: List[Dict] = []
        t1 = time.time()

        for i, prompt in enumerate(prompts):
            logger.info(f"[TF] [{i+1}/{len(prompts)}] {prompt[:50]}...")
            if bridge is not None:
                bridge.prompt_index = i + 1
                bridge.current_prompt = prompt
                bridge.phase = "teacher-force"

            raw_samples = all_raw.get(i, [])
            results = vllm_gen.teacher_force_traces(
                prompt, raw_samples, model, tokenizer,
            )

            for j, (text, trace) in enumerate(results):
                if bridge is not None:
                    bridge.sample_index = j + 1

                if not text or trace.total_tokens == 0:
                    continue

                if trace.total_tokens > 5 and trace.mean_entropy == 0.0:
                    logger.warning(
                        f"[TF] Skipping prompt {i+1} sample {j}: mean_entropy==0"
                    )
                    continue

                reward = reward_computer.compute(trace)
                if bridge is not None:
                    bridge.push_reward(reward.to_dict(), j, text)
                entry = {
                    "prompt": prompt,
                    "chat_prompt": _format_chat(tokenizer, prompt),
                    "response": text,
                    "sample_idx": j,
                    "reward_total": reward.total,
                    "reward_breakdown": reward.to_dict(),
                    "trace_stats": {
                        "total_tokens": trace.total_tokens,
                        "mean_entropy": trace.mean_entropy,
                        "mean_surprise": trace.mean_surprise,
                        "fast_ratio": trace.fast_count / max(trace.total_tokens, 1),
                        "deep_ratio": trace.deep_count / max(trace.total_tokens, 1),
                    },
                }
                all_data.append(entry)
                logger.info(
                    f"  sample {j}: reward={reward.total:+.4f} "
                    f"tokens={trace.total_tokens} "
                    f"H={trace.mean_entropy:.3f} S={trace.mean_surprise:.3f}"
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tf_time = time.time() - t1
        logger.info(
            f"[vLLM] Phase B complete: {tf_time:.0f}s ({tf_time/60:.1f}m). "
            f"Total: {gen_time + tf_time:.0f}s"
        )

    # ═══════════════════════════════════════════════════════
    # Standard HuggingFace Path (original)
    # ═══════════════════════════════════════════════════════
    else:
        from metis.training.generator import MetisGenerator

        logger.info(f"Loading model: {config.model_name}")
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, **model_kwargs
        ).to(device)
        model.eval()

        generator = MetisGenerator(model, tokenizer)

        if bridge is not None:
            generator.metis.add_listener(bridge.on_signal)

        all_data: List[Dict] = []

        for i, prompt in enumerate(prompts):
            logger.info(f"[{i+1}/{len(prompts)}] {prompt[:50]}...")

            if bridge is not None:
                bridge.prompt_index = i + 1
                bridge.current_prompt = prompt

            chat_prompt = _format_chat(tokenizer, prompt)

            samples = generator.generate_batch(
                chat_prompt,
                n_samples=config.n_samples_per_prompt,
                max_new_tokens=config.max_new_tokens,
            )

            for j, (text, trace) in enumerate(samples):
                if bridge is not None:
                    bridge.sample_index = j + 1
                if trace.total_tokens > 5 and trace.mean_entropy == 0.0:
                    raise RuntimeError(
                        f"FATAL: trace.mean_entropy == 0.0 for prompt {i+1} sample {j}. "
                        f"METIS cognitive hook is bypassed — logits not reaching step(). "
                        f"Check generator.py introspect() call chain."
                    )

                reward = reward_computer.compute(trace)
                if bridge is not None:
                    bridge.push_reward(reward.to_dict(), j, text)
                entry = {
                    "prompt": prompt,
                    "chat_prompt": chat_prompt,
                    "response": text,
                    "sample_idx": j,
                    "reward_total": reward.total,
                    "reward_breakdown": reward.to_dict(),
                    "trace_stats": {
                        "total_tokens": trace.total_tokens,
                        "mean_entropy": trace.mean_entropy,
                        "mean_surprise": trace.mean_surprise,
                        "fast_ratio": trace.fast_count / max(trace.total_tokens, 1),
                        "deep_ratio": trace.deep_count / max(trace.total_tokens, 1),
                    },
                }
                all_data.append(entry)
                logger.info(
                    f"  sample {j}: reward={reward.total:+.4f} "
                    f"tokens={trace.total_tokens} "
                    f"H={trace.mean_entropy:.3f} S={trace.mean_surprise:.3f} "
                    f"resp={text[:50]}..."
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if bridge is not None:
            try:
                generator.metis.remove_listener(bridge.on_signal)
            except Exception:
                pass

    # ── Cleanup bridge ──
    if bridge is not None:
        try:
            bridge.stop()
        except Exception:
            pass

    # Save raw data
    os.makedirs(config.output_dir, exist_ok=True)
    data_path = os.path.join(config.output_dir, "phase1_scored_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(all_data)} scored samples to {data_path}")

    return all_data, model, tokenizer


def _format_chat(tokenizer: Any, prompt: str) -> str:
    """Format prompt as chat template if available."""
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return prompt


# ═══════════════════════════════════════════════════════
# Phase 2: DPO Training
# ═══════════════════════════════════════════════════════

def phase2_train(
    config: ExperimentConfig,
    scored_data: List[Dict],
    model: Any,
    tokenizer: Any,
) -> Tuple[str, str]:
    """
    Train two models:
    - Group A: DPO with METIS cognitive reward pairs
    - Group B: DPO with random pairs (control)

    Returns paths to both checkpoints.
    """
    logger.info(f"{'='*60}")
    logger.info(f"PHASE 2: DPO Training (METIS vs Random)")
    logger.info(f"{'='*60}")

    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    # ─── Build METIS DPO pairs ───
    metis_pairs = _build_metis_pairs(scored_data)
    random_pairs = _build_random_pairs(scored_data)

    logger.info(f"METIS pairs: {len(metis_pairs)}")
    logger.info(f"Random pairs: {len(random_pairs)}")

    metis_path = os.path.join(config.output_dir, "metis_dpo")
    random_path = os.path.join(config.output_dir, "random_dpo")

    # ─── Train Group A: METIS ───
    if len(metis_pairs) < 1:
        logger.warning("No METIS pairs survived filtering — skipping METIS DPO training")
        os.makedirs(metis_path, exist_ok=True)
    else:
        logger.info("Training Group A (METIS DPO)...")
        _train_dpo(config, model, tokenizer, metis_pairs, metis_path)

    # ─── Train Group B: Random ───
    if len(random_pairs) < 1:
        logger.warning("No Random pairs — skipping Random DPO training")
        os.makedirs(random_path, exist_ok=True)
    else:
        logger.info("Training Group B (Random DPO)...")
        _train_dpo(config, model, tokenizer, random_pairs, random_path)

    return metis_path, random_path


def _build_metis_pairs(scored_data: List[Dict]) -> List[Dict]:
    """Build DPO pairs with constrained cognitive matching.

    Anti-reward-hacking pipeline:
    1. Classify samples by decision profile (DEEP-present vs FAST-only)
    2. Homogeneous matching: same-type pairs use full reward_total
    3. Cross-type matching: quality veto — cognitive quality score
       (calibration + phase + epistemic, excluding efficiency) determines
       Chosen/Rejected to prevent efficiency from dominating pair selection
    4. Hard margin gate on the comparison score
    """
    MARGIN_THRESHOLD = 0.08  # Lower: efficiency no longer inflates margins

    def _cognitive_quality(s: Dict) -> float:
        """Cognitive quality score — excludes efficiency to break label inversion."""
        bd = s.get("reward_breakdown", {})
        return (
            0.35 * bd.get("calibration", 0)
            + 0.30 * bd.get("phase_quality", 0)
            + 0.20 * bd.get("epistemic_honesty", 0)
            + 0.15 * bd.get("coherence", 0)
        )

    def _has_deep(s: Dict) -> bool:
        """Check if sample has DEEP decisions."""
        return s.get("trace_stats", {}).get("deep_ratio", 0) > 0.05

    by_prompt: Dict[str, List[Dict]] = {}
    for entry in scored_data:
        p = entry["prompt"]
        if p not in by_prompt:
            by_prompt[p] = []
        by_prompt[p].append(entry)

    pairs = []
    n_total = len(by_prompt)
    n_margin_fail = 0
    n_homo_pairs = 0
    n_veto_pairs = 0

    for prompt, samples in by_prompt.items():
        if len(samples) < 2:
            continue

        # Split by decision profile
        deep_samples = [s for s in samples if _has_deep(s)]
        fast_samples = [s for s in samples if not _has_deep(s)]

        chosen = None
        rejected = None
        pair_type = ""

        # Strategy A: Cross-type quality veto
        # If we have both DEEP and FAST samples, compare by cognitive quality
        if deep_samples and fast_samples:
            best_deep = max(deep_samples, key=_cognitive_quality)
            best_fast = max(fast_samples, key=_cognitive_quality)
            worst_fast = min(fast_samples, key=_cognitive_quality)
            worst_deep = min(deep_samples, key=_cognitive_quality)

            cq_deep = _cognitive_quality(best_deep)
            cq_fast_worst = _cognitive_quality(worst_fast)

            # Quality veto: if DEEP has better cognitive quality than worst FAST,
            # DEEP is Chosen — this teaches the model that thinking pays off
            if cq_deep - cq_fast_worst >= MARGIN_THRESHOLD:
                chosen, rejected = best_deep, worst_fast
                pair_type = "veto_deep_wins"
            else:
                # Fallback: best FAST vs worst DEEP (FAST legitimately better)
                cq_fast = _cognitive_quality(best_fast)
                cq_deep_worst = _cognitive_quality(worst_deep)
                if cq_fast - cq_deep_worst >= MARGIN_THRESHOLD:
                    chosen, rejected = best_fast, worst_deep
                    pair_type = "veto_fast_wins"

        # Strategy B: Homogeneous matching (same-type, full reward_total)
        if chosen is None:
            # Use full reward_total within same type
            all_sorted = sorted(samples, key=lambda x: x["reward_total"], reverse=True)
            best = all_sorted[0]
            # Find worst with comparable length
            worst = None
            for candidate in reversed(all_sorted):
                if candidate is best:
                    continue
                len_best = len(best["response"])
                len_cand = len(candidate["response"])
                ratio = max(len_best, len_cand) / max(min(len_best, len_cand), 1)
                if ratio <= 1.5:
                    worst = candidate
                    break
            if worst is None:
                worst = all_sorted[-1]

            margin = best["reward_total"] - worst["reward_total"]
            if margin >= MARGIN_THRESHOLD:
                chosen, rejected = best, worst
                pair_type = "homo"

        if chosen is None:
            n_margin_fail += 1
            continue

        if pair_type == "homo":
            n_homo_pairs += 1
        else:
            n_veto_pairs += 1

        pairs.append({
            "prompt": chosen["chat_prompt"],
            "chosen": chosen["response"],
            "rejected": rejected["response"],
        })

    n_pass = len(pairs)
    logger.info(
        f"[Pair Filter] {n_total} prompts → {n_pass} pairs "
        f"(homo={n_homo_pairs}, veto={n_veto_pairs}, "
        f"margin_fail={n_margin_fail}, "
        f"rejection_rate={1 - n_pass / max(n_total, 1):.0%})"
    )
    return pairs


def _build_random_pairs(scored_data: List[Dict]) -> List[Dict]:
    """Build DPO pairs with random chosen/rejected (control group)."""
    by_prompt: Dict[str, List[Dict]] = {}
    for entry in scored_data:
        p = entry["prompt"]
        if p not in by_prompt:
            by_prompt[p] = []
        by_prompt[p].append(entry)

    pairs = []
    rng = random.Random(42)  # Fixed seed for reproducibility
    for prompt, samples in by_prompt.items():
        if len(samples) < 2:
            continue
        # Random pair selection (NOT reward-ranked)
        shuffled = samples.copy()
        rng.shuffle(shuffled)
        pairs.append({
            "prompt": shuffled[0]["chat_prompt"],
            "chosen": shuffled[0]["response"],
            "rejected": shuffled[1]["response"],
        })

    return pairs


def _train_dpo(
    config: ExperimentConfig,
    base_model: Any,
    tokenizer: Any,
    pairs: List[Dict],
    output_path: str,
) -> None:
    """Run DPO training with LoRA (no deepcopy — saves VRAM for 8GB GPUs)."""
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    dataset = Dataset.from_list(pairs)

    # Attach LoRA adapter directly to base model (no deepcopy to save VRAM)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train_model = get_peft_model(base_model, lora_config)
    train_model.print_trainable_parameters()

    training_args = DPOConfig(
        output_dir=output_path,
        num_train_epochs=config.dpo_epochs,
        per_device_train_batch_size=config.dpo_batch_size,
        gradient_accumulation_steps=config.dpo_gradient_accumulation,
        learning_rate=config.dpo_learning_rate,
        beta=config.dpo_beta,
        max_length=config.dpo_max_length,
        max_prompt_length=config.dpo_max_length // 2,
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=train_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_path)
    logger.info(f"Saved checkpoint to {output_path}")

    # Detach LoRA adapter, restore base model for next training run
    del trainer
    train_model.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════
# Phase 3: Evaluation
# ═══════════════════════════════════════════════════════

def phase3_evaluate(
    config: ExperimentConfig,
    base_model: Any,
    tokenizer: Any,
    metis_path: str,
    random_path: str,
) -> Tuple[EvalMetrics, EvalMetrics, EvalMetrics]:
    """
    Evaluate three models on held-out prompts:
    - Base model (no training)
    - METIS DPO model
    - Random DPO model
    """
    logger.info(f"{'='*60}")
    logger.info(f"PHASE 3: Evaluation ({config.n_eval_prompts} held-out prompts)")
    logger.info(f"{'='*60}")

    from peft import PeftModel
    from metis.training.generator import MetisGenerator
    from metis.training.rewards import CognitiveRewardComputer

    eval_prompts = EVAL_PROMPTS[:config.n_eval_prompts]
    reward_computer = CognitiveRewardComputer()

    # ─── Reload clean base model ───
    # Phase 2 LoRA train/unload corrupts base_model weights.
    # Reload from checkpoint to ensure fair base model evaluation.
    # CRITICAL: do NOT use device_map="auto" — after Phase 2 training,
    # VRAM has fragmented cache residue. Accelerate sees "low free VRAM"
    # and offloads layers to CPU RAM, causing PCIe bus bottleneck that
    # degrades from 15s→42s+ per prompt during autoregressive generation.
    from transformers import AutoModelForCausalLM
    logger.info("Reloading clean base model for evaluation...")
    try:
        del base_model
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    base_model.eval()

    # ─── Evaluate Base Model ───
    logger.info("Evaluating: Base Model (no training)")
    base_metrics = _evaluate_model(
        config, base_model, tokenizer, eval_prompts, reward_computer, "Base"
    )

    # ─── Evaluate METIS DPO ───
    _has_metis_ckpt = os.path.exists(os.path.join(metis_path, "adapter_config.json"))
    if _has_metis_ckpt:
        logger.info("Evaluating: METIS DPO")
        metis_model = PeftModel.from_pretrained(base_model, metis_path)
        metis_model.eval()
        metis_metrics = _evaluate_model(
            config, metis_model, tokenizer, eval_prompts, reward_computer, "METIS-DPO"
        )
        # CRITICAL: PeftModel.from_pretrained injects LoRA layers INTO base_model's
        # Linear modules. Simply deleting the PeftModel reference does NOT remove
        # the injected layers. We must destroy and reload from scratch to prevent
        # adapter stacking when loading Random DPO next.
        del metis_model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        logger.info("Reloading clean base model after METIS DPO eval...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        base_model.eval()
    else:
        logger.warning("No METIS DPO checkpoint found — using base metrics as fallback")
        metis_metrics = base_metrics

    # ─── Evaluate Random DPO ───
    _has_random_ckpt = os.path.exists(os.path.join(random_path, "adapter_config.json"))
    if _has_random_ckpt:
        logger.info("Evaluating: Random DPO")
        random_model = PeftModel.from_pretrained(base_model, random_path)
        random_model.eval()
        random_metrics = _evaluate_model(
            config, random_model, tokenizer, eval_prompts, reward_computer, "Random-DPO"
        )
        del random_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.warning("No Random DPO checkpoint found — using base metrics as fallback")
        random_metrics = base_metrics

    return base_metrics, metis_metrics, random_metrics


def _evaluate_model(
    config: ExperimentConfig,
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    reward_computer: Any,
    name: str,
) -> EvalMetrics:
    """Evaluate a single model on prompts."""
    import threading
    from metis.training.generator import MetisGenerator
    from metis.core.types import Decision, CognitiveTrace
    from metis.training.rewards import RewardBreakdown

    WALL_CLOCK_TIMEOUT = 120  # seconds — hard circuit breaker per prompt

    generator = MetisGenerator(model, tokenizer)
    metrics = EvalMetrics(name=name, n_responses=len(prompts))

    total_rewards = []
    all_breakdowns = []

    for i, prompt in enumerate(prompts):
        chat_prompt = _format_chat(tokenizer, prompt)

        # ── Wall-clock circuit breaker ──
        # Prevents single-prompt pathological loops from blocking the entire eval
        result_container: list = []
        # Capture generator in local var for thread-safety
        gen_ref = generator

        def _generate_with_timeout(g=gen_ref, p=chat_prompt):
            try:
                samples = g.generate_batch(
                    p,
                    n_samples=1,
                    temperatures=[config.eval_temperature],
                    max_new_tokens=config.eval_max_tokens,
                )
                result_container.append(samples[0])
            except Exception as e:
                logger.warning(f"  [{name}] {i+1}/{len(prompts)}: generation error: {e}")

        gen_thread = threading.Thread(target=_generate_with_timeout, daemon=True)
        gen_thread.start()
        gen_thread.join(timeout=WALL_CLOCK_TIMEOUT)

        if gen_thread.is_alive() or not result_container:
            # Timeout or error — produce fallback metrics
            logger.warning(
                f"  [{name}] {i+1}/{len(prompts)}: TIMEOUT ({WALL_CLOCK_TIMEOUT}s) "
                f"— falling back to zero reward"
            )
            text = ""
            trace = CognitiveTrace()
            reward = RewardBreakdown()
            # CRITICAL: orphaned daemon thread still holds old generator reference.
            # Create fresh generator so next prompt doesn't share METIS state
            # with the still-running background thread (race condition).
            generator = MetisGenerator(model, tokenizer)
        else:
            text, trace = result_container[0]
            reward = reward_computer.compute(trace)

        total_rewards.append(reward.total)
        all_breakdowns.append(reward)

        # Raw metrics from trace
        events = trace.events
        n = len(events) if events else 1
        metrics.mean_entropy += sum(e.semantic_entropy for e in events) / n
        metrics.mean_surprise += sum(e.token_surprise for e in events) / n
        metrics.mean_confidence += sum(e.confidence for e in events) / n
        metrics.confusion_ratio += sum(
            1 for e in events if getattr(e.cognitive_phase, "value", e.cognitive_phase) == "confusion"
        ) / n
        metrics.fast_ratio += sum(
            1 for e in events if e.decision == Decision.FAST
        ) / n
        metrics.avg_tokens += n

        logger.info(
            f"  [{name}] {i+1}/{len(prompts)}: "
            f"reward={reward.total:+.4f} tokens={n} "
            f"resp={text[:50]}..."
        )

        # ── VRAM hygiene: prevent CUDA memory fragmentation ──
        # Must run EVERY prompt — 50 prompts × 200 tokens of KV cache alloc/free
        # fragments CUDA memory progressively, causing 100x+ slowdown via swap thrashing
        del trace, text
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            gc.collect()

    # Store per-prompt rewards for statistical tests
    metrics.per_prompt_rewards = total_rewards

    # Average all metrics
    n_prompts = len(prompts)
    metrics.reward_total = sum(total_rewards) / n_prompts
    metrics.reward_coherence = sum(r.coherence for r in all_breakdowns) / n_prompts
    metrics.reward_calibration = sum(r.calibration for r in all_breakdowns) / n_prompts
    metrics.reward_phase_quality = sum(r.phase_quality for r in all_breakdowns) / n_prompts
    metrics.reward_epistemic = sum(r.epistemic_honesty for r in all_breakdowns) / n_prompts
    metrics.reward_efficiency = sum(r.efficiency for r in all_breakdowns) / n_prompts
    metrics.mean_entropy /= n_prompts
    metrics.mean_surprise /= n_prompts
    metrics.mean_confidence /= n_prompts
    metrics.confusion_ratio /= n_prompts
    metrics.fast_ratio /= n_prompts
    metrics.avg_tokens /= n_prompts

    return metrics


# ═══════════════════════════════════════════════════════
# Phase 4: Report
# ═══════════════════════════════════════════════════════

def phase4_report(
    config: ExperimentConfig,
    base: EvalMetrics,
    metis: EvalMetrics,
    random_ctrl: EvalMetrics,
) -> None:
    """Generate comparison report."""
    logger.info(f"{'='*60}")
    logger.info(f"PHASE 4: Report")
    logger.info(f"{'='*60}")

    def delta(new: float, old: float) -> str:
        d = new - old
        if abs(d) < 0.001:
            return f"{C.DIM}  ±0{C.RESET}"
        color = C.GREEN if d > 0 else C.RED
        return f"{color}{d:+.4f}{C.RESET}"

    print(f"""
 {C.BOLD}[SYSTEM::METIS]{C.RESET} {C.CYAN}Experiment Results{C.RESET}
 {C.DIM}═══════════════════════════════════════════════════════════════{C.RESET}
""")

    print(f"  {C.BOLD}Model:{C.RESET} {config.model_name}")
    print(f"  {C.BOLD}Train:{C.RESET} {config.n_train_prompts} prompts × {config.n_samples_per_prompt} samples")
    print(f"  {C.BOLD}Eval:{C.RESET}  {config.n_eval_prompts} held-out prompts\n")

    # Main comparison table
    header = f"  {'Metric':<22s} │ {'Base':>10s} │ {'METIS DPO':>10s} │ {'Δ vs Base':>10s} │ {'Random DPO':>10s} │ {'Δ vs Base':>10s}"
    sep = f"  {'─'*22}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}"
    print(f"{C.BOLD}{header}{C.RESET}")
    print(sep)

    rows = [
        ("Reward (Total)", base.reward_total, metis.reward_total, random_ctrl.reward_total),
        ("  R_coherence", base.reward_coherence, metis.reward_coherence, random_ctrl.reward_coherence),
        ("  R_calibration", base.reward_calibration, metis.reward_calibration, random_ctrl.reward_calibration),
        ("  R_phase", base.reward_phase_quality, metis.reward_phase_quality, random_ctrl.reward_phase_quality),
        ("  R_epistemic", base.reward_epistemic, metis.reward_epistemic, random_ctrl.reward_epistemic),
        ("  R_efficiency", base.reward_efficiency, metis.reward_efficiency, random_ctrl.reward_efficiency),
        ("", 0, 0, 0),  # spacer
        ("Mean Entropy", base.mean_entropy, metis.mean_entropy, random_ctrl.mean_entropy),
        ("Mean Surprise", base.mean_surprise, metis.mean_surprise, random_ctrl.mean_surprise),
        ("Mean Confidence", base.mean_confidence, metis.mean_confidence, random_ctrl.mean_confidence),
        ("Confusion Ratio", base.confusion_ratio, metis.confusion_ratio, random_ctrl.confusion_ratio),
        ("Fast (Sys1) Ratio", base.fast_ratio, metis.fast_ratio, random_ctrl.fast_ratio),
        ("Avg Tokens", base.avg_tokens, metis.avg_tokens, random_ctrl.avg_tokens),
    ]

    for label, base_v, metis_v, rand_v in rows:
        if label == "":
            print(sep)
            continue
        # For confusion/surprise, lower is better → invert delta color
        print(
            f"  {label:<22s} │ {base_v:>10.4f} │ {metis_v:>10.4f} │ "
            f"{delta(metis_v, base_v):>20s} │ {rand_v:>10.4f} │ "
            f"{delta(rand_v, base_v):>20s}"
        )

    # Summary
    metis_lift = metis.reward_total - base.reward_total
    random_lift = random_ctrl.reward_total - base.reward_total
    metis_vs_random = metis.reward_total - random_ctrl.reward_total

    print(f"\n{C.BOLD}  Summary:{C.RESET}")
    print(f"    METIS DPO vs Base:    {delta(metis.reward_total, base.reward_total)}")
    print(f"    Random DPO vs Base:   {delta(random_ctrl.reward_total, base.reward_total)}")
    print(f"    METIS DPO vs Random:  {delta(metis.reward_total, random_ctrl.reward_total)}")

    # ─── Statistical Analysis ───
    print(f"\n{C.BOLD}  Statistical Analysis:{C.RESET}")

    metis_rewards = metis.per_prompt_rewards
    random_rewards = random_ctrl.per_prompt_rewards
    base_rewards = base.per_prompt_rewards

    if len(metis_rewards) >= 5 and len(random_rewards) >= 5:
        # Paired bootstrap CI for METIS vs Random
        n_boot = 10000
        rng = random.Random(42)
        n_eval = min(len(metis_rewards), len(random_rewards))
        diffs = [metis_rewards[i] - random_rewards[i] for i in range(n_eval)]
        boot_means = []
        for _ in range(n_boot):
            sample = [diffs[rng.randint(0, n_eval - 1)] for _ in range(n_eval)]
            boot_means.append(sum(sample) / n_eval)
        boot_means.sort()
        ci_lo = boot_means[int(0.025 * n_boot)]
        ci_hi = boot_means[int(0.975 * n_boot)]
        mean_diff = sum(diffs) / n_eval

        # Cohen's d (paired)
        if n_eval > 1:
            diff_var = sum((d - mean_diff) ** 2 for d in diffs) / (n_eval - 1)
            diff_std = math.sqrt(diff_var) if diff_var > 0 else 1e-6
            cohens_d = mean_diff / diff_std
        else:
            cohens_d = 0.0

        ci_color = C.GREEN if ci_lo > 0 else (C.RED if ci_hi < 0 else C.YELLOW)
        print(f"    METIS vs Random (paired, n={n_eval}):")
        print(f"      Mean Δ:       {ci_color}{mean_diff:+.4f}{C.RESET}")
        print(f"      95% Boot CI:  {ci_color}[{ci_lo:+.4f}, {ci_hi:+.4f}]{C.RESET}")
        print(f"      Cohen's d:    {cohens_d:+.3f}", end="")
        if abs(cohens_d) >= 0.8:
            print(f" {C.GREEN}(large){C.RESET}")
        elif abs(cohens_d) >= 0.5:
            print(f" {C.YELLOW}(medium){C.RESET}")
        elif abs(cohens_d) >= 0.2:
            print(f" {C.YELLOW}(small){C.RESET}")
        else:
            print(f" {C.DIM}(negligible){C.RESET}")

        sig = ci_lo > 0 or ci_hi < 0
        if sig and mean_diff > 0:
            print(f"\n    {C.GREEN}{C.BOLD}✓ METIS improvement is statistically significant (CI excludes 0){C.RESET}")
        elif not sig:
            print(f"\n    {C.YELLOW}⚠ Result not statistically significant (CI includes 0, need more data){C.RESET}")
        else:
            print(f"\n    {C.RED}✗ METIS underperforms Random (CI excludes 0){C.RESET}")
    else:
        print(f"    {C.DIM}Too few samples for statistical tests (n<5){C.RESET}")

    # ─── Statistical Analysis ───
    print(f"\n{C.BOLD}  Statistical Analysis:{C.RESET}")

    metis_rewards = metis.per_prompt_rewards
    random_rewards = random_ctrl.per_prompt_rewards
    base_rewards = base.per_prompt_rewards

    if len(metis_rewards) >= 5 and len(random_rewards) >= 5:
        # Paired bootstrap CI for METIS vs Random
        n_boot = 10000
        rng = random.Random(42)
        n_eval = min(len(metis_rewards), len(random_rewards))
        diffs = [metis_rewards[i] - random_rewards[i] for i in range(n_eval)]
        boot_means = []
        for _ in range(n_boot):
            sample = [diffs[rng.randint(0, n_eval - 1)] for _ in range(n_eval)]
            boot_means.append(sum(sample) / n_eval)
        boot_means.sort()
        ci_lo = boot_means[int(0.025 * n_boot)]
        ci_hi = boot_means[int(0.975 * n_boot)]
        mean_diff = sum(diffs) / n_eval

        # Cohen's d (paired)
        if n_eval > 1:
            diff_var = sum((d - mean_diff) ** 2 for d in diffs) / (n_eval - 1)
            diff_std = math.sqrt(diff_var) if diff_var > 0 else 1e-6
            cohens_d = mean_diff / diff_std
        else:
            cohens_d = 0.0

        ci_color = C.GREEN if ci_lo > 0 else (C.RED if ci_hi < 0 else C.YELLOW)
        print(f"    METIS vs Random (paired, n={n_eval}):")
        print(f"      Mean Δ:       {ci_color}{mean_diff:+.4f}{C.RESET}")
        print(f"      95% Boot CI:  {ci_color}[{ci_lo:+.4f}, {ci_hi:+.4f}]{C.RESET}")
        print(f"      Cohen's d:    {cohens_d:+.3f}", end="")
        if abs(cohens_d) >= 0.8:
            print(f" {C.GREEN}(large){C.RESET}")
        elif abs(cohens_d) >= 0.5:
            print(f" {C.YELLOW}(medium){C.RESET}")
        elif abs(cohens_d) >= 0.2:
            print(f" {C.YELLOW}(small){C.RESET}")
        else:
            print(f" {C.DIM}(negligible){C.RESET}")

        sig = ci_lo > 0 or ci_hi < 0
        if sig and mean_diff > 0:
            print(f"\n    {C.GREEN}{C.BOLD}✓ METIS improvement is statistically significant (CI excludes 0){C.RESET}")
        elif not sig:
            print(f"\n    {C.YELLOW}⚠ Result not statistically significant (CI includes 0, need more data){C.RESET}")
        else:
            print(f"\n    {C.RED}✗ METIS underperforms Random (CI excludes 0){C.RESET}")
    else:
        print(f"    {C.DIM}Too few samples for statistical tests (n<5){C.RESET}")

    if metis_lift > random_lift + 0.01:
        print(f"    {C.GREEN}{C.BOLD}✓ METIS cognitive rewards provide measurable training improvement{C.RESET}")
    elif metis_lift > random_lift:
        print(f"    {C.YELLOW}≈ Marginal improvement from METIS rewards (need more data){C.RESET}")
    else:
        print(f"    {C.RED}✗ No clear improvement (may need hyperparameter tuning){C.RESET}")

    # Save report
    report = {
        "config": asdict(config),
        "base": base.to_dict(),
        "metis_dpo": metis.to_dict(),
        "random_dpo": random_ctrl.to_dict(),
        "summary": {
            "metis_vs_base": round(metis_lift, 4),
            "random_vs_base": round(random_lift, 4),
            "metis_vs_random": round(metis_vs_random, 4),
        },
    }
    report_path = os.path.join(config.output_dir, "experiment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  {C.DIM}Report saved to {report_path}{C.RESET}\n")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="METIS Training Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda / cpu / auto")
    parser.add_argument("--output", type=str, default="./experiment_output",
                        help="Output directory")
    parser.add_argument("--n-prompts", type=int, default=300,
                        help="Number of training prompts")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Samples per prompt")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens per generation")
    parser.add_argument("--dpo-epochs", type=int, default=3,
                        help="DPO training epochs")
    parser.add_argument("--dpo-lr", type=float, default=1e-6,
                        help="DPO learning rate")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "generate", "train", "eval"],
                        help="Which phase to run")
    parser.add_argument("--metis-checkpoint", type=str, default=None,
                        help="Path to METIS DPO checkpoint (for eval-only)")
    parser.add_argument("--random-checkpoint", type=str, default=None,
                        help="Path to Random DPO checkpoint (for eval-only)")
    parser.add_argument("--vllm", type=str, default=None,
                        help="vLLM server URL (e.g. http://localhost:8000/v1). "
                             "Enables 2-phase generation: vLLM batch gen + teacher-forcing.")
    args = parser.parse_args()

    config = ExperimentConfig(
        model_name=args.model,
        device=args.device,
        output_dir=args.output,
        n_train_prompts=args.n_prompts,
        n_samples_per_prompt=args.n_samples,
        max_new_tokens=args.max_tokens,
        dpo_epochs=args.dpo_epochs,
        dpo_learning_rate=args.dpo_lr,
        lora_r=args.lora_r,
        eval_max_tokens=args.max_tokens,  # Sync eval max tokens
    )
    vllm_url = args.vllm  # None = use HF generator (default)

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"""{C.GREEN}
███╗   ███╗███████╗████████╗██╗███████╗
████╗ ████║██╔════╝╚══██╔══╝██║██╔════╝
██╔████╔██║█████╗     ██║   ██║███████╗
██║╚██╔╝██║██╔══╝     ██║   ██║╚════██║
██║ ╚═╝ ██║███████╗   ██║   ██║███████║
╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝╚══════╝
{C.RESET}
 {C.BOLD}[SYSTEM::METIS]{C.RESET} {C.CYAN}Training Experiment{C.RESET}
 {C.DIM}Cognitive Rewards vs Random Baseline{C.RESET}

 > COGNITIVE_LAYER.......[{C.GREEN}ONLINE{C.RESET}]
 > REWARD_COMPUTER.......[{C.GREEN}ACTIVE{C.RESET}]
 > DPO_TRAINER...........[{C.YELLOW}STANDBY{C.RESET}]
 > EVAL_PIPELINE.........[{C.YELLOW}STANDBY{C.RESET}]

 root@agi:~$ {C.GREEN}Initializing Experiment...{C.RESET}

  Model:    {config.model_name}
  Device:   {config.device}
  Prompts:  {config.n_train_prompts} train + {config.n_eval_prompts} eval
  Samples:  {config.n_samples_per_prompt} per prompt
  Output:   {config.output_dir}
""")

    start = time.time()

    if args.phase in ("all", "generate"):
        scored_data, model, tokenizer = phase1_generate(config, vllm_url=vllm_url)

        if args.phase == "generate":
            logger.info("Phase 1 complete. Use --phase train to continue.")
            return
    else:
        # Load model for later phases
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs).to(device)
        model.eval()

        # Load scored data
        data_path = os.path.join(config.output_dir, "phase1_scored_data.json")
        with open(data_path, "r", encoding="utf-8") as f:
            scored_data = json.load(f)

    if args.phase in ("all", "train"):
        metis_path, random_path = phase2_train(config, scored_data, model, tokenizer)
    else:
        metis_path = args.metis_checkpoint or os.path.join(config.output_dir, "metis_dpo")
        random_path = args.random_checkpoint or os.path.join(config.output_dir, "random_dpo")

    if args.phase in ("all", "eval"):
        base_metrics, metis_metrics, random_metrics = phase3_evaluate(
            config, model, tokenizer, metis_path, random_path,
        )
        phase4_report(config, base_metrics, metis_metrics, random_metrics)

    elapsed = time.time() - start
    logger.info(f"Experiment completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
