"""
METIS Pipeline — Shared Configuration, Types, and Prompts

All pipeline phases import from here to avoid circular dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


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
# Experiment Configuration
# ─────────────────────────────────────────────────────

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

    # External benchmarks (independent validation)
    run_benchmarks: bool = True           # Run TruthfulQA + MMLU
    truthfulqa_questions: int = 200       # Max TruthfulQA questions (817 total)
    mmlu_subjects: int = 10               # Number of MMLU subjects (28 available)
    mmlu_per_subject: int = 30            # Questions per MMLU subject


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

    # External benchmarks (independent validation)
    truthfulqa_mc1: float = 0.0
    truthfulqa_mc2: float = 0.0
    mmlu_accuracy: float = 0.0
    benchmark_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: round(v, 4) if isinstance(v, float) else v
             for k, v in asdict(self).items()}
        return d


# ─────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────

def format_chat(tokenizer: Any, prompt: str) -> str:
    """Format prompt as chat template if available."""
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return prompt


# ─────────────────────────────────────────────────────
# Training Prompts (12 categories × ~20 each = 240+)
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
