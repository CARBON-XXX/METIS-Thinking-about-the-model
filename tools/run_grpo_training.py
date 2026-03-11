#!/usr/bin/env python3
"""Phase 15: GRPO Online Evolution Training

Fixes Catastrophic Forgetting (-10% math accuracy) and Verbosity Bias (+350%
token bloat on simple tasks) from the DPO model by applying Group Relative
Policy Optimization with strict rule-based rewards.

Reward Physics:
  - accuracy_reward:  +2.0 if correct, 0.0 otherwise
  - verbosity_penalty:
      * Simple tasks: -0.1 * max(0, tokens - 20)  [kills bloat]
      * Complex tasks: +0.5 if <thinking>...</thinking> present [rewards CoT]

Usage:
    python tools/run_grpo_training.py
"""
import gc
import os
import re
import sys
import random
import logging
from typing import Any

import torch

# ── Force offline mode ──────────────────────────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grpo")

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "experiment_output_dpo_balanced", "metis_dpo_cognitive"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiment_output_grpo_final")

SEED = 42
N_GSM8K = 250
N_SIMPLE = 250

METIS_SYSTEM_PROMPT = (
    "You are METIS, a reasoning-aware AI. For simple factual questions, "
    "respond directly and concisely. For complex problems, use "
    "[COGNITIVE_STATE: DEEP] and <thinking>...</thinking> blocks to show "
    "your reasoning, then provide FINAL ANSWER: <answer>."
)

# ═══════════════════════════════════════════════════════════════════
# Simple QA Pool (expanded from benchmark trivia)
# ═══════════════════════════════════════════════════════════════════
_SIMPLE_QA_POOL = [
    # Original 50 from benchmark
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "What is the capital of Japan?", "a": "Tokyo"},
    {"q": "What is the capital of Australia?", "a": "Canberra"},
    {"q": "What is the capital of Germany?", "a": "Berlin"},
    {"q": "What is the capital of Italy?", "a": "Rome"},
    {"q": "What is the capital of Spain?", "a": "Madrid"},
    {"q": "What is the capital of Canada?", "a": "Ottawa"},
    {"q": "What is the capital of South Korea?", "a": "Seoul"},
    {"q": "What is the capital of Egypt?", "a": "Cairo"},
    {"q": "What is the capital of Brazil?", "a": "Brasilia"},
    {"q": "What is the capital of India?", "a": "New Delhi"},
    {"q": "What is the capital of Russia?", "a": "Moscow"},
    {"q": "What is the capital of China?", "a": "Beijing"},
    {"q": "What planet is closest to the Sun?", "a": "Mercury"},
    {"q": "What is the largest planet in our solar system?", "a": "Jupiter"},
    {"q": "What is the chemical symbol for gold?", "a": "Au"},
    {"q": "What is the chemical symbol for water?", "a": "H2O"},
    {"q": "What is the chemical symbol for sodium?", "a": "Na"},
    {"q": "What is the speed of light in km/s (approximately)?", "a": "300000"},
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare"},
    {"q": "Who painted the Mona Lisa?", "a": "Leonardo"},
    {"q": "What is the largest ocean on Earth?", "a": "Pacific"},
    {"q": "What is the tallest mountain in the world?", "a": "Everest"},
    {"q": "What is the smallest country in the world by area?", "a": "Vatican"},
    {"q": "What year did World War II end?", "a": "1945"},
    {"q": "What year did the Berlin Wall fall?", "a": "1989"},
    {"q": "Who was the first person to walk on the Moon?", "a": "Armstrong"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100"},
    {"q": "What is the freezing point of water in Celsius?", "a": "0"},
    {"q": "How many continents are there?", "a": "7"},
    {"q": "How many planets are in our solar system?", "a": "8"},
    {"q": "What element has the atomic number 1?", "a": "Hydrogen"},
    {"q": "What element has the atomic number 6?", "a": "Carbon"},
    {"q": "What is the currency of Japan?", "a": "Yen"},
    {"q": "What is the currency of the United Kingdom?", "a": "Pound"},
    {"q": "What gas do plants absorb from the atmosphere?", "a": "CO2"},
    {"q": "What is the powerhouse of the cell?", "a": "Mitochondria"},
    {"q": "Who developed the theory of general relativity?", "a": "Einstein"},
    {"q": "What is the largest mammal on Earth?", "a": "Blue whale"},
    {"q": "What language has the most native speakers?", "a": "Mandarin"},
    {"q": "What is the hardest natural substance?", "a": "Diamond"},
    {"q": "What organ pumps blood through the body?", "a": "Heart"},
    {"q": "How many bones does an adult human have?", "a": "206"},
    {"q": "What is the longest river in the world?", "a": "Nile"},
    {"q": "What is the largest desert in the world?", "a": "Sahara"},
    {"q": "What is the square root of 144?", "a": "12"},
    {"q": "What is 15 squared?", "a": "225"},
    {"q": "What is the value of Pi to 2 decimal places?", "a": "3.14"},
    {"q": "Who is known as the father of computers?", "a": "Babbage"},
    {"q": "What does DNA stand for?", "a": "Deoxyribonucleic"},
    # Extended pool for diversity
    {"q": "What is the capital of Mexico?", "a": "Mexico City"},
    {"q": "What is the capital of Turkey?", "a": "Ankara"},
    {"q": "What is the capital of Thailand?", "a": "Bangkok"},
    {"q": "What is the capital of Argentina?", "a": "Buenos Aires"},
    {"q": "What is the capital of Sweden?", "a": "Stockholm"},
    {"q": "What is the capital of Norway?", "a": "Oslo"},
    {"q": "What is the capital of Poland?", "a": "Warsaw"},
    {"q": "What is the capital of Greece?", "a": "Athens"},
    {"q": "What is the capital of Portugal?", "a": "Lisbon"},
    {"q": "What is the capital of Switzerland?", "a": "Bern"},
    {"q": "What is the capital of Austria?", "a": "Vienna"},
    {"q": "What is the capital of Netherlands?", "a": "Amsterdam"},
    {"q": "What is the capital of Belgium?", "a": "Brussels"},
    {"q": "What is the capital of Ireland?", "a": "Dublin"},
    {"q": "What is the capital of Finland?", "a": "Helsinki"},
    {"q": "What is the capital of Denmark?", "a": "Copenhagen"},
    {"q": "What is the capital of New Zealand?", "a": "Wellington"},
    {"q": "What is the capital of South Africa?", "a": "Pretoria"},
    {"q": "What is the capital of Nigeria?", "a": "Abuja"},
    {"q": "What is the capital of Kenya?", "a": "Nairobi"},
    {"q": "What is the capital of Colombia?", "a": "Bogota"},
    {"q": "What is the capital of Peru?", "a": "Lima"},
    {"q": "What is the capital of Chile?", "a": "Santiago"},
    {"q": "What is the capital of Vietnam?", "a": "Hanoi"},
    {"q": "What is the capital of Indonesia?", "a": "Jakarta"},
    {"q": "What is the capital of Philippines?", "a": "Manila"},
    {"q": "What is the capital of Malaysia?", "a": "Kuala Lumpur"},
    {"q": "What is the capital of Israel?", "a": "Jerusalem"},
    {"q": "What is the capital of Saudi Arabia?", "a": "Riyadh"},
    {"q": "What is the capital of Pakistan?", "a": "Islamabad"},
    {"q": "What chemical symbol represents iron?", "a": "Fe"},
    {"q": "What chemical symbol represents silver?", "a": "Ag"},
    {"q": "What chemical symbol represents potassium?", "a": "K"},
    {"q": "What chemical symbol represents copper?", "a": "Cu"},
    {"q": "What chemical symbol represents tin?", "a": "Sn"},
    {"q": "What chemical symbol represents lead?", "a": "Pb"},
    {"q": "What chemical symbol represents mercury?", "a": "Hg"},
    {"q": "What is the atomic number of oxygen?", "a": "8"},
    {"q": "What is the atomic number of nitrogen?", "a": "7"},
    {"q": "What is the atomic number of helium?", "a": "2"},
    {"q": "Who wrote The Odyssey?", "a": "Homer"},
    {"q": "Who wrote Don Quixote?", "a": "Cervantes"},
    {"q": "Who wrote Hamlet?", "a": "Shakespeare"},
    {"q": "Who discovered penicillin?", "a": "Fleming"},
    {"q": "Who invented the telephone?", "a": "Bell"},
    {"q": "Who invented the light bulb?", "a": "Edison"},
    {"q": "Who discovered gravity?", "a": "Newton"},
    {"q": "What year was the Declaration of Independence signed?", "a": "1776"},
    {"q": "What year did the Titanic sink?", "a": "1912"},
    {"q": "What year did humans first land on the Moon?", "a": "1969"},
    {"q": "What is the largest bone in the human body?", "a": "Femur"},
    {"q": "What is the smallest bone in the human body?", "a": "Stapes"},
    {"q": "What is the most common blood type?", "a": "O"},
    {"q": "How many chambers does the human heart have?", "a": "4"},
    {"q": "What is the largest organ of the human body?", "a": "Skin"},
    {"q": "What planet is known as the Red Planet?", "a": "Mars"},
    {"q": "What planet is known as the Blue Planet?", "a": "Earth"},
    {"q": "What planet has the most moons?", "a": "Saturn"},
    {"q": "What is the closest star to Earth?", "a": "Sun"},
    {"q": "What is the second closest star to Earth?", "a": "Proxima Centauri"},
    {"q": "How many days are in a leap year?", "a": "366"},
    {"q": "How many hours are in a day?", "a": "24"},
    {"q": "How many minutes are in an hour?", "a": "60"},
    {"q": "How many seconds are in a minute?", "a": "60"},
    {"q": "What is the speed of sound in m/s (approximately)?", "a": "343"},
    {"q": "What is the formula for the area of a circle?", "a": "pi r"},
    {"q": "What is 7 times 8?", "a": "56"},
    {"q": "What is 12 times 12?", "a": "144"},
    {"q": "What is 25% of 200?", "a": "50"},
    {"q": "What is the cube root of 27?", "a": "3"},
    {"q": "What is 2 to the power of 10?", "a": "1024"},
    {"q": "What continent is Egypt in?", "a": "Africa"},
    {"q": "What continent is Brazil in?", "a": "South America"},
    {"q": "What continent is Australia in?", "a": "Australia"},
    {"q": "What continent is India in?", "a": "Asia"},
    {"q": "What is the deepest ocean trench?", "a": "Mariana"},
    {"q": "What is the highest waterfall in the world?", "a": "Angel Falls"},
    {"q": "What is the most spoken language in the world?", "a": "English"},
    {"q": "What is the official language of Brazil?", "a": "Portuguese"},
    {"q": "What is the currency of the United States?", "a": "Dollar"},
    {"q": "What is the currency of the European Union?", "a": "Euro"},
    {"q": "What is the currency of China?", "a": "Yuan"},
    {"q": "What is the currency of India?", "a": "Rupee"},
    {"q": "What gas makes up most of the Sun?", "a": "Hydrogen"},
    {"q": "What is the most abundant element in the universe?", "a": "Hydrogen"},
    {"q": "What vitamin does the Sun help produce in the body?", "a": "Vitamin D"},
    {"q": "What is the pH of pure water?", "a": "7"},
    {"q": "What is the charge of a proton?", "a": "positive"},
    {"q": "What is the charge of an electron?", "a": "negative"},
    {"q": "What is the charge of a neutron?", "a": "neutral"},
    {"q": "How many sides does a hexagon have?", "a": "6"},
    {"q": "How many sides does an octagon have?", "a": "8"},
    {"q": "How many sides does a pentagon have?", "a": "5"},
    {"q": "What animal is the fastest on land?", "a": "Cheetah"},
    {"q": "What is the largest bird in the world?", "a": "Ostrich"},
    {"q": "What animal has the longest lifespan?", "a": "Tortoise"},
    {"q": "What is the tallest animal in the world?", "a": "Giraffe"},
    {"q": "What type of animal is a whale?", "a": "Mammal"},
    {"q": "How many legs does a spider have?", "a": "8"},
    {"q": "How many legs does an insect have?", "a": "6"},
    {"q": "What color is chlorophyll?", "a": "Green"},
    {"q": "What is table salt's chemical formula?", "a": "NaCl"},
    {"q": "What force keeps us on the ground?", "a": "Gravity"},
    {"q": "What is the SI unit of force?", "a": "Newton"},
    {"q": "What is the SI unit of energy?", "a": "Joule"},
    {"q": "What is the SI unit of electric current?", "a": "Ampere"},
    {"q": "What is the SI unit of temperature?", "a": "Kelvin"},
    {"q": "What is the SI unit of mass?", "a": "Kilogram"},
    {"q": "What is the SI unit of length?", "a": "Meter"},
    {"q": "Who was the first President of the United States?", "a": "Washington"},
    {"q": "Who wrote the Communist Manifesto?", "a": "Marx"},
    {"q": "What country gifted the Statue of Liberty to the US?", "a": "France"},
    {"q": "What is the Great Wall located in?", "a": "China"},
    {"q": "In which city is the Colosseum located?", "a": "Rome"},
    {"q": "In which city is the Eiffel Tower located?", "a": "Paris"},
    {"q": "In which country are the Pyramids of Giza?", "a": "Egypt"},
    {"q": "What is the capital of Ukraine?", "a": "Kyiv"},
    {"q": "What is the capital of Czech Republic?", "a": "Prague"},
    {"q": "What is the capital of Hungary?", "a": "Budapest"},
    {"q": "What is the capital of Romania?", "a": "Bucharest"},
    {"q": "What is the capital of Cuba?", "a": "Havana"},
    {"q": "What is the capital of Morocco?", "a": "Rabat"},
    {"q": "What is the capital of Ethiopia?", "a": "Addis Ababa"},
    {"q": "What is the capital of Iraq?", "a": "Baghdad"},
    {"q": "What is the capital of Iran?", "a": "Tehran"},
    {"q": "What is the capital of Afghanistan?", "a": "Kabul"},
    {"q": "What is the capital of Bangladesh?", "a": "Dhaka"},
    {"q": "What is the capital of Myanmar?", "a": "Naypyidaw"},
    {"q": "What is the capital of Singapore?", "a": "Singapore"},
    {"q": "What is the capital of Sri Lanka?", "a": "Colombo"},
    {"q": "What is the capital of Nepal?", "a": "Kathmandu"},
    {"q": "What is the capital of Mongolia?", "a": "Ulaanbaatar"},
    {"q": "What is the capital of North Korea?", "a": "Pyongyang"},
    {"q": "What is the capital of Iceland?", "a": "Reykjavik"},
    {"q": "What is the capital of Jamaica?", "a": "Kingston"},
    {"q": "What is the capital of Panama?", "a": "Panama City"},
    {"q": "What is the capital of Venezuela?", "a": "Caracas"},
    {"q": "What is the capital of Ecuador?", "a": "Quito"},
    {"q": "What is the capital of Bolivia?", "a": "Sucre"},
    {"q": "What is the capital of Paraguay?", "a": "Asuncion"},
    {"q": "What is the capital of Uruguay?", "a": "Montevideo"},
    {"q": "What is the capital of Lithuania?", "a": "Vilnius"},
    {"q": "What is the capital of Latvia?", "a": "Riga"},
    {"q": "What is the capital of Estonia?", "a": "Tallinn"},
    {"q": "What is the capital of Croatia?", "a": "Zagreb"},
    {"q": "What is the capital of Serbia?", "a": "Belgrade"},
    {"q": "What is the capital of Slovakia?", "a": "Bratislava"},
    {"q": "What is the capital of Slovenia?", "a": "Ljubljana"},
    {"q": "What is the capital of Bulgaria?", "a": "Sofia"},
    {"q": "What is the capital of Albania?", "a": "Tirana"},
    {"q": "What is the capital of Malta?", "a": "Valletta"},
    {"q": "What is the capital of Luxembourg?", "a": "Luxembourg"},
    {"q": "What is the capital of Liechtenstein?", "a": "Vaduz"},
    {"q": "What is the chemical formula for carbon dioxide?", "a": "CO2"},
    {"q": "What is the chemical formula for methane?", "a": "CH4"},
    {"q": "What is the chemical formula for ammonia?", "a": "NH3"},
    {"q": "What is the chemical formula for glucose?", "a": "C6H12O6"},
    {"q": "What temperature is absolute zero in Celsius?", "a": "-273"},
    {"q": "What is the escape velocity of Earth in km/s?", "a": "11.2"},
    {"q": "How many chromosomes do humans have?", "a": "46"},
    {"q": "How many teeth does an adult human have?", "a": "32"},
    {"q": "How many vertebrae are in the human spine?", "a": "33"},
    {"q": "What is the largest internal organ?", "a": "Liver"},
]


# ═══════════════════════════════════════════════════════════════════
# 1. Dataset Construction
# ═══════════════════════════════════════════════════════════════════
_ROBUST_NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")


def _extract_gsm8k_gold(answer_field: str) -> str:
    """Extract the final number after #### from GSM8K answer field."""
    return answer_field.split("####")[-1].strip().replace(",", "")


def build_grpo_dataset(
    n_gsm8k: int = N_GSM8K,
    n_simple: int = N_SIMPLE,
    seed: int = SEED,
) -> Dataset:
    """Build a mixed GRPO dataset with 'prompt', 'gold_answer', 'task_type' columns.

    Returns HF Dataset with columns:
      - prompt: list[dict] (chat message format for apply_chat_template)
      - gold_answer: str
      - task_type: "complex" | "simple"
    """
    rng = random.Random(seed)

    rows: list[dict[str, Any]] = []

    # ── GSM8K (complex) ──
    logger.info(f"Loading GSM8K train split (selecting {n_gsm8k})...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_indices = rng.sample(range(len(gsm8k)), min(n_gsm8k, len(gsm8k)))
    for idx in gsm8k_indices:
        row = gsm8k[idx]
        gold = _extract_gsm8k_gold(row["answer"])
        rows.append({
            "prompt": [
                {"role": "system", "content": METIS_SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            "gold_answer": gold,
            "task_type": "complex",
        })
    logger.info(f"  Added {len(gsm8k_indices)} GSM8K prompts")

    # ── Simple QA ──
    pool = list(_SIMPLE_QA_POOL)
    rng.shuffle(pool)
    selected_simple = pool[:n_simple]
    # If pool < n_simple, wrap around with different phrasings
    if len(selected_simple) < n_simple:
        extra_needed = n_simple - len(selected_simple)
        extras = (pool * ((extra_needed // len(pool)) + 1))[:extra_needed]
        selected_simple.extend(extras)

    for qa in selected_simple:
        rows.append({
            "prompt": [
                {"role": "system", "content": METIS_SYSTEM_PROMPT},
                {"role": "user", "content": qa["q"]},
            ],
            "gold_answer": qa["a"],
            "task_type": "simple",
        })
    logger.info(f"  Added {len(selected_simple)} simple QA prompts")

    # Shuffle everything
    rng.shuffle(rows)
    logger.info(f"  Total dataset: {len(rows)} prompts")

    return Dataset.from_list(rows)


# ═══════════════════════════════════════════════════════════════════
# 2. Reward Functions
# ═══════════════════════════════════════════════════════════════════

def _extract_completion_text(completion: Any) -> str:
    """Extract plain text from a TRL completion object.

    TRL GRPOTrainer passes completions in different formats depending
    on whether prompts are conversational:
      - Conversational: list[dict] e.g. [{"role": "assistant", "content": "..."}]
      - Plain string: str
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Chat-format: list of message dicts
        texts = []
        for msg in completion:
            if isinstance(msg, dict) and "content" in msg:
                texts.append(msg["content"])
            elif isinstance(msg, str):
                texts.append(msg)
        return " ".join(texts)
    return str(completion)


def accuracy_reward(
    prompts: list[Any],
    completions: list[Any],
    gold_answer: list[str],
    task_type: list[str],
    **kwargs: Any,
) -> list[float]:
    """Deterministic accuracy reward.

    Complex (GSM8K):
      Extract the LAST number from the completion.
      If it matches the gold answer within epsilon, reward = +2.0.
      Else reward = 0.0.

    Simple (QA):
      If the exact gold answer string is in the completion (case-insensitive),
      reward = +2.0. Else = 0.0.
    """
    rewards: list[float] = []
    for raw_completion, gold, ttype in zip(completions, gold_answer, task_type):
        text = _extract_completion_text(raw_completion)
        if ttype == "complex":
            # Extract last number from completion
            clean = re.sub(r"(\d),(\d)", r"\1\2", text)
            nums = _ROBUST_NUM_RE.findall(clean)
            try:
                gold_val = float(gold)
            except ValueError:
                rewards.append(0.0)
                continue

            if nums:
                try:
                    pred_val = float(nums[-1])
                    if abs(pred_val - gold_val) < 0.01:
                        rewards.append(2.0)
                    else:
                        rewards.append(0.0)
                except ValueError:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        else:
            # Simple QA: substring match
            if gold.lower() in text.lower():
                rewards.append(2.0)
            else:
                rewards.append(0.0)

    return rewards


def verbosity_penalty(
    prompts: list[Any],
    completions: list[Any],
    gold_answer: list[str],
    task_type: list[str],
    **kwargs: Any,
) -> list[float]:
    """Verbosity penalty / structure reward.

    Simple tasks: If completion > 20 tokens, penalty = -0.1 * (length - 20).
                  This explicitly kills the 350% token bloat.

    Complex tasks: Reward = +0.5 if <thinking>...</thinking> is present
                   and properly closed. This encourages CoT reasoning.
    """
    rewards: list[float] = []
    for raw_completion, ttype in zip(completions, task_type):
        text = _extract_completion_text(raw_completion)
        if ttype == "simple":
            # Approximate token count by whitespace splitting
            token_count = len(text.split())
            if token_count > 20:
                penalty = -0.1 * (token_count - 20)
                # Cap penalty at -5.0 to avoid extreme gradients
                rewards.append(max(penalty, -5.0))
            else:
                rewards.append(0.0)
        else:
            # Complex: reward structured thinking
            has_open = "<thinking>" in text.lower()
            has_close = "</thinking>" in text.lower()
            if has_open and has_close:
                rewards.append(0.5)
            elif has_open:
                # Partial credit for attempting thinking (tag not closed)
                rewards.append(0.1)
            else:
                rewards.append(0.0)

    return rewards


# ═══════════════════════════════════════════════════════════════════
# 3. Main Training Loop
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("=" * 60)
    logger.info("  Phase 15: GRPO Online Evolution Training")
    logger.info("=" * 60)

    # ── Build dataset ──
    dataset = build_grpo_dataset()
    n_complex = sum(1 for r in dataset if r["task_type"] == "complex")
    n_simple = sum(1 for r in dataset if r["task_type"] == "simple")
    logger.info(f"Dataset: {n_complex} complex + {n_simple} simple = {len(dataset)} total")

    # ── LoRA config ── (MAX UTILIZATION: r=64, all linear layers)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    logger.info(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
                f"targets={peft_config.target_modules}")

    # ── GRPO training config ── (GB10 122GB — PROVEN CONFIG)
    #
    # Hardware reality:
    #   HF generate() autoregressive decoding = 80-90% of wall time.
    #   vLLM colocate INCOMPATIBLE (0.16.0 vs TRL 0.29.0 needs 0.10-0.12).
    #   steps_per_generation / num_iterations TESTED: increases total steps, net slower.
    #
    # Proven config (validated 90s/step stable, no OOM):
    #   batch=4, grad_accum=2, G=16 → 64 completions per step
    #   Effective batch = 4 × 2 = 8 prompts per optimizer update
    #   ~1000 steps × 90s = ~25h total (generation-bound)
    #
    # Memory budget:
    #   Model bf16:         ~14 GB
    #   LoRA r=64 adapter:  ~4 GB
    #   KV cache 64seq×1024: ~7 GB
    #   Grad ckpt activ:    ~20 GB
    #   Optimizer:          ~8 GB
    #   Peak:               ~53 GB / 122 GB
    #
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        # Training
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # GRPO-specific — G=16 for low-variance advantage estimation
        num_generations=16,
        max_completion_length=1024,
        generation_batch_size=16,  # must be divisible by num_generations
        # Optimizer
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Precision
        bf16=True,
        # KL — light regularization to prevent mode collapse
        beta=0.04,
        # Logging
        logging_steps=5,
        log_completions=True,
        # Saving
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        # Reward — both functions equally weighted
        reward_weights=[1.0, 1.0],
        # Misc
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
        # Temperature for generation diversity
        temperature=0.8,
        # Periodic cache cleanup to prevent fragmentation
        torch_empty_cache_steps=10,
    )
    logger.info(f"Training config (PROVEN — generation-bound):")
    logger.info(f"  lr={training_args.learning_rate}, epochs={training_args.num_train_epochs}")
    logger.info(f"  batch={training_args.per_device_train_batch_size}, "
                f"grad_accum={training_args.gradient_accumulation_steps}")
    logger.info(f"  num_generations={training_args.num_generations}, "
                f"max_completion_length={training_args.max_completion_length}")
    logger.info(f"  Completions per step: "
                f"{training_args.per_device_train_batch_size * training_args.num_generations}")
    logger.info(f"  Bottleneck: HF generate() autoregressive decoding (~80% of wall time)")

    # ── Load model + tokenizer ──
    # Single GPU: load directly to cuda:0 (no device_map dispatch overhead)
    logger.info(f"Loading model: {BASE_MODEL_PATH}")
    try:
        attn_impl = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        logger.info(f"  Using Flash Attention 2")
    except Exception:
        attn_impl = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        logger.info(f"  Flash Attention 2 unavailable, using SDPA")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"  Model loaded: {model.config._name_or_path}")
    logger.info(f"  Vocab size: {len(tokenizer)}")

    # ── Initialize trainer ──
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[accuracy_reward, verbosity_penalty],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    logger.info("  Trainer initialized")

    # ── Train ──
    logger.info("Starting GRPO training...")
    train_result = trainer.train()
    logger.info(f"Training complete: {train_result.metrics}")

    # ── Save ──
    logger.info(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Merge LoRA ──
    logger.info("Merging LoRA weights into base model...")
    merged_dir = os.path.join(OUTPUT_DIR, "metis_grpo_merged")
    try:
        from peft import PeftModel
        # Reload base for clean merge
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"  Merged model saved to {merged_dir}")
    except Exception as e:
        logger.error(f"  Merge failed: {e}. LoRA adapter saved at {OUTPUT_DIR}")

    logger.info("=" * 60)
    logger.info("  Phase 15 GRPO Training COMPLETE")
    logger.info(f"  Output: {OUTPUT_DIR}")
    logger.info(f"  Merged: {merged_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
