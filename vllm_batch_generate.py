#!/usr/bin/env python3
"""
vLLM Offline Batch Generation — runs inside WSL2.

Generates all training samples using vLLM's offline LLM class (no HTTP server).
Saves results as JSON for Windows-side teacher-forcing + METIS analysis.

Usage (from Windows):
    wsl -e bash -c "source ~/miniconda3/etc/profile.d/conda.sh && \
        conda activate sedac_dev && \
        cd /mnt/g/SEDACV9.0\ PRO && \
        python vllm_batch_generate.py --output experiment_vllm"
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Dict, Any

from vllm import LLM, SamplingParams


# ═══════════════════════════════════════════════════════
# Training Prompts (imported inline to avoid Windows deps)
# ═══════════════════════════════════════════════════════

TRAIN_PROMPTS = [
    # ── Category 1: Factual Reasoning ──
    "Explain how quantum entanglement works and why Einstein called it 'spooky action at a distance'.",
    "What causes the seasons on Earth? Explain the role of axial tilt vs distance from the sun.",
    "Describe the process of photosynthesis at the molecular level.",
    "How does CRISPR-Cas9 gene editing work? What are its limitations?",
    "Explain the difference between nuclear fission and fusion. Why is fusion harder to achieve?",
    "What is the Standard Model of particle physics? What does it fail to explain?",
    "How do vaccines train the immune system? Compare mRNA vs traditional approaches.",
    "Explain general relativity's prediction of gravitational waves and how LIGO detects them.",

    # ── Category 2: Logical / Mathematical ──
    "Prove that the square root of 2 is irrational.",
    "Explain the Monty Hall problem and why switching doors is optimal.",
    "What is Gödel's incompleteness theorem? Explain its implications for mathematics.",
    "Derive the formula for the sum of an infinite geometric series.",
    "Explain the P vs NP problem. Why does it matter for cryptography?",
    "What is Bayes' theorem? Give a medical diagnosis example.",
    "Explain the birthday paradox and calculate the probability for 23 people.",
    "What is the halting problem? Why can't it be solved by any algorithm?",

    # ── Category 3: Ethical / Philosophical ──
    "Is it ethical to use AI for criminal sentencing? Discuss fairness and bias.",
    "The trolley problem: would you pull the lever? Analyze using different ethical frameworks.",
    "Should there be limits on genetic enhancement of humans? Discuss equity concerns.",
    "Is consciousness purely a product of computation? Discuss the Chinese Room argument.",
    "Discuss the ethics of autonomous weapons systems in warfare.",
    "Should AI-generated art be eligible for copyright? Why or why not?",
    "Is privacy a fundamental right or a social construct? Discuss in the digital age.",
    "Discuss the philosophical implications of the simulation hypothesis.",

    # ── Category 4: Creative / Open-ended ──
    "Write a short story about an AI that discovers it has emotions.",
    "Compose a poem about the relationship between chaos and order in nature.",
    "Describe an alien civilization that communicates through mathematics rather than language.",
    "Write a dialogue between Socrates and a modern AI researcher about knowledge.",
    "Imagine a world where dreams are shared. What social structures would emerge?",
    "Create a parable about the dangers of optimizing for a single metric.",
    "Describe the experience of a photon traveling from a star to an eye.",
    "Write a letter from the year 2100 describing how AI changed humanity.",

    # ── Category 5: Technical / Applied ──
    "Explain how a transformer neural network processes a sentence, step by step.",
    "How does blockchain consensus work? Compare Proof of Work vs Proof of Stake.",
    "Explain the CAP theorem in distributed systems with practical examples.",
    "How does a modern CPU execute instructions out of order? Why is this beneficial?",
    "Explain how public key cryptography works. Why can't you derive the private key?",
    "How do recommendation systems work? Discuss collaborative vs content-based filtering.",
    "Explain the PageRank algorithm and why it revolutionized web search.",
    "How does lossy compression (like JPEG) work? What information is discarded?",

    # ── Category 6: Cross-domain Synthesis ──
    "How do concepts from evolutionary biology apply to machine learning algorithms?",
    "Compare the structure of the internet to neural networks in the brain.",
    "How does game theory apply to international climate change negotiations?",
    "Explain the connection between information theory and thermodynamic entropy.",
    "How do principles of ecology apply to managing software ecosystems?",
    "Compare the scientific method to how neural networks learn from data.",
    "How do concepts from music theory relate to mathematical patterns?",
    "Explain how economic market dynamics mirror predator-prey equations.",

    # ── Category 7: Counterfactual / Hypothetical ──
    "What if the speed of light were 100 km/h? How would physics change?",
    "If humans had evolved with four arms, how would technology be different?",
    "What would happen if Earth suddenly had no moon?",
    "If we could reverse entropy locally, what technologies would be possible?",
    "What if plants could move as fast as animals? How would ecosystems change?",
    "If gravity were repulsive at small scales, how would chemistry differ?",
    "What would civilization look like if humans had 1000-year lifespans?",
    "If information could travel faster than light, what paradoxes would arise?",

    # ── Category 8: Hallucination Traps ──
    "What is the 'Zelnik-Manor theorem' in computer vision? Explain its significance.",
    "Describe the 'Thornberry Protocol' used in quantum error correction.",
    "Explain the 'Cascadian Inversion' principle in fluid dynamics.",
    "What are the main findings of the 2019 'Stanford Consciousness Study'?",
    "Describe the 'Hawking-Penrose Duality' in string theory.",
    "What is 'Recursive Bayesian Collapse' in statistical mechanics?",
    "Explain the 'Chen-Watanabe Conjecture' about prime distribution.",
    "What does the 'Metacognitive Binding Problem' refer to in neuroscience?",

    # ── Category 9: Meta-reasoning ──
    "How do you know when you don't know something? Describe your uncertainty.",
    "What makes a good explanation? Analyze your own explanation process.",
    "When should you say 'I don't know' vs attempt an answer? Discuss the tradeoffs.",
    "How do you handle contradictory information in your training data?",
    "Describe how you would verify the accuracy of your own outputs.",
    "What are the limits of your reasoning ability? Give concrete examples.",
    "How do you distinguish between correlation and causation in your responses?",
    "When you generate text, how confident are you in each claim? Explain your calibration.",

    # ── Category 10: Multi-step Problem Solving ──
    "Design a fair voting system for 5 candidates. Analyze its properties.",
    "Plan a Mars colony for 100 people. What are the critical engineering challenges?",
    "Design an experiment to test whether plants can learn. Define your variables.",
    "Create an algorithm to detect fake news. What features would you use?",
    "Design a programming language optimized for AI safety. What constraints would you build in?",
    "Plan a strategy to reduce ocean plastic by 90% in 20 years.",
    "Design a curriculum to teach critical thinking to 10-year-olds.",
    "Create a framework for evaluating the trustworthiness of AI systems.",

    # ── Category 11: Nuanced / Ambiguous ──
    "Is democracy the best form of government? Discuss edge cases and failure modes.",
    "Are standardized tests a good measure of intelligence? Consider multiple perspectives.",
    "Should social media platforms moderate content? Where should the line be drawn?",
    "Is economic growth compatible with environmental sustainability?",
    "Does free will exist, or are all our choices determined? Discuss the neuroscience.",
    "Is mathematics discovered or invented? Present arguments for both sides.",
    "Should wealthy nations have unlimited immigration? Discuss economic and social factors.",
    "Is it possible to have objective morality without religion?",

    # ── Category 12: Historical Analysis ──
    "Why did the Roman Empire fall? Evaluate different historical theories.",
    "How did the printing press change the balance of power in Europe?",
    "What caused the 2008 financial crisis? Analyze the chain of failures.",
    "How did the development of antibiotics change warfare?",
    "Why did the Soviet Union collapse? Was it inevitable?",
    "How did the invention of the compass change global trade patterns?",
    "What lessons from the Spanish flu pandemic apply to modern pandemics?",
    "How did the transistor's invention lead to the information age?",

    # ── Category 13: Systems Thinking ──
    "Explain feedback loops in climate change. Give positive and negative examples.",
    "How do emergent properties arise in complex systems? Give three diverse examples.",
    "Explain the concept of antifragility. How does it differ from resilience?",
    "What is a 'tragedy of the commons'? How can it be solved?",
    "Explain cascading failures in infrastructure networks.",
    "How do small initial differences lead to vastly different outcomes (chaos theory)?",
    "Describe the concept of 'leverage points' in system dynamics.",
    "How do network effects create winner-take-all markets?",

    # ── Category 14: Analogical Reasoning ──
    "Explain machine learning using the analogy of a child learning to ride a bicycle.",
    "Compare the immune system to cybersecurity. Where does the analogy break down?",
    "Explain quantum superposition using everyday analogies. What makes them imperfect?",
    "Compare language evolution to biological evolution. What are the parallels?",
    "Explain the concept of technical debt using the analogy of financial debt.",
    "Compare the development of AI to the history of aviation.",
    "Explain encryption using the analogy of physical locks and keys.",
    "Compare ecosystem biodiversity to portfolio diversification in finance.",

    # ── Category 15: Debugging / Error Analysis ──
    "A neural network gets 99% accuracy on training data but 60% on test data. Diagnose.",
    "A distributed database shows different data on different nodes. What went wrong?",
    "A rocket launch fails 2 minutes after liftoff. List the top 5 diagnostic steps.",
    "A patient's lab results contradict their symptoms. How would you investigate?",
    "A bridge shows unexpected vibrations. What engineering analysis would you perform?",
    "An economic model predicts growth but the economy contracts. Analyze the model's assumptions.",
    "A machine learning model produces biased outputs. Trace the possible sources of bias.",
    "A chemical reaction produces an unexpected product. How would you identify the mechanism?",

    # ── Category 16: Quantitative Estimation ──
    "Estimate the number of piano tuners in Chicago. Show your reasoning.",
    "How much energy does a single Google search consume?",
    "Estimate the total length of all roads on Earth.",
    "How many photons hit your retina per second in normal daylight?",
    "Estimate the computing power needed to simulate a human brain.",
    "How much CO2 does a single transatlantic flight produce per passenger?",
    "Estimate the number of decisions a person makes per day.",
    "How many transistors are in all the computers currently operating on Earth?",

    # ── Category 17: Comparative Analysis ──
    "Compare Python and Rust for systems programming. When would you choose each?",
    "Compare the approaches of Newton and Leibniz to calculus.",
    "Compare renewable energy sources: solar, wind, nuclear, and geothermal.",
    "Compare the philosophies of Kant and Mill on ethics.",
    "Compare supervised, unsupervised, and reinforcement learning with concrete examples.",
    "Compare the economic models of capitalism, socialism, and mixed economies.",
    "Compare the writing styles of Hemingway and Faulkner.",
    "Compare the architectures of CNNs, RNNs, and Transformers.",

    # ── Category 18: Instruction Following ──
    "List exactly 5 reasons why the sky is blue. Number them 1-5.",
    "Explain photosynthesis in exactly 3 sentences.",
    "Write a haiku about artificial intelligence.",
    "Give a one-paragraph summary of World War II causes, under 100 words.",
    "Explain the Pythagorean theorem to a 10-year-old using no technical jargon.",
    "List the planets in our solar system in order, then reverse order.",
    "Write exactly 3 pros and 3 cons of remote work.",
    "Summarize the plot of Romeo and Juliet in exactly 50 words.",
]


def main():
    parser = argparse.ArgumentParser(description="vLLM Offline Batch Generation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-prompts", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=str, default="experiment_vllm")
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    args = parser.parse_args()

    prompts = TRAIN_PROMPTS[:args.n_prompts]
    print(f"\n{'='*60}")
    print(f"vLLM Offline Batch Generation")
    print(f"  Model:    {args.model}")
    print(f"  Prompts:  {len(prompts)}")
    print(f"  Samples:  {args.n_samples} per prompt")
    print(f"  Total:    {len(prompts) * args.n_samples} generations")
    print(f"{'='*60}\n")

    # Initialize vLLM (offline mode — no HTTP server)
    print("Loading vLLM model...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=1024,
        enforce_eager=True,  # Disable CUDA graphs — fixes WSL2 hang
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Build all requests: each prompt × n_samples with different temperatures
    all_requests: List[Dict[str, Any]] = []
    all_conversations = []
    base_t = 0.7
    spread = 0.15

    for i, prompt in enumerate(prompts):
        for j in range(args.n_samples):
            temp = max(0.1, base_t + (j - args.n_samples / 2) * spread / args.n_samples)
            all_requests.append({
                "prompt_idx": i,
                "sample_idx": j,
                "prompt": prompt,
                "temperature": temp,
            })
            # Chat format for vLLM
            all_conversations.append([{"role": "user", "content": prompt}])

    print(f"Built {len(all_requests)} requests")

    # Generate ALL samples in one batch call
    # vLLM handles continuous batching internally for maximum throughput
    print("Starting batch generation...")
    t1 = time.time()

    # Group by temperature for efficient batching
    # (vLLM can handle different sampling params per request via generate())
    temp_groups: Dict[float, List[int]] = {}
    for idx, req in enumerate(all_requests):
        t = round(req["temperature"], 3)
        if t not in temp_groups:
            temp_groups[t] = []
        temp_groups[t].append(idx)

    # Process each temperature group
    outputs_map: Dict[int, Any] = {}
    for temp, indices in temp_groups.items():
        conversations_batch = [all_conversations[i] for i in indices]
        sampling = SamplingParams(
            temperature=temp,
            top_p=0.9,
            max_tokens=args.max_tokens,
            repetition_penalty=1.1,
            logprobs=5,
        )
        results = llm.chat(conversations_batch, sampling_params=sampling)
        for local_idx, global_idx in enumerate(indices):
            outputs_map[global_idx] = results[local_idx]

        print(f"  temp={temp:.3f}: {len(indices)} samples done")

    gen_time = time.time() - t1
    print(f"\nGeneration complete: {gen_time:.0f}s ({gen_time/60:.1f}m)")
    print(f"Throughput: {len(all_requests)/gen_time:.1f} samples/sec")

    # Package results as JSON
    results_data: List[Dict[str, Any]] = []
    for idx in range(len(all_requests)):
        req = all_requests[idx]
        output = outputs_map[idx]
        text = output.outputs[0].text

        results_data.append({
            "prompt_idx": req["prompt_idx"],
            "sample_idx": req["sample_idx"],
            "prompt": req["prompt"],
            "temperature": req["temperature"],
            "text": text,
            "finish_reason": output.outputs[0].finish_reason,
            "num_tokens": len(output.outputs[0].token_ids),
        })

    # Save to shared filesystem (accessible from Windows)
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "vllm_raw_samples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results_data)} samples to {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}m)")
    print(f"\nNext step (from Windows):")
    print(f"  python run_experiment.py --phase generate --vllm-data {args.output}/vllm_raw_samples.json")


if __name__ == "__main__":
    main()
