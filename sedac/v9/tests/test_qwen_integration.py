"""
SEDAC V9.0 - Qwen Integration Test
Complete end-to-end test with Qwen2.5-3B
"""
import torch
import time
import sys
sys.path.insert(0, "G:/SEDACV9.0 PRO")

from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("SEDAC V9.0 - Qwen2.5 Integration Test")
print("=" * 70)

# Load model
model_name = "Qwen/Qwen2.5-3B-Instruct"
print(f"\nLoading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()

print(f"Model loaded: {model.config.num_hidden_layers} layers")

# Import SEDAC
from sedac.v9.core import create_sedac_engine, SEDACConfig

# Create SEDAC engine
config = SEDACConfig(
    exit_threshold=0.65,
    min_exit_layer=4,
    anchor_interval=6,
    protect_first_n=2,
    protect_last_n=1,
    attention_sink_tokens=4,
    verbose=True,
)

engine = create_sedac_engine(model, config.exit_threshold, config.min_exit_layer, config.anchor_interval, True)

print(f"\nSEDAC Engine ready")
print(f"  Anchor layers: {sorted(engine.anchor_layers)}")

# Test prompts
test_cases = [
    {"prompt": "What is 2+2?", "expected_easy": True},
    {"prompt": "Hello!", "expected_easy": True},
    {"prompt": "Explain the theory of general relativity in detail.", "expected_easy": False},
    {"prompt": "Write a complex recursive algorithm.", "expected_easy": False},
]

print("\n" + "=" * 70)
print("Forward Pass Test")
print("=" * 70)

for case in test_cases:
    prompt = case["prompt"]
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        result = engine.forward(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    
    avg_exit = result["avg_exit_layer"]
    skip_ratio = result["skip_ratio"]
    
    print(f"\nPrompt: '{prompt[:50]}...'")
    print(f"  Time: {elapsed:.1f}ms")
    print(f"  Avg Exit Layer: {avg_exit:.1f}/{engine.num_layers}")
    print(f"  Skip Ratio: {skip_ratio*100:.1f}%")
    print(f"  Expected Easy: {case['expected_easy']}")

# Baseline comparison
print("\n" + "=" * 70)
print("Baseline Comparison (No SEDAC)")
print("=" * 70)

for case in test_cases[:2]:
    prompt = case["prompt"]
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            return_dict=True,
        )
    
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) * 1000
    
    # SEDAC
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        result = engine.forward(inputs.input_ids, attention_mask=inputs.attention_mask)
    
    torch.cuda.synchronize()
    sedac_time = (time.perf_counter() - start) * 1000
    
    speedup = baseline_time / sedac_time if sedac_time > 0 else 0
    
    print(f"\nPrompt: '{prompt[:30]}...'")
    print(f"  Baseline: {baseline_time:.1f}ms")
    print(f"  SEDAC:    {sedac_time:.1f}ms")
    print(f"  Speedup:  {speedup:.2f}x")

# Statistics
stats = engine.get_stats()
print("\n" + "=" * 70)
print("SEDAC Statistics")
print("=" * 70)
print(f"  Entropy Mean: {stats['entropy_mean']:.3f}")
print(f"  Entropy Std:  {stats['entropy_std']:.3f}")
print(f"  Sample Count: {stats['entropy_count']}")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
