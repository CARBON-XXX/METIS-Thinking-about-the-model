"""Test SEDAC with Qwen2.5-3B"""
import torch
import time
import sys
sys.path.insert(0, "G:/SEDACV9.0 PRO")

# Load CUDA kernels
import sedac_cuda
print(f"SEDAC CUDA Kernels: {sedac_cuda}")

from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("SEDAC V9.0 - Qwen2.5-3B Real Model Test")
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

print(f"Model loaded: {model.config.num_hidden_layers} layers, hidden={model.config.hidden_size}")

# Test input
test_prompts = [
    "What is 2+2?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
]

print("\n" + "-" * 70)
print("Baseline Inference (No SEDAC)")
print("-" * 70)

for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nPrompt: {prompt[:50]}...")
    print(f"Response: {response[:100]}...")
    print(f"Time: {elapsed:.1f}ms")

print("\n" + "-" * 70)
print("SEDAC CUDA Kernel Integration Test")
print("-" * 70)

# Test CUDA kernel with model's actual hidden states
with torch.no_grad():
    messages = [{"role": "user", "content": "Hello"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Get hidden states
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    
    # Extract hidden states from layer 10 and 11
    hidden_10 = outputs.hidden_states[10].squeeze(0)  # [seq_len, hidden]
    hidden_11 = outputs.hidden_states[11].squeeze(0)
    logits = outputs.logits.squeeze(0)  # [seq_len, vocab]
    
    print(f"\nHidden states shape: {hidden_10.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Run SEDAC decision kernel
    entropy, confidence, decision, load = sedac_cuda.fused_entropy_decision(
        logits.float(),
        hidden_11.float(),
        hidden_10.float(),
        3.0,  # mean_entropy
        1.0,  # std_entropy
        0.5,  # layer_progress
        0.7,  # threshold
    )
    
    print(f"\nSEDAC Decision Results:")
    print(f"  Entropy: mean={entropy.mean().item():.3f}, std={entropy.std().item():.3f}")
    print(f"  Confidence: mean={confidence.mean().item():.3f}")
    print(f"  Exit ratio: {decision.float().mean().item()*100:.1f}%")
    print(f"  Cognitive Load: mean={load.mean().item():.3f}")

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
