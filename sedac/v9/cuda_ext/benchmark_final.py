"""
SEDAC V9.0 - Final Benchmark
Compare CUDA kernel vs PyTorch on real model hidden states
"""
import torch
import time
import sedac_cuda
import torch.nn.functional as F
import math

from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("SEDAC V9.0 - CUDA Kernel Final Benchmark")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")

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

num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
vocab_size = model.config.vocab_size

print(f"Model: {num_layers} layers, hidden={hidden_size}, vocab={vocab_size}")

# Test prompts
prompts = [
    "What is 2+2?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
]

print("\n" + "=" * 70)
print("Extracting Hidden States & Running Decision Kernels")
print("=" * 70)

def pytorch_decision(logits, hidden, prev_hidden, mean_ent, std_ent, progress, thresh):
    """PyTorch baseline implementation"""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -torch.sum(probs * log_probs, dim=-1) / math.log(2)
    
    diff = hidden - prev_hidden
    diff_norm = torch.norm(diff, p=2, dim=-1)
    hidden_norm = torch.norm(hidden, p=2, dim=-1)
    stability = 1.0 / (1.0 + diff_norm / (hidden_norm + 1e-6))
    
    z_score = (mean_ent - entropy) / (std_ent + 1e-6)
    confidence = torch.sigmoid(z_score * 2.0)
    
    load = (1.0 - confidence) * 0.5 + (1.0 - stability) * 0.3 + (1.0 - progress) * 0.2
    current_thresh = thresh - progress * 0.2
    decision = (confidence * stability * progress) > current_thresh
    
    return entropy, confidence, decision, load

total_cuda_time = 0
total_pytorch_time = 0
total_tokens = 0

for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Get all hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    
    seq_len = inputs.input_ids.shape[1]
    print(f"\nPrompt: '{prompt[:40]}...' (seq_len={seq_len})")
    
    # Test decision kernel at each layer
    for layer_idx in range(4, num_layers - 1, 4):  # Sample every 4th layer
        hidden = outputs.hidden_states[layer_idx + 1].squeeze(0).float()
        prev_hidden = outputs.hidden_states[layer_idx].squeeze(0).float()
        logits = outputs.logits.squeeze(0).float()
        
        layer_progress = layer_idx / num_layers
        
        # CUDA kernel
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            e1, c1, d1, l1 = sedac_cuda.fused_entropy_decision(
                logits, hidden, prev_hidden, 3.0, 1.0, layer_progress, 0.7
            )
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) * 1000 / 10
        
        # PyTorch
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            e2, c2, d2, l2 = pytorch_decision(
                logits, hidden, prev_hidden, 3.0, 1.0, layer_progress, 0.7
            )
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) * 1000 / 10
        
        speedup = pytorch_time / cuda_time
        
        print(f"  Layer {layer_idx:2d}: CUDA={cuda_time:.2f}ms, PyTorch={pytorch_time:.2f}ms, Speedup={speedup:.1f}x, "
              f"Exit={d1.float().mean().item()*100:.0f}%")
        
        total_cuda_time += cuda_time
        total_pytorch_time += pytorch_time
        total_tokens += seq_len

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total CUDA time:    {total_cuda_time:.1f}ms")
print(f"Total PyTorch time: {total_pytorch_time:.1f}ms")
print(f"Average Speedup:    {total_pytorch_time / total_cuda_time:.1f}x")
print(f"Total tokens:       {total_tokens}")
print("=" * 70)

# Large batch test
print("\n" + "=" * 70)
print("Large Batch Stress Test (4096 tokens)")
print("=" * 70)

N = 4096
logits = torch.randn(N, vocab_size, device="cuda", dtype=torch.float32)
hidden = torch.randn(N, hidden_size, device="cuda", dtype=torch.float32)
prev_hidden = torch.randn(N, hidden_size, device="cuda", dtype=torch.float32)

# Warmup
for _ in range(10):
    sedac_cuda.fused_entropy_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5, 0.7)
torch.cuda.synchronize()

# CUDA
start = time.perf_counter()
for _ in range(100):
    sedac_cuda.fused_entropy_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5, 0.7)
torch.cuda.synchronize()
cuda_time = (time.perf_counter() - start) * 1000 / 100

# PyTorch
for _ in range(10):
    pytorch_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5, 0.7)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(100):
    pytorch_decision(logits, hidden, prev_hidden, 3.0, 1.0, 0.5, 0.7)
torch.cuda.synchronize()
pytorch_time = (time.perf_counter() - start) * 1000 / 100

print(f"CUDA Kernel:  {cuda_time:.2f}ms ({N/cuda_time*1000:.0f} tok/s)")
print(f"PyTorch:      {pytorch_time:.2f}ms ({N/pytorch_time*1000:.0f} tok/s)")
print(f"Speedup:      {pytorch_time/cuda_time:.1f}x")
print("=" * 70)
