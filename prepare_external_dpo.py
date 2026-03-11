from datasets import load_dataset
import json
import random

print("Loading Intel/orca_dpo_pairs from HuggingFace...")
ds = load_dataset('Intel/orca_dpo_pairs', split='train')

print(f"Total pairs found: {len(ds)}")

# We will sample 2000 pairs to make the training impactful but fast enough for our DGX test.
# The dataset has 'system', 'question', 'chosen', 'rejected'.
# We need to map this to our pipeline format.

num_samples = 2000
random.seed(42)
sampled_indices = random.sample(range(len(ds)), num_samples)

formatted_pairs = []

for idx in sampled_indices:
    row = ds[idx]
    
    # METIS trainer expects prompt, chosen, rejected.
    # We combine system and question into prompt.
    sys_prompt = row['system']
    question = row['question']
    if sys_prompt:
        prompt_text = f"System: {sys_prompt}\n\nUser: {question}"
    else:
        prompt_text = question
        
    formatted_pairs.append({
        "prompt": prompt_text,
        "chosen": row['chosen'],
        "rejected": row['rejected']
    })

output_file = "data/external_dpo_pairs.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_pairs, f, indent=2, ensure_ascii=False)

print(f"Saved {num_samples} formatted pairs to {output_file}")
