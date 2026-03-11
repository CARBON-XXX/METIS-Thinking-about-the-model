import json
import os
from datasets import load_dataset

def main():
    print("Downloading diverse prompts from tatsu-lab/alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    
    # Filter: we want standalone instructions (no external input context)
    # and we want them to be reasonably substantial (length > 30 chars)
    prompts = []
    for ex in ds:
        if ex["input"].strip() == "" and len(ex["instruction"]) > 30:
            prompts.append(ex["instruction"])
            if len(prompts) >= 1500:
                break
                
    os.makedirs("data", exist_ok=True)
    with open("data/extended_prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(prompts)} prompts to data/extended_prompts.json")

if __name__ == "__main__":
    main()
