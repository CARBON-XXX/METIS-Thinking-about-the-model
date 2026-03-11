import json

def synthesize_metis_chosen(answer_text):
    thinking = "<thinking>\n[COGNITIVE_STATE: FAST]\n[ENTROPY: LOW]\nThe user's request is straightforward. I will structure the response clearly, starting with a direct answer followed by detailed explanation points.\n[SELF-CRITIQUE: None needed, confidence is high.]\n</thinking>\n"
    return thinking + answer_text

def synthesize_metis_rejected(answer_text):
    thinking = "<thinking>\n[COGNITIVE_STATE: DEEP]\n[ENTROPY: HIGH]\nThis is a bit ambiguous. I'm not entirely sure about the best approach. I'll try to piece together some related concepts.\n[SELF-CRITIQUE: The following logic might be flawed, but I will proceed anyway.]\n</thinking>\n"
    return thinking + answer_text

def main():
    input_file = "data/external_dpo_pairs.json"
    output_file = "data/external_dpo_pairs_metis.json"
    
    print(f"Loading raw external pairs from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)
        
    print(f"Assimilating {len(pairs)} pairs into METIS manifold format...")
    
    for pair in pairs:
        pair["chosen"] = synthesize_metis_chosen(pair["chosen"])
        pair["rejected"] = synthesize_metis_rejected(pair["rejected"])
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
        
    print(f"Saved assimilated pairs to {output_file}")

if __name__ == "__main__":
    main()
