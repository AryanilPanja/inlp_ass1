import os
import random
from tokenizers import WhitespaceTokenizer, RegexTokenizer, BPETokenizer
from language_models import LanguageModel

# --- SETTINGS ---
LANG = "mn"  # 'en' or 'mn'
DATA_TYPE = "sample"  # 'sample' or 'final'
BPE_TRAIN_LIMIT = 50000  # Must match the limit used during BPE training
NUM_MERGES = 1000  # Must match the merges used during BPE training

# Derived paths
TEST_FILE = f"./data/datasets/{LANG}_{DATA_TYPE}_test.txt"
BPE_RULES_FILE = f"./data/models/{LANG}_{DATA_TYPE}_bpe_rules_{BPE_TRAIN_LIMIT}_{NUM_MERGES}.json"

# Model combinations
TOKENIZERS = ["Whitespace", "Regex", "BPE"]
SMOOTHINGS = [None, "witten-bell", "kneser-ney"]

def load_tokenizer(tokenizer_type, lang):
    """Load and return appropriate tokenizer."""
    if tokenizer_type == "Whitespace":
        return WhitespaceTokenizer()
    elif tokenizer_type == "Regex":
        return RegexTokenizer()
    elif tokenizer_type == "BPE":
        bpe = BPETokenizer(num_merges=NUM_MERGES)
        if os.path.exists(BPE_RULES_FILE):
            bpe.load_model(BPE_RULES_FILE)
        else:
            print(f"Warning: BPE rules file not found at {BPE_RULES_FILE}")
        return bpe

def decode_output(generated_string, tokenizer_type, tokenizer):
    """Decode generated output based on tokenizer type."""
    if tokenizer_type == "BPE":
        tokens = generated_string.split(" ")
        return tokenizer.decode(tokens)
    else:
        return generated_string

def main():
    # 1. Load Test Data
    print(f"Loading test data from {TEST_FILE}...")
    if not os.path.exists(TEST_FILE):
        print(f"Error: {TEST_FILE} not found. Run clean.py first.")
        return
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(lines)} test samples.\n")

    # 2. Pick random sentences for testing
    print("="*80)
    print(f"VISUAL VERIFICATION | Language: {LANG.upper()} | Data: {DATA_TYPE.upper()}")
    print("="*80)
    
    # Pick 3 random sentences
    num_samples = min(3, len(lines))
    random_indices = random.sample(range(len(lines)), num_samples)
    
    for sample_idx, idx in enumerate(random_indices, 1):
        original = lines[idx]
        words = original.split()
        
        # We need at least 4 words to make a valid prompt
        if len(words) < 4:
            continue

        # Use first 3 words as prompt
        prompt = " ".join(words[:3])
        
        print(f"\n{'='*80}")
        print(f"SAMPLE {sample_idx}")
        print(f"{'='*80}")
        print(f"Original:  {original}")
        print(f"Prompt:    '{prompt}'")
        print(f"\n{'-'*80}")
        print(f"{'Tokenizer':<15} | {'Smoothing':<15} | Generated Output")
        print(f"{'-'*80}")

        # 3. Loop through all 9 model combinations
        for tokenizer_type in TOKENIZERS:
            tokenizer = load_tokenizer(tokenizer_type, LANG)
            
            for smoothing in SMOOTHINGS:
                smoothing_str = smoothing if smoothing else "None"
                model_file = f"./data/models/lm_{LANG}_{DATA_TYPE}_{tokenizer_type}_{smoothing}.pkl"
                
                # Try to load and generate
                if not os.path.exists(model_file):
                    print(f"{tokenizer_type:<15} | {smoothing_str:<15} | [Model not found]")
                    continue
                
                try:
                    lm = LanguageModel.load_model(model_file)
                    generated_string = lm.generate_sentence(prompt, tokenizer, max_len=15)
                    decoded = decode_output(generated_string, tokenizer_type, tokenizer)
                    print(f"{tokenizer_type:<15} | {smoothing_str:<15} | {decoded}")
                except Exception as e:
                    print(f"{tokenizer_type:<15} | {smoothing_str:<15} | [Error: {str(e)}]")

if __name__ == "__main__":
    main()