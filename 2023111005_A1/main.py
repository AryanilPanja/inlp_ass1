# final main.py code but with toggle switches at the start for loading and running separate parts of the model.

import os
import math
from tqdm import tqdm
from tokenizers import WhitespaceTokenizer, RegexTokenizer, BPETokenizer
from language_models import LanguageModel


# --- USER CONFIGURATION (TOGGLES) ---

# 1. Language Settings
LANG = "en"  # 'en' or 'mn'

# 2. Data Type
DATA_TYPE = "final"  # 'sample' or 'final'

# 5. Limits
LM_TRAIN_LIMIT = 500000 
BPE_TRAIN_LIMIT = 100000  # Lines to train BPE on
NUM_MERGES = 2000  # Number of BPE merges
# Test size to calculate perplexity.
TEST_SIZE = 50000

# 3. Data Files
TRAIN_FILE = f"./data/datasets/{LANG}_{DATA_TYPE}_train.txt"
TEST_FILE = f"./data/datasets/{LANG}_{DATA_TYPE}_test.txt"
BPE_RULES_FILE = f"./data/models/{LANG}_{DATA_TYPE}_bpe_rules_{BPE_TRAIN_LIMIT}_{NUM_MERGES}.json"

# 4. Execution Toggles
RUN_WHITESPACE = True
RUN_REGEX      = True
RUN_BPE        = True

RUN_SMOOTHING_NONE = True
RUN_SMOOTHING_WB   = True  # Witten-Bell
RUN_SMOOTHING_KN   = True  # Kneser-Ney

# 4. Training Settings
# Set to False if you already have the .pkl files and just want to print results
TRAIN_MODELS = True 

# Set to False if you want to skip BPE training (if rules file exists)
TRAIN_BPE_RULES = True





# ==========================================

def load_data_lines(path, limit=None):
    if not os.path.exists(path):
        print(f"Error: {path} missing. Run clean.py first.")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for i, line in enumerate(f) if line.strip()]
        if limit: return lines[:limit]
        return lines

def main():
    print(f"--- ASSIGNMENT RUNNER | LANG: {LANG.upper()} ---")
    
    # 1. Load Data
    print(f"Loading Data (Train Limit: {LM_TRAIN_LIMIT} | BPE Limit: {BPE_TRAIN_LIMIT} | Test Limit: {TEST_SIZE})...")
    train_data = load_data_lines(TRAIN_FILE, limit=LM_TRAIN_LIMIT)
    test_data = load_data_lines(TEST_FILE, limit=TEST_SIZE)
    print(f"Loaded {len(train_data)} training lines and {len(test_data)} test lines.")

    # 2. Setup Tokenizers
    tokenizers = {}

    if RUN_WHITESPACE:
        print("Initializing Whitespace Tokenizer...")
        tokenizers["Whitespace"] = WhitespaceTokenizer()

    if RUN_REGEX:
        print("Initializing Regex Tokenizer...")
        tokenizers["Regex"] = RegexTokenizer()

    if RUN_BPE:
        print("Initializing BPE Tokenizer...")
        bpe = BPETokenizer(num_merges=NUM_MERGES)
        
        if os.path.exists(BPE_RULES_FILE):
            print(f"  -> Loading BPE rules from {BPE_RULES_FILE}")
            bpe.load_model(BPE_RULES_FILE)
        elif TRAIN_BPE_RULES:
            print(f"  -> Training BPE rules (on first {BPE_TRAIN_LIMIT} lines)...")
            bpe.train(train_data[:BPE_TRAIN_LIMIT])
            bpe.save_model(BPE_RULES_FILE)
        else:
            print("  -> Warning: BPE rules missing and training disabled.")
            
        tokenizers["BPE"] = bpe

    # 3. Determine Smoothing Methods to Run
    smoothings = []
    if RUN_SMOOTHING_NONE: smoothings.append(None)
    if RUN_SMOOTHING_WB:   smoothings.append('witten-bell')
    if RUN_SMOOTHING_KN:   smoothings.append('kneser-ney')

    # 4. Master Loop
    results = {}

    for tok_name, tokenizer in tokenizers.items():
        print(f"\n" + "="*50)
        print(f"TOKENIZER: {tok_name}")
        print("="*50)

        # Optimization: Tokenize once per tokenizer
        print(f"Tokenizing {len(train_data)} lines...")
        tokenized_train = [tokenizer.tokenize(s) for s in tqdm(train_data, desc=f"Tok {tok_name} ({len(train_data)} lines)")]
        
        for smooth in smoothings:
            print(f"\n   -> Smoothing: {smooth}")
            lm_filename = f"./data/models/lm_{LANG}_{DATA_TYPE}_{tok_name}_{smooth}.pkl"
            
            lm = None
            
            # Try loading if exists
            if os.path.exists(lm_filename):
                lm = LanguageModel.load_model(lm_filename)
                
            # Train if missing or forced
            if lm is None and TRAIN_MODELS:
                print(f"      Training new model...")
                lm = LanguageModel(n=4, smoothing=smooth)
                lm.train(tokenized_train)
                lm.save_model(lm_filename)
            elif lm is None and not TRAIN_MODELS:
                print("      Skipping (Model missing and training disabled).")
                continue

            # Calculate Perplexity
            ppl = lm.score_perplexity(test_data, tokenizer)
            results[f"{tok_name} + {smooth}"] = ppl
            print(f"      PERPLEXITY: {ppl:.4f}")

    # 5. Final Report
    print("\n" + "="*50)
    print("FINAL PERPLEXITY RESULTS")
    print("="*50)
    print(f"{'Model':<30} | {'Perplexity':<15}")
    print("-" * 45)
    for model_name, score in results.items():
        print(f"{model_name:<30} | {score:.4f}")

if __name__ == "__main__":
    main()