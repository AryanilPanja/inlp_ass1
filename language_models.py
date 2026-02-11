import math
import json
from collections import Counter, defaultdict
from tqdm import tqdm
import pickle
import os

class LanguageModel:
    def __init__(self, n=4, smoothing=None):
        self.n = n
        self.smoothing = smoothing # None, 'witten-bell', or 'kneser-ney'
        
        # We now store counts for all levels: 1, 2, 3, and 4-grams
        # counts[order][context] = Counter({next_word: count})
        self.counts = {i: defaultdict(Counter) for i in range(1, n + 1)}
        
        # Total occurrences of each context: counts[order][context].total()
        self.context_totals = {i: Counter() for i in range(1, n + 1)}

        # Kneser-Ney specific: Continuation counts for Unigrams
        self.kn_continuation_counts = Counter()
        self.kn_total_continuation = 0
        
        self.vocab = set()
        self.total_words = 0

    def train(self, tokenized_sentences):
        for tokens in tqdm(tokenized_sentences, desc="Training LM", unit=" lines"):
            # Pad for the highest order
            padded = (['<s>'] * (self.n - 1)) + tokens + ['</s>']
            self.total_words += len(tokens) + 1 # +1 for </s>
            
            for i in range(len(padded)):
                word = padded[i]
                if word != '<s>': self.vocab.add(word)
                
                # Count for every order from 1 to N
                for order in range(1, self.n + 1):
                    if i - order + 1 >= 0:
                        context = tuple(padded[i - order + 1 : i])
                        self.counts[order][context][word] += 1
                        self.context_totals[order][context] += 1
        # --- FIX for Kneser-Ney: Calculate Continuation Counts ---
        # A word's continuation count is the number of unique Bigram contexts it completes.
        # We derive this from our order=2 counts.
        if self.smoothing == 'kneser-ney':
            print("Calculating Kneser-Ney continuation counts...")
            for context_tuple, next_words_counter in self.counts[2].items():
                # context_tuple is like ('previous_word',)
                for next_word in next_words_counter:
                    # Increment simply because this (prev, next) pair exists
                    self.kn_continuation_counts[next_word] += 1
                    self.kn_total_continuation += 1

    def get_probability(self, word, context, order=None):
        """Calculates the probability of a word given a context using the selected smoothing."""
        # Small epsilon to avoid zero probabilities
        EPSILON = 1e-10
        
        if order is None: order = self.n
        context = tuple(context[-(order-1):]) if order > 1 else ()

        if order is None: order = self.n

        # --- FIX: Base Case to prevent KeyError: 0 ---
        # If we back off past Unigrams (order 1), return Uniform Probability (1 / Vocab Size)
        if order == 0:
            return 1 / len(self.vocab) if len(self.vocab) > 0 else EPSILON

        # Adjust context to match the current order
        # e.g. for order 3, we only need the last 2 words of context
        context = tuple(context[-(order-1):]) if order > 1 else ()


        # --- 1. No Smoothing (MLE) ---
        if self.smoothing is None:
            count = self.counts[order][context][word]
            total = self.context_totals[order][context]
            return count / total if total > 0 else EPSILON

        # --- 2. Witten-Bell Smoothing ---
        elif self.smoothing == 'witten-bell':
            count = self.counts[order][context][word]
            total = self.context_totals[order][context]
            
            # T is the number of unique words seen after this context
            T = len(self.counts[order][context])
            
            if total > 0:
                if count > 0:
                    # Seen n-gram
                    return count / (total + T)
                else:
                    # Unseen n-gram: back off to order-1
                    weight = T / (total + T)
                    return weight * self.get_probability(word, context, order - 1)
            else:
                # Context never seen: back off or use uniform unigram
                if order > 1:
                    return self.get_probability(word, context, order - 1)
                else:
                    return 1 / len(self.vocab) if self.vocab else EPSILON

        # --- 3. Kneser-Ney Smoothing (Simplified version) ---
        elif self.smoothing == 'kneser-ney':
            # --- FIX: Unigram Base Case uses Continuation Counts ---
            if order == 1:
                count = self.kn_continuation_counts[word]
                total = self.kn_total_continuation
                # Fallback to uniform if total is 0 (shouldn't happen with trained data)
                return count / total if total > 0 else (1 / len(self.vocab) if self.vocab else EPSILON)

            # Recursive Step for orders > 1
            count = self.counts[order][context][word]
            total = self.context_totals[order][context]
            D = 0.75 # Discount

            if total > 0:
                # Number of unique words following this context
                T = len(self.counts[order][context]) 
                lambda_weight = (D / total) * T
                
                prob_lower = self.get_probability(word, context, order - 1)
                
                # Formula: Interpolated Kneser-Ney
                first_term = max(count - D, 0) / total
                return first_term + (lambda_weight * prob_lower)
            else:
                # Context unseen, fully backoff
                return self.get_probability(word, context, order - 1)

    def generate_sentence(self, prefix_text, tokenizer, max_len=20):
        tokens = tokenizer.tokenize(prefix_text)
        result_tokens = list(tokens)
        
        for _ in range(max_len):
            context = result_tokens[-(self.n - 1):]
            
            # Get probabilities for all words in vocab
            # Note: For efficiency, we only check words that appeared in this context
            # plus a back-off probability.
            candidates = {}
            # Check words seen in this 4-gram context
            context_tuple = tuple(context)
            for word in self.counts[self.n][context_tuple]:
                candidates[word] = self.get_probability(word, context)
            
            # If smoothing is active, we also look at words in lower-order contexts
            if self.smoothing and not candidates:
                # This is a fallback to make sure we always get *something*
                for word in ['the', 'a', '.', 'is']: # Common fallback words
                    candidates[word] = self.get_probability(word, context)

            if not candidates: break
            
            next_word = max(candidates, key=candidates.get)
            if next_word == '</s>': break
            result_tokens.append(next_word)
            
        return " ".join(result_tokens)
    
    def save_model(self, filepath):
        """Saves the entire model object to a file."""
        print(f"Saving model to {filepath}...")
        with open(filepath, 'wb') as f: # 'wb' means write binary
            pickle.dump(self, f)

    def score_perplexity(self, test_sentences, tokenizer):
        """
        Calculates the Perplexity of the model on a list of test sentences.
        PP(W) = 2^(-1/N * sum(log2(P(word|context))))
        """
        total_log_prob = 0
        total_words = 0
        
        # Small epsilon to avoid log(0) and infinite perplexity
        EPSILON = 1e-10
        
        # We assume test_sentences is a list of raw strings
        for sentence in tqdm(test_sentences, desc="Calculating Perplexity", leave=False):
            tokens = tokenizer.tokenize(sentence)
            # We include </s> in the evaluation, but not <s>
            # The context needs padding
            padded_tokens = (['<s>'] * (self.n - 1)) + tokens + ['</s>']
            
            # Evaluate P(word | context) for every word in the sentence + </s>
            # We start measuring from the first real word (after padding)
            start_index = self.n - 1 
            
            for i in range(start_index, len(padded_tokens)):
                word = padded_tokens[i]
                context = padded_tokens[i - (self.n - 1) : i]
                
                prob = self.get_probability(word, context)
                
                # Apply epsilon floor to prevent log(0)
                if prob == 0:
                    prob = EPSILON
                
                total_log_prob += math.log2(prob)
                total_words += 1
                
        if total_words == 0:
            return 0

        avg_log_prob = total_log_prob / total_words
        perplexity = 2 ** (-avg_log_prob)
        return perplexity

    @staticmethod
    def load_model(filepath):
        # Get file size in Megabytes
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        print(f"Loading model from {filepath}...")
        print(f"  -> File size: {file_size_mb:.2f} MB (This may take a while)")
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
        print("  -> Model loaded successfully!")
        return model
    
