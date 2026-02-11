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
        

        self.counts = {i: defaultdict(Counter) for i in range(1, n + 1)}
        
        self.context_totals = {i: Counter() for i in range(1, n + 1)}

        self.kn_continuation_counts = Counter()
        self.kn_total_continuation = 0
        
        self.vocab = set()
        self.total_words = 0

    def train(self, tokenized_sentences):
        for tokens in tqdm(tokenized_sentences, desc="Training LM", unit=" lines"):
            padded = (['<s>'] * (self.n - 1)) + tokens + ['</s>']
            self.total_words += len(tokens) + 1 # +1 for </s>
            
            for i in range(len(padded)):
                word = padded[i]
                if word != '<s>': self.vocab.add(word)
                
                for order in range(1, self.n + 1):
                    if i - order + 1 >= 0:
                        context = tuple(padded[i - order + 1 : i])
                        self.counts[order][context][word] += 1
                        self.context_totals[order][context] += 1

        if self.smoothing == 'kneser-ney':
            print("Calculating Kneser-Ney continuation counts...")
            for context_tuple, next_words_counter in self.counts[2].items():
                for next_word in next_words_counter:
                    self.kn_continuation_counts[next_word] += 1
                    self.kn_total_continuation += 1

    def get_probability(self, word, context, order=None):
        """Calculates the probability of a word given a context using the selected smoothing."""
        # Small epsilon to avoid zero probabilities
        EPSILON = 1e-10
        
        if order is None: order = self.n
        context = tuple(context[-(order-1):]) if order > 1 else ()

        if order is None: order = self.n


        if order == 0:
            return 1 / len(self.vocab) if len(self.vocab) > 0 else EPSILON

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
            
            T = len(self.counts[order][context])
            
            if total > 0:
                if count > 0:
                    return count / (total + T)
                else:
                    weight = T / (total + T)
                    return weight * self.get_probability(word, context, order - 1)
            else:
                if order > 1:
                    return self.get_probability(word, context, order - 1)
                else:
                    return 1 / len(self.vocab) if self.vocab else EPSILON

        # --- 3. Kneser-Ney Smoothing (Simplified version) ---
        elif self.smoothing == 'kneser-ney':
            if order == 1:
                count = self.kn_continuation_counts[word]
                total = self.kn_total_continuation
                return count / total if total > 0 else (1 / len(self.vocab) if self.vocab else EPSILON)

            count = self.counts[order][context][word]
            total = self.context_totals[order][context]
            D = 0.75 # Discount

            if total > 0:
                T = len(self.counts[order][context]) 
                lambda_weight = (D / total) * T
                
                prob_lower = self.get_probability(word, context, order - 1)
                
                first_term = max(count - D, 0) / total
                return first_term + (lambda_weight * prob_lower)
            else:
                return self.get_probability(word, context, order - 1)

    def generate_sentence(self, prefix_text, tokenizer, max_len=20):
        tokens = tokenizer.tokenize(prefix_text)
        result_tokens = list(tokens)
        
        for _ in range(max_len):
            context = result_tokens[-(self.n - 1):]
            
            candidates = {}
            context_tuple = tuple(context)
            for word in self.counts[self.n][context_tuple]:
                candidates[word] = self.get_probability(word, context)
            
            if self.smoothing and not candidates:
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
        
        EPSILON = 1e-10
        
        for sentence in tqdm(test_sentences, desc="Calculating Perplexity", leave=False):
            tokens = tokenizer.tokenize(sentence)

            padded_tokens = (['<s>'] * (self.n - 1)) + tokens + ['</s>']
            
            start_index = self.n - 1 
            
            for i in range(start_index, len(padded_tokens)):
                word = padded_tokens[i]
                context = padded_tokens[i - (self.n - 1) : i]
                
                prob = self.get_probability(word, context)
                
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
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        print(f"Loading model from {filepath}...")
        print(f"  -> File size: {file_size_mb:.2f} MB (This may take a while)")
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
        print("  -> Model loaded successfully!")
        return model
    
