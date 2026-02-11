import re
import json
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not installed, define a dummy function that just returns the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable

class WhitespaceTokenizer:
    def __init__(self):
        self.punctuation_regex = re.compile(r'([^\w\s])')

    def tokenize(self, text):
        text_with_spaces = self.punctuation_regex.sub(r' \1 ', text)
        return text_with_spaces.split()
    
    def decode(self, tokens):
        return " ".join(tokens)

class RegexTokenizer:
    def __init__(self):
        self.pattern = re.compile(pattern=r'\w+|[^\w\s]')
        #self.pattern = re.compile(r"[a-zA-Z']+[^\w\s]?")

    def tokenize(self, text):
        return self.pattern.findall(text)
    
    def decode(self, tokens):
        """Joins tokens back into a sentence."""
        return " ".join(tokens)

class BPETokenizer:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.merges = []

    def train(self, corpus):
        word_freqs = Counter()
        for text in tqdm(corpus, desc="BPE: Pre-processing Corpus", unit=" lines"):
            words = text.split()
            for word in words:
                char_word = ""
                for char in word:
                    char_word += char + " "
                char_word += "</w>"
                word_freqs[char_word] += 1

        # 2. Iterative Merging
        # Wrap the range in tqdm to see progress of the merges
        for i in tqdm(range(self.num_merges), desc="BPE: Learning Merges", unit=" merge"):
            pairs = Counter()
            for word, freq in word_freqs.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j+1])
                    pairs[pair] += freq
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            # Update vocab
            new_freqs = {}
            bigram = ' '.join(best_pair)
            replacement = ''.join(best_pair)
            pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
            for word in word_freqs:
                new_word = pattern.sub(replacement, word)
                new_freqs[new_word] = word_freqs[word]
            word_freqs = new_freqs
            
        print(f"BPE training complete with {len(self.merges)} merges.")

    def tokenize(self, text):
        
        words = text.split()
        final_tokens = []
        for word in words:
            chars = []
            for c in word:
                chars.append(c)
            chars.append("</w>")
            
            for pair in self.merges:
                p1, p2 = pair
                new_chars = []
                idx = 0
                while idx < len(chars):
                    if idx < len(chars)-1 and chars[idx] == p1 and chars[idx+1] == p2:
                        new_chars.append(p1 + p2)
                        idx += 2
                    else:
                        new_chars.append(chars[idx])
                        idx += 1
                chars = new_chars
            for t in chars:
                final_tokens.append(t)
        return final_tokens

    def save_model(self, filename):
        """Saves the learned merges to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.merges, f)
        print(f"BPE model saved to {filename}")

    def load_model(self, filename):
        """Loads merges from a JSON file so you don't have to retrain."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert lists back to tuples
            self.merges = []
            for item in data:
                self.merges.append(tuple(item))
        print(f"BPE model loaded from {filename} with {len(self.merges)} merges.")

    def decode(self, tokens):
        """
        Converts BPE tokens back to a string.
        1. Join all tokens together (e.g. "The</w>R" -> "The</w>R")
        2. Replace </w> with a space ("The R")
        3. Strip extra spaces
        """
        concatenated = "".join(tokens)
        decoded = concatenated.replace("</w>", " ")
        return decoded.strip()