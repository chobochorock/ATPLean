#!/usr/bin/env python3
"""Debug tokenizer to identify empty sequence issue."""

from goal2vec_model import MathTokenizer

def test_tokenizer():
    tokenizer = MathTokenizer()
    
    # Test samples from simple_goal2vec_training.py
    test_cases = [
        "∀ n : ℕ, n + 0 = n",
        "by simp",
        "simp",
        "ring", 
        "norm_num",
        "∀ a b : ℕ, a + b = b + a",
        "by ring"
    ]
    
    print("=== Tokenizer Debug ===")
    for i, text in enumerate(test_cases):
        tokens = tokenizer.tokenize(text)
        print(f"{i+1}. Text: '{text}'")
        print(f"   Tokens: {tokens}")
        print(f"   Token count: {len(tokens)}")
        print()
    
    print(f"Final vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Vocabulary: {sorted(list(tokenizer.vocab))}")

if __name__ == "__main__":
    test_tokenizer()