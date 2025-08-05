"""
Goal2Vec: Vector representations for mathematical goals and tactics.
A novel embedding model that captures the relationship between proof goals and tactics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from collections import defaultdict, Counter
import re
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.word2vec_inner import FAST_VERSION

from definition_database import DefinitionDatabase, MathDefinition
from reinforcement_learning_node import RLStateNode, TacticAction
from minif2f_processor import MinIF2FProcessor, MathProblem


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    embedding_dim: int = 128
    window_size: int = 5
    min_count: int = 2
    workers: int = 4
    epochs: int = 100
    learning_rate: float = 0.001
    negative_samples: int = 5
    batch_size: int = 64


class MathTokenizer:
    """Tokenizer for mathematical expressions and goals."""
    
    def __init__(self):
        self.vocab = set()
        self.token_patterns = [
            r'∀|∃|→|↔|∧|∨|¬',  # Logic symbols
            r'≤|≥|≠|∈|∉|⊆|⊇|∪|∩',  # Math symbols
            r'[a-zA-Z_][a-zA-Z0-9_]*',  # Identifiers
            r'\d+\.?\d*',  # Numbers
            r'[+\-*/^=<>()]',  # Operators and parentheses
            r'[,;:]',  # Punctuation
        ]
        self.compiled_patterns = [re.compile(pattern) for pattern in self.token_patterns]
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize mathematical text."""
        tokens = []
        text = text.strip()
        i = 0
        
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            
            matched = False
            for pattern in self.compiled_patterns:
                match = pattern.match(text, i)
                if match:
                    token = match.group()
                    tokens.append(token)
                    self.vocab.add(token)
                    i = match.end()
                    matched = True
                    break
            
            if not matched:
                # Unknown character, add as single token
                tokens.append(text[i])
                self.vocab.add(text[i])
                i += 1
        
        return tokens
    
    def tokenize_goal_tactic_pair(self, goal: str, tactic: str) -> Tuple[List[str], List[str]]:
        """Tokenize goal-tactic pair."""
        goal_tokens = self.tokenize(goal)
        tactic_tokens = self.tokenize(tactic)
        return goal_tokens, tactic_tokens
    
    def build_vocab_from_corpus(self, corpus: List[str]) -> None:
        """Build vocabulary from a corpus of mathematical texts."""
        for text in corpus:
            self.tokenize(text)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file."""
        with open(filepath, 'w') as f:
            json.dump(list(self.vocab), f, indent=2)
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file."""
        with open(filepath, 'r') as f:
            vocab_list = json.load(f)
            self.vocab = set(vocab_list)


class Goal2VecModel(nn.Module):
    """
    Goal2Vec: Neural network model for learning goal-tactic embeddings.
    
    This model learns vector representations that capture the semantic
    relationship between mathematical goals and the tactics used to solve them.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embeddings for goals and tactics
        self.goal_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.tactic_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context encoders
        self.goal_encoder = nn.LSTM(embedding_dim, embedding_dim // 2, bidirectional=True, batch_first=True)
        self.tactic_encoder = nn.LSTM(embedding_dim, embedding_dim // 2, bidirectional=True, batch_first=True)
        
        # Attention mechanism
        self.goal_attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.tactic_attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        
        # Projection layers
        self.goal_projection = nn.Linear(embedding_dim, embedding_dim)
        self.tactic_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Similarity scoring
        self.similarity_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for embedding in [self.goal_embeddings, self.tactic_embeddings]:
            nn.init.uniform_(embedding.weight, -0.1, 0.1)
        
        for layer in [self.goal_projection, self.tactic_projection]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def encode_sequence(self, token_ids: torch.Tensor, 
                       embedding_layer: nn.Embedding,
                       encoder: nn.LSTM,
                       attention: nn.MultiheadAttention,
                       projection: nn.Linear) -> torch.Tensor:
        """Encode a sequence of tokens into a fixed-size embedding."""
        # Get embeddings
        embeddings = embedding_layer(token_ids)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM encoding
        lstm_out, (hidden, _) = encoder(embeddings)
        
        # Self-attention
        attended, _ = attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling (mean)
        pooled = attended.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Projection
        projected = projection(pooled)
        
        return projected
    
    def encode_goal(self, goal_token_ids: torch.Tensor) -> torch.Tensor:
        """Encode goal into embedding vector."""
        return self.encode_sequence(
            goal_token_ids, 
            self.goal_embeddings,
            self.goal_encoder,
            self.goal_attention,
            self.goal_projection
        )
    
    def encode_tactic(self, tactic_token_ids: torch.Tensor) -> torch.Tensor:
        """Encode tactic into embedding vector."""
        return self.encode_sequence(
            tactic_token_ids,
            self.tactic_embeddings,
            self.tactic_encoder,
            self.tactic_attention,
            self.tactic_projection
        )
    
    def forward(self, goal_token_ids: torch.Tensor, 
                tactic_token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute goal-tactic compatibility score."""
        goal_embedding = self.encode_goal(goal_token_ids)
        tactic_embedding = self.encode_tactic(tactic_token_ids)
        
        # Concatenate embeddings
        combined = torch.cat([goal_embedding, tactic_embedding], dim=-1)
        
        # Compute compatibility score
        score = self.similarity_mlp(combined)
        
        return score.squeeze(-1)
    
    def get_goal_embedding(self, goal_tokens: List[str], tokenizer: MathTokenizer) -> np.ndarray:
        """Get embedding for a goal string."""
        # Convert tokens to IDs (simplified)
        token_to_id = {token: i for i, token in enumerate(tokenizer.vocab)}
        token_ids = [token_to_id.get(token, 0) for token in goal_tokens]
        
        # Pad sequence
        max_len = 50
        if len(token_ids) < max_len:
            token_ids.extend([0] * (max_len - len(token_ids)))
        else:
            token_ids = token_ids[:max_len]
        
        token_tensor = torch.LongTensor([token_ids])
        
        with torch.no_grad():
            embedding = self.encode_goal(token_tensor)
        
        return embedding.numpy().flatten()


class Goal2VecTrainer:
    """Trainer for Goal2Vec model."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.tokenizer = MathTokenizer()
        self.model = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training data
        self.goal_tactic_pairs = []
        self.vocab_to_id = {}
        self.id_to_vocab = {}
        
        # Training statistics
        self.training_losses = []
        self.validation_losses = []
    
    def prepare_training_data(self, problems: List[MathProblem], 
                            rl_trees: List[RLStateNode] = None) -> None:
        """Prepare training data from problems and RL trees."""
        print("Preparing training data...")
        
        # Extract goal-tactic pairs from problems
        for problem in problems:
            if problem.proof and problem.proof != "sorry":
                # Simple proof parsing - extract tactics
                tactics = self._extract_tactics_from_proof(problem.proof)
                for tactic in tactics:
                    self.goal_tactic_pairs.append((problem.formal_statement, tactic))
        
        # Extract goal-tactic pairs from RL trees
        if rl_trees:
            for tree in rl_trees:
                self._extract_pairs_from_tree(tree)
        
        # Build vocabulary
        all_text = []
        for goal, tactic in self.goal_tactic_pairs:
            all_text.extend([goal, tactic])
        
        self.tokenizer.build_vocab_from_corpus(all_text)
        
        # Create vocabulary mappings
        vocab_list = list(self.tokenizer.vocab)
        self.vocab_to_id = {token: i for i, token in enumerate(vocab_list)}
        self.id_to_vocab = {i: token for i, token in enumerate(vocab_list)}
        
        print(f"Prepared {len(self.goal_tactic_pairs)} goal-tactic pairs")
        print(f"Vocabulary size: {len(self.tokenizer.vocab)}")
    
    def _extract_tactics_from_proof(self, proof: str) -> List[str]:
        """Extract tactics from proof string."""
        # Simple tactic extraction
        tactics = []
        tactic_patterns = [
            r'\bsimp\b', r'\brw\b', r'\bring\b', r'\blinarith\b',
            r'\bexact\b', r'\bapply\b', r'\bintro\b', r'\bcases\b',
            r'\binduction\b', r'\bunfold\b', r'\bnorm_num\b',
            r'\bomega\b', r'\btauto\b', r'\bconstructor\b'
        ]
        
        for pattern in tactic_patterns:
            if re.search(pattern, proof, re.IGNORECASE):
                tactics.append(pattern.replace(r'\b', '').replace('\\', ''))
        
        return tactics if tactics else ["sorry"]
    
    def _extract_pairs_from_tree(self, tree: RLStateNode) -> None:
        """Extract goal-tactic pairs from RL tree."""
        def traverse(node):
            if node.tactic_action:
                self.goal_tactic_pairs.append((node.goal, node.tactic_action.tactic))
            
            for child in node.children:
                traverse(child)
        
        traverse(tree)
    
    def create_model(self) -> None:
        """Create Goal2Vec model."""
        vocab_size = len(self.tokenizer.vocab)
        self.model = Goal2VecModel(vocab_size, self.config.embedding_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def prepare_batch(self, goal_tactic_pairs: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare a batch of training data."""
        goals = []
        tactics = []
        labels = []
        
        for goal_text, tactic_text in goal_tactic_pairs:
            # Positive example
            goal_tokens = self.tokenizer.tokenize(goal_text)
            tactic_tokens = self.tokenizer.tokenize(tactic_text)
            
            goal_ids = self._tokens_to_ids(goal_tokens)
            tactic_ids = self._tokens_to_ids(tactic_tokens)
            
            goals.append(goal_ids)
            tactics.append(tactic_ids)
            labels.append(1.0)
            
            # Negative example (random tactic)
            random_tactic = np.random.choice([pair[1] for pair in self.goal_tactic_pairs])
            random_tactic_tokens = self.tokenizer.tokenize(random_tactic)
            random_tactic_ids = self._tokens_to_ids(random_tactic_tokens)
            
            goals.append(goal_ids)
            tactics.append(random_tactic_ids)
            labels.append(0.0)
        
        # Pad sequences
        max_goal_len = max(len(g) for g in goals)
        max_tactic_len = max(len(t) for t in tactics)
        
        goal_tensor = torch.zeros(len(goals), max_goal_len, dtype=torch.long)
        tactic_tensor = torch.zeros(len(tactics), max_tactic_len, dtype=torch.long)
        
        for i, (goal_ids, tactic_ids) in enumerate(zip(goals, tactics)):
            goal_tensor[i, :len(goal_ids)] = torch.LongTensor(goal_ids)
            tactic_tensor[i, :len(tactic_ids)] = torch.LongTensor(tactic_ids)
        
        return goal_tensor.to(self.device), tactic_tensor.to(self.device), torch.FloatTensor(labels).to(self.device)
    
    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return [self.vocab_to_id.get(token, 0) for token in tokens]
    
    def train(self, epochs: Optional[int] = None) -> None:
        """Train the Goal2Vec model."""
        if epochs is None:
            epochs = self.config.epochs
        
        if not self.model:
            self.create_model()
        
        print(f"Training Goal2Vec model for {epochs} epochs...")
        
        # Split data into train/validation
        split_idx = int(0.8 * len(self.goal_tactic_pairs))
        train_pairs = self.goal_tactic_pairs[:split_idx]
        val_pairs = self.goal_tactic_pairs[split_idx:]
        
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle training data
            np.random.shuffle(train_pairs)
            
            for i in range(0, len(train_pairs), self.config.batch_size):
                batch_pairs = train_pairs[i:i + self.config.batch_size]
                
                goal_tensor, tactic_tensor, labels = self.prepare_batch(batch_pairs)
                
                # Forward pass
                scores = self.model(goal_tensor, tactic_tensor)
                loss = criterion(scores, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            self.training_losses.append(avg_train_loss)
            
            # Validation
            if val_pairs:
                val_loss = self._validate(val_pairs, criterion)
                self.validation_losses.append(val_loss)
            
            # Log progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        print("Training completed!")
    
    def _validate(self, val_pairs: List[Tuple[str, str]], criterion) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_pairs), self.config.batch_size):
                batch_pairs = val_pairs[i:i + self.config.batch_size]
                
                goal_tensor, tactic_tensor, labels = self.prepare_batch(batch_pairs)
                scores = self.model(goal_tensor, tactic_tensor)
                loss = criterion(scores, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def get_goal_embedding(self, goal: str) -> np.ndarray:
        """Get embedding for a goal."""
        goal_tokens = self.tokenizer.tokenize(goal)
        return self.model.get_goal_embedding(goal_tokens, self.tokenizer)
    
    def find_similar_goals(self, query_goal: str, candidate_goals: List[str], 
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """Find goals similar to query goal."""
        query_embedding = self.get_goal_embedding(query_goal)
        
        similarities = []
        for candidate in candidate_goals:
            candidate_embedding = self.get_goal_embedding(candidate)
            
            # Cosine similarity
            similarity = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )
            similarities.append((candidate, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def recommend_tactics(self, goal: str, candidate_tactics: List[str], 
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """Recommend tactics for a given goal."""
        goal_tokens = self.tokenizer.tokenize(goal)
        goal_ids = self._tokens_to_ids(goal_tokens)
        
        # Pad goal sequence
        max_len = 50
        if len(goal_ids) < max_len:
            goal_ids.extend([0] * (max_len - len(goal_ids)))
        else:
            goal_ids = goal_ids[:max_len]
        
        goal_tensor = torch.LongTensor([goal_ids]).to(self.device)
        
        tactic_scores = []
        
        with torch.no_grad():
            for tactic in candidate_tactics:
                tactic_tokens = self.tokenizer.tokenize(tactic)
                tactic_ids = self._tokens_to_ids(tactic_tokens)
                
                # Pad tactic sequence
                if len(tactic_ids) < max_len:
                    tactic_ids.extend([0] * (max_len - len(tactic_ids)))
                else:
                    tactic_ids = tactic_ids[:max_len]
                
                tactic_tensor = torch.LongTensor([tactic_ids]).to(self.device)
                
                score = self.model(goal_tensor, tactic_tensor)
                tactic_scores.append((tactic, torch.sigmoid(score).item()))
        
        # Sort by score
        tactic_scores.sort(key=lambda x: x[1], reverse=True)
        return tactic_scores[:top_k]
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "vocab_to_id": self.vocab_to_id,
            "id_to_vocab": self.id_to_vocab,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses
        }
        
        torch.save(checkpoint, filepath)
        
        # Save tokenizer vocabulary
        vocab_path = str(Path(filepath).with_suffix('.vocab'))
        self.tokenizer.save_vocab(vocab_path)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint["config"]
        self.vocab_to_id = checkpoint["vocab_to_id"]
        self.id_to_vocab = checkpoint["id_to_vocab"]
        self.training_losses = checkpoint["training_losses"]
        self.validation_losses = checkpoint["validation_losses"]
        
        # Load tokenizer vocabulary
        vocab_path = str(Path(filepath).with_suffix('.vocab'))
        if Path(vocab_path).exists():
            self.tokenizer.load_vocab(vocab_path)
        
        # Create and load model
        vocab_size = len(self.vocab_to_id)
        self.model = Goal2VecModel(vocab_size, self.config.embedding_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Model loaded from {filepath}")
    
    def visualize_embeddings(self, goals: List[str], save_path: str = "/tmp/goal_embeddings.png") -> None:
        """Visualize goal embeddings using t-SNE."""
        if not goals:
            return
        
        # Get embeddings
        embeddings = []
        valid_goals = []
        
        for goal in goals:
            try:
                embedding = self.get_goal_embedding(goal)
                embeddings.append(embedding)
                valid_goals.append(goal[:50] + "..." if len(goal) > 50 else goal)
            except Exception as e:
                print(f"Error getting embedding for goal: {e}")
                continue
        
        if len(embeddings) < 2:
            print("Need at least 2 valid embeddings for visualization")
            return
        
        embeddings = np.array(embeddings)
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        # Add labels
        for i, goal in enumerate(valid_goals):
            plt.annotate(goal, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.title("Goal2Vec Embeddings Visualization")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Embedding visualization saved to {save_path}")


# Integration with existing Word2Vec
class Word2VecGoalComparator:
    """Compare Goal2Vec with traditional Word2Vec for mathematical texts."""
    
    def __init__(self, tokenizer: MathTokenizer):
        self.tokenizer = tokenizer
        self.word2vec_model = None
        self.goal2vec_trainer = None
    
    def train_word2vec(self, corpus: List[str]) -> None:
        """Train Word2Vec model on mathematical corpus."""
        # Tokenize corpus
        tokenized_corpus = [self.tokenizer.tokenize(text) for text in corpus]
        
        # Train Word2Vec
        self.word2vec_model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=128,
            window=5,
            min_count=2,
            workers=4,
            epochs=100
        )
        
        print("Word2Vec model trained")
    
    def compare_similarity(self, goal1: str, goal2: str) -> Dict[str, float]:
        """Compare similarity using both models."""
        results = {}
        
        # Word2Vec similarity
        if self.word2vec_model:
            tokens1 = self.tokenizer.tokenize(goal1)
            tokens2 = self.tokenizer.tokenize(goal2)
            
            # Average word vectors
            def get_avg_vector(tokens):
                vectors = []
                for token in tokens:
                    if token in self.word2vec_model.wv:
                        vectors.append(self.word2vec_model.wv[token])
                
                if vectors:
                    return np.mean(vectors, axis=0)
                else:
                    return np.zeros(self.word2vec_model.vector_size)
            
            vec1 = get_avg_vector(tokens1)
            vec2 = get_avg_vector(tokens2)
            
            # Cosine similarity
            word2vec_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            results["word2vec"] = float(word2vec_sim)
        
        # Goal2Vec similarity
        if self.goal2vec_trainer and self.goal2vec_trainer.model:
            emb1 = self.goal2vec_trainer.get_goal_embedding(goal1)
            emb2 = self.goal2vec_trainer.get_goal_embedding(goal2)
            
            goal2vec_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            results["goal2vec"] = float(goal2vec_sim)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print("=== Goal2Vec Model Testing ===")
    
    # Create configuration
    config = EmbeddingConfig(
        embedding_dim=64,  # Smaller for testing
        epochs=20,
        batch_size=8
    )
    
    # Create trainer
    trainer = Goal2VecTrainer(config)
    
    # Create sample training data
    sample_problems = [
        MathProblem(
            problem_id="test1",
            statement="∀ n : ℕ, n + 0 = n",
            proof="by simp"
        ),
        MathProblem(
            problem_id="test2", 
            statement="∀ a b : ℕ, a + b = b + a",
            proof="by ring"
        ),
        MathProblem(
            problem_id="test3",
            statement="∀ n : ℕ, 0 + n = n",
            proof="by simp"
        )
    ]
    
    # Prepare training data
    trainer.prepare_training_data(sample_problems)
    
    # Train model
    trainer.train(epochs=10)
    
    # Test embeddings
    test_goal = "∀ n : ℕ, n + 0 = n"
    embedding = trainer.get_goal_embedding(test_goal)
    print(f"Goal embedding shape: {embedding.shape}")
    print(f"Goal embedding sample: {embedding[:5]}")
    
    # Test tactic recommendation
    candidate_tactics = ["simp", "ring", "linarith", "exact", "rw"]
    recommendations = trainer.recommend_tactics(test_goal, candidate_tactics, top_k=3)
    
    print(f"\nTactic recommendations for '{test_goal}':")
    for tactic, score in recommendations:
        print(f"  {tactic}: {score:.3f}")
    
    # Test goal similarity
    candidate_goals = [
        "∀ n : ℕ, 0 + n = n",
        "∀ a b : ℕ, a + b = b + a",
        "∀ x : ℝ, x + 0 = x"
    ]
    
    similar_goals = trainer.find_similar_goals(test_goal, candidate_goals, top_k=2)
    print(f"\nSimilar goals to '{test_goal}':")
    for goal, similarity in similar_goals:
        print(f"  {similarity:.3f}: {goal}")
    
    # Save model
    trainer.save_model("/tmp/goal2vec_model.pth")
    
    # Test Word2Vec comparison
    print(f"\n=== Word2Vec Comparison ===")
    comparator = Word2VecGoalComparator(trainer.tokenizer)
    
    # Create corpus for Word2Vec
    corpus = [p.statement for p in sample_problems] + [p.proof for p in sample_problems]
    comparator.train_word2vec(corpus)
    comparator.goal2vec_trainer = trainer
    
    # Compare similarities
    goal1 = "∀ n : ℕ, n + 0 = n"
    goal2 = "∀ n : ℕ, 0 + n = n"
    
    similarities = comparator.compare_similarity(goal1, goal2)
    print(f"Similarity between goals:")
    print(f"  Goal 1: {goal1}")
    print(f"  Goal 2: {goal2}")
    for model, sim in similarities.items():
        print(f"  {model}: {sim:.3f}")
    
    print("Goal2Vec testing completed!")