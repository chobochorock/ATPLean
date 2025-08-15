#!/usr/bin/env python3
"""
Mathlib Goal2Vec Trainer: Enhanced Goal2Vec training with Mathlib data.

This module extends the Goal2Vec model to train on comprehensive Mathlib data,
providing better mathematical understanding and tactic recommendation capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from pathlib import Path
import sqlite3
import json
import logging
from collections import defaultdict, Counter
import time
import re

from goal2vec_model import Goal2VecTrainer, Goal2VecModel, EmbeddingConfig, MathTokenizer
from mathlib_data_loader import MathlibDataLoader, LeanTheorem, TacticUsage
from minif2f_processor import MathProblem, ProblemDifficulty
from definition_database import DefinitionDatabase, MathDefinition, DefinitionType


logger = logging.getLogger(__name__)


class MathlibTokenizer(MathTokenizer):
    """Enhanced tokenizer for Mathlib mathematical expressions."""
    
    def __init__(self):
        super().__init__()
        
        # Add Mathlib-specific patterns
        self.mathlib_patterns = [
            r'[⟨⟩⟪⟫]',  # Angle brackets
            r'[αβγδεζηθικλμνξοπρστυφχψω]',  # Greek letters
            r'[ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]',  # Capital Greek letters
            r'[ℕℤℚℝℂ]',  # Number systems
            r'[₀₁₂₃₄₅₆₇₈₉]',  # Subscripts
            r'[⁰¹²³⁴⁵⁶⁷⁸⁹]',  # Superscripts
            r'[∑∏∫∮]',  # Mathematical operators
            r'[⊕⊗⊘⊙⊚⊛]',  # Binary operators
            r'[⊢⊨⊩⊪⊫⊬⊭⊮⊯]',  # Logical symbols
            r'[■□▪▫]',  # Proof symbols
        ]
        
        # Compile Mathlib patterns
        self.compiled_mathlib_patterns = [re.compile(pattern) for pattern in self.mathlib_patterns]
        
        # Add Mathlib-specific tokens to vocabulary during initialization
        self.mathlib_vocab = {
            'simp', 'rw', 'ring', 'norm_num', 'linarith', 'omega', 'exact', 'apply',
            'intro', 'intros', 'cases', 'induction', 'constructor', 'tauto', 'decide',
            'assumption', 'rfl', 'unfold', 'fold', 'conv', 'abel', 'group', 'field_simp',
            'norm_cast', 'push_neg', 'by_contra', 'contrapose', 'use', 'refine', 'obtain',
            'have', 'suffices', 'wlog', 'rcases', 'rintro', 'ext', 'funext', 'congruence',
            'library_search', 'aesop', 'polyrith', 'linear_combination', 'nlinarith',
            'positivity', 'Nat', 'Int', 'Real', 'Complex', 'List', 'Set', 'Function',
            'Type', 'Prop', 'Sort', 'theorem', 'lemma', 'def', 'example', 'inductive',
            'structure', 'class', 'instance', 'namespace', 'open', 'variable', 'universe'
        }
        self.vocab.update(self.mathlib_vocab)
    
    def tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with Mathlib-specific patterns."""
        tokens = []
        text = text.strip()
        i = 0
        
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            
            matched = False
            
            # Try Mathlib-specific patterns first
            for pattern in self.compiled_mathlib_patterns:
                match = pattern.match(text, i)
                if match:
                    token = match.group()
                    tokens.append(token)
                    self.vocab.add(token)
                    i = match.end()
                    matched = True
                    break
            
            if not matched:
                # Fall back to base tokenization
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


class MathlibGoal2VecTrainer(Goal2VecTrainer):
    """Enhanced Goal2Vec trainer with Mathlib integration."""
    
    def __init__(self, config: EmbeddingConfig, mathlib_db_path: str = "mathlib_training_data.db"):
        # Use enhanced tokenizer
        super().__init__(config)
        self.tokenizer = MathlibTokenizer()
        
        self.mathlib_db_path = mathlib_db_path
        self.mathlib_data = None
        
        # Enhanced training data structures
        self.theorem_embeddings = {}
        self.tactic_embeddings = {}
        self.difficulty_weights = {
            ProblemDifficulty.EASY: 1.0,
            ProblemDifficulty.MEDIUM: 1.5,
            ProblemDifficulty.HARD: 2.0
        }
        
        # Training metrics
        self.mathlib_stats = {
            'theorems_processed': 0,
            'tactics_learned': 0,
            'avg_statement_length': 0,
            'avg_proof_length': 0
        }
    
    def load_mathlib_data(self) -> None:
        """Load processed Mathlib data from database."""
        if not Path(self.mathlib_db_path).exists():
            logger.warning(f"Mathlib database not found: {self.mathlib_db_path}")
            return
        
        logger.info("Loading Mathlib data from database...")
        
        conn = sqlite3.connect(self.mathlib_db_path)
        cursor = conn.cursor()
        
        # Load theorems
        cursor.execute("""
            SELECT name, statement, proof, tactics_used, difficulty 
            FROM theorems 
            WHERE statement IS NOT NULL AND statement != ''
        """)
        
        theorems = []
        for name, statement, proof, tactics_json, difficulty in cursor.fetchall():
            try:
                tactics_used = json.loads(tactics_json) if tactics_json else []
                diff_enum = ProblemDifficulty(difficulty) if difficulty else ProblemDifficulty.MEDIUM
                
                theorem = LeanTheorem(
                    name=name,
                    statement=statement,
                    proof=proof or "",
                    tactics_used=tactics_used,
                    difficulty=diff_enum
                )
                theorems.append(theorem)
            except Exception as e:
                logger.warning(f"Error processing theorem {name}: {e}")
        
        # Load tactic usage patterns
        cursor.execute("SELECT tactic, context, theorem_name FROM tactic_usage")
        tactic_usage = []
        for tactic, context, theorem_name in cursor.fetchall():
            usage = TacticUsage(
                tactic=tactic,
                goal_before=context,
                theorem_name=theorem_name
            )
            tactic_usage.append(usage)
        
        conn.close()
        
        self.mathlib_data = {
            'theorems': theorems,
            'tactic_usage': tactic_usage
        }
        
        logger.info(f"Loaded {len(theorems)} theorems and {len(tactic_usage)} tactic usages")
    
    def prepare_mathlib_training_data(self) -> None:
        """Prepare training data from Mathlib theorems and proofs."""
        if not self.mathlib_data:
            self.load_mathlib_data()
        
        if not self.mathlib_data:
            logger.error("No Mathlib data available")
            return
        
        logger.info("Preparing Mathlib training data...")
        
        # Extract goal-tactic pairs from theorems
        for theorem in self.mathlib_data['theorems']:
            # Create pairs from theorem statement and tactics used
            if theorem.tactics_used:
                for tactic in theorem.tactics_used:
                    # Weight by difficulty
                    weight = self.difficulty_weights.get(theorem.difficulty, 1.0)
                    
                    # Add multiple instances for harder problems
                    for _ in range(int(weight)):
                        self.goal_tactic_pairs.append((theorem.statement, tactic))
        
        # Extract pairs from proof contexts
        for usage in self.mathlib_data['tactic_usage']:
            if usage.goal_before and usage.tactic:
                self.goal_tactic_pairs.append((usage.goal_before, usage.tactic))
        
        # Build enhanced vocabulary from all mathematical text
        all_text = []
        for theorem in self.mathlib_data['theorems']:
            all_text.append(theorem.statement)
            if theorem.proof:
                all_text.append(theorem.proof)
        
        self.tokenizer.build_vocab_from_corpus(all_text)
        
        # Create vocabulary mappings
        vocab_list = list(self.tokenizer.vocab)
        self.vocab_to_id = {token: i for i, token in enumerate(vocab_list)}
        self.id_to_vocab = {i: token for i, token in enumerate(vocab_list)}
        
        # Update statistics
        self.mathlib_stats['theorems_processed'] = len(self.mathlib_data['theorems'])
        self.mathlib_stats['tactics_learned'] = len(set(usage.tactic for usage in self.mathlib_data['tactic_usage']))
        
        if self.mathlib_data['theorems']:
            avg_stmt_len = np.mean([len(t.statement) for t in self.mathlib_data['theorems']])
            avg_proof_len = np.mean([len(t.proof or "") for t in self.mathlib_data['theorems']])
            self.mathlib_stats['avg_statement_length'] = avg_stmt_len
            self.mathlib_stats['avg_proof_length'] = avg_proof_len
        
        logger.info(f"Prepared {len(self.goal_tactic_pairs)} goal-tactic pairs")
        logger.info(f"Vocabulary size: {len(self.tokenizer.vocab)}")
        logger.info(f"Average statement length: {self.mathlib_stats['avg_statement_length']:.1f}")
        logger.info(f"Average proof length: {self.mathlib_stats['avg_proof_length']:.1f}")
    
    def create_enhanced_model(self) -> None:
        """Create enhanced Goal2Vec model with Mathlib-specific features."""
        vocab_size = len(self.tokenizer.vocab)
        
        # Create model with larger embedding dimension for mathematical complexity
        enhanced_config = EmbeddingConfig(
            embedding_dim=min(256, self.config.embedding_dim * 2),  # Larger embeddings for math
            window_size=self.config.window_size,
            min_count=max(1, self.config.min_count // 2),  # Lower threshold for math terms
            workers=self.config.workers,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            negative_samples=self.config.negative_samples,
            batch_size=self.config.batch_size
        )
        
        self.model = Goal2VecModel(vocab_size, enhanced_config.embedding_dim).to(self.device)
        self.optimizer = optim.AdamW(  # Use AdamW for better generalization
            self.model.parameters(), 
            lr=enhanced_config.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=enhanced_config.epochs,
            eta_min=enhanced_config.learning_rate * 0.1
        )
    
    def train_with_mathlib(self, epochs: Optional[int] = None) -> None:
        """Enhanced training with Mathlib data and curriculum learning."""
        if epochs is None:
            epochs = self.config.epochs
        
        # Prepare data if not already done
        if not self.goal_tactic_pairs:
            self.prepare_mathlib_training_data()
        
        if not self.model:
            self.create_enhanced_model()
        
        logger.info(f"Training enhanced Goal2Vec model on Mathlib data for {epochs} epochs...")
        
        # Curriculum learning: start with easier problems
        easy_pairs = []
        medium_pairs = []
        hard_pairs = []
        
        # Categorize training pairs by difficulty (based on source theorem)
        for goal, tactic in self.goal_tactic_pairs:
            # Simple heuristic: longer statements are harder
            if len(goal) < 50:
                easy_pairs.append((goal, tactic))
            elif len(goal) < 150:
                medium_pairs.append((goal, tactic))
            else:
                hard_pairs.append((goal, tactic))
        
        # Split data
        all_pairs = easy_pairs + medium_pairs + hard_pairs
        split_idx = int(0.8 * len(all_pairs))
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]
        
        criterion = nn.BCEWithLogitsLoss()
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training loop with curriculum learning
        for epoch in range(epochs):
            start_time = time.time()
            
            # Curriculum learning schedule
            if epoch < epochs // 3:
                # Start with easy problems
                epoch_pairs = easy_pairs + medium_pairs[:len(easy_pairs)//2]
            elif epoch < 2 * epochs // 3:
                # Add medium problems
                epoch_pairs = easy_pairs + medium_pairs + hard_pairs[:len(medium_pairs)//2]
            else:
                # Full dataset
                epoch_pairs = train_pairs
            
            # Training
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle training data
            np.random.shuffle(epoch_pairs)
            
            for i in range(0, len(epoch_pairs), self.config.batch_size):
                batch_pairs = epoch_pairs[i:i + self.config.batch_size]
                
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
            
            # Learning rate scheduling
            self.scheduler.step()
            
            avg_train_loss = epoch_loss / max(num_batches, 1)
            self.training_losses.append(avg_train_loss)
            
            # Validation
            val_loss = 0.0
            if val_pairs:
                val_loss = self._validate(val_pairs, criterion)
                self.validation_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self._save_checkpoint("best_mathlib_goal2vec.pth")
                else:
                    patience_counter += 1
            
            epoch_time = time.time() - start_time
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Logging
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch}/{epochs}: "
                           f"Train Loss={avg_train_loss:.4f}, "
                           f"Val Loss={val_loss:.4f}, "
                           f"LR={current_lr:.6f}, "
                           f"Time={epoch_time:.1f}s")
            
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Enhanced Goal2Vec training completed!")
        
        # Load best model
        if Path("best_mathlib_goal2vec.pth").exists():
            self._load_checkpoint("best_mathlib_goal2vec.pth")
            logger.info("Loaded best model from checkpoint")
    
    def _save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "vocab_to_id": self.vocab_to_id,
            "id_to_vocab": self.id_to_vocab,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "mathlib_stats": self.mathlib_stats
        }
        torch.save(checkpoint, filepath)
    
    def _load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    def recommend_tactics_with_confidence(self, goal: str, candidate_tactics: List[str] = None, 
                                        top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Enhanced tactic recommendation with confidence scores and explanations."""
        if candidate_tactics is None:
            # Use common Mathlib tactics
            candidate_tactics = [
                'simp', 'rw', 'ring', 'norm_num', 'linarith', 'omega', 'exact', 'apply',
                'intro', 'intros', 'cases', 'induction', 'constructor', 'tauto', 'decide',
                'assumption', 'rfl', 'unfold', 'abel', 'group', 'field_simp', 'norm_cast',
                'push_neg', 'by_contra', 'use', 'refine', 'obtain', 'have', 'ext', 'funext'
            ]
        
        goal_tokens = self.tokenizer.tokenize(goal)
        goal_ids = self._tokens_to_ids(goal_tokens)
        
        # Pad goal sequence
        max_len = 100  # Longer sequences for complex math
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
                confidence = torch.sigmoid(score).item()
                
                # Generate explanation based on tactic type
                explanation = self._generate_tactic_explanation(tactic, goal, confidence)
                
                tactic_scores.append((tactic, confidence, explanation))
        
        # Sort by confidence score
        tactic_scores.sort(key=lambda x: x[1], reverse=True)
        return tactic_scores[:top_k]
    
    def _generate_tactic_explanation(self, tactic: str, goal: str, confidence: float) -> str:
        """Generate explanation for tactic recommendation."""
        explanations = {
            'simp': "Simplification tactic for basic algebraic manipulations",
            'rw': "Rewrite using equalities and equivalences",
            'ring': "Solve equations in commutative rings",
            'norm_num': "Normalize numerical expressions",
            'linarith': "Linear arithmetic solver",
            'omega': "Integer linear arithmetic",
            'exact': "Provide exact proof term",
            'apply': "Apply theorem or hypothesis",
            'intro': "Introduce hypothesis or variable",
            'cases': "Case analysis on data type",
            'induction': "Proof by induction",
            'constructor': "Apply constructor of inductive type",
            'tauto': "Propositional tautology solver",
            'assumption': "Use available hypothesis"
        }
        
        base_explanation = explanations.get(tactic, f"Apply {tactic} tactic")
        
        if confidence > 0.8:
            return f"{base_explanation} (high confidence)"
        elif confidence > 0.6:
            return f"{base_explanation} (medium confidence)"
        else:
            return f"{base_explanation} (low confidence)"
    
    def evaluate_on_mathlib_test(self) -> Dict[str, float]:
        """Evaluate model performance on Mathlib test set."""
        if not self.mathlib_data:
            logger.error("No Mathlib data loaded")
            return {}
        
        # Create test set from unseen theorems
        test_theorems = self.mathlib_data['theorems'][-100:]  # Last 100 theorems
        
        correct_predictions = 0
        total_predictions = 0
        
        for theorem in test_theorems:
            if not theorem.tactics_used:
                continue
            
            # Predict tactics for theorem statement
            predicted_tactics = self.recommend_tactics_with_confidence(
                theorem.statement, top_k=len(theorem.tactics_used)
            )
            
            predicted_tactic_names = [t[0] for t in predicted_tactics]
            
            # Check if any predicted tactic matches actual tactics
            for actual_tactic in theorem.tactics_used:
                if actual_tactic in predicted_tactic_names:
                    correct_predictions += 1
                    break
            
            total_predictions += 1
        
        accuracy = correct_predictions / max(total_predictions, 1)
        
        metrics = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'test_theorems': len(test_theorems)
        }
        
        logger.info(f"Mathlib test evaluation: {accuracy:.3f} accuracy "
                   f"({correct_predictions}/{total_predictions})")
        
        return metrics
    
    def save_enhanced_model(self, filepath: str) -> None:
        """Save enhanced model with Mathlib-specific data."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            "config": self.config,
            "vocab_to_id": self.vocab_to_id,
            "id_to_vocab": self.id_to_vocab,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "mathlib_stats": self.mathlib_stats,
            "mathlib_db_path": self.mathlib_db_path
        }
        
        torch.save(checkpoint, filepath)
        
        # Save enhanced vocabulary
        vocab_path = str(Path(filepath).with_suffix('.vocab'))
        self.tokenizer.save_vocab(vocab_path)
        
        logger.info(f"Enhanced model saved to {filepath}")
    
    def get_mathlib_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        base_stats = {
            'training_pairs': len(self.goal_tactic_pairs),
            'vocabulary_size': len(self.tokenizer.vocab),
            'training_epochs': len(self.training_losses),
            'final_train_loss': self.training_losses[-1] if self.training_losses else None,
            'final_val_loss': self.validation_losses[-1] if self.validation_losses else None
        }
        
        base_stats.update(self.mathlib_stats)
        return base_stats


# Example usage and testing
if __name__ == "__main__":
    print("=== Enhanced Mathlib Goal2Vec Training ===")
    
    # Configuration
    config = EmbeddingConfig(
        embedding_dim=256,  # Larger for mathematical complexity
        epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Initialize enhanced trainer
    trainer = MathlibGoal2VecTrainer(config, "mathlib_training_data.db")
    
    # Load and prepare Mathlib data
    print("Loading Mathlib data...")
    trainer.load_mathlib_data()
    trainer.prepare_mathlib_training_data()
    
    # Train enhanced model
    print("Training enhanced Goal2Vec model...")
    trainer.train_with_mathlib(epochs=50)  # Reduced for demo
    
    # Evaluate model
    print("Evaluating model on Mathlib test set...")
    metrics = trainer.evaluate_on_mathlib_test()
    
    # Test enhanced tactic recommendations
    test_goals = [
        "∀ n : ℕ, n + 0 = n",
        "∀ a b : ℕ, a + b = b + a", 
        "∀ x : ℝ, x * 0 = 0",
        "∀ P Q : Prop, P ∧ Q → Q ∧ P"
    ]
    
    print("\nEnhanced tactic recommendations:")
    for goal in test_goals:
        recommendations = trainer.recommend_tactics_with_confidence(goal, top_k=3)
        print(f"\nGoal: {goal}")
        for i, (tactic, confidence, explanation) in enumerate(recommendations):
            print(f"  {i+1}. {tactic}: {confidence:.3f} - {explanation}")
    
    # Save enhanced model
    trainer.save_enhanced_model("enhanced_mathlib_goal2vec.pth")
    
    # Print statistics
    stats = trainer.get_mathlib_statistics()
    print(f"\nTraining Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("Enhanced Mathlib Goal2Vec training completed!")