#!/usr/bin/env python3
"""
Training script for Goal2Vec model.
Trains the Goal2Vec embeddings on sample mathematical problems.
"""

import sys
import json
from pathlib import Path
from goal2vec_model import Goal2VecTrainer, EmbeddingConfig, Word2VecGoalComparator
from minif2f_processor import MathProblem, ProblemDifficulty
from definition_database import DefinitionDatabase, MathDefinition, DefinitionType

def create_sample_problems():
    """Create sample mathematical problems for training."""
    problems = [
        # Basic arithmetic
        MathProblem("nat_add_zero", "∀ n : ℕ, n + 0 = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_zero_add", "∀ n : ℕ, 0 + n = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_add_comm", "∀ a b : ℕ, a + b = b + a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        MathProblem("nat_add_assoc", "∀ a b c : ℕ, (a + b) + c = a + (b + c)", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        
        # Multiplication
        MathProblem("nat_mul_zero", "∀ n : ℕ, n * 0 = 0", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_mul_one", "∀ n : ℕ, n * 1 = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_mul_comm", "∀ a b : ℕ, a * b = b * a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        
        # Real numbers
        MathProblem("real_add_zero", "∀ x : ℝ, x + 0 = x", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("real_mul_zero", "∀ x : ℝ, x * 0 = 0", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("real_add_comm", "∀ a b : ℝ, a + b = b + a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        
        # Inequalities
        MathProblem("nat_le_refl", "∀ n : ℕ, n ≤ n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_le_trans", "∀ a b c : ℕ, a ≤ b → b ≤ c → a ≤ c", proof="by intros; omega", difficulty=ProblemDifficulty.HARD),
        
        # Logic
        MathProblem("and_comm", "∀ P Q : Prop, P ∧ Q ↔ Q ∧ P", proof="by simp [and_comm]", difficulty=ProblemDifficulty.MEDIUM),
        MathProblem("or_comm", "∀ P Q : Prop, P ∨ Q ↔ Q ∨ P", proof="by simp [or_comm]", difficulty=ProblemDifficulty.MEDIUM),
        
        # Functions
        MathProblem("function_comp", "∀ f g : ℕ → ℕ, ∀ x, (f ∘ g) x = f (g x)", proof="by simp", difficulty=ProblemDifficulty.MEDIUM),
        
        # Sets (simplified)
        MathProblem("set_union_comm", "∀ A B : Set α, A ∪ B = B ∪ A", proof="by simp [Set.union_comm]", difficulty=ProblemDifficulty.MEDIUM),
        
        # Example problem from the request
        MathProblem("mathd_algebra_10", "abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10", 
                   proof="by norm_num", difficulty=ProblemDifficulty.MEDIUM),
        
        # Tree theorem (generic version, not MyTree specific)
        MathProblem("tree_vertex_edge", "∀ t : Tree, vertices t = edges t + 1", 
                   proof="by induction t; simp [vertices, edges]; ring", difficulty=ProblemDifficulty.HARD),
    ]
    return problems

def create_sample_definitions():
    """Create sample mathematical definitions."""
    definitions = [
        MathDefinition("Natural", DefinitionType.INDUCTIVE, 
                      "inductive Nat where | zero : Nat | succ : Nat → Nat",
                      "Natural numbers starting from zero", "number_theory"),
        MathDefinition("Real", DefinitionType.STRUCTURE,
                      "structure Real extends ...",
                      "Real number system", "analysis"),
        MathDefinition("List", DefinitionType.INDUCTIVE,
                      "inductive List (α : Type) where | nil : List α | cons : α → List α → List α",
                      "Linked list data structure", "data_structures"),
        MathDefinition("Tree", DefinitionType.INDUCTIVE,
                      "inductive Tree where | leaf : Tree | branch : List Tree → Tree",
                      "Generic tree data structure", "data_structures"),
        MathDefinition("Set", DefinitionType.FUNCTION,
                      "def Set (α : Type) : Type := α → Prop",
                      "Mathematical sets", "set_theory"),
    ]
    return definitions

def main():
    """Main training function."""
    print("=== Goal2Vec Training Script ===\n")
    
    # Configuration
    config = EmbeddingConfig(
        embedding_dim=128,
        epochs=50,  # Reduced for demo
        batch_size=32,
        learning_rate=0.001
    )
    
    print("1. Creating training data...")
    problems = create_sample_problems()
    definitions = create_sample_definitions()
    
    print(f"   Created {len(problems)} problems")
    print(f"   Created {len(definitions)} definitions")
    
    # Initialize trainer
    print("2. Initializing Goal2Vec trainer...")
    trainer = Goal2VecTrainer(config)
    
    # Add definitions to database
    print("3. Setting up definition database...")
    db = DefinitionDatabase("training_definitions.db")
    for definition in definitions:
        db.add_definition(definition)
    
    trainer.definition_db = db
    
    # Prepare training data
    print("4. Preparing training data...")
    trainer.prepare_training_data(problems)
    
    print(f"   Vocabulary size: {len(trainer.tokenizer.vocab)}")
    print(f"   Training pairs: {len(trainer.training_pairs)}")
    
    # Train the model
    print("5. Training Goal2Vec model...")
    print("   This may take a few minutes...")
    
    try:
        trainer.train()
        print("   ✓ Goal2Vec training completed successfully!")
        
        # Save the model
        model_path = "goal2vec_model.pth"
        trainer.save_model(model_path)
        print(f"   ✓ Model saved to {model_path}")
        
        # Test the model
        print("6. Testing trained model...")
        
        # Test similarity
        test_goals = [
            "∀ n : ℕ, n + 0 = n",
            "∀ a b : ℕ, a + b = b + a",
            "∀ x : ℝ, x + 0 = x"
        ]
        
        for goal in test_goals:
            try:
                similar = trainer.find_similar_goals(goal, top_k=3)
                print(f"   Similar to '{goal}':")
                for i, (sim_goal, score) in enumerate(similar):
                    print(f"     {i+1}. {score:.3f}: {sim_goal}")
            except Exception as e:
                print(f"   Error finding similar goals: {e}")
        
        # Test tactic recommendations
        print("7. Testing tactic recommendations...")
        for goal in test_goals[:2]:  # Test first two goals
            try:
                tactics = trainer.recommend_tactics(goal, top_k=3)
                print(f"   Tactics for '{goal}':")
                for i, (tactic, score) in enumerate(tactics):
                    print(f"     {i+1}. {tactic}: {score:.3f}")
            except Exception as e:
                print(f"   Error getting tactic recommendations: {e}")
        
        print("\n8. Comparing with Word2Vec baseline...")
        try:
            # Create Word2Vec comparator
            comparator = Word2VecGoalComparator()
            comparator.prepare_training_data(problems)
            comparator.train()
            
            # Compare performance on a test goal
            test_goal = "∀ n : ℕ, n + 0 = n"
            goal2vec_similar = trainer.find_similar_goals(test_goal, top_k=3)
            word2vec_similar = comparator.find_similar_goals(test_goal, top_k=3)
            
            print(f"   Goal2Vec results for '{test_goal}':")
            for goal, score in goal2vec_similar:
                print(f"     {score:.3f}: {goal}")
            
            print(f"   Word2Vec results for '{test_goal}':")
            for goal, score in word2vec_similar:
                print(f"     {score:.3f}: {goal}")
                
        except Exception as e:
            print(f"   Word2Vec comparison failed: {e}")
        
        # Training statistics
        stats = trainer.get_training_stats()
        print(f"\n9. Training Statistics:")
        print(f"   Final loss: {stats.get('final_loss', 'N/A'):.6f}")
        print(f"   Training time: {stats.get('training_time', 'N/A'):.2f}s")
        print(f"   Epochs completed: {stats.get('epochs_completed', 'N/A')}")
        
        print("\n=== Goal2Vec Training Completed Successfully! ===")
        return True
        
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)