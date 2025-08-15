#!/usr/bin/env python3
"""
Simplified Goal2Vec training script without database complications.
"""

import sys
from goal2vec_model import Goal2VecTrainer, EmbeddingConfig
from minif2f_processor import MathProblem, ProblemDifficulty

def create_sample_problems():
    """Create sample mathematical problems for training."""
    problems = [
        # Basic arithmetic and algebra
        MathProblem("nat_add_zero", "∀ n : ℕ, n + 0 = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_zero_add", "∀ n : ℕ, 0 + n = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_add_comm", "∀ a b : ℕ, a + b = b + a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        MathProblem("nat_add_assoc", "∀ a b c : ℕ, (a + b) + c = a + (b + c)", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        
        MathProblem("nat_mul_zero", "∀ n : ℕ, n * 0 = 0", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_mul_one", "∀ n : ℕ, n * 1 = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_mul_comm", "∀ a b : ℕ, a * b = b * a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        
        # Real numbers
        MathProblem("real_add_zero", "∀ x : ℝ, x + 0 = x", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("real_mul_zero", "∀ x : ℝ, x * 0 = 0", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("real_add_comm", "∀ a b : ℝ, a + b = b + a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        
        # Logic
        MathProblem("and_comm", "∀ P Q : Prop, P ∧ Q ↔ Q ∧ P", proof="by simp [and_comm]", difficulty=ProblemDifficulty.MEDIUM),
        MathProblem("or_comm", "∀ P Q : Prop, P ∨ Q ↔ Q ∨ P", proof="by simp [or_comm]", difficulty=ProblemDifficulty.MEDIUM),
        
        # Example from request
        MathProblem("mathd_algebra_10", "abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10", 
                   proof="by norm_num", difficulty=ProblemDifficulty.MEDIUM),
        
        # More varied examples
        MathProblem("list_length", "∀ l : List α, length (l ++ []) = length l", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("function_comp", "∀ f g : ℕ → ℕ, ∀ x, (f ∘ g) x = f (g x)", proof="by simp", difficulty=ProblemDifficulty.MEDIUM),
    ]
    return problems

def main():
    """Main training function."""
    print("=== Simplified Goal2Vec Training ===\n")
    
    # Configuration for quick training
    config = EmbeddingConfig(
        embedding_dim=64,  # Smaller for demo
        epochs=20,         # Fewer epochs for demo
        batch_size=16,
        learning_rate=0.01,
        min_count=1        # Include all tokens
    )
    
    print("1. Creating training data...")
    problems = create_sample_problems()
    print(f"   Created {len(problems)} problems")
    
    # Initialize trainer without database
    print("2. Initializing Goal2Vec trainer...")
    trainer = Goal2VecTrainer(config)
    trainer.definition_db = None  # Skip database for simplicity
    
    # Prepare training data
    print("3. Preparing training data...")
    try:
        trainer.prepare_training_data(problems)
        print(f"   Vocabulary size: {len(trainer.tokenizer.vocab)}")
        print(f"   Training pairs: {len(trainer.training_pairs) if hasattr(trainer, 'training_pairs') else 'Generated'}")
    except Exception as e:
        print(f"   Error preparing data: {e}")
        return False
    
    # Train the model
    print("4. Training Goal2Vec model...")
    print("   Training in progress...")
    
    try:
        trainer.train()
        print("   ✓ Goal2Vec training completed successfully!")
        
        # Save the model
        model_path = "simple_goal2vec_model.pth"
        trainer.save_model(model_path)
        print(f"   ✓ Model saved to {model_path}")
        
        # Test the model with simple queries
        print("5. Testing trained model...")
        
        test_goals = [
            "∀ n : ℕ, n + 0 = n",
            "∀ a b : ℕ, a + b = b + a",
        ]
        
        for goal in test_goals:
            try:
                # Test similarity search
                similar = trainer.find_similar_goals_from_training(goal, top_k=3)
                print(f"   Similar to '{goal}':")
                for i, (sim_goal, score) in enumerate(similar):
                    print(f"     {i+1}. {score:.3f}: {sim_goal}")
                
                # Test tactic recommendations
                tactics = trainer.recommend_tactics_from_training(goal, top_k=2)
                print(f"   Recommended tactics:")
                for tactic, score in tactics:
                    print(f"     {tactic}: {score:.3f}")
                    
            except Exception as e:
                print(f"   Error testing goal '{goal}': {e}")
        
        # 새로운 Word2Vec 스타일 아날로지 테스트
        print("6. Testing Word2Vec style analogy features...")
        
        try:
            # 테스트 케이스: 자연수 → 실수 아날로지
            print("\n   테스트: '자연수 + 0' → '실수 + 0' 아날로지")
            print("   질문: ∀ n : ℕ, n + 0 = n (simp) : ∀ x : ℝ, x + 0 = x (?)")
            
            analogy_results = trainer.solve_analogy(
                "∀ n : ℕ, n + 0 = n", "simp", "∀ x : ℝ, x + 0 = x", top_k=3
            )
            
            print("   아날로지 결과:")
            for i, (tactic, score) in enumerate(analogy_results):
                print(f"     {i+1}. {tactic}: {score:.3f}")
                
        except Exception as e:
            print(f"   아날로지 테스트 에러: {e}")
        
        try:    
            # 여러 방법 비교
            print("\n   다양한 아날로지 방법 비교:")
            comparison = trainer.compare_analogy_methods(
                "∀ n : ℕ, n + 0 = n", "simp", "∀ a b : ℕ, a + b = b + a", top_k=2
            )
            
            for method, results in comparison.items():
                print(f"   {method}:")
                for tactic, score in results:
                    print(f"     - {tactic}: {score:.3f}")
                    
        except Exception as e:
            print(f"   방법 비교 에러: {e}")
        
        # Show training statistics
        try:
            stats = trainer.get_training_stats()
            print(f"\n7. Training Statistics:")
            print(f"   Final loss: {stats.get('final_loss', 'N/A')}")
            print(f"   Training time: {stats.get('training_time', 'N/A')}s")
            print(f"   Epochs completed: {stats.get('epochs_completed', 'N/A')}")
        except Exception as e:
            print(f"   Could not retrieve training stats: {e}")
        
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