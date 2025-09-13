"""
Main entry point for the ATPLean integrated theorem prover system.
Demonstrates basic usage of the integrated system components.
"""

from integrated_theorem_prover import IntegratedTheoremProver
from minif2f_processor import MathProblem, ProblemDifficulty

def main():
    """Demonstrate the integrated theorem prover system."""
    
    print("=== ATPLean Integrated Theorem Prover Demo ===\n")
    
    # Initialize the system
    print("1. Initializing integrated theorem prover...")
    prover = IntegratedTheoremProver()
    
    # Create sample problems
    print("2. Loading sample problems...")
    sample_problems = [
        MathProblem(
            "sample_1", 
            "∀ n : ℕ, n + 0 = n", 
            proof="by simp",
            difficulty=ProblemDifficulty.EASY
        ),
        MathProblem(
            "sample_2",
            "∀ a b : ℕ, a + b = b + a", 
            proof="by ring",
            difficulty=ProblemDifficulty.MEDIUM
        ),
        MathProblem(
            "mathd_algebra_10",
            "abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10",
            proof="sorry",
            difficulty=ProblemDifficulty.MEDIUM
        )
    ]
    
    prover.problems = sample_problems
    print(f"   Loaded {len(sample_problems)} sample problems")
    
    # Create RL environments
    print("3. Creating RL environments...")
    prover.create_environments()
    print(f"   Created {len(prover.environments)} environments")
    
    # Demonstrate system components
    print("4. Testing system components...")
    
    # Test Goal2Vec training preparation
    print("   - Preparing Goal2Vec training data...")
    prover.goal2vec_trainer.prepare_training_data(sample_problems)
    
    # Test similar theorem search
    print("   - Testing similar theorem search...")
    try:
        similar = prover.find_similar_theorems("∀ n : ℕ, n + 0 = n", top_k=2)
        print(f"     Found {len(similar)} similar theorems")
        for i, (problem, similarity) in enumerate(similar):
            print(f"     {i+1}. {similarity:.3f}: {problem.statement}")
    except Exception as e:
        print(f"     Similar theorem search not available: {e}")
    
    # Test tactic recommendations
    print("   - Testing tactic recommendations...")
    try:
        recommendations = prover.get_tactic_recommendations("∀ n : ℕ, n + 0 = n", top_k=3)
        print(f"     Generated {len(recommendations)} tactic recommendations:")
        for i, (tactic, score) in enumerate(recommendations):
            print(f"     {i+1}. {tactic}: {score:.3f}")
    except Exception as e:
        print(f"     Tactic recommendations not available: {e}")
    
    # Test proof attempt
    print("5. Attempting theorem proof...")
    try:
        result = prover.prove_theorem(
            "∀ n : ℕ, n + 0 = n", 
            max_steps=10, 
            use_goal2vec=False  # Skip Goal2Vec for demo
        )
        print(f"   Proof attempt completed:")
        print(f"   - Success: {result['success']}")
        print(f"   - Steps: {result['proof_length']}")
        print(f"   - Reward: {result['total_reward']}")
        
        if result['success']:
            print(f"   - Proof tree: {result['proof_tree']}")
    except Exception as e:
        print(f"   Proof attempt failed: {e}")
    
    print("\n6. System evaluation...")
    try:
        evaluation = prover.evaluate_system(num_problems=len(sample_problems))
        print(f"   Success rate: {evaluation['success_rate']:.2%}")
        print(f"   Average proof length: {evaluation['avg_proof_length']:.1f}")
        print(f"   Average reward: {evaluation['avg_reward']:.2f}")
    except Exception as e:
        print(f"   System evaluation failed: {e}")
    
    print("\n=== Demo completed ===")
    print("For external API integration, see openrouter_api_client.py")

if __name__ == "__main__":
    main()
