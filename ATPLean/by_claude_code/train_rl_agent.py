#!/usr/bin/env python3
"""
Training script for RL agent components.
"""

import sys
from rl_agent import ProofAgent, ProofTrainer, AgentType, TrainingConfig
from reinforcement_learning_node import RLProofEnvironment, RLStateNode, TacticAction
from minif2f_processor import MathProblem, ProblemDifficulty

def create_sample_environments():
    """Create sample RL environments for training."""
    problems = [
        MathProblem("nat_add_zero", "∀ n : ℕ, n + 0 = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_zero_add", "∀ n : ℕ, 0 + n = n", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("nat_add_comm", "∀ a b : ℕ, a + b = b + a", proof="by ring", difficulty=ProblemDifficulty.MEDIUM),
        MathProblem("real_add_zero", "∀ x : ℝ, x + 0 = x", proof="by simp", difficulty=ProblemDifficulty.EASY),
        MathProblem("real_mul_zero", "∀ x : ℝ, x * 0 = 0", proof="by simp", difficulty=ProblemDifficulty.EASY),
    ]
    
    environments = []
    for problem in problems:
        env = RLProofEnvironment(problem.statement)
        environments.append(env)
    
    return environments

def main():
    """Main RL training function."""
    print("=== RL Agent Training ===\n")
    
    # Training configuration
    config = TrainingConfig(
        learning_rate=0.001,
        max_episodes=50,      # Reduced for demo
        max_steps_per_episode=20,
        batch_size=16,
        memory_size=1000
    )
    
    print("1. Creating training environments...")
    environments = create_sample_environments()
    print(f"   Created {len(environments)} environments")
    
    # Initialize agent and trainer
    print("2. Initializing RL agent and trainer...")
    
    # Get action and state sizes from first environment
    sample_env = environments[0]
    
    # Get initial state to understand state structure
    initial_state = sample_env.reset()
    state_features = sample_env.current_node.get_state_features()
    
    # Estimate sizes (simplified)
    state_size = len(str(initial_state.get('observation', '')))  # Use string length as proxy
    action_size = len(sample_env.get_available_actions())
    
    print(f"   State size (estimated): {state_size}")
    print(f"   Action size: {action_size}")
    
    try:
        agent = ProofAgent(AgentType.DQN, config)
        trainer = ProofTrainer(agent, config)
        trainer.environments = environments  # Set environments
        
        print("3. Starting RL training...")
        print("   This may take a few minutes...")
        
        # Train agent
        trainer.train(config.max_episodes)
        
        print("   ✓ RL training completed!")
        
        # Show training results
        print("4. Training Results:")
        try:
            stats = trainer.get_training_stats()
            print(f"   Episodes completed: {config.max_episodes}")
            print(f"   Training stats: {stats}")
        except Exception as e:
            print(f"   Could not retrieve training stats: {e}")
        
        # Save the trained agent
        model_path = "trained_rl_agent.pth"
        agent.save_model(model_path)
        print(f"   ✓ Model saved to {model_path}")
        
        # Test the agent on a simple problem
        print("5. Testing trained agent...")
        test_env = environments[0]  # Use first environment for testing
        
        try:
            # Reset environment and get initial state
            initial_state = test_env.reset()
            current_state = test_env.current_node
            
            print(f"   Testing on: {test_env.initial_theorem}")
            print(f"   Initial state: {current_state.goal}")
            
            # Run a few test steps
            for step in range(5):
                action = agent.select_action(current_state, training=False)
                next_state, reward, done = test_env.step(action)
                
                print(f"   Step {step + 1}: Action {action}, Reward {reward:.3f}")
                if done:
                    if test_env.is_solved():
                        print("   ✓ Problem solved!")
                    else:
                        print("   Problem not solved yet.")
                    break
                    
                current_state = next_state
                
        except Exception as e:
            print(f"   Testing failed: {e}")
        
        print("\n=== RL Agent Training Completed Successfully! ===")
        return True
        
    except Exception as e:
        print(f"   ✗ RL training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)