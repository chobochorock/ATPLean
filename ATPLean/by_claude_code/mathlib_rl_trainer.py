#!/usr/bin/env python3
"""
Mathlib RL Trainer: Enhanced reinforcement learning for theorem proving with Mathlib data.

This module integrates the RL proof search system with comprehensive Mathlib data,
providing better theorem proving capabilities through enhanced training environments.
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
from collections import defaultdict, deque
import time
import random

from reinforcement_learning_node import RLStateNode, TacticAction, RLProofEnvironment
from rl_agent import (
    ProofAgent,
    ProofTrainer,
    TrainingConfig,
)  # AgentConfig에서 TrainingConfig로 전환
from mathlib_data_loader import MathlibDataLoader, LeanTheorem, TacticUsage
from mathlib_goal2vec_trainer import MathlibGoal2VecTrainer, EmbeddingConfig
from minif2f_processor import MathProblem, ProblemDifficulty


logger = logging.getLogger(__name__)


class MathlibProofEnvironment(RLProofEnvironment):
    """Enhanced proof environment with Mathlib-derived knowledge."""

    def __init__(
        self,
        initial_goal: str,
        goal2vec_trainer: Optional[MathlibGoal2VecTrainer] = None,
    ):
        super().__init__(initial_goal)
        self.goal2vec_trainer = goal2vec_trainer
        self.mathlib_tactics = self._load_mathlib_tactics()
        self.tactic_success_history = defaultdict(lambda: {"success": 0, "total": 0})

        # Enhanced action space with Mathlib tactics
        self.action_space = self.mathlib_tactics

        # Advanced metrics
        self.theorem_complexity = self._estimate_theorem_complexity(initial_goal)
        self.proof_depth_limit = max(20, int(self.theorem_complexity * 10))

    def _load_mathlib_tactics(self) -> List[str]:
        """Load comprehensive Mathlib tactic set."""
        core_tactics = [
            "simp",
            "rw",
            "ring",
            "norm_num",
            "linarith",
            "omega",
            "exact",
            "apply",
            "intro",
            "intros",
            "cases",
            "induction",
            "constructor",
            "tauto",
            "decide",
            "assumption",
            "rfl",
            "unfold",
            "fold",
            "conv",
            "abel",
            "group",
            "field_simp",
            "norm_cast",
            "push_neg",
            "by_contra",
            "contrapose",
            "use",
            "refine",
            "obtain",
            "have",
            "suffices",
            "wlog",
            "rcases",
            "rintro",
            "ext",
            "funext",
            "congruence",
        ]

        advanced_tactics = [
            "library_search",
            "aesop",
            "polyrith",
            "linear_combination",
            "nlinarith",
            "positivity",
            "continuity",
            "differentiability",
            "measurability",
            "mono",
            "gcongr",
            "abel_rw",
            "compute_deriv",
            "field_simp",
            "ring_nf",
            "simp_rw",
            "simp_all",
            "tauto!",
            "decide!",
            "norm_fin",
            "interval_cases",
            "fin_cases",
            "mod_cases",
            "lift",
            "squeeze_simp",
            "abel!",
            "group!",
            "ring!",
        ]

        return core_tactics + advanced_tactics

    def _estimate_theorem_complexity(self, goal: str) -> float:
        """Estimate theorem complexity based on syntactic features."""
        complexity = 0.5  # Base complexity

        # Quantifier complexity
        complexity += goal.count("∀") * 0.2
        complexity += goal.count("∃") * 0.3

        # Logical connectives
        complexity += goal.count("→") * 0.1
        complexity += goal.count("↔") * 0.2
        complexity += goal.count("∧") * 0.1
        complexity += goal.count("∨") * 0.1

        # Mathematical structures
        if any(sym in goal for sym in ["ℕ", "ℤ", "ℚ", "ℝ", "ℂ"]):
            complexity += 0.3

        # Length factor
        complexity += len(goal) / 200.0

        return min(complexity, 3.0)  # Cap at 3.0

    def get_recommended_tactics(
        self, current_goal: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get Goal2Vec-recommended tactics for current goal."""
        if not self.goal2vec_trainer:
            # Fallback to heuristic recommendations
            return self._heuristic_tactic_recommendation(current_goal, top_k)

        try:
            recommendations = self.goal2vec_trainer.recommend_tactics_with_confidence(
                current_goal, candidate_tactics=self.mathlib_tactics, top_k=top_k
            )
            return [(tactic, confidence) for tactic, confidence, _ in recommendations]
        except Exception as e:
            logger.warning(f"Error getting Goal2Vec recommendations: {e}")
            return self._heuristic_tactic_recommendation(current_goal, top_k)

    def _heuristic_tactic_recommendation(
        self, goal: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Heuristic tactic recommendation based on goal patterns."""
        recommendations = []

        # Pattern-based recommendations
        if any(op in goal for op in ["+", "-", "*", "/", "^"]):
            recommendations.extend([("ring", 0.8), ("norm_num", 0.7), ("simp", 0.6)])

        if any(sym in goal for sym in ["=", "≤", "≥", "<", ">"]):
            recommendations.extend([("linarith", 0.7), ("omega", 0.6)])

        if "∀" in goal:
            recommendations.extend([("intro", 0.9), ("intros", 0.8)])

        if "∃" in goal:
            recommendations.extend([("use", 0.8), ("refine", 0.7)])

        if any(
            word in goal.lower()
            for word in ["continuous", "differentiable", "measurable"]
        ):
            recommendations.extend([("continuity", 0.8), ("differentiability", 0.7)])

        # Add some general tactics
        recommendations.extend([("exact", 0.5), ("apply", 0.5), ("assumption", 0.4)])

        # Remove duplicates and sort by confidence
        unique_recommendations = {}
        for tactic, conf in recommendations:
            if (
                tactic not in unique_recommendations
                or unique_recommendations[tactic] < conf
            ):
                unique_recommendations[tactic] = conf

        sorted_recommendations = sorted(
            unique_recommendations.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_recommendations[:top_k]

    def step(self, action: TacticAction) -> Tuple[RLStateNode, float, bool, Dict]:
        """Enhanced step function with Mathlib-aware rewards."""
        next_state, reward, done, info = super().step(action)

        # Update tactic success history
        tactic_name = action.tactic
        self.tactic_success_history[tactic_name]["total"] += 1
        if reward > 0:
            self.tactic_success_history[tactic_name]["success"] += 1

        # Enhanced reward calculation
        enhanced_reward = self._calculate_enhanced_reward(
            action, next_state, reward, done
        )

        # Add Mathlib-specific information
        info["mathlib_tactics_used"] = len(
            set(
                node.tactic_action.tactic
                for node in self._get_proof_path()
                if node.tactic_action
            )
        )
        info["theorem_complexity"] = self.theorem_complexity
        info["tactic_success_rate"] = self._get_tactic_success_rate(tactic_name)

        return next_state, enhanced_reward, done, info

    def _calculate_enhanced_reward(
        self,
        action: TacticAction,
        next_state: RLStateNode,
        base_reward: float,
        done: bool,
    ) -> float:
        """Calculate enhanced reward based on Mathlib patterns."""
        reward = base_reward

        # Bonus for using appropriate tactics
        if self.goal2vec_trainer:
            recommendations = self.get_recommended_tactics(
                self.current_state.goal, top_k=10
            )
            tactic_scores = {tactic: score for tactic, score in recommendations}

            if action.tactic in tactic_scores:
                # Bonus for using recommended tactics
                recommendation_bonus = tactic_scores[action.tactic] * 0.2
                reward += recommendation_bonus

        # Penalty for repeated failed tactics
        success_rate = self._get_tactic_success_rate(action.tactic)
        if (
            success_rate < 0.3
            and self.tactic_success_history[action.tactic]["total"] > 5
        ):
            reward -= 0.1  # Penalty for persisting with low-success tactics

        # Bonus for proof completion with reasonable depth
        if done and base_reward > 0:
            proof_depth = len(self._get_proof_path())
            if proof_depth <= self.proof_depth_limit:
                # Bonus for concise proofs
                efficiency_bonus = 0.5 * (1.0 - proof_depth / self.proof_depth_limit)
                reward += efficiency_bonus

        # Penalty for excessive depth
        current_depth = len(self._get_proof_path())
        if current_depth > self.proof_depth_limit:
            reward -= 0.2

        return reward

    def _get_tactic_success_rate(self, tactic: str) -> float:
        """Get success rate for a specific tactic."""
        history = self.tactic_success_history[tactic]
        if history["total"] == 0:
            return 0.5  # Neutral for untried tactics
        return history["success"] / history["total"]

    def _get_proof_path(self) -> List[RLStateNode]:
        """Get the current proof path from root to current state."""
        path = []
        current = self.current_state
        while current.parent:
            path.append(current)
            current = current.parent
        path.append(current)  # Add root
        return list(reversed(path))


class MathlibProofAgent(ProofAgent):
    """Enhanced proof agent with Mathlib knowledge integration."""

    def __init__(self, state_dim: int, action_dim: int, config: TrainingConfig):
        super().__init__(state_dim, action_dim, config)
        self.mathlib_tactic_embeddings = {}
        self.experience_buffer = deque(
            maxlen=100000
        )  # Larger buffer for complex proofs

    def set_mathlib_embeddings(self, goal2vec_trainer: MathlibGoal2VecTrainer):
        """Set Mathlib tactic embeddings from Goal2Vec trainer."""
        if not goal2vec_trainer or not goal2vec_trainer.model:
            return

        # Extract tactic embeddings
        tactics = [
            "simp",
            "rw",
            "ring",
            "norm_num",
            "linarith",
            "omega",
            "exact",
            "apply",
            "intro",
            "intros",
            "cases",
            "induction",
            "constructor",
            "tauto",
            "decide",
        ]

        for tactic in tactics:
            try:
                embedding = goal2vec_trainer.get_goal_embedding(tactic)
                self.mathlib_tactic_embeddings[tactic] = embedding
            except Exception as e:
                logger.warning(f"Error getting embedding for tactic {tactic}: {e}")

    def get_enhanced_state_representation(
        self, env: MathlibProofEnvironment
    ) -> np.ndarray:
        """Get enhanced state representation including Mathlib features."""
        base_state = self.get_state_representation(env)

        # Add Mathlib-specific features
        mathlib_features = []

        # Theorem complexity
        mathlib_features.append(env.theorem_complexity)

        # Proof depth ratio
        current_depth = len(env._get_proof_path())
        depth_ratio = current_depth / env.proof_depth_limit
        mathlib_features.append(depth_ratio)

        # Tactic success rates (for last few tactics)
        recent_tactics = [
            node.tactic_action.tactic
            for node in env._get_proof_path()[-3:]
            if node.tactic_action
        ]
        avg_success_rate = (
            np.mean([env._get_tactic_success_rate(tactic) for tactic in recent_tactics])
            if recent_tactics
            else 0.5
        )
        mathlib_features.append(avg_success_rate)

        # Goal2Vec recommendations alignment
        if env.goal2vec_trainer:
            recommendations = env.get_recommended_tactics(
                env.current_state.goal, top_k=5
            )
            avg_confidence = (
                np.mean([conf for _, conf in recommendations])
                if recommendations
                else 0.5
            )
            mathlib_features.append(avg_confidence)
        else:
            mathlib_features.append(0.5)

        # Combine features
        enhanced_state = np.concatenate([base_state, np.array(mathlib_features)])
        return enhanced_state


class MathlibProofTrainer(ProofTrainer):
    """Enhanced proof trainer with Mathlib integration."""

    def __init__(
        self,
        config: TrainingConfig,
        goal2vec_trainer: Optional[MathlibGoal2VecTrainer] = None,
    ):
        super().__init__(config)
        self.goal2vec_trainer = goal2vec_trainer
        self.mathlib_training_problems = []
        self.training_metrics = {
            "proofs_completed": 0,
            "average_proof_length": 0,
            "tactic_usage_stats": defaultdict(int),
            "difficulty_success_rates": defaultdict(lambda: {"success": 0, "total": 0}),
        }

    def load_mathlib_problems(
        self, mathlib_db_path: str, max_problems: int = 1000
    ) -> None:
        """Load training problems from Mathlib database."""
        if not Path(mathlib_db_path).exists():
            logger.warning(f"Mathlib database not found: {mathlib_db_path}")
            return

        conn = sqlite3.connect(mathlib_db_path)
        cursor = conn.cursor()

        # Load theorems with proofs
        cursor.execute(
            """
            SELECT name, statement, proof, difficulty 
            FROM theorems 
            WHERE statement IS NOT NULL AND statement != '' 
                AND proof IS NOT NULL AND proof != ''
            ORDER BY RANDOM()
            LIMIT ?
        """,
            (max_problems,),
        )

        for name, statement, proof, difficulty in cursor.fetchall():
            try:
                diff_enum = (
                    ProblemDifficulty(difficulty)
                    if difficulty
                    else ProblemDifficulty.MEDIUM
                )
                problem = MathProblem(
                    problem_id=name,
                    statement=statement,
                    proof=proof,
                    difficulty=diff_enum,
                )
                self.mathlib_training_problems.append(problem)
            except Exception as e:
                logger.warning(f"Error loading problem {name}: {e}")

        conn.close()
        logger.info(
            f"Loaded {len(self.mathlib_training_problems)} Mathlib training problems"
        )

    def train_on_mathlib(
        self, episodes: int = 1000, curriculum_learning: bool = True
    ) -> Dict[str, Any]:
        """Train RL agent on Mathlib problems with curriculum learning."""
        if not self.mathlib_training_problems:
            logger.error("No Mathlib problems loaded")
            return {}

        logger.info(
            f"Training RL agent on {len(self.mathlib_training_problems)} Mathlib problems"
        )

        # Sort problems by difficulty for curriculum learning
        if curriculum_learning:
            easy_problems = [
                p
                for p in self.mathlib_training_problems
                if p.difficulty == ProblemDifficulty.EASY
            ]
            medium_problems = [
                p
                for p in self.mathlib_training_problems
                if p.difficulty == ProblemDifficulty.MEDIUM
            ]
            hard_problems = [
                p
                for p in self.mathlib_training_problems
                if p.difficulty == ProblemDifficulty.HARD
            ]

            logger.info(
                f"Problem distribution: {len(easy_problems)} easy, {len(medium_problems)} medium, {len(hard_problems)} hard"
            )
        else:
            all_problems = self.mathlib_training_problems.copy()

        training_start_time = time.time()

        for episode in range(episodes):
            # Curriculum learning schedule
            if curriculum_learning:
                if episode < episodes // 3:
                    # Start with easy problems
                    problem_pool = (
                        easy_problems + medium_problems[: len(easy_problems) // 2]
                    )
                elif episode < 2 * episodes // 3:
                    # Add medium problems
                    problem_pool = (
                        easy_problems
                        + medium_problems
                        + hard_problems[: len(medium_problems) // 2]
                    )
                else:
                    # Full problem set
                    problem_pool = self.mathlib_training_problems
            else:
                problem_pool = all_problems

            if not problem_pool:
                continue

            # Select random problem
            problem = random.choice(problem_pool)

            # Create enhanced environment
            env = MathlibProofEnvironment(problem.statement, self.goal2vec_trainer)

            # Create enhanced agent if needed
            if not hasattr(self, "enhanced_agent"):
                # Determine state and action dimensions
                sample_state = env.get_state_representation()
                state_dim = len(sample_state) + 4  # +4 for Mathlib features
                action_dim = len(env.action_space)

                self.enhanced_agent = MathlibProofAgent(
                    state_dim, action_dim, self.config
                )
                if self.goal2vec_trainer:
                    self.enhanced_agent.set_mathlib_embeddings(self.goal2vec_trainer)

            # Run episode
            self._run_enhanced_episode(env, episode, episodes)

            # Update metrics
            if episode % 100 == 0:
                self._log_training_progress(
                    episode, episodes, time.time() - training_start_time
                )

        training_time = time.time() - training_start_time

        # Final metrics
        final_metrics = {
            "training_episodes": episodes,
            "training_time": training_time,
            "problems_used": len(self.mathlib_training_problems),
            "proofs_completed": self.training_metrics["proofs_completed"],
            "success_rate": self.training_metrics["proofs_completed"] / episodes,
            "average_proof_length": self.training_metrics["average_proof_length"],
            "difficulty_success_rates": dict(
                self.training_metrics["difficulty_success_rates"]
            ),
        }

        logger.info(f"Mathlib RL training completed: {final_metrics}")
        return final_metrics

    def _run_enhanced_episode(
        self, env: MathlibProofEnvironment, episode: int, total_episodes: int
    ) -> None:
        """Run single training episode with enhanced features."""
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        max_steps = 50

        proof_path = []

        while not done and steps < max_steps:
            # Get enhanced state representation
            state_repr = self.enhanced_agent.get_enhanced_state_representation(env)

            # Get action from agent
            action_idx = self.enhanced_agent.select_action(
                state_repr, epsilon=self._get_epsilon(episode, total_episodes)
            )

            # Get recommended tactics for guidance
            recommended_tactics = env.get_recommended_tactics(
                env.current_state.goal, top_k=5
            )

            # Convert action index to tactic (simplified)
            if action_idx < len(env.action_space):
                tactic = env.action_space[action_idx]
            else:
                # Fallback to recommended tactic
                tactic = recommended_tactics[0][0] if recommended_tactics else "simp"

            action = TacticAction(tactic)
            proof_path.append((env.current_state.goal, tactic))

            # Take step
            next_state, reward, done, info = env.step(action)

            # Get next state representation
            next_state_repr = self.enhanced_agent.get_enhanced_state_representation(env)

            # Store experience
            self.enhanced_agent.experience_buffer.append(
                {
                    "state": state_repr,
                    "action": action_idx,
                    "reward": reward,
                    "next_state": next_state_repr,
                    "done": done,
                    "info": info,
                }
            )

            episode_reward += reward
            steps += 1

            # Update tactic usage statistics
            self.training_metrics["tactic_usage_stats"][tactic] += 1

        # Update training metrics
        if done and episode_reward > 0:
            self.training_metrics["proofs_completed"] += 1

            # Update average proof length
            current_avg = self.training_metrics["average_proof_length"]
            completed_proofs = self.training_metrics["proofs_completed"]
            self.training_metrics["average_proof_length"] = (
                current_avg * (completed_proofs - 1) + steps
            ) / completed_proofs

        # Train agent on batch
        if len(self.enhanced_agent.experience_buffer) >= self.config.batch_size:
            self._train_agent_batch()

    def _get_epsilon(self, episode: int, total_episodes: int) -> float:
        """Get epsilon for epsilon-greedy exploration with decay."""
        epsilon_start = 0.9
        epsilon_end = 0.1
        epsilon_decay = 0.995

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (epsilon_decay**episode)
        return max(epsilon, epsilon_end)

    def _train_agent_batch(self) -> None:
        """Train agent on a batch of experiences."""
        if len(self.enhanced_agent.experience_buffer) < self.config.batch_size:
            return

        # Sample batch
        batch = random.sample(
            list(self.enhanced_agent.experience_buffer), self.config.batch_size
        )

        # Extract batch data
        states = torch.FloatTensor([exp["state"] for exp in batch])
        actions = torch.LongTensor([exp["action"] for exp in batch])
        rewards = torch.FloatTensor([exp["reward"] for exp in batch])
        next_states = torch.FloatTensor([exp["next_state"] for exp in batch])
        dones = torch.BoolTensor([exp["done"] for exp in batch])

        # Train agent (simplified - would need actual DQN/A3C implementation)
        # This is a placeholder for the actual training logic
        pass

    def _log_training_progress(
        self, episode: int, total_episodes: int, elapsed_time: float
    ) -> None:
        """Log training progress."""
        success_rate = self.training_metrics["proofs_completed"] / max(episode, 1)
        avg_length = self.training_metrics["average_proof_length"]

        logger.info(
            f"Episode {episode}/{total_episodes}: "
            f"Success rate: {success_rate:.3f}, "
            f"Avg proof length: {avg_length:.1f}, "
            f"Time: {elapsed_time:.1f}s"
        )

    def save_trained_agent(self, filepath: str) -> None:
        """Save trained RL agent."""
        if hasattr(self, "enhanced_agent"):
            checkpoint = {
                "agent_state_dict": (
                    self.enhanced_agent.state_dict()
                    if hasattr(self.enhanced_agent, "state_dict")
                    else None
                ),
                "config": self.config,
                "training_metrics": dict(self.training_metrics),
                "mathlib_tactic_embeddings": self.enhanced_agent.mathlib_tactic_embeddings,
            }
            torch.save(checkpoint, filepath)
            logger.info(f"Trained agent saved to {filepath}")


# Example usage and integration
if __name__ == "__main__":
    print("=== Mathlib RL Training Pipeline ===")

    # Configuration
    embedding_config = EmbeddingConfig(embedding_dim=128, epochs=50, batch_size=16)

    agent_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        gamma=0.99,
        epsilon_start=0.9,
        epsilon_end=0.1,
        epsilon_decay=0.995,
    )

    # Load Mathlib data
    print("1. Loading Mathlib data...")
    mathlib_loader = MathlibDataLoader(
        "/home/chorock/Projects/ATPLean/.lake/packages/mathlib"
    )

    # Train Goal2Vec model
    print("2. Training Goal2Vec model...")
    goal2vec_trainer = MathlibGoal2VecTrainer(embedding_config)
    goal2vec_trainer.load_mathlib_data()
    goal2vec_trainer.prepare_mathlib_training_data()
    goal2vec_trainer.train_with_mathlib(epochs=20)  # Reduced for demo

    # Train RL agent
    print("3. Training RL agent with Mathlib...")
    rl_trainer = MathlibProofTrainer(agent_config, goal2vec_trainer)
    rl_trainer.load_mathlib_problems("mathlib_training_data.db", max_problems=100)

    metrics = rl_trainer.train_on_mathlib(episodes=200, curriculum_learning=True)

    # Save models
    print("4. Saving trained models...")
    goal2vec_trainer.save_enhanced_model("mathlib_goal2vec_rl.pth")
    rl_trainer.save_trained_agent("mathlib_rl_agent.pth")

    # Test the system
    print("5. Testing integrated system...")
    test_goals = [
        "∀ n : ℕ, n + 0 = n",
        "∀ a b : ℕ, a + b = b + a",
        "∀ x : ℝ, x * 0 = 0",
    ]

    for goal in test_goals:
        print(f"\nTesting goal: {goal}")

        # Create test environment
        env = MathlibProofEnvironment(goal, goal2vec_trainer)

        # Get recommendations
        recommendations = env.get_recommended_tactics(goal, top_k=3)
        print("Recommended tactics:")
        for tactic, confidence in recommendations:
            print(f"  {tactic}: {confidence:.3f}")

    print("\nMathlib RL training pipeline completed!")
    print(f"Final metrics: {metrics}")

