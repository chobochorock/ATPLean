"""
Reinforcement Learning Node Structure for Mathematical Problem Solving.
Extends the basic StateNode with RL-specific attributes for goal-tactic learning.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Set
from enum import Enum
import json
from dataclasses import dataclass, field
from lean_problem_structure import StateNode


class NodeStatus(Enum):
    """Status of a proof node in the search tree."""
    OPEN = "open"           # Goal not yet solved
    SOLVED = "solved"       # Goal successfully solved
    FAILED = "failed"       # Tactic failed, dead end
    EXPLORED = "explored"   # Node has been expanded but not solved
    PROMISING = "promising" # High-value node for exploration


@dataclass
class TacticAction:
    """Represents a tactic action with metadata."""
    tactic: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "manual"  # manual, learned, heuristic
    success_rate: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "tactic": self.tactic,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "source": self.source,
            "success_rate": self.success_rate
        }


@dataclass
class NodeReward:
    """Reward structure for RL training."""
    immediate_reward: float = 0.0
    completion_reward: float = 0.0
    efficiency_reward: float = 0.0
    novelty_reward: float = 0.0
    total_reward: float = 0.0
    
    def calculate_total(self) -> float:
        """Calculate total reward from components."""
        self.total_reward = (
            self.immediate_reward + 
            self.completion_reward + 
            self.efficiency_reward + 
            self.novelty_reward
        )
        return self.total_reward


class RLStateNode(StateNode):
    """
    Reinforcement Learning State Node for mathematical problem solving.
    Extends StateNode with RL-specific attributes and methods.
    """
    
    def __init__(self, goal: str, parent: Optional["RLStateNode"] = None, 
                 tactic_action: Optional[TacticAction] = None):
        # Use goal as content for parent StateNode
        tactic_str = tactic_action.tactic if tactic_action else None
        super().__init__(goal, parent, tactic_str)
        
        # RL-specific attributes
        self.goal = goal  # Current goal/state to solve
        self.tactic_action = tactic_action  # Action that led to this state
        self.status = NodeStatus.OPEN
        
        # Value estimates and statistics
        self.value_estimate: float = 0.0  # Estimated value of this state
        self.visit_count: int = 0  # Number of times visited
        self.success_count: int = 0  # Number of successful completions from here
        self.avg_completion_length: float = 0.0  # Average steps to completion
        
        # Action space and policy
        self.available_tactics: List[TacticAction] = []
        self.tried_tactics: Set[str] = set()
        self.policy_probabilities: Dict[str, float] = {}
        
        # Reward and learning
        self.reward = NodeReward()
        self.q_values: Dict[str, float] = {}  # Q-values for each available tactic
        
        # Context and features
        self.goal_embedding: Optional[np.ndarray] = None  # Vector representation
        self.context_features: Dict[str, Any] = {}  # Additional context
        
        # Search tree metadata
        self.depth_from_root: int = self.get_depth()
        self.exploration_priority: float = 0.0
        
        # Performance tracking
        self.solution_path: List[str] = []  # Tactics in successful solution
        self.failed_tactics: List[TacticAction] = []  # Failed attempts
        
    def set_goal_embedding(self, embedding: np.ndarray) -> None:
        """Set the vector embedding for this goal."""
        self.goal_embedding = embedding
        
    def add_available_tactic(self, tactic_action: TacticAction) -> None:
        """Add a tactic to the available action space."""
        self.available_tactics.append(tactic_action)
        self.q_values[tactic_action.tactic] = 0.0
        
    def select_best_tactic(self, exploration_rate: float = 0.1) -> Optional[TacticAction]:
        """
        Select best tactic using epsilon-greedy or UCB strategy.
        
        Args:
            exploration_rate: Epsilon for epsilon-greedy exploration
            
        Returns:
            Selected tactic action or None if no tactics available
        """
        if not self.available_tactics:
            return None
            
        # Filter out already tried tactics if we want to avoid repetition
        untried_tactics = [t for t in self.available_tactics if t.tactic not in self.tried_tactics]
        
        # Prefer untried tactics for exploration
        if untried_tactics and np.random.random() < exploration_rate:
            return np.random.choice(untried_tactics)
        
        # Otherwise select based on Q-values and confidence
        best_tactic = max(self.available_tactics, 
                         key=lambda t: self.q_values.get(t.tactic, 0.0) + t.confidence)
        return best_tactic
    
    def update_q_value(self, tactic: str, reward: float, learning_rate: float = 0.1) -> None:
        """Update Q-value for a specific tactic."""
        current_q = self.q_values.get(tactic, 0.0)
        self.q_values[tactic] = current_q + learning_rate * (reward - current_q)
    
    def mark_tactic_tried(self, tactic: str, success: bool = False) -> None:
        """Mark a tactic as tried and update statistics."""
        self.tried_tactics.add(tactic)
        self.visit_count += 1
        
        if success:
            self.success_count += 1
            self.status = NodeStatus.SOLVED
        else:
            # Add to failed tactics
            failed_action = next((t for t in self.available_tactics if t.tactic == tactic), None)
            if failed_action:
                self.failed_tactics.append(failed_action)
    
    def calculate_exploration_priority(self) -> float:
        """Calculate exploration priority using UCB1 or similar."""
        if self.visit_count == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        # UCB1 formula: Q(s,a) + c * sqrt(ln(N) / n)
        c = 1.4  # Exploration constant
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration_bonus = c * np.sqrt(np.log(parent_visits) / self.visit_count)
        
        self.exploration_priority = self.value_estimate + exploration_bonus
        return self.exploration_priority
    
    def propagate_reward_up(self, reward: float, decay_factor: float = 0.95) -> None:
        """Propagate reward up the tree to parent nodes."""
        current_node = self
        current_reward = reward
        
        while current_node is not None:
            current_node.reward.immediate_reward += current_reward
            current_node.reward.calculate_total()
            
            # Update value estimate
            current_node.value_estimate = (
                current_node.value_estimate * current_node.visit_count + current_reward
            ) / (current_node.visit_count + 1)
            
            # Move to parent with decayed reward
            current_reward *= decay_factor
            current_node = current_node.parent
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (solved or failed)."""
        return self.status in [NodeStatus.SOLVED, NodeStatus.FAILED]
    
    def get_solution_path(self) -> List[TacticAction]:
        """Get the path of tactics from root to this node."""
        path = []
        current = self
        
        while current.parent is not None:
            if current.tactic_action:
                path.append(current.tactic_action)
            current = current.parent
            
        return list(reversed(path))
    
    def add_rl_child(self, goal: str, tactic_action: TacticAction) -> "RLStateNode":
        """Add a child node with RL-specific initialization."""
        child = RLStateNode(goal, self, tactic_action)
        return child
    
    def get_state_features(self) -> Dict[str, Any]:
        """Extract features for ML model training."""
        return {
            "goal": self.goal,
            "depth": self.depth_from_root,
            "visit_count": self.visit_count,
            "success_rate": self.success_count / self.visit_count if self.visit_count > 0 else 0,
            "value_estimate": self.value_estimate,
            "num_available_tactics": len(self.available_tactics),
            "num_tried_tactics": len(self.tried_tactics),
            "exploration_priority": self.exploration_priority,
            "goal_embedding": self.goal_embedding.tolist() if self.goal_embedding is not None else None,
            "context_features": self.context_features
        }
    
    def to_training_example(self) -> Dict[str, Any]:
        """Convert node to training example for RL model."""
        return {
            "state": self.get_state_features(),
            "action": self.tactic_action.to_dict() if self.tactic_action else None,
            "reward": self.reward.total_reward,
            "next_state": self.children[0].get_state_features() if self.children else None,
            "done": self.is_terminal(),
            "q_values": self.q_values.copy()
        }
    
    def visualize_rl_tree(self, max_width: int = 100, show_values: bool = True) -> str:
        """Enhanced tree visualization with RL information."""
        return self._visualize_rl_recursive("", True, max_width, show_values)
    
    def _visualize_rl_recursive(self, prefix: str, is_last: bool, max_width: int, show_values: bool) -> str:
        """Recursive helper for RL tree visualization."""
        # Prepare content display
        goal_display = self.goal[:50] + "..." if len(self.goal) > 50 else self.goal
        
        # Status and value info
        status_symbol = {
            NodeStatus.OPEN: "○",
            NodeStatus.SOLVED: "✓",
            NodeStatus.FAILED: "✗",
            NodeStatus.EXPLORED: "◐",
            NodeStatus.PROMISING: "⭐"
        }.get(self.status, "?")
        
        # Build info string
        info_parts = [f"[{self.index}] {status_symbol}"]
        if show_values:
            info_parts.append(f"V:{self.value_estimate:.2f}")
            info_parts.append(f"N:{self.visit_count}")
            if self.success_count > 0:
                info_parts.append(f"S:{self.success_count}")
        
        tactic_display = f" [{self.tactic_action.tactic}]" if self.tactic_action else ""
        
        # Current node line
        connector = "└── " if is_last else "├── "
        info_str = " ".join(info_parts)
        result = f"{prefix}{connector}{info_str} {goal_display}{tactic_display}\n"
        
        # Children
        if self.children:
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(self.children):
                is_child_last = (i == len(self.children) - 1)
                result += child._visualize_rl_recursive(new_prefix, is_child_last, max_width, show_values)
        
        return result
    
    def save_to_json(self, filepath: str) -> None:
        """Save the RL tree to JSON for persistence."""
        tree_data = {
            "root": self._to_json_recursive(),
            "metadata": {
                "total_nodes": self._count_nodes(),
                "max_depth": self._get_max_depth(),
                "solved_nodes": self._count_solved_nodes()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(tree_data, f, indent=2, default=str)
    
    def _to_json_recursive(self) -> Dict[str, Any]:
        """Recursive helper for JSON serialization."""
        return {
            "index": self.index,
            "goal": self.goal,
            "status": self.status.value,
            "tactic_action": self.tactic_action.to_dict() if self.tactic_action else None,
            "value_estimate": self.value_estimate,
            "visit_count": self.visit_count,
            "success_count": self.success_count,
            "q_values": self.q_values,
            "reward": {
                "immediate": self.reward.immediate_reward,
                "completion": self.reward.completion_reward,
                "efficiency": self.reward.efficiency_reward,
                "novelty": self.reward.novelty_reward,
                "total": self.reward.total_reward
            },
            "children": [child._to_json_recursive() for child in self.children]
        }
    
    def _count_nodes(self) -> int:
        """Count total nodes in subtree."""
        return 1 + sum(child._count_nodes() for child in self.children)
    
    def _get_max_depth(self) -> int:
        """Get maximum depth in subtree."""
        if not self.children:
            return self.depth_from_root
        return max(child._get_max_depth() for child in self.children)
    
    def _count_solved_nodes(self) -> int:
        """Count solved nodes in subtree."""
        count = 1 if self.status == NodeStatus.SOLVED else 0
        return count + sum(child._count_solved_nodes() for child in self.children)


class RLProofEnvironment:
    """
    Reinforcement Learning Environment for theorem proving.
    Manages the state space and action space for RL agents.
    """
    
    def __init__(self, initial_theorem: str):
        self.initial_theorem = initial_theorem
        self.root_node = RLStateNode(initial_theorem)
        self.current_node = self.root_node
        self.available_tactics = self._initialize_tactics()
        
        # RL environment parameters  
        self.max_depth = 50
        self.episode_length = 0
        self.total_reward = 0.0
        
    def _initialize_tactics(self) -> List[TacticAction]:
        """Initialize available tactics for the environment."""
        common_tactics = [
            TacticAction("simp", confidence=0.8),
            TacticAction("rw", confidence=0.7),
            TacticAction("ring", confidence=0.6),
            TacticAction("linarith", confidence=0.6),
            TacticAction("exact", confidence=0.5),
            TacticAction("apply", confidence=0.5),
            TacticAction("intro", confidence=0.7),
            TacticAction("cases", confidence=0.6),
            TacticAction("induction", confidence=0.4),
            TacticAction("unfold", confidence=0.5),
            TacticAction("norm_num", confidence=0.8),
            TacticAction("omega", confidence=0.7),
            TacticAction("tauto", confidence=0.6),
            TacticAction("constructor", confidence=0.5),
            TacticAction("left", confidence=0.6),
            TacticAction("right", confidence=0.6),
            TacticAction("split", confidence=0.6),
            TacticAction("exists", confidence=0.4),
            TacticAction("use", confidence=0.4),
            TacticAction("have", confidence=0.3)
        ]
        return common_tactics
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to initial state."""
        self.current_node = self.root_node
        self.episode_length = 0
        self.total_reward = 0.0
        return self.current_node.get_state_features()
    
    def step(self, tactic_action: TacticAction) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Returns:
            (next_state, reward, done, info)
        """
        self.episode_length += 1
        
        # Mark tactic as tried
        self.current_node.mark_tactic_tried(tactic_action.tactic)
        
        # Simulate tactic execution (in real system, would call Lean)
        success, new_goals = self._simulate_tactic_execution(tactic_action)
        
        # Calculate reward
        reward = self._calculate_reward(success, tactic_action)
        self.total_reward += reward
        
        # Update Q-value
        self.current_node.update_q_value(tactic_action.tactic, reward)
        
        if success:
            if new_goals:
                # Tactic succeeded but created new subgoals
                child_node = self.current_node.add_rl_child(new_goals[0], tactic_action)
                self.current_node = child_node
                done = False
            else:
                # Theorem completely solved!
                self.current_node.status = NodeStatus.SOLVED
                done = True
        else:
            # Tactic failed
            self.current_node.status = NodeStatus.FAILED
            done = True
        
        # Propagate reward up the tree
        self.current_node.propagate_reward_up(reward)
        
        # Check termination conditions
        done = done or self.episode_length >= self.max_depth
        
        info = {
            "episode_length": self.episode_length,
            "total_reward": self.total_reward,
            "success": success,
            "node_depth": self.current_node.depth_from_root
        }
        
        return self.current_node.get_state_features(), reward, done, info
    
    def _simulate_tactic_execution(self, tactic_action: TacticAction) -> Tuple[bool, List[str]]:
        """
        Simulate tactic execution. In real system, this would call Lean.
        Returns (success, new_goals)
        """
        # Simple simulation based on tactic confidence and randomness
        success_prob = tactic_action.confidence + np.random.normal(0, 0.1)
        success = np.random.random() < max(0, min(1, success_prob))
        
        new_goals = []
        if success:
            # Sometimes tactics solve completely, sometimes create subgoals
            if np.random.random() < 0.3:  # 30% chance of complete solution
                new_goals = []
            else:
                # Create 1-2 new subgoals
                num_subgoals = np.random.choice([1, 2], p=[0.7, 0.3])
                for i in range(num_subgoals):
                    new_goals.append(f"subgoal_{self.episode_length}_{i}")
        
        return success, new_goals
    
    def _calculate_reward(self, success: bool, tactic_action: TacticAction) -> float:
        """Calculate reward for the action taken."""
        reward = 0.0
        
        if success:
            reward += 1.0  # Base success reward
            reward += tactic_action.confidence * 0.5  # Bonus for confident tactics
        else:
            reward -= 0.5  # Penalty for failure
        
        # Efficiency bonus (shorter proofs are better)
        if self.episode_length > 0:
            reward += max(0, (self.max_depth - self.episode_length) / self.max_depth * 0.2)
        
        return reward
    
    def get_available_actions(self) -> List[TacticAction]:
        """Get available actions for current state."""
        return [t for t in self.available_tactics if t.tactic not in self.current_node.tried_tactics]
    
    def render(self) -> str:
        """Render the current proof tree."""
        return self.root_node.visualize_rl_tree()


# Example usage and testing
if __name__ == "__main__":
    print("=== Reinforcement Learning Node Testing ===")
    
    # Create RL environment
    theorem = "∀ n : ℕ, n + 0 = n"
    env = RLProofEnvironment(theorem)
    
    print(f"Initial theorem: {theorem}")
    print(f"Available tactics: {[t.tactic for t in env.available_tactics[:10]]}")
    
    # Simulate some proof steps
    state = env.reset()
    print(f"\nInitial state: {env.current_node.goal}")
    
    for step in range(5):
        available_actions = env.get_available_actions()
        if not available_actions:
            break
            
        # Select random action for demo
        action = np.random.choice(available_actions)
        print(f"\nStep {step + 1}: Trying tactic '{action.tactic}'")
        
        next_state, reward, done, info = env.step(action)
        print(f"Reward: {reward:.2f}, Done: {done}")
        print(f"Current goal: {env.current_node.goal}")
        
        if done:
            break
    
    print("\n=== Final Proof Tree ===")
    print(env.render())
    
    # Test node serialization
    print("\n=== JSON Serialization Test ===")
    env.root_node.save_to_json("/tmp/test_proof_tree.json")
    print("Proof tree saved to /tmp/test_proof_tree.json")
    
    # Test training example generation
    print("\n=== Training Example ===")
    training_example = env.root_node.to_training_example()
    print(json.dumps(training_example, indent=2, default=str)[:500] + "...")