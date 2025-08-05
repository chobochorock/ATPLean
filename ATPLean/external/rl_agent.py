"""
Reinforcement Learning Agent for Mathematical Theorem Proving.
Implements various RL algorithms for learning proof strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import deque, defaultdict
import random
import json
from dataclasses import dataclass
from enum import Enum
import pickle

from reinforcement_learning_node import RLStateNode, TacticAction, RLProofEnvironment, NodeStatus


class AgentType(Enum):
    """Types of RL agents."""
    DQN = "dqn"
    ACTOR_CRITIC = "actor_critic"
    PPO = "ppo"
    MCTS = "mcts"
    RANDOM = "random"


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    learning_rate: float = 0.001
    batch_size: int = 32
    memory_size: int = 10000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    gamma: float = 0.99
    target_update_freq: int = 100
    max_episodes: int = 1000
    max_steps_per_episode: int = 50
    hidden_size: int = 256
    embedding_size: int = 128


class GoalEncoder(nn.Module):
    """Neural network for encoding mathematical goals into embeddings."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state
        output = self.fc(hidden[-1])  # (batch_size, embedding_dim)
        return self.dropout(output)


class DQNNetwork(nn.Module):
    """Deep Q-Network for tactic selection."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.q_head = nn.Linear(hidden_size, action_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        q_values = self.q_head(x)
        return q_values


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for policy gradient methods."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_size, hidden_size)
        self.actor_head = nn.Linear(hidden_size, action_dim)
        
        # Critic head (value function)
        self.critic_fc = nn.Linear(hidden_size, hidden_size)
        self.critic_head = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shared layers
        x = F.relu(self.shared_fc1(state))
        x = self.dropout(x)
        x = F.relu(self.shared_fc2(x))
        shared_features = self.dropout(x)
        
        # Actor (policy)
        actor_x = F.relu(self.actor_fc(shared_features))
        action_probs = F.softmax(self.actor_head(actor_x), dim=-1)
        
        # Critic (value)
        critic_x = F.relu(self.critic_fc(shared_features))
        state_value = self.critic_head(critic_x)
        
        return action_probs, state_value


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class ProofAgent:
    """Base class for RL agents that learn to prove theorems."""
    
    def __init__(self, agent_type: AgentType, config: TrainingConfig):
        self.agent_type = agent_type
        self.config = config
        self.tactic_to_id = {}  # Map tactic names to IDs
        self.id_to_tactic = {}  # Map IDs to tactic names
        self.vocab_size = 1000  # For goal encoding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training statistics
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0.0,
            "success_rate": 0.0,
            "avg_episode_length": 0.0,
            "losses": []
        }
        
        self._build_networks()
        self._initialize_optimizers()
        
    def _build_networks(self) -> None:
        """Build neural networks based on agent type."""
        state_dim = self.config.embedding_size + 10  # Goal embedding + additional features
        
        if self.agent_type == AgentType.DQN:
            self.q_network = DQNNetwork(state_dim, 100, self.config.hidden_size).to(self.device)
            self.target_network = DQNNetwork(state_dim, 100, self.config.hidden_size).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.replay_buffer = ReplayBuffer(self.config.memory_size)
            
        elif self.agent_type == AgentType.ACTOR_CRITIC:
            self.ac_network = ActorCriticNetwork(state_dim, 100, self.config.hidden_size).to(self.device)
            
        # Goal encoder for all agent types
        self.goal_encoder = GoalEncoder(self.vocab_size, self.config.embedding_size, 
                                       self.config.hidden_size).to(self.device)
    
    def _initialize_optimizers(self) -> None:
        """Initialize optimizers for networks."""
        if self.agent_type == AgentType.DQN:
            self.optimizer = optim.Adam(
                list(self.q_network.parameters()) + list(self.goal_encoder.parameters()),
                lr=self.config.learning_rate
            )
        elif self.agent_type == AgentType.ACTOR_CRITIC:
            self.optimizer = optim.Adam(
                list(self.ac_network.parameters()) + list(self.goal_encoder.parameters()),
                lr=self.config.learning_rate
            )
    
    def encode_goal(self, goal: str) -> torch.Tensor:
        """Encode goal string into embedding vector."""
        # Simple tokenization (in practice, use proper tokenizer)
        tokens = goal.split()[:50]  # Limit length
        token_ids = []
        
        for token in tokens:
            # Simple hash-based token ID (in practice, use proper vocabulary)
            token_id = hash(token) % self.vocab_size
            token_ids.append(token_id)
        
        # Pad sequence
        while len(token_ids) < 50:
            token_ids.append(0)  # Padding token
        
        input_tensor = torch.LongTensor([token_ids]).to(self.device)
        with torch.no_grad():
            embedding = self.goal_encoder(input_tensor)
        
        return embedding.squeeze(0)
    
    def state_to_features(self, state: Dict[str, Any]) -> torch.Tensor:
        """Convert state dictionary to feature vector."""
        # Extract goal embedding
        goal = state.get("goal", "")
        goal_embedding = self.encode_goal(goal)
        
        # Additional features
        additional_features = torch.FloatTensor([
            state.get("depth", 0) / 10.0,  # Normalized depth
            state.get("visit_count", 0) / 100.0,  # Normalized visit count
            state.get("success_rate", 0.0),
            state.get("value_estimate", 0.0),
            len(state.get("goal", "")) / 100.0,  # Normalized goal length
            state.get("num_available_tactics", 0) / 20.0,  # Normalized tactic count
            state.get("num_tried_tactics", 0) / 20.0,
            state.get("exploration_priority", 0.0),
            1.0 if "∀" in goal else 0.0,  # Has universal quantifier
            1.0 if "∃" in goal else 0.0,  # Has existential quantifier
        ]).to(self.device)
        
        # Concatenate goal embedding and additional features
        features = torch.cat([goal_embedding, additional_features])
        return features
    
    def get_tactic_id(self, tactic: str) -> int:
        """Get or create ID for tactic."""
        if tactic not in self.tactic_to_id:
            tactic_id = len(self.tactic_to_id)
            self.tactic_to_id[tactic] = tactic_id
            self.id_to_tactic[tactic_id] = tactic
        return self.tactic_to_id[tactic]
    
    def select_action(self, state: Dict[str, Any], available_actions: List[TacticAction], 
                     epsilon: float = 0.0) -> TacticAction:
        """Select action based on current policy."""
        if self.agent_type == AgentType.RANDOM:
            return random.choice(available_actions)
        
        # Convert state to features
        state_features = self.state_to_features(state).unsqueeze(0)
        
        if self.agent_type == AgentType.DQN:
            return self._dqn_select_action(state_features, available_actions, epsilon)
        elif self.agent_type == AgentType.ACTOR_CRITIC:
            return self._ac_select_action(state_features, available_actions)
        else:
            return random.choice(available_actions)
    
    def _dqn_select_action(self, state_features: torch.Tensor, 
                          available_actions: List[TacticAction], epsilon: float) -> TacticAction:
        """DQN action selection with epsilon-greedy."""
        if random.random() < epsilon:
            return random.choice(available_actions)
        
        with torch.no_grad():
            q_values = self.q_network(state_features)
        
        # Get Q-values for available actions only
        available_q_values = []
        for action in available_actions:
            action_id = self.get_tactic_id(action.tactic)
            if action_id < q_values.size(1):
                available_q_values.append((q_values[0, action_id].item(), action))
            else:
                available_q_values.append((0.0, action))  # Unknown action
        
        # Select action with highest Q-value
        best_action = max(available_q_values, key=lambda x: x[0])[1]
        return best_action
    
    def _ac_select_action(self, state_features: torch.Tensor, 
                         available_actions: List[TacticAction]) -> TacticAction:
        """Actor-Critic action selection."""
        with torch.no_grad():
            action_probs, _ = self.ac_network(state_features)
        
        # Get probabilities for available actions
        available_probs = []
        for action in available_actions:
            action_id = self.get_tactic_id(action.tactic)
            if action_id < action_probs.size(1):
                prob = action_probs[0, action_id].item()
            else:
                prob = 1.0 / len(available_actions)  # Uniform for unknown actions
            available_probs.append(prob)
        
        # Normalize probabilities
        total_prob = sum(available_probs)
        if total_prob > 0:
            available_probs = [p / total_prob for p in available_probs]
        else:
            available_probs = [1.0 / len(available_actions)] * len(available_actions)
        
        # Sample action based on probabilities
        action_idx = np.random.choice(len(available_actions), p=available_probs)
        return available_actions[action_idx]
    
    def update(self, experiences: List[Dict[str, Any]]) -> float:
        """Update agent based on experiences."""
        if self.agent_type == AgentType.DQN:
            return self._update_dqn(experiences)
        elif self.agent_type == AgentType.ACTOR_CRITIC:
            return self._update_actor_critic(experiences)
        else:
            return 0.0
    
    def _update_dqn(self, experiences: List[Dict[str, Any]]) -> float:
        """Update DQN networks."""
        # Add experiences to replay buffer
        for exp in experiences:
            state = self.state_to_features(exp["state"]).cpu().numpy()
            next_state = self.state_to_features(exp["next_state"]).cpu().numpy()
            action_id = self.get_tactic_id(exp["action"]["tactic"])
            
            self.replay_buffer.push(
                state, action_id, exp["reward"], next_state, exp["done"]
            )
        
        # Update if buffer has enough samples
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _update_actor_critic(self, experiences: List[Dict[str, Any]]) -> float:
        """Update Actor-Critic network."""
        if not experiences:
            return 0.0
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in experiences:
            states.append(self.state_to_features(exp["state"]))
            actions.append(self.get_tactic_id(exp["action"]["tactic"]))
            rewards.append(exp["reward"])
            next_states.append(self.state_to_features(exp["next_state"]))
            dones.append(exp["done"])
        
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Forward pass
        action_probs, state_values = self.ac_network(states)
        _, next_state_values = self.ac_network(next_states)
        
        # Calculate advantages
        with torch.no_grad():
            target_values = rewards + self.config.gamma * next_state_values.squeeze() * ~dones
            advantages = target_values - state_values.squeeze()
        
        # Actor loss (policy gradient)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        actor_loss = -(action_log_probs * advantages).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(state_values.squeeze(), target_values)
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def update_target_network(self) -> None:
        """Update target network for DQN."""
        if self.agent_type == AgentType.DQN:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        checkpoint = {
            "agent_type": self.agent_type.value,
            "config": self.config,
            "tactic_to_id": self.tactic_to_id,
            "id_to_tactic": self.id_to_tactic,
            "training_stats": self.training_stats
        }
        
        if self.agent_type == AgentType.DQN:
            checkpoint["q_network"] = self.q_network.state_dict()
            checkpoint["target_network"] = self.target_network.state_dict()
        elif self.agent_type == AgentType.ACTOR_CRITIC:
            checkpoint["ac_network"] = self.ac_network.state_dict()
        
        checkpoint["goal_encoder"] = self.goal_encoder.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.tactic_to_id = checkpoint["tactic_to_id"]
        self.id_to_tactic = checkpoint["id_to_tactic"]
        self.training_stats = checkpoint["training_stats"]
        
        if self.agent_type == AgentType.DQN and "q_network" in checkpoint:
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
        elif self.agent_type == AgentType.ACTOR_CRITIC and "ac_network" in checkpoint:
            self.ac_network.load_state_dict(checkpoint["ac_network"])
        
        if "goal_encoder" in checkpoint:
            self.goal_encoder.load_state_dict(checkpoint["goal_encoder"])
        
        print(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats.copy()


class ProofTrainer:
    """Trainer for RL agents on theorem proving tasks."""
    
    def __init__(self, agent: ProofAgent, environments: List[RLProofEnvironment]):
        self.agent = agent
        self.environments = environments
        self.training_history = []
        
    def train(self, num_episodes: int, save_interval: int = 100) -> None:
        """Train the agent on the environments."""
        epsilon = self.agent.config.epsilon_start
        episode_rewards = deque(maxlen=100)
        episode_lengths = deque(maxlen=100)
        success_count = 0
        
        for episode in range(num_episodes):
            # Select environment
            env = random.choice(self.environments)
            
            # Run episode
            experiences, total_reward, episode_length, success = self._run_episode(env, epsilon)
            
            # Update statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            if success:
                success_count += 1
            
            # Update agent
            if experiences:
                loss = self.agent.update(experiences)
                self.agent.training_stats["losses"].append(loss)
            
            # Update target network
            if episode % self.agent.config.target_update_freq == 0:
                self.agent.update_target_network()
            
            # Decay epsilon
            epsilon = max(self.agent.config.epsilon_end, 
                         epsilon * self.agent.config.epsilon_decay)
            
            # Log progress
            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                avg_length = np.mean(episode_lengths) if episode_lengths else 0
                success_rate = success_count / (episode + 1)
                
                print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, "
                      f"Avg Length={avg_length:.1f}, Success Rate={success_rate:.2f}, "
                      f"Epsilon={epsilon:.3f}")
                
                # Update training stats
                self.agent.training_stats.update({
                    "episodes": episode + 1,
                    "total_reward": avg_reward,
                    "success_rate": success_rate,
                    "avg_episode_length": avg_length
                })
            
            # Save model
            if episode % save_interval == 0 and episode > 0:
                self.agent.save_model(f"proof_agent_episode_{episode}.pth")
            
            # Record training history
            self.training_history.append({
                "episode": episode,
                "reward": total_reward,
                "length": episode_length,
                "success": success,
                "epsilon": epsilon
            })
    
    def _run_episode(self, env: RLProofEnvironment, epsilon: float) -> Tuple[List[Dict[str, Any]], float, int, bool]:
        """Run a single training episode."""
        experiences = []
        state = env.reset()
        total_reward = 0.0
        episode_length = 0
        
        for step in range(self.agent.config.max_steps_per_episode):
            # Get available actions
            available_actions = env.get_available_actions()
            if not available_actions:
                break
            
            # Select action
            action = self.agent.select_action(state, available_actions, epsilon)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            experience = {
                "state": state.copy(),
                "action": action.to_dict(),
                "reward": reward,
                "next_state": next_state.copy(),
                "done": done,
                "info": info
            }
            experiences.append(experience)
            
            total_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        success = env.current_node.status == NodeStatus.SOLVED
        return experiences, total_reward, episode_length, success
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance."""
        total_rewards = []
        episode_lengths = []
        success_count = 0
        
        for _ in range(num_episodes):
            env = random.choice(self.environments)
            _, total_reward, episode_length, success = self._run_episode(env, epsilon=0.0)
            
            total_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            if success:
                success_count += 1
        
        return {
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "avg_length": np.mean(episode_lengths),
            "success_rate": success_count / num_episodes
        }
    
    def save_training_history(self, filepath: str) -> None:
        """Save training history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)


# Example usage and testing
if __name__ == "__main__":
    print("=== Reinforcement Learning Agent Testing ===")
    
    # Create training configuration
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=16,
        memory_size=1000,
        max_episodes=100,
        max_steps_per_episode=20
    )
    
    # Create agent
    agent = ProofAgent(AgentType.DQN, config)
    print(f"Created {agent.agent_type.value} agent")
    
    # Create sample environments
    theorems = [
        "∀ n : ℕ, n + 0 = n",
        "∀ a b : ℕ, a + b = b + a",
        "∀ n : ℕ, 0 + n = n"
    ]
    
    environments = []
    for theorem in theorems:
        env = RLProofEnvironment(theorem)
        environments.append(env)
    
    print(f"Created {len(environments)} training environments")
    
    # Test action selection
    test_env = environments[0]
    state = test_env.reset()
    available_actions = test_env.get_available_actions()
    
    print(f"\nTesting action selection:")
    print(f"Current goal: {test_env.current_node.goal}")
    print(f"Available actions: {[a.tactic for a in available_actions[:5]]}")
    
    selected_action = agent.select_action(state, available_actions, epsilon=0.5)
    print(f"Selected action: {selected_action.tactic}")
    
    # Test training
    print(f"\n=== Starting Training ===")
    trainer = ProofTrainer(agent, environments)
    trainer.train(num_episodes=50, save_interval=25)
    
    # Evaluate performance
    print(f"\n=== Evaluation ===")
    eval_results = trainer.evaluate(num_episodes=10)
    for key, value in eval_results.items():
        print(f"{key}: {value:.3f}")
    
    # Save model and training history
    agent.save_model("/tmp/trained_proof_agent.pth")
    trainer.save_training_history("/tmp/training_history.json")
    
    print("RL agent testing completed!")