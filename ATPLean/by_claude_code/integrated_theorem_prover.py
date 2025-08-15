"""
Integrated Theorem Prover System.
Combines all components: RL agents, Goal2Vec embeddings, definition database, and Lean integration.
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime

# Import all our components
from reinforcement_learning_node import RLStateNode, TacticAction, RLProofEnvironment, NodeStatus
from rl_agent import ProofAgent, ProofTrainer, AgentType, TrainingConfig
from goal2vec_model import Goal2VecTrainer, EmbeddingConfig, Word2VecGoalComparator
from minif2f_processor import MinIF2FProcessor, MathProblem, ProblemDifficulty
from definition_database import DefinitionDatabase, MathDefinition, DefinitionType
# Updated to use lean_interact versions
from lean_problem_reader_interact import LeanInteractReader as LeanReader
from lean_problem_parser import LeanInteractParser, load_lean_file_with_interact

# Original imports (commented out for backup)
# from lean_problem_reader import LeanReader
# from lean_problem_parser import LeanInteractParser, load_lean_file_with_interact


class IntegratedTheoremProver:
    """
    Main system that integrates all components for automated theorem proving.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the integrated system."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize components
        self.definition_db = DefinitionDatabase(self.config.get("db_path", "theorem_prover.db"))
        self.minif2f_processor = MinIF2FProcessor(self.config.get("minif2f_path"))
        self.goal2vec_trainer = Goal2VecTrainer(EmbeddingConfig(**self.config.get("embedding", {})))
        self.rl_agent = None
        self.trainer = None
        
        # Data storage
        self.problems = []
        self.environments = []
        self.training_trees = []
        
        # System state
        self.is_trained = False
        self.training_stats = {}
        
        self.logger.info("Integrated Theorem Prover initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "db_path": "theorem_prover.db",
            "minif2f_path": None,
            "model_save_path": "models/",
            "data_save_path": "data/",
            "embedding": {
                "embedding_dim": 128,
                "epochs": 100,
                "batch_size": 32
            },
            "rl_training": {
                "agent_type": "dqn",
                "learning_rate": 0.001,
                "max_episodes": 1000,
                "max_steps_per_episode": 50
            },
            "logging": {
                "level": "INFO",
                "file": "theorem_prover.log"
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge configurations
            def merge_dicts(default, user):
                for key, value in user.items():
                    if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                        merge_dicts(default[key], value)
                    else:
                        default[key] = value
            
            merge_dicts(default_config, user_config)
        
        return default_config
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "theorem_prover.log")
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("TheoremProver")
    
    def load_problems(self, source: str = "minif2f", limit: Optional[int] = None) -> None:
        """
        Load mathematical problems from various sources.
        
        Args:
            source: Source of problems ("minif2f", "file", or "samples")
            limit: Maximum number of problems to load
        """
        self.logger.info(f"Loading problems from {source}")
        
        if source == "minif2f":
            self.problems = self.minif2f_processor.load_minif2f_problems(limit)
        elif source == "samples":
            self.problems = self.minif2f_processor._create_sample_problems()
        elif source.endswith(".lean"):
            # Load from Lean file using lean_interact
            parser = LeanInteractParser()
            if parser.load_lean_file(source):
                problem_structure = parser.get_problem_structure()
                # Convert to MathProblem objects
                for i, theorem in enumerate(problem_structure.theorems):
                    problem = MathProblem(
                        problem_id=f"lean_{i}",
                        statement=theorem,
                        source="lean_file"
                    )
                    self.problems.append(problem)
        
        self.logger.info(f"Loaded {len(self.problems)} problems")
        
        # Import definitions to database
        self._import_definitions_from_problems()
    
    def _import_definitions_from_problems(self) -> None:
        """Import definitions from loaded problems to the database."""
        self.logger.info("Importing definitions from problems")
        
        for problem in self.problems:
            # Create definition for the theorem
            definition = MathDefinition(
                name=problem.problem_id,
                definition_type=DefinitionType.THEOREM,
                formal_statement=problem.formal_statement,
                informal_description=problem.informal_statement,
                category=problem.category,
                proof_sketch=problem.proof if problem.proof != "sorry" else ""
            )
            
            try:
                self.definition_db.add_definition(definition)
            except Exception as e:
                self.logger.warning(f"Failed to add definition {problem.problem_id}: {e}")
    
    def create_environments(self) -> None:
        """Create RL environments from loaded problems."""
        self.logger.info("Creating RL environments")
        
        self.environments = []
        for problem in self.problems:
            env = self.minif2f_processor.convert_to_rl_environment(problem)
            self.environments.append(env)
        
        self.logger.info(f"Created {len(self.environments)} environments")
    
    def train_goal2vec(self, use_existing_trees: bool = True) -> None:
        """
        Train Goal2Vec embeddings.
        
        Args:
            use_existing_trees: Whether to use existing RL trees for training
        """
        self.logger.info("Training Goal2Vec embeddings")
        
        # Prepare training data
        rl_trees = self.training_trees if use_existing_trees else None
        self.goal2vec_trainer.prepare_training_data(self.problems, rl_trees)
        
        # Train the model
        self.goal2vec_trainer.train()
        
        # Save model
        model_path = Path(self.config["model_save_path"]) / "goal2vec_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.goal2vec_trainer.save_model(str(model_path))
        
        self.logger.info("Goal2Vec training completed")
    
    def train_rl_agent(self) -> None:
        """Train the RL agent for theorem proving."""
        self.logger.info("Training RL agent")
        
        if not self.environments:
            self.create_environments()
        
        # Create RL agent
        rl_config = TrainingConfig(**self.config["rl_training"])
        agent_type = AgentType(self.config["rl_training"]["agent_type"])
        self.rl_agent = ProofAgent(agent_type, rl_config)
        
        # Create trainer
        self.trainer = ProofTrainer(self.rl_agent, self.environments)
        
        # Train
        self.trainer.train(
            num_episodes=rl_config.max_episodes,
            save_interval=100
        )
        
        # Save final model
        model_path = Path(self.config["model_save_path"]) / "rl_agent_final.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.rl_agent.save_model(str(model_path))
        
        # Save training trees for Goal2Vec
        self._extract_training_trees()
        
        self.is_trained = True
        self.training_stats = self.rl_agent.get_training_stats()
        
        self.logger.info("RL agent training completed")
    
    def _extract_training_trees(self) -> None:
        """Extract training trees from environments after RL training."""
        self.training_trees = []
        for env in self.environments:
            if env.root_node.children:  # Only non-trivial trees
                self.training_trees.append(env.root_node)
        
        self.logger.info(f"Extracted {len(self.training_trees)} training trees")
    
    def prove_theorem(self, theorem_statement: str, max_steps: int = 50, 
                     use_goal2vec: bool = True) -> Dict[str, Any]:
        """
        Attempt to prove a theorem using the trained system.
        
        Args:
            theorem_statement: The theorem to prove
            max_steps: Maximum proof steps
            use_goal2vec: Whether to use Goal2Vec for tactic recommendation
            
        Returns:
            Dictionary with proof result and metadata
        """
        self.logger.info(f"Attempting to prove: {theorem_statement}")
        
        if not self.is_trained or not self.rl_agent:
            raise ValueError("System must be trained before proving theorems")
        
        # Create environment for this theorem
        env = RLProofEnvironment(theorem_statement)
        
        # Add embeddings if Goal2Vec is available
        if use_goal2vec and self.goal2vec_trainer.model:
            goal_embedding = self.goal2vec_trainer.get_goal_embedding(theorem_statement)
            env.root_node.set_goal_embedding(goal_embedding)
        
        # Attempt proof
        state = env.reset()
        proof_steps = []
        total_reward = 0.0
        
        for step in range(max_steps):
            # Get available actions
            available_actions = env.get_available_actions()
            if not available_actions:
                break
            
            # Enhanced action selection with Goal2Vec
            if use_goal2vec and self.goal2vec_trainer.model:
                # Get Goal2Vec recommendations
                tactic_names = [a.tactic for a in available_actions]
                recommendations = self.goal2vec_trainer.recommend_tactics(
                    env.current_node.goal, tactic_names, top_k=len(tactic_names)
                )
                
                # Update action confidences with Goal2Vec scores
                rec_dict = {tactic: score for tactic, score in recommendations}
                for action in available_actions:
                    if action.tactic in rec_dict:
                        action.confidence = max(action.confidence, rec_dict[action.tactic])
            
            # Select action using RL agent
            action = self.rl_agent.select_action(state, available_actions, epsilon=0.1)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Record step
            proof_steps.append({
                "step": step,
                "goal": env.current_node.goal,
                "tactic": action.tactic,
                "reward": reward,
                "confidence": action.confidence
            })
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Analyze result
        success = env.current_node.status == NodeStatus.SOLVED
        proof_length = len(proof_steps)
        
        result = {
            "theorem": theorem_statement,
            "success": success,
            "proof_length": proof_length,
            "total_reward": total_reward,
            "proof_steps": proof_steps,
            "final_status": env.current_node.status.value,
            "proof_tree": env.render() if success else None
        }
        
        if success:
            self.logger.info(f"Theorem proved in {proof_length} steps!")
        else:
            self.logger.info(f"Failed to prove theorem after {proof_length} steps")
        
        return result
    
    def find_similar_theorems(self, query_theorem: str, top_k: int = 5) -> List[Tuple[MathProblem, float]]:
        """
        Find theorems similar to the query using Goal2Vec embeddings.
        
        Args:
            query_theorem: Theorem to find similarities for
            top_k: Number of similar theorems to return
            
        Returns:
            List of (problem, similarity_score) tuples
        """
        if not self.goal2vec_trainer.model:
            raise ValueError("Goal2Vec model must be trained first")
        
        candidate_statements = [p.formal_statement for p in self.problems]
        similar_goals = self.goal2vec_trainer.find_similar_goals(
            query_theorem, candidate_statements, top_k
        )
        
        # Map back to problems
        statement_to_problem = {p.formal_statement: p for p in self.problems}
        similar_problems = []
        
        for statement, similarity in similar_goals:
            if statement in statement_to_problem:
                similar_problems.append((statement_to_problem[statement], similarity))
        
        return similar_problems
    
    def get_tactic_recommendations(self, goal: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get tactic recommendations for a goal using Goal2Vec."""
        if not self.goal2vec_trainer.model:
            raise ValueError("Goal2Vec model must be trained first")
        
        # Common tactics from all environments
        all_tactics = set()
        for env in self.environments:
            all_tactics.update(t.tactic for t in env.available_tactics)
        
        return self.goal2vec_trainer.recommend_tactics(goal, list(all_tactics), top_k)
    
    def evaluate_system(self, test_problems: Optional[List[MathProblem]] = None, 
                       num_problems: int = 10) -> Dict[str, Any]:
        """
        Evaluate the integrated system on test problems.
        
        Args:
            test_problems: Specific problems to test (if None, use random subset)
            num_problems: Number of problems to test
            
        Returns:
            Evaluation results
        """
        self.logger.info("Evaluating system performance")
        
        if test_problems is None:
            # Use random subset of loaded problems
            import random
            test_problems = random.sample(self.problems, min(num_problems, len(self.problems)))
        
        results = []
        success_count = 0
        total_steps = 0
        total_reward = 0.0
        
        for problem in test_problems:
            try:
                result = self.prove_theorem(problem.formal_statement, max_steps=30)
                results.append(result)
                
                if result["success"]:
                    success_count += 1
                
                total_steps += result["proof_length"]
                total_reward += result["total_reward"]
                
            except Exception as e:
                self.logger.error(f"Error evaluating problem {problem.problem_id}: {e}")
                continue
        
        # Calculate statistics
        num_tested = len(results)
        success_rate = success_count / num_tested if num_tested > 0 else 0
        avg_steps = total_steps / num_tested if num_tested > 0 else 0
        avg_reward = total_reward / num_tested if num_tested > 0 else 0
        
        evaluation_results = {
            "num_problems_tested": num_tested,
            "success_count": success_count,
            "success_rate": success_rate,
            "avg_proof_length": avg_steps,
            "avg_reward": avg_reward,
            "individual_results": results
        }
        
        self.logger.info(f"Evaluation completed: {success_rate:.2%} success rate")
        return evaluation_results
    
    def save_system_state(self, filepath: str) -> None:
        """Save the current system state."""
        state = {
            "config": self.config,
            "num_problems": len(self.problems),
            "num_environments": len(self.environments),
            "is_trained": self.is_trained,
            "training_stats": self.training_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"System state saved to {filepath}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive system report."""
        report = []
        report.append("=" * 50)
        report.append("INTEGRATED THEOREM PROVER REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System configuration
        report.append("SYSTEM CONFIGURATION:")
        report.append(f"  Agent Type: {self.config['rl_training']['agent_type']}")
        report.append(f"  Embedding Dim: {self.config['embedding']['embedding_dim']}")
        report.append(f"  Max Episodes: {self.config['rl_training']['max_episodes']}")
        report.append("")
        
        # Data statistics
        report.append("DATA STATISTICS:")
        report.append(f"  Problems Loaded: {len(self.problems)}")
        report.append(f"  Environments Created: {len(self.environments)}")
        report.append(f"  Training Trees: {len(self.training_trees)}")
        
        # Problem breakdown by category
        if self.problems:
            categories = {}
            difficulties = {}
            for problem in self.problems:
                categories[problem.category] = categories.get(problem.category, 0) + 1
                difficulties[problem.difficulty.value] = difficulties.get(problem.difficulty.value, 0) + 1
            
            report.append("  Problem Categories:")
            for cat, count in categories.items():
                report.append(f"    {cat}: {count}")
            
            report.append("  Problem Difficulties:")
            for diff, count in difficulties.items():
                report.append(f"    {diff}: {count}")
        
        report.append("")
        
        # Training status
        report.append("TRAINING STATUS:")
        report.append(f"  System Trained: {self.is_trained}")
        if self.training_stats:
            report.append(f"  Episodes Completed: {self.training_stats.get('episodes', 0)}")
            report.append(f"  Success Rate: {self.training_stats.get('success_rate', 0):.2%}")
            report.append(f"  Avg Episode Length: {self.training_stats.get('avg_episode_length', 0):.1f}")
        report.append("")
        
        # Component status
        report.append("COMPONENT STATUS:")
        report.append(f"  Definition Database: {self.definition_db.get_statistics()['total_definitions']} definitions")
        report.append(f"  Goal2Vec Model: {'Trained' if self.goal2vec_trainer.model else 'Not trained'}")
        report.append(f"  RL Agent: {'Trained' if self.rl_agent else 'Not trained'}")
        report.append("")
        
        return "\n".join(report)


def main():
    """Main function for running the integrated system."""
    print("Starting Integrated Theorem Prover...")
    
    # Initialize system
    prover = IntegratedTheoremProver()
    
    # Load problems
    prover.load_problems(source="samples", limit=5)
    
    # Create environments
    prover.create_environments()
    
    # Train Goal2Vec (with minimal data)
    prover.train_goal2vec(use_existing_trees=False)
    
    # Train RL agent
    prover.train_rl_agent()
    
    # Test the system
    test_theorem = "∀ n : ℕ, n + 0 = n"
    result = prover.prove_theorem(test_theorem)
    
    print(f"\nProof attempt result:")
    print(f"Success: {result['success']}")
    print(f"Steps: {result['proof_length']}")
    print(f"Reward: {result['total_reward']:.2f}")
    
    if result['success']:
        print(f"Proof tree:\n{result['proof_tree']}")
    
    # Evaluate system
    evaluation = prover.evaluate_system(num_problems=3)
    print(f"\nSystem evaluation:")
    print(f"Success rate: {evaluation['success_rate']:.2%}")
    print(f"Average steps: {evaluation['avg_proof_length']:.1f}")
    
    # Generate report
    report = prover.generate_report()
    print("\n" + report)
    
    # Save system state
    prover.save_system_state("system_state.json")
    
    print("\nIntegrated Theorem Prover demo completed!")


if __name__ == "__main__":
    main()