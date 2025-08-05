"""
MinIF2F Dataset Processor for Reinforcement Learning.
Processes MinIF2F mathematical problems and converts them to RL-compatible structures.
"""

import re
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from lean_problem_reader import LeanReader
from lean_problem_parser import LeanProblemParser
from reinforcement_learning_node import RLStateNode, TacticAction, RLProofEnvironment


class ProblemDifficulty(Enum):
    """Difficulty levels for mathematical problems."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


@dataclass
class MathProblem:
    """Represents a mathematical problem from MinIF2F or similar datasets."""
    problem_id: str
    statement: str
    informal_statement: str = ""
    formal_statement: str = ""
    proof: str = ""
    difficulty: ProblemDifficulty = ProblemDifficulty.MEDIUM
    category: str = "general"
    source: str = "minif2f"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MinIF2FProcessor:
    """
    Processor for MinIF2F dataset problems.
    Converts mathematical problems to RL-compatible training data.
    """
    
    def __init__(self, minif2f_path: Optional[str] = None):
        self.minif2f_path = Path(minif2f_path) if minif2f_path else None
        self.problems: List[MathProblem] = []
        self.category_patterns = self._initialize_category_patterns()
        self.difficulty_heuristics = self._initialize_difficulty_heuristics()
        
    def _initialize_category_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for categorizing problems."""
        return {
            "algebra": ["polynomial", "equation", "inequality", "sqrt", "^", "**"],
            "number_theory": ["prime", "divisible", "gcd", "lcm", "mod", "∣"],
            "geometry": ["triangle", "circle", "angle", "area", "perimeter", "distance"],
            "combinatorics": ["combination", "permutation", "choose", "factorial"],
            "calculus": ["derivative", "integral", "limit", "continuous", "differentiable"],
            "logic": ["∀", "∃", "→", "∧", "∨", "¬", "iff"],
            "set_theory": ["∈", "⊆", "∪", "∩", "∅", "complement"],
            "analysis": ["supremum", "infimum", "bounded", "convergent", "sequence"],
            "linear_algebra": ["matrix", "vector", "eigenvalue", "determinant", "rank"],
            "topology": ["open", "closed", "compact", "connected", "homeomorphism"]
        }
    
    def _initialize_difficulty_heuristics(self) -> List[Tuple[str, ProblemDifficulty, int]]:
        """Initialize heuristics for difficulty assessment."""
        return [
            # (keyword/pattern, difficulty, weight)
            ("induction", ProblemDifficulty.HARD, 3),
            ("∀.*∃", ProblemDifficulty.HARD, 3),  # Nested quantifiers
            ("supremum|infimum", ProblemDifficulty.HARD, 2),
            ("continuous", ProblemDifficulty.MEDIUM, 2),
            ("differentiable", ProblemDifficulty.HARD, 2),
            ("bijective|injective|surjective", ProblemDifficulty.MEDIUM, 2),
            ("prime", ProblemDifficulty.MEDIUM, 1),
            ("polynomial", ProblemDifficulty.EASY, 1),
            ("triangle", ProblemDifficulty.EASY, 1),
            ("sqrt", ProblemDifficulty.EASY, 1),
        ]
    
    def load_minif2f_problems(self, limit: Optional[int] = None) -> List[MathProblem]:
        """
        Load problems from MinIF2F dataset directory.
        
        Args:
            limit: Maximum number of problems to load
            
        Returns:
            List of loaded MathProblem objects
        """
        if not self.minif2f_path or not self.minif2f_path.exists():
            print("MinIF2F path not found, creating sample problems")
            return self._create_sample_problems()
        
        problems = []
        
        # Look for .lean files in the dataset
        lean_files = list(self.minif2f_path.rglob("*.lean"))
        
        for i, lean_file in enumerate(lean_files):
            if limit and i >= limit:
                break
                
            try:
                problem = self._parse_minif2f_file(lean_file)
                if problem:
                    problems.append(problem)
            except Exception as e:
                print(f"Error parsing {lean_file}: {e}")
                continue
        
        self.problems.extend(problems)
        return problems
    
    def _parse_minif2f_file(self, file_path: Path) -> Optional[MathProblem]:
        """Parse a single MinIF2F .lean file."""
        try:
            reader = LeanReader(str(file_path))
            if not reader.read_file():
                return None
            
            content = reader.get_all_content()
            
            # Extract problem ID from filename
            problem_id = file_path.stem
            
            # Look for theorem statements
            theorems = content.get("theorems", [])
            if not theorems:
                return None
            
            # Use the first theorem as the main problem
            main_theorem = theorems[0]
            
            # Extract informal statement from comments
            informal_statement = self._extract_informal_statement(reader.raw_content)
            
            # Categorize the problem
            category = self._categorize_problem(main_theorem["statement"])
            
            # Assess difficulty
            difficulty = self._assess_difficulty(main_theorem["statement"])
            
            problem = MathProblem(
                problem_id=problem_id,
                statement=main_theorem["statement"],
                informal_statement=informal_statement,
                formal_statement=main_theorem["statement"],
                proof=main_theorem.get("proof", ""),
                difficulty=difficulty,
                category=category,
                source="minif2f",
                metadata={
                    "file_path": str(file_path),
                    "num_theorems": len(theorems),
                    "has_proof": bool(main_theorem.get("proof")),
                    "theorem_name": main_theorem["name"]
                }
            )
            
            return problem
            
        except Exception as e:
            print(f"Error parsing MinIF2F file {file_path}: {e}")
            return None
    
    def _extract_informal_statement(self, content: str) -> str:
        """Extract informal problem statement from comments."""
        # Look for common comment patterns in MinIF2F
        patterns = [
            r'/\*\s*(.*?)\s*\*/',  # /* ... */
            r'--\s*Problem:\s*(.*?)(?=\n|$)',  # -- Problem: ...
            r'--\s*(.*?)(?=\n--|$)',  # -- ... 
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                # Return the first non-empty match
                for match in matches:
                    cleaned = re.sub(r'\s+', ' ', match.strip())
                    if len(cleaned) > 10:  # Minimum length check
                        return cleaned
        
        return ""
    
    def _categorize_problem(self, statement: str) -> str:
        """Categorize problem based on keywords in statement."""
        statement_lower = statement.lower()
        
        category_scores = {category: 0 for category in self.category_patterns}
        
        for category, keywords in self.category_patterns.items():
            for keyword in keywords:
                if keyword.lower() in statement_lower:
                    category_scores[category] += 1
        
        # Return category with highest score, or "general" if no matches
        best_category = max(category_scores.items(), key=lambda x: x[1])
        return best_category[0] if best_category[1] > 0 else "general"
    
    def _assess_difficulty(self, statement: str) -> ProblemDifficulty:
        """Assess problem difficulty using heuristics."""
        difficulty_score = 0
        
        for pattern, difficulty, weight in self.difficulty_heuristics:
            if re.search(pattern, statement, re.IGNORECASE):
                difficulty_score += weight * (difficulty.value == "hard" and 3 or 
                                            difficulty.value == "medium" and 2 or 1)
        
        # Also consider statement complexity
        complexity_indicators = [
            len(statement.split()) > 50,  # Long statement
            statement.count('∀') + statement.count('∃') > 2,  # Many quantifiers
            statement.count('(') > 5,  # Many parentheses
            '→' in statement and ('∀' in statement or '∃' in statement),  # Complex logic
        ]
        difficulty_score += sum(complexity_indicators)
        
        if difficulty_score >= 8:
            return ProblemDifficulty.VERY_HARD
        elif difficulty_score >= 5:
            return ProblemDifficulty.HARD
        elif difficulty_score >= 2:
            return ProblemDifficulty.MEDIUM
        else:
            return ProblemDifficulty.EASY
    
    def _create_sample_problems(self) -> List[MathProblem]:
        """Create sample problems for testing when MinIF2F is not available."""
        samples = [
            MathProblem(
                problem_id="sample_001",
                statement="∀ n : ℕ, n + 0 = n",
                informal_statement="For any natural number n, n plus 0 equals n",
                formal_statement="∀ n : ℕ, n + 0 = n",
                proof="by simp",
                difficulty=ProblemDifficulty.EASY,
                category="algebra",
                source="sample"
            ),
            MathProblem(
                problem_id="sample_002", 
                statement="∀ a b : ℕ, a + b = b + a",
                informal_statement="Addition of natural numbers is commutative",
                formal_statement="∀ a b : ℕ, a + b = b + a",
                proof="by ring",
                difficulty=ProblemDifficulty.EASY,
                category="algebra",
                source="sample"
            ),
            MathProblem(
                problem_id="sample_003",
                statement="∀ n : ℕ, n > 0 → ∃ m : ℕ, m * m = n ∨ (∃ k : ℕ, m * m < n ∧ n < (m + 1) * (m + 1))",
                informal_statement="For any positive natural number, either it's a perfect square or lies between two consecutive perfect squares",
                formal_statement="∀ n : ℕ, n > 0 → ∃ m : ℕ, m * m = n ∨ (∃ k : ℕ, m * m < n ∧ n < (m + 1) * (m + 1))",
                proof="sorry",
                difficulty=ProblemDifficulty.HARD,
                category="number_theory",
                source="sample"
            )
        ]
        return samples
    
    def convert_to_rl_environment(self, problem: MathProblem) -> RLProofEnvironment:
        """
        Convert a MathProblem to an RLProofEnvironment.
        
        Args:
            problem: MathProblem to convert
            
        Returns:
            RLProofEnvironment ready for training
        """
        env = RLProofEnvironment(problem.formal_statement)
        
        # Add problem metadata to root node
        env.root_node.context_features.update({
            "problem_id": problem.problem_id,
            "category": problem.category,
            "difficulty": problem.difficulty.value,
            "source": problem.source,
            "informal_statement": problem.informal_statement
        })
        
        # Customize available tactics based on problem category
        category_tactics = self._get_category_specific_tactics(problem.category)
        env.available_tactics.extend(category_tactics)
        
        # Adjust tactic confidences based on difficulty
        self._adjust_tactic_confidences(env.available_tactics, problem.difficulty)
        
        return env
    
    def _get_category_specific_tactics(self, category: str) -> List[TacticAction]:
        """Get tactics specific to problem category."""
        category_tactics = {
            "algebra": [
                TacticAction("ring", confidence=0.9),
                TacticAction("field_simp", confidence=0.7),
                TacticAction("norm_num", confidence=0.8),
                TacticAction("algebraize", confidence=0.6)
            ],
            "number_theory": [
                TacticAction("omega", confidence=0.8),
                TacticAction("norm_num", confidence=0.9),
                TacticAction("divisibility", confidence=0.6),
                TacticAction("mod_cases", confidence=0.5)
            ],
            "geometry": [
                TacticAction("angle_chase", confidence=0.6),
                TacticAction("coordinate_geometry", confidence=0.5),
                TacticAction("triangle_inequality", confidence=0.7)
            ],
            "logic": [
                TacticAction("tauto", confidence=0.9),
                TacticAction("classical", confidence=0.6),
                TacticAction("by_contra", confidence=0.7),
                TacticAction("push_neg", confidence=0.8)
            ],
            "analysis": [
                TacticAction("continuity", confidence=0.6),
                TacticAction("differentiability", confidence=0.5),
                TacticAction("squeeze_theorem", confidence=0.4),
                TacticAction("mean_value_theorem", confidence=0.4)
            ]
        }
        
        return category_tactics.get(category, [])
    
    def _adjust_tactic_confidences(self, tactics: List[TacticAction], difficulty: ProblemDifficulty) -> None:
        """Adjust tactic confidences based on problem difficulty."""
        difficulty_multipliers = {
            ProblemDifficulty.EASY: 1.2,
            ProblemDifficulty.MEDIUM: 1.0,
            ProblemDifficulty.HARD: 0.8,
            ProblemDifficulty.VERY_HARD: 0.6
        }
        
        multiplier = difficulty_multipliers[difficulty]
        
        for tactic in tactics:
            tactic.confidence = min(1.0, tactic.confidence * multiplier)
    
    def create_training_dataset(self, problems: List[MathProblem], 
                              episodes_per_problem: int = 10) -> List[Dict[str, Any]]:
        """
        Create training dataset from problems.
        
        Args:
            problems: List of problems to convert
            episodes_per_problem: Number of training episodes per problem
            
        Returns:
            List of training examples
        """
        training_examples = []
        
        for problem in problems:
            print(f"Processing problem {problem.problem_id} ({problem.category})")
            
            env = self.convert_to_rl_environment(problem)
            
            # Generate training episodes
            for episode in range(episodes_per_problem):
                episode_examples = self._generate_episode_data(env, max_steps=20)
                training_examples.extend(episode_examples)
                
                # Reset environment for next episode
                env.reset()
        
        return training_examples
    
    def _generate_episode_data(self, env: RLProofEnvironment, max_steps: int = 20) -> List[Dict[str, Any]]:
        """Generate training data from one episode of interaction."""
        episode_data = []
        state = env.reset()
        
        for step in range(max_steps):
            # Get available actions
            available_actions = env.get_available_actions()
            if not available_actions:
                break
            
            # Select action (random for data generation)
            action = np.random.choice(available_actions)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Create training example
            example = {
                "state": state.copy(),
                "action": action.to_dict(),
                "reward": reward,
                "next_state": next_state.copy(),
                "done": done,
                "step": step,
                "episode_info": info
            }
            episode_data.append(example)
            
            state = next_state
            
            if done:
                break
        
        return episode_data
    
    def save_processed_dataset(self, dataset: List[Dict[str, Any]], filepath: str) -> None:
        """Save processed dataset to JSON file."""
        output_data = {
            "metadata": {
                "num_examples": len(dataset),
                "num_problems": len(self.problems),
                "categories": list(set(p.category for p in self.problems)),
                "difficulties": [p.difficulty.value for p in self.problems]
            },
            "problems": [
                {
                    "problem_id": p.problem_id,
                    "statement": p.statement,
                    "category": p.category,
                    "difficulty": p.difficulty.value,
                    "source": p.source
                } for p in self.problems
            ],
            "training_data": dataset
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Dataset saved to {filepath}")
        print(f"Total examples: {len(dataset)}")
        print(f"Problems by category: {dict(zip(*np.unique([p.category for p in self.problems], return_counts=True)))}")
    
    def load_processed_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """Load processed dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct problems
        self.problems = []
        for p_data in data["problems"]:
            problem = MathProblem(
                problem_id=p_data["problem_id"],
                statement=p_data["statement"],
                category=p_data["category"],
                difficulty=ProblemDifficulty(p_data["difficulty"]),
                source=p_data["source"]
            )
            self.problems.append(problem)
        
        return data["training_data"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded problems."""
        if not self.problems:
            return {"message": "No problems loaded"}
        
        categories = [p.category for p in self.problems]
        difficulties = [p.difficulty.value for p in self.problems]
        sources = [p.source for p in self.problems]
        
        return {
            "total_problems": len(self.problems),
            "categories": dict(zip(*np.unique(categories, return_counts=True))),
            "difficulties": dict(zip(*np.unique(difficulties, return_counts=True))),
            "sources": dict(zip(*np.unique(sources, return_counts=True))),
            "avg_statement_length": np.mean([len(p.statement) for p in self.problems]),
            "problems_with_proof": sum(1 for p in self.problems if p.proof and p.proof != "sorry")
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== MinIF2F Processor Testing ===")
    
    # Initialize processor
    processor = MinIF2FProcessor()
    
    # Load sample problems (since we don't have actual MinIF2F dataset)
    print("Loading sample problems...")
    problems = processor.load_minif2f_problems(limit=5)
    
    print(f"Loaded {len(problems)} problems")
    
    # Show statistics
    stats = processor.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test problem conversion to RL environment
    print(f"\n=== Testing Problem Conversion ===")
    test_problem = problems[0]
    print(f"Converting problem: {test_problem.problem_id}")
    print(f"Statement: {test_problem.statement}")
    print(f"Category: {test_problem.category}")
    print(f"Difficulty: {test_problem.difficulty.value}")
    
    # Convert to RL environment
    env = processor.convert_to_rl_environment(test_problem)
    print(f"Available tactics: {[t.tactic for t in env.available_tactics[:10]]}")
    
    # Test training dataset creation
    print(f"\n=== Creating Training Dataset ===")
    training_data = processor.create_training_dataset(problems[:2], episodes_per_problem=3)
    print(f"Generated {len(training_data)} training examples")
    
    # Save dataset
    output_file = "/tmp/minif2f_training_data.json"
    processor.save_processed_dataset(training_data, output_file)
    
    # Show sample training example
    if training_data:
        print("\nSample training example:")
        example = training_data[0]
        print(f"  State goal: {example['state'].get('goal', 'N/A')}")
        print(f"  Action: {example['action']['tactic']}")
        print(f"  Reward: {example['reward']}")
        print(f"  Done: {example['done']}")