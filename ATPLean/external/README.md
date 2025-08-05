# Integrated Theorem Prover System

An automated theorem proving system using reinforcement learning and Goal2Vec embeddings for mathematical problem solving.

## System Architecture

### 1. Reinforcement Learning Tree Node Structure (`reinforcement_learning_node.py`)

**RLStateNode**: Enhanced state node for reinforcement learning
- `goal`: Current proof goal/state to be solved
- `tactic_action`: Tactic used to reach this node from parent
- `parent/children`: Tree relationship information
- Value estimation, visit counts, Q-values, and other RL attributes

**RLProofEnvironment**: RL environment for theorem proving
- State space and action space management  
- Reward calculation and episode management
- Tactic success simulation and proof tree construction

### 2. Lean Problem Parsing and Preprocessing System

**MinIF2FProcessor** (`minif2f_processor.py`): MinIF2F dataset processing
- Mathematical problem classification and difficulty assessment
- Automatic conversion to RL environments
- Category-specific tactic recommendations

**DefinitionDatabase** (`definition_database.py`): Mathematical definition database
- Storage and retrieval of mathematical definitions
- Dependency tracking between definitions
- Vector embeddings for word2vec comparison

### 3. Reinforcement Learning Foundation (`rl_agent.py`)

**ProofAgent**: Multi-algorithm RL agent support
- DQN, Actor-Critic, PPO, MCTS algorithms
- Neural networks for goal encoding
- Tactic selection and policy learning

**ProofTrainer**: Complete training pipeline
- Multi-environment parallel training
- Performance evaluation and model persistence
- Training statistics and progress tracking

### 4. Goal2Vec Model (`goal2vec_model.py`)

**Goal2VecModel**: Novel embedding model for goal-tactic relationships
- Semantic relationship learning between mathematical goals and tactics
- LSTM + Attention-based encoder architecture
- Similarity scoring and tactic recommendation

**Word2VecGoalComparator**: Comparison with traditional Word2Vec
- Performance benchmarking against standard word embeddings
- Mathematical text tokenization and vocabulary building

## Usage

### 1. Basic Setup

```python
from integrated_theorem_prover import IntegratedTheoremProver

# Initialize system
prover = IntegratedTheoremProver()

# Load problems (sample data)
prover.load_problems(source="samples", limit=10)

# Create RL environments
prover.create_environments()
```

### 2. Model Training

```python
# Train Goal2Vec embeddings
prover.train_goal2vec()

# Train reinforcement learning agent
prover.train_rl_agent()
```

### 3. Theorem Proving

```python
# Attempt to prove a theorem
theorem = "∀ n : ℕ, n + 0 = n"
result = prover.prove_theorem(theorem, max_steps=30, use_goal2vec=True)

print(f"Proof successful: {result['success']}")
print(f"Proof steps: {result['proof_length']}")
print(f"Total reward: {result['total_reward']}")

if result['success']:
    print(f"Proof tree:\n{result['proof_tree']}")
```

### 4. Similar Theorem Search

```python
# Find similar theorems using Goal2Vec
similar_theorems = prover.find_similar_theorems(
    "∀ n : ℕ, n + 0 = n", 
    top_k=5
)

for problem, similarity in similar_theorems:
    print(f"{similarity:.3f}: {problem.statement}")
```

### 5. Tactic Recommendations

```python
# Get tactic recommendations for a goal
recommendations = prover.get_tactic_recommendations(
    "∀ n : ℕ, n + 0 = n", 
    top_k=5
)

for tactic, score in recommendations:
    print(f"{tactic}: {score:.3f}")
```

### 6. System Evaluation

```python
# Evaluate system performance
evaluation = prover.evaluate_system(num_problems=10)
print(f"Success rate: {evaluation['success_rate']:.2%}")
print(f"Average steps: {evaluation['avg_proof_length']:.1f}")
```

## Individual Component Usage

### Direct RLStateNode Usage

```python
from reinforcement_learning_node import RLStateNode, TacticAction

# Create root node
root = RLStateNode("∀ n : ℕ, n + 0 = n")

# Create tactic action
tactic = TacticAction("simp", confidence=0.8)

# Add child node
child = root.add_rl_child("n + 0 = n", tactic)

# Visualize tree
print(root.visualize_rl_tree())
```

### Direct Goal2Vec Training

```python
from goal2vec_model import Goal2VecTrainer, EmbeddingConfig
from minif2f_processor import MathProblem

# Configuration
config = EmbeddingConfig(embedding_dim=128, epochs=50)
trainer = Goal2VecTrainer(config)

# Sample problems
problems = [
    MathProblem("test1", "∀ n : ℕ, n + 0 = n", proof="by simp"),
    MathProblem("test2", "∀ a b : ℕ, a + b = b + a", proof="by ring")
]

# Prepare training data and train
trainer.prepare_training_data(problems)
trainer.train()

# Save model
trainer.save_model("goal2vec_model.pth")
```

### Definition Database Usage

```python
from definition_database import DefinitionDatabase, MathDefinition, DefinitionType

# Initialize database
db = DefinitionDatabase("my_definitions.db")

# Add definition
definition = MathDefinition(
    name="MyTree",
    definition_type=DefinitionType.INDUCTIVE,
    formal_statement="inductive MyTree where | leaf : MyTree | branch : List MyTree → MyTree",
    informal_description="A tree data structure",
    category="data_structures"
)

db.add_definition(definition)

# Search definitions
results = db.search_definitions("tree", limit=5)
for def_obj, relevance in results:
    print(f"{def_obj.name}: {relevance:.2f}")
```

## Configuration File Format

Example `config.json` file:

```json
{
  "db_path": "theorem_prover.db",
  "minif2f_path": "/path/to/minif2f/dataset",
  "model_save_path": "models/",
  "data_save_path": "data/",
  "embedding": {
    "embedding_dim": 128,
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "rl_training": {
    "agent_type": "dqn",
    "learning_rate": 0.001,
    "max_episodes": 1000,
    "max_steps_per_episode": 50,
    "batch_size": 32,
    "memory_size": 10000
  },
  "logging": {
    "level": "INFO",
    "file": "theorem_prover.log"
  }
}
```

## Dependencies

### Python Packages (included in pyproject.toml)
- `torch` >= 1.9.0: Neural network implementation
- `numpy` >= 1.21.0: Numerical computations
- `scikit-learn` >= 1.0.0: Machine learning utilities
- `gensim` >= 4.0.0: Word2Vec implementation
- `matplotlib` >= 3.5.0: Visualization
- `lean-interact` >= 0.8.0: Lean interface

### System Requirements
- Python 3.9+
- Lean 4 (for proof verification)
- CUDA (optional, for GPU acceleration)

## File Structure

```
ATPLean/external/
├── reinforcement_learning_node.py    # RL node structure
├── rl_agent.py                       # RL agent and training
├── minif2f_processor.py              # MinIF2F data processing
├── definition_database.py            # Definition database
├── goal2vec_model.py                 # Goal2Vec embedding model
├── integrated_theorem_prover.py      # Integrated system
├── lean_problem_reader.py            # Lean file reading
├── lean_problem_parser.py            # Lean problem parsing
├── lean_problem_structure.py         # Basic structures (existing)
├── lean_expression_parser.py         # Mathematical expression parser
├── lean_problem_solver.py            # Lean integration (existing)
├── main.py                           # Main execution file
├── pyproject.toml                    # Dependency management
└── README.md                         # This file
```

## Key Features

1. **Integrated Workflow**: Complete pipeline from problem loading to proof generation
2. **Multiple RL Algorithms**: Support for DQN, Actor-Critic, PPO, and more
3. **Goal2Vec Embeddings**: Novel embedding model for goal-tactic relationships
4. **Extensible Architecture**: Easy addition of new datasets and tactics
5. **Performance Tracking**: Detailed training and evaluation metrics
6. **Modular Design**: Independent components that can be used separately

## Running Examples

```bash
# Run basic demo
cd ATPLean/external
python integrated_theorem_prover.py

# Test individual components
python reinforcement_learning_node.py
python goal2vec_model.py
python minif2f_processor.py
python definition_database.py
python rl_agent.py
```

## Implementation Status

✅ **Completed Components:**
1. **RL Tree Node Structure**: Complete with goal/tactic/parent-children relationships
2. **Lean Problem Parsing**: MinIF2F processor and general Lean file parsing
3. **RL Foundation**: Multi-algorithm agents with complete training pipeline
4. **Goal2Vec Model**: Novel embedding model with Word2Vec comparison
5. **Integrated System**: Main system connecting all components

## Future Improvements

1. **Real Lean Server Integration**: Currently simulated, needs actual Lean proof verification
2. **Additional Datasets**: Support for IMO, AMC, and other mathematical competitions
3. **Advanced RL Algorithms**: AlphaZero, MuZero, and other tree-search methods
4. **Distributed Training**: Scale to larger datasets with distributed processing
5. **Web Interface**: User-friendly GUI for interactive theorem proving

## Research Applications

This system provides a foundation for:
- Automated theorem proving research
- Mathematical reasoning with AI
- Goal-tactic relationship learning
- Reinforcement learning in formal mathematics
- Comparison studies between embedding methods