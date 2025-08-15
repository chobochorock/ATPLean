# Mathlib-Enhanced Goal2Vec and RL Training System

This system provides comprehensive training capabilities for mathematical theorem proving using **Mathlib** data (~7000 Lean files) to train both **Goal2Vec** embeddings and **Reinforcement Learning** agents.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mathlib Training Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│  📁 .lake/packages/mathlib (7000+ Lean files)                  │
│           ↓                                                     │
│  🔍 MathlibDataLoader: Extract theorems, proofs, tactics       │
│           ↓                                                     │
│  🧠 MathlibGoal2VecTrainer: Train embeddings on math content   │
│           ↓                                                     │
│  🤖 MathlibRLTrainer: Train RL agents with Goal2Vec guidance   │
│           ↓                                                     │
│  📊 Integrated System: Complete theorem proving pipeline       │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Basic Training (Quick Mode)
```bash
cd ATPLean/external/
python train_mathlib_system.py --quick
```

### 2. Full Training Pipeline
```bash
cd ATPLean/external/
python train_mathlib_system.py \
    --mathlib-path /path/to/.lake/packages/mathlib \
    --output-dir ./mathlib_trained_models
```

### 3. Custom Configuration
```bash
# Create config.json with your settings
python train_mathlib_system.py --config config.json
```

## 📋 Training Components

### 1. **MathlibDataLoader** (`mathlib_data_loader.py`)
- **Purpose**: Extract mathematical content from Mathlib files
- **Features**:
  - Parallel processing of 7000+ Lean files
  - Extract theorems, definitions, proofs, and tactic usage
  - SQLite database storage for efficient access
  - Difficulty estimation and categorization
  - Comprehensive error handling and logging

**Key Outputs**:
- `mathlib_training_data.db`: SQLite database with extracted content
- `mathlib_data_export.json`: JSON export for analysis
- Processing statistics and error reports

### 2. **MathlibGoal2VecTrainer** (`mathlib_goal2vec_trainer.py`)
- **Purpose**: Train enhanced Goal2Vec embeddings on mathematical content
- **Features**:
  - Enhanced tokenizer for mathematical symbols and Lean syntax
  - Curriculum learning (easy → medium → hard problems)
  - Attention mechanisms and LSTM encoders
  - Confidence-based tactic recommendations
  - Integration with Mathlib-specific tactic vocabulary

**Key Outputs**:
- `mathlib_goal2vec.pth`: Trained Goal2Vec model
- Enhanced mathematical tokenizer with Mathlib vocabulary
- Tactic recommendation system with explanations

### 3. **MathlibRLTrainer** (`mathlib_rl_trainer.py`)
- **Purpose**: Train RL agents for theorem proving with Mathlib knowledge
- **Features**:
  - Enhanced proof environments with Goal2Vec integration
  - Curriculum learning based on theorem difficulty
  - Mathlib-aware reward systems
  - Tactic success tracking and adaptation
  - Multi-algorithm RL support (DQN, A3C, PPO, MCTS)

**Key Outputs**:
- `mathlib_rl_agent.pth`: Trained RL agent
- Proof search metrics and performance statistics
- Tactic usage analytics

### 4. **Complete Training Pipeline** (`train_mathlib_system.py`)
- **Purpose**: Orchestrate end-to-end training process
- **Features**:
  - Five-phase training pipeline
  - Comprehensive logging and monitoring
  - System integration and evaluation
  - Configuration management
  - Results analysis and export

## 📊 Training Phases

### Phase 1: Data Extraction 🔍
```
Input:  .lake/packages/mathlib (7000+ files)
Process: Parse Lean files → Extract math content → Store in database
Output: mathlib_training_data.db (theorems, definitions, tactics)
Time:   ~30-60 minutes (parallel processing)
```

### Phase 2: Goal2Vec Training 🧠
```
Input:  Extracted mathematical content
Process: Tokenize → Build vocab → Train embeddings → Evaluate
Output: mathlib_goal2vec.pth (trained embeddings)
Time:   ~2-4 hours (100 epochs, curriculum learning)
```

### Phase 3: RL Training 🤖
```
Input:  Goal2Vec model + Mathlib problems
Process: Create environments → Train agents → Evaluate performance
Output: mathlib_rl_agent.pth (trained RL agent)
Time:   ~3-6 hours (1000 episodes, curriculum learning)
```

### Phase 4: System Evaluation 📊
```
Input:  Trained models + test problems
Process: Integrate systems → Run evaluations → Generate metrics
Output: Performance metrics and system config
Time:   ~10-30 minutes
```

### Phase 5: System Packaging 💾
```
Input:  All trained components
Process: Package models → Create configs → Generate documentation
Output: Complete deployable system
Time:   ~5 minutes
```

## ⚙️ Configuration

### Default Configuration
```json
{
  "mathlib_path": "/path/to/.lake/packages/mathlib",
  "output_dir": "./mathlib_training_output",
  "extraction": {
    "max_workers": 4,
    "batch_size": 100
  },
  "goal2vec": {
    "embedding_dim": 256,
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "rl_training": {
    "episodes": 1000,
    "max_problems": 1000,
    "curriculum_learning": true,
    "learning_rate": 0.001
  }
}
```

### Quick Mode Settings
- Reduced epochs (20 vs 100)
- Smaller embedding dimension (128 vs 256)
- Fewer training episodes (200 vs 1000)
- Limited problem set (100 vs 1000)

## 📈 Expected Performance

### Data Extraction
- **Files processed**: ~7000 Lean files
- **Theorems extracted**: ~15,000-25,000
- **Definitions extracted**: ~5,000-10,000
- **Tactics identified**: ~50,000-100,000 usage patterns
- **Processing time**: 30-60 minutes (4 workers)

### Goal2Vec Training
- **Vocabulary size**: ~10,000-20,000 mathematical tokens
- **Training pairs**: ~50,000-200,000 goal-tactic pairs
- **Test accuracy**: 60-80% (tactic recommendation accuracy)
- **Training time**: 2-4 hours (100 epochs)

### RL Training
- **Success rate**: 40-70% (theorem completion rate)
- **Average proof length**: 5-15 steps
- **Training episodes**: 1000 (curriculum learning)
- **Training time**: 3-6 hours

## 🔧 Advanced Usage

### Individual Component Training
```python
# Train only Goal2Vec
from mathlib_goal2vec_trainer import MathlibGoal2VecTrainer, EmbeddingConfig

config = EmbeddingConfig(embedding_dim=256, epochs=50)
trainer = MathlibGoal2VecTrainer(config)
trainer.load_mathlib_data()
trainer.prepare_mathlib_training_data()
trainer.train_with_mathlib()
trainer.save_enhanced_model("goal2vec_model.pth")
```

```python
# Train only RL Agent
from mathlib_rl_trainer import MathlibProofTrainer, AgentConfig

config = AgentConfig(learning_rate=0.001, batch_size=32)
trainer = MathlibProofTrainer(config)
trainer.load_mathlib_problems("mathlib_data.db", max_problems=500)
trainer.train_on_mathlib(episodes=500)
trainer.save_trained_agent("rl_agent.pth")
```

### Custom Data Extraction
```python
from mathlib_data_loader import MathlibDataLoader

loader = MathlibDataLoader("/path/to/mathlib", "custom_data.db")
stats = loader.process_all_files(max_workers=8, batch_size=200)
problems = loader.get_training_problems(limit=2000, difficulty=ProblemDifficulty.HARD)
```

## 📝 Output Files

After training completion, the output directory contains:

```
mathlib_training_output/
├── mathlib_training_data.db        # Extracted Mathlib content
├── mathlib_data_export.json        # Human-readable data export
├── mathlib_goal2vec.pth            # Trained Goal2Vec model
├── mathlib_rl_agent.pth            # Trained RL agent
├── system_config.json              # System configuration
├── training_results.json           # Comprehensive training metrics
├── system_info.json                # System package information
└── mathlib_training.log            # Detailed training logs
```

## 🔬 System Integration

### Loading Trained System
```python
from integrated_theorem_prover import IntegratedTheoremProver

# Load complete system
prover = IntegratedTheoremProver.load_from_config("system_config.json")

# Prove theorems
result = prover.prove_theorem("∀ n : ℕ, n + 0 = n")
print(f"Proof found: {result.proof_found}")
print(f"Proof steps: {result.proof_steps}")

# Get tactic recommendations
recommendations = prover.recommend_tactics("∀ a b : ℕ, a + b = b + a")
for tactic, confidence in recommendations:
    print(f"{tactic}: {confidence:.3f}")
```

### Custom Problem Solving
```python
# Solve custom problems
custom_goals = [
    "∀ x : ℝ, x * 0 = 0",
    "∀ P Q : Prop, P ∧ Q → Q ∧ P",
    "∀ n m : ℕ, n + m = m + n"
]

for goal in custom_goals:
    result = prover.prove_theorem(goal, max_steps=20)
    if result.proof_found:
        print(f"✅ Proved: {goal}")
        print(f"   Steps: {len(result.proof_steps)}")
        print(f"   Tactics: {[step.tactic for step in result.proof_steps]}")
    else:
        print(f"❌ Failed: {goal}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce `batch_size` in configuration
   - Use `--quick` mode for initial testing
   - Increase system RAM or use smaller datasets

2. **Long Training Times**
   - Use `--quick` mode for testing
   - Reduce `epochs` and `episodes` in configuration
   - Increase `max_workers` for parallel processing

3. **Mathlib Path Issues**
   - Ensure Mathlib is built: `lake build` in ATPLean directory
   - Verify path: `/path/to/.lake/packages/mathlib` should contain `Mathlib/` directory
   - Check file permissions for read access

4. **CUDA/GPU Issues**
   - Training automatically detects and uses GPU if available
   - For CPU-only training, models will automatically use CPU
   - Monitor GPU memory usage with `nvidia-smi`

### Performance Optimization

```python
# For faster development/testing
config = {
    "goal2vec": {"epochs": 10, "embedding_dim": 64},
    "rl_training": {"episodes": 100, "max_problems": 50},
    "extraction": {"max_workers": 8, "batch_size": 200}
}
```

## 📚 Research Applications

This system enables research in:

1. **Mathematical Reasoning**: Large-scale analysis of mathematical proof patterns
2. **Automated Theorem Proving**: AI-assisted formal verification
3. **Educational Technology**: Intelligent tutoring for mathematics
4. **Proof Mining**: Discovery of new mathematical insights from existing proofs
5. **Neural-Symbolic Integration**: Combining neural networks with formal methods

## 🤝 Contributing

To extend the system:

1. **Add New Tactic Types**: Extend `MathlibTokenizer` and tactic vocabulary
2. **Improve Parsing**: Enhance `LeanParser` for better content extraction
3. **Custom Reward Functions**: Modify reward calculation in `MathlibProofEnvironment`
4. **New RL Algorithms**: Implement additional RL methods in `MathlibProofAgent`
5. **Evaluation Metrics**: Add domain-specific evaluation in training pipeline

## 📖 Citation

If you use this system in research, please cite:

```bibtex
@software{mathlib_goal2vec_rl,
  title={Mathlib-Enhanced Goal2Vec and RL Training System},
  author={ATPLean Project},
  year={2024},
  url={https://github.com/your-repo/ATPLean}
}
```

## 📄 License

This project is released under the same license as the ATPLean project.

---

**🎯 Happy Training!** For questions or issues, please check the logs in `mathlib_training.log` and refer to the troubleshooting section above.