#!/usr/bin/env python3
"""
Comprehensive Mathlib Training System: End-to-end pipeline for training Goal2Vec and RL on Mathlib.

This script coordinates the complete training pipeline:
1. Extract mathematical content from Mathlib (~7000 files)
2. Train Goal2Vec embeddings on mathematical goals and tactics
3. Train RL agents for theorem proving with Mathlib-derived knowledge
4. Evaluate and save the complete system

Usage:
    python train_mathlib_system.py [--mathlib-path PATH] [--output-dir DIR] [--config CONFIG]
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from mathlib_data_loader import MathlibDataLoader
from mathlib_goal2vec_trainer import MathlibGoal2VecTrainer, EmbeddingConfig
from mathlib_rl_trainer import MathlibProofTrainer, TrainingConfig
from integrated_theorem_prover import IntegratedTheoremProver


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mathlib_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MathlibTrainingPipeline:
    """Complete training pipeline for Mathlib-based mathematical reasoning system."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mathlib_path = config["mathlib_path"]
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(exist_ok=True)

        # Training components
        self.data_loader = None
        self.goal2vec_trainer = None
        self.rl_trainer = None

        # Training results
        self.results = {
            "data_extraction": {},
            "goal2vec_training": {},
            "rl_training": {},
            "evaluation": {},
            "total_time": 0,
        }

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("MATHLIB TRAINING PIPELINE STARTED")
        logger.info("=" * 60)

        try:
            # Phase 1: Data Extraction
            logger.info("\n🔍 PHASE 1: EXTRACTING MATHLIB DATA")
            self._extract_mathlib_data()

            # Phase 2: Goal2Vec Training
            logger.info("\n🧠 PHASE 2: TRAINING GOAL2VEC MODEL")
            self._train_goal2vec()

            # Phase 3: RL Training
            logger.info("\n🤖 PHASE 3: TRAINING RL AGENT")
            self._train_rl_agent()

            # Phase 4: System Integration and Evaluation
            logger.info("\n📊 PHASE 4: SYSTEM EVALUATION")
            self._evaluate_system()

            # Phase 5: Save Complete System
            logger.info("\n💾 PHASE 5: SAVING TRAINED SYSTEM")
            self._save_system()

            self.results["total_time"] = time.time() - start_time

            logger.info("\n" + "=" * 60)
            logger.info("MATHLIB TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(
                f"Total training time: {self.results['total_time']:.2f} seconds"
            )
            logger.info("=" * 60)

            return self.results

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _extract_mathlib_data(self) -> None:
        """Extract mathematical content from Mathlib files."""
        logger.info(f"Extracting data from Mathlib at: {self.mathlib_path}")

        # Initialize data loader
        db_path = str(self.output_dir / "mathlib_training_data.db")
        self.data_loader = MathlibDataLoader(self.mathlib_path, db_path)

        # Configure extraction parameters
        max_workers = self.config.get("extraction", {}).get("max_workers", 4)
        batch_size = self.config.get("extraction", {}).get("batch_size", 100)

        # Extract data
        extraction_start = time.time()
        stats = self.data_loader.process_all_files(
            max_workers=max_workers, batch_size=batch_size
        )
        extraction_time = time.time() - extraction_start

        # Store results
        self.results["data_extraction"] = {
            "files_processed": stats.processed_files,
            "total_files": stats.total_files,
            "theorems_extracted": stats.total_theorems,
            "definitions_extracted": stats.total_definitions,
            "tactics_extracted": stats.total_tactics,
            "extraction_time": extraction_time,
            "error_files": len(stats.error_files),
            "success_rate": stats.processed_files / max(stats.total_files, 1),
        }

        logger.info(f"✅ Data extraction completed:")
        logger.info(
            f"   📁 Files processed: {stats.processed_files}/{stats.total_files}"
        )
        logger.info(f"   📚 Theorems: {stats.total_theorems}")
        logger.info(f"   📝 Definitions: {stats.total_definitions}")
        logger.info(f"   🔧 Tactics: {stats.total_tactics}")
        logger.info(f"   ⏱️  Time: {extraction_time:.2f}s")

        # Export data for analysis
        export_path = str(self.output_dir / "mathlib_data_export.json")
        self.data_loader.export_to_json(export_path)
        logger.info(f"   📄 Data exported to: {export_path}")

    def _train_goal2vec(self) -> None:
        """Train Goal2Vec model on extracted Mathlib data."""
        logger.info("Training Goal2Vec model on Mathlib data...")

        # Load configuration
        goal2vec_config = self.config.get("goal2vec", {})
        embedding_config = EmbeddingConfig(
            embedding_dim=goal2vec_config.get("embedding_dim", 256),
            epochs=goal2vec_config.get("epochs", 100),
            batch_size=goal2vec_config.get("batch_size", 32),
            learning_rate=goal2vec_config.get("learning_rate", 0.001),
            window_size=goal2vec_config.get("window_size", 5),
            min_count=goal2vec_config.get("min_count", 2),
        )

        # Initialize trainer
        db_path = str(self.output_dir / "mathlib_training_data.db")
        self.goal2vec_trainer = MathlibGoal2VecTrainer(embedding_config, db_path)

        # Load and prepare data
        data_prep_start = time.time()
        self.goal2vec_trainer.load_mathlib_data()
        self.goal2vec_trainer.prepare_mathlib_training_data()
        data_prep_time = time.time() - data_prep_start

        # Train model
        training_start = time.time()
        self.goal2vec_trainer.train_with_mathlib(epochs=embedding_config.epochs)
        training_time = time.time() - training_start

        # Evaluate model
        eval_start = time.time()
        metrics = self.goal2vec_trainer.evaluate_on_mathlib_test()
        eval_time = time.time() - eval_start

        # Get training statistics
        stats = self.goal2vec_trainer.get_mathlib_statistics()

        # Store results
        self.results["goal2vec_training"] = {
            "data_preparation_time": data_prep_time,
            "training_time": training_time,
            "evaluation_time": eval_time,
            "total_time": data_prep_time + training_time + eval_time,
            "training_pairs": stats.get("training_pairs", 0),
            "vocabulary_size": stats.get("vocabulary_size", 0),
            "final_train_loss": stats.get("final_train_loss", 0),
            "final_val_loss": stats.get("final_val_loss", 0),
            "test_accuracy": metrics.get("accuracy", 0),
            "mathlib_theorems": stats.get("theorems_processed", 0),
            "mathlib_tactics": stats.get("tactics_learned", 0),
        }

        logger.info(f"✅ Goal2Vec training completed:")
        logger.info(f"   📚 Training pairs: {stats.get('training_pairs', 0)}")
        logger.info(f"   📖 Vocabulary size: {stats.get('vocabulary_size', 0)}")
        logger.info(f"   📈 Test accuracy: {metrics.get('accuracy', 0):.3f}")
        logger.info(f"   ⏱️  Training time: {training_time:.2f}s")

        # Save Goal2Vec model
        model_path = str(self.output_dir / "mathlib_goal2vec.pth")
        self.goal2vec_trainer.save_enhanced_model(model_path)
        logger.info(f"   💾 Model saved to: {model_path}")

    def _train_rl_agent(self) -> None:
        """Train RL agent with Mathlib knowledge."""
        logger.info("Training RL agent with Mathlib integration...")

        # Load configuration
        rl_config = self.config.get("rl_training", {})
        agent_config = TrainingConfig(
            learning_rate=rl_config.get("learning_rate", 0.001),
            batch_size=rl_config.get("batch_size", 32),
            gamma=rl_config.get("gamma", 0.99),
            epsilon_start=rl_config.get("epsilon_start", 0.9),
            epsilon_end=rl_config.get("epsilon_end", 0.1),
            epsilon_decay=rl_config.get("epsilon_decay", 0.995),
        )

        # Initialize RL trainer
        self.rl_trainer = MathlibProofTrainer(agent_config, self.goal2vec_trainer)

        # Load Mathlib problems
        db_path = str(self.output_dir / "mathlib_training_data.db")
        max_problems = rl_config.get("max_problems", 1000)
        self.rl_trainer.load_mathlib_problems(db_path, max_problems)

        # Train RL agent
        training_start = time.time()
        episodes = rl_config.get("episodes", 1000)
        curriculum_learning = rl_config.get("curriculum_learning", True)

        rl_metrics = self.rl_trainer.train_on_mathlib(
            episodes=episodes, curriculum_learning=curriculum_learning
        )
        training_time = time.time() - training_start

        # Store results
        self.results["rl_training"] = {
            "training_time": training_time,
            "training_episodes": episodes,
            "max_problems_used": max_problems,
            "curriculum_learning": curriculum_learning,
            **rl_metrics,
        }

        logger.info(f"✅ RL training completed:")
        logger.info(f"   🎯 Episodes: {episodes}")
        logger.info(f"   📊 Success rate: {rl_metrics.get('success_rate', 0):.3f}")
        logger.info(
            f"   📏 Avg proof length: {rl_metrics.get('average_proof_length', 0):.1f}"
        )
        logger.info(f"   ⏱️  Training time: {training_time:.2f}s")

        # Save RL agent
        agent_path = str(self.output_dir / "mathlib_rl_agent.pth")
        self.rl_trainer.save_trained_agent(agent_path)
        logger.info(f"   💾 Agent saved to: {agent_path}")

    def _evaluate_system(self) -> None:
        """Evaluate the complete integrated system."""
        logger.info("Evaluating integrated theorem proving system...")

        # Create integrated system
        config_path = str(self.output_dir / "system_config.json")
        system_config = {
            "goal2vec_model_path": str(self.output_dir / "mathlib_goal2vec.pth"),
            "rl_agent_path": str(self.output_dir / "mathlib_rl_agent.pth"),
            "mathlib_db_path": str(self.output_dir / "mathlib_training_data.db"),
            "max_proof_steps": 50,
            "beam_search_width": 5,
        }

        with open(config_path, "w") as f:
            json.dump(system_config, f, indent=2)

        # Test problems for evaluation
        test_problems = [
            "∀ n : ℕ, n + 0 = n",
            "∀ a b : ℕ, a + b = b + a",
            "∀ x : ℝ, x * 0 = 0",
            "∀ P Q : Prop, P ∧ Q → Q ∧ P",
            "∀ n : ℕ, 0 ≤ n",
        ]

        eval_start = time.time()

        # Evaluate Goal2Vec recommendations
        goal2vec_results = []
        for problem in test_problems:
            try:
                recommendations = (
                    self.goal2vec_trainer.recommend_tactics_with_confidence(
                        problem, top_k=5
                    )
                )
                goal2vec_results.append(
                    {
                        "problem": problem,
                        "recommendations": [
                            (tactic, float(conf), exp)
                            for tactic, conf, exp in recommendations
                        ],
                    }
                )
            except Exception as e:
                logger.warning(f"Error evaluating problem '{problem}': {e}")

        eval_time = time.time() - eval_start

        # Store evaluation results
        self.results["evaluation"] = {
            "evaluation_time": eval_time,
            "test_problems": len(test_problems),
            "goal2vec_results": goal2vec_results,
            "system_config_path": config_path,
        }

        logger.info(f"✅ System evaluation completed:")
        logger.info(f"   🧪 Test problems: {len(test_problems)}")
        logger.info(f"   ⏱️  Evaluation time: {eval_time:.2f}s")
        logger.info(f"   ⚙️  Config saved to: {config_path}")

        # Log sample recommendations
        if goal2vec_results:
            sample_result = goal2vec_results[0]
            logger.info(
                f"   📝 Sample recommendations for '{sample_result['problem']}':"
            )
            for i, (tactic, conf, exp) in enumerate(
                sample_result["recommendations"][:3]
            ):
                logger.info(f"      {i+1}. {tactic}: {conf:.3f} - {exp}")

    def _save_system(self) -> None:
        """Save the complete trained system."""
        logger.info("Saving complete system and results...")

        # Save training results
        results_path = str(self.output_dir / "training_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Create system package info
        system_info = {
            "system_name": "Mathlib-Enhanced Theorem Prover",
            "version": "1.0.0",
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "components": {
                "goal2vec_model": "mathlib_goal2vec.pth",
                "rl_agent": "mathlib_rl_agent.pth",
                "training_data": "mathlib_training_data.db",
                "data_export": "mathlib_data_export.json",
                "system_config": "system_config.json",
                "training_results": "training_results.json",
            },
            "training_statistics": self.results,
            "usage_instructions": {
                "load_system": "Use IntegratedTheoremProver.load_from_config('system_config.json')",
                "prove_theorem": "prover.prove_theorem('your theorem statement')",
                "get_recommendations": "prover.recommend_tactics('your goal')",
            },
        }

        system_info_path = str(self.output_dir / "system_info.json")
        with open(system_info_path, "w") as f:
            json.dump(system_info, f, indent=2, default=str)

        logger.info(f"✅ System saved successfully:")
        logger.info(f"   📊 Results: {results_path}")
        logger.info(f"   ℹ️  System info: {system_info_path}")
        logger.info(f"   📁 All files in: {self.output_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    if Path(config_path).exists():  # 뒤쪽 부분은 에러를 제거하기 위한 임시방편
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            "mathlib_path": "/home/chorock/Projects/ATPLean/.lake/packages/mathlib",
            "output_dir": "./mathlib_training_output",
            "extraction": {"max_workers": 4, "batch_size": 100},
            "goal2vec": {
                "embedding_dim": 256,
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "window_size": 5,
                "min_count": 2,
            },
            "rl_training": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "gamma": 0.99,
                "epsilon_start": 0.9,
                "epsilon_end": 0.1,
                "epsilon_decay": 0.995,
                "episodes": 1000,
                "max_problems": 1000,
                "curriculum_learning": True,
            },
        }


def main():
    """Main entry point for Mathlib training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train Goal2Vec and RL on Mathlib data"
    )
    parser.add_argument(
        "--mathlib-path",
        type=str,
        default="/home/chorock/Projects/ATPLean/.lake/packages/mathlib",
        help="Path to Mathlib directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mathlib_training_output",
        help="Output directory for trained models",
    )
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode with reduced parameters",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config(
            "None"
        )  # Use default config, ""은 항상 True를 내놓는 것 같아서 None

    # Override with command line arguments
    config["mathlib_path"] = args.mathlib_path
    config["output_dir"] = args.output_dir

    # Quick mode adjustments
    if args.quick:
        config["extraction"]["batch_size"] = 50
        config["goal2vec"]["epochs"] = 20
        config["goal2vec"]["embedding_dim"] = 128
        config["rl_training"]["episodes"] = 200
        config["rl_training"]["max_problems"] = 100
        logger.info("🚀 Quick training mode enabled")

    # Log configuration
    logger.info("Training configuration:")
    for section, settings in config.items():
        if isinstance(settings, dict):
            logger.info(f"  {section}:")
            for key, value in settings.items():
                logger.info(f"    {key}: {value}")
        else:
            logger.info(f"  {section}: {settings}")

    # Run training pipeline
    pipeline = MathlibTrainingPipeline(config)
    results = pipeline.run_complete_pipeline()

    # Print final summary
    print("\n" + "=" * 80)
    print("🎉 MATHLIB TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📁 Output directory: {config['output_dir']}")
    print(f"⏱️  Total time: {results['total_time']:.2f} seconds")
    print(
        f"📚 Theorems extracted: {results['data_extraction'].get('theorems_extracted', 0)}"
    )
    print(
        f"🧠 Goal2Vec accuracy: {results['goal2vec_training'].get('test_accuracy', 0):.3f}"
    )
    print(f"🤖 RL success rate: {results['rl_training'].get('success_rate', 0):.3f}")
    print("=" * 80)
    print("\n💡 Usage:")
    print("   from integrated_theorem_prover import IntegratedTheoremProver")
    print(
        f"   prover = IntegratedTheoremProver.load_from_config('{config['output_dir']}/system_config.json')"
    )
    print("   result = prover.prove_theorem('∀ n : ℕ, n + 0 = n')")
    print("=" * 80)


if __name__ == "__main__":
    main()

