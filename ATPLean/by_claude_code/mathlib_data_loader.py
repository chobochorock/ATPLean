#!/usr/bin/env python3
"""
Mathlib Data Loader: Extract mathematical content from Mathlib for Goal2Vec and RL training.

This module provides functionality to parse Lean 4 files from Mathlib and extract:
- Theorems and lemmas with their statements and proofs
- Definitions and inductive types
- Tactic usage patterns
- Goal-tactic pairs for training

Supports the full Mathlib library (~7000 files) for comprehensive training data.
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Union, Iterator
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import logging

from minif2f_processor import MathProblem, ProblemDifficulty
from definition_database import DefinitionDatabase, MathDefinition, DefinitionType


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LeanTheorem:
    """Represents a theorem or lemma from Lean."""
    name: str
    statement: str
    proof: Optional[str] = None
    file_path: str = ""
    line_number: int = 0
    theorem_type: str = "theorem"  # theorem, lemma, example, def
    namespace: str = ""
    dependencies: List[str] = field(default_factory=list)
    tactics_used: List[str] = field(default_factory=list)
    difficulty: ProblemDifficulty = ProblemDifficulty.MEDIUM


@dataclass
class LeanDefinition:
    """Represents a definition from Lean."""
    name: str
    definition_text: str
    definition_type: str  # def, inductive, structure, class
    file_path: str = ""
    line_number: int = 0
    namespace: str = ""
    parameters: List[str] = field(default_factory=list)


@dataclass
class TacticUsage:
    """Represents tactic usage in a specific context."""
    tactic: str
    goal_before: str
    goal_after: Optional[str] = None
    context: str = ""
    success: bool = True
    theorem_name: str = ""


@dataclass
class MathlibStats:
    """Statistics about Mathlib processing."""
    total_files: int = 0
    processed_files: int = 0
    total_theorems: int = 0
    total_definitions: int = 0
    total_tactics: int = 0
    processing_time: float = 0.0
    error_files: List[str] = field(default_factory=list)


class LeanParser:
    """Parser for Lean 4 files with focus on mathematical content."""
    
    def __init__(self):
        self.theorem_patterns = [
            r'^(theorem|lemma|example)\s+([a-zA-Z_][a-zA-Z0-9_\'\.]*)(\s*\[.*?\])?\s*(\([^:]*\))?\s*:\s*(.*?)(?:by|:=|\s*$)',
            r'^(theorem|lemma|example)\s+([a-zA-Z_][a-zA-Z0-9_\'\.]*)(\s*\{.*?\})?\s*(\([^:]*\))?\s*:\s*(.*?)(?:by|:=|\s*$)',
        ]
        
        self.definition_patterns = [
            r'^(def|inductive|structure|class)\s+([a-zA-Z_][a-zA-Z0-9_\'\.]*)(\s*\([^:]*\))?\s*:?\s*(.*?)(?::=|\s*where|\s*$)',
            r'^(def|inductive|structure|class)\s+([a-zA-Z_][a-zA-Z0-9_\'\.]*)(\s*\{.*?\})?\s*(\([^:]*\))?\s*:?\s*(.*?)(?::=|\s*where|\s*$)',
        ]
        
        self.tactic_patterns = [
            r'\b(simp|rw|ring|norm_num|linarith|omega|exact|apply|intro|intros|cases|induction|constructor|tauto|decide|assumption|rfl)\b',
            r'\b(unfold|fold|conv|abel|group|field_simp|norm_cast|push_neg|by_contra|contrapose)\b',
            r'\b(use|refine|obtain|have|suffices|wlog|rcases|rintro|ext|funext|congruence)\b',
            r'\b(library_search|aesop|polyrith|linear_combination|nlinarith|positivity)\b'
        ]
        
        self.namespace_pattern = r'^namespace\s+([a-zA-Z_][a-zA-Z0-9_\.]*)'
        self.import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)'
        
        # Compiled patterns for efficiency
        self.compiled_theorem_patterns = [re.compile(p, re.MULTILINE) for p in self.theorem_patterns]
        self.compiled_definition_patterns = [re.compile(p, re.MULTILINE) for p in self.definition_patterns]
        self.compiled_tactic_patterns = [re.compile(p) for p in self.tactic_patterns]
        self.compiled_namespace_pattern = re.compile(self.namespace_pattern, re.MULTILINE)
        self.compiled_import_pattern = re.compile(self.import_pattern, re.MULTILINE)
    
    def parse_file(self, file_path: Path) -> Tuple[List[LeanTheorem], List[LeanDefinition], List[TacticUsage]]:
        """Parse a single Lean file and extract mathematical content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            theorems = self._extract_theorems(content, str(file_path))
            definitions = self._extract_definitions(content, str(file_path))
            tactics = self._extract_tactic_usage(content, str(file_path))
            
            return theorems, definitions, tactics
            
        except Exception as e:
            logger.warning(f"Error parsing file {file_path}: {e}")
            return [], [], []
    
    def _extract_theorems(self, content: str, file_path: str) -> List[LeanTheorem]:
        """Extract theorems and lemmas from file content."""
        theorems = []
        lines = content.split('\n')
        current_namespace = self._get_current_namespace(content)
        
        for pattern in self.compiled_theorem_patterns:
            for match in pattern.finditer(content):
                theorem_type = match.group(1)
                name = match.group(2)
                statement = match.group(5) if match.group(5) else ""
                
                # Clean up statement
                statement = self._clean_statement(statement)
                
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                
                # Extract proof
                proof = self._extract_proof_after_match(content, match.end(), lines, line_num)
                
                # Extract tactics used
                tactics_used = self._extract_tactics_from_text(proof) if proof else []
                
                # Estimate difficulty
                difficulty = self._estimate_difficulty(statement, proof, tactics_used)
                
                theorem = LeanTheorem(
                    name=name,
                    statement=statement,
                    proof=proof,
                    file_path=file_path,
                    line_number=line_num,
                    theorem_type=theorem_type,
                    namespace=current_namespace,
                    tactics_used=tactics_used,
                    difficulty=difficulty
                )
                
                theorems.append(theorem)
        
        return theorems
    
    def _extract_definitions(self, content: str, file_path: str) -> List[LeanDefinition]:
        """Extract definitions from file content."""
        definitions = []
        current_namespace = self._get_current_namespace(content)
        
        for pattern in self.compiled_definition_patterns:
            for match in pattern.finditer(content):
                def_type = match.group(1)
                name = match.group(2)
                def_text = match.group(4) if match.group(4) else ""
                
                # Clean up definition text
                def_text = self._clean_statement(def_text)
                
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                
                definition = LeanDefinition(
                    name=name,
                    definition_text=def_text,
                    definition_type=def_type,
                    file_path=file_path,
                    line_number=line_num,
                    namespace=current_namespace
                )
                
                definitions.append(definition)
        
        return definitions
    
    def _extract_tactic_usage(self, content: str, file_path: str) -> List[TacticUsage]:
        """Extract tactic usage patterns from proofs."""
        tactics = []
        
        # Look for proof blocks
        proof_blocks = self._find_proof_blocks(content)
        
        for proof_text, context in proof_blocks:
            tactic_matches = []
            for pattern in self.compiled_tactic_patterns:
                tactic_matches.extend(pattern.findall(proof_text))
            
            # Flatten and deduplicate
            unique_tactics = list(set([t for sublist in tactic_matches for t in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]))
            
            for tactic in unique_tactics:
                tactic_usage = TacticUsage(
                    tactic=tactic,
                    goal_before="",  # Would need more sophisticated parsing
                    context=context,
                    theorem_name=self._extract_theorem_name_from_context(context)
                )
                tactics.append(tactic_usage)
        
        return tactics
    
    def _get_current_namespace(self, content: str) -> str:
        """Get the current namespace from file content."""
        namespaces = self.compiled_namespace_pattern.findall(content)
        return namespaces[-1] if namespaces else ""
    
    def _clean_statement(self, statement: str) -> str:
        """Clean and normalize statement text."""
        if not statement:
            return ""
        
        # Remove extra whitespace
        statement = ' '.join(statement.split())
        
        # Remove common prefixes/suffixes
        statement = statement.strip()
        
        return statement
    
    def _extract_proof_after_match(self, content: str, start_pos: int, lines: List[str], line_num: int) -> Optional[str]:
        """Extract proof text starting from a position."""
        # Simple heuristic: look for proof starting with 'by' or ':='
        remaining_text = content[start_pos:]
        
        # Look for 'by' keyword
        by_match = re.search(r'\bby\b', remaining_text)
        if by_match:
            proof_start = start_pos + by_match.start()
            # Extract until next theorem/definition or end of file
            proof_text = self._extract_proof_block(content, proof_start)
            return proof_text
        
        # Look for ':=' pattern
        assign_match = re.search(r':=', remaining_text)
        if assign_match:
            proof_start = start_pos + assign_match.start()
            proof_text = self._extract_proof_block(content, proof_start)
            return proof_text
        
        return None
    
    def _extract_proof_block(self, content: str, start_pos: int) -> str:
        """Extract a complete proof block using brace matching."""
        remaining = content[start_pos:]
        
        # Simple heuristic: extract until next top-level declaration
        end_patterns = [
            r'\n(theorem|lemma|def|example|inductive|structure|class)\s+',
            r'\nend\s+',
            r'\n\n'
        ]
        
        min_end = len(remaining)
        for pattern in end_patterns:
            match = re.search(pattern, remaining)
            if match:
                min_end = min(min_end, match.start())
        
        return remaining[:min_end].strip()
    
    def _find_proof_blocks(self, content: str) -> List[Tuple[str, str]]:
        """Find all proof blocks in content."""
        proof_blocks = []
        
        # Look for 'by' proofs
        by_matches = re.finditer(r'\bby\b', content)
        for match in by_matches:
            context_start = max(0, match.start() - 200)
            context = content[context_start:match.start()]
            proof = self._extract_proof_block(content, match.start())
            proof_blocks.append((proof, context))
        
        return proof_blocks
    
    def _extract_tactics_from_text(self, text: str) -> List[str]:
        """Extract tactics from proof text."""
        tactics = []
        for pattern in self.compiled_tactic_patterns:
            matches = pattern.findall(text)
            tactics.extend([m for m in matches if isinstance(m, str)])
        
        return list(set(tactics))  # Remove duplicates
    
    def _estimate_difficulty(self, statement: str, proof: Optional[str], tactics: List[str]) -> ProblemDifficulty:
        """Estimate problem difficulty based on statement and proof complexity."""
        if not statement:
            return ProblemDifficulty.EASY
        
        # Count complexity indicators
        complexity_score = 0
        
        # Statement complexity
        if len(statement) > 100:
            complexity_score += 1
        if '∀' in statement or '∃' in statement:
            complexity_score += 1
        if any(sym in statement for sym in ['→', '↔', '∧', '∨']):
            complexity_score += 1
        
        # Proof complexity
        if proof:
            if len(proof) > 100:
                complexity_score += 1
            if len(tactics) > 3:
                complexity_score += 1
            
            # Advanced tactics indicate higher difficulty
            advanced_tactics = ['induction', 'cases', 'contrapose', 'wlog', 'obtain']
            if any(tactic in tactics for tactic in advanced_tactics):
                complexity_score += 2
        
        # Classify based on score
        if complexity_score <= 1:
            return ProblemDifficulty.EASY
        elif complexity_score <= 3:
            return ProblemDifficulty.MEDIUM
        else:
            return ProblemDifficulty.HARD
    
    def _extract_theorem_name_from_context(self, context: str) -> str:
        """Extract theorem name from context."""
        # Look for theorem/lemma declarations in context
        for pattern in self.compiled_theorem_patterns:
            match = pattern.search(context)
            if match and match.group(2):
                return match.group(2)
        return ""


class MathlibDataLoader:
    """Main class for loading and processing Mathlib data."""
    
    def __init__(self, mathlib_path: str, output_db: str = "mathlib_data.db"):
        self.mathlib_path = Path(mathlib_path)
        self.output_db = output_db
        self.parser = LeanParser()
        self.stats = MathlibStats()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing extracted data."""
        conn = sqlite3.connect(self.output_db)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS theorems (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                statement TEXT,
                proof TEXT,
                file_path TEXT,
                line_number INTEGER,
                theorem_type TEXT,
                namespace TEXT,
                tactics_used TEXT,
                difficulty TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS definitions (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                definition_text TEXT,
                definition_type TEXT,
                file_path TEXT,
                line_number INTEGER,
                namespace TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tactic_usage (
                id INTEGER PRIMARY KEY,
                tactic TEXT,
                goal_before TEXT,
                goal_after TEXT,
                context TEXT,
                theorem_name TEXT,
                success BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY,
                total_files INTEGER,
                processed_files INTEGER,
                total_theorems INTEGER,
                total_definitions INTEGER,
                total_tactics INTEGER,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_theorems_name ON theorems(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_theorems_difficulty ON theorems(difficulty)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_definitions_name ON definitions(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tactic_usage_tactic ON tactic_usage(tactic)')
        
        conn.commit()
        conn.close()
    
    def find_lean_files(self) -> List[Path]:
        """Find all Lean files in Mathlib directory."""
        lean_files = []
        
        # Look in Mathlib directory
        mathlib_dir = self.mathlib_path / "Mathlib"
        if mathlib_dir.exists():
            lean_files.extend(mathlib_dir.glob("**/*.lean"))
        
        # Also check Archive directory for examples
        archive_dir = self.mathlib_path / "Archive"
        if archive_dir.exists():
            lean_files.extend(archive_dir.glob("**/*.lean"))
        
        # MathlibTest for additional examples
        test_dir = self.mathlib_path / "MathlibTest"
        if test_dir.exists():
            lean_files.extend(test_dir.glob("**/*.lean"))
        
        return sorted(lean_files)
    
    def process_file(self, file_path: Path) -> Tuple[int, int, int]:
        """Process a single file and return counts."""
        try:
            theorems, definitions, tactics = self.parser.parse_file(file_path)
            
            # Store in database
            self._store_theorems(theorems)
            self._store_definitions(definitions)
            self._store_tactics(tactics)
            
            return len(theorems), len(definitions), len(tactics)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.stats.error_files.append(str(file_path))
            return 0, 0, 0
    
    def process_all_files(self, max_workers: int = 4, batch_size: int = 100) -> MathlibStats:
        """Process all Lean files in Mathlib with parallel processing."""
        start_time = time.time()
        
        lean_files = self.find_lean_files()
        self.stats.total_files = len(lean_files)
        
        logger.info(f"Found {len(lean_files)} Lean files to process")
        
        # Process files in batches to avoid memory issues
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, len(lean_files), batch_size):
                batch_files = lean_files[i:i + batch_size]
                
                # Submit batch
                futures = [executor.submit(self.process_file, file_path) for file_path in batch_files]
                
                # Collect results
                for future, file_path in zip(futures, batch_files):
                    try:
                        theorem_count, def_count, tactic_count = future.result(timeout=30)
                        
                        self.stats.processed_files += 1
                        self.stats.total_theorems += theorem_count
                        self.stats.total_definitions += def_count
                        self.stats.total_tactics += tactic_count
                        
                        if self.stats.processed_files % 100 == 0:
                            logger.info(f"Processed {self.stats.processed_files}/{self.stats.total_files} files")
                    
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        self.stats.error_files.append(str(file_path))
        
        self.stats.processing_time = time.time() - start_time
        
        # Store final statistics
        self._store_stats()
        
        logger.info(f"Processing completed: {self.stats.processed_files} files, "
                   f"{self.stats.total_theorems} theorems, "
                   f"{self.stats.total_definitions} definitions, "
                   f"{self.stats.total_tactics} tactics")
        
        return self.stats
    
    def _store_theorems(self, theorems: List[LeanTheorem]):
        """Store theorems in database."""
        if not theorems:
            return
        
        conn = sqlite3.connect(self.output_db)
        cursor = conn.cursor()
        
        for theorem in theorems:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO theorems 
                    (name, statement, proof, file_path, line_number, theorem_type, namespace, tactics_used, difficulty)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    theorem.name,
                    theorem.statement,
                    theorem.proof,
                    theorem.file_path,
                    theorem.line_number,
                    theorem.theorem_type,
                    theorem.namespace,
                    json.dumps(theorem.tactics_used),
                    theorem.difficulty.value
                ))
            except sqlite3.Error as e:
                logger.warning(f"Error storing theorem {theorem.name}: {e}")
        
        conn.commit()
        conn.close()
    
    def _store_definitions(self, definitions: List[LeanDefinition]):
        """Store definitions in database."""
        if not definitions:
            return
        
        conn = sqlite3.connect(self.output_db)
        cursor = conn.cursor()
        
        for definition in definitions:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO definitions 
                    (name, definition_text, definition_type, file_path, line_number, namespace)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    definition.name,
                    definition.definition_text,
                    definition.definition_type,
                    definition.file_path,
                    definition.line_number,
                    definition.namespace
                ))
            except sqlite3.Error as e:
                logger.warning(f"Error storing definition {definition.name}: {e}")
        
        conn.commit()
        conn.close()
    
    def _store_tactics(self, tactics: List[TacticUsage]):
        """Store tactic usage in database."""
        if not tactics:
            return
        
        conn = sqlite3.connect(self.output_db)
        cursor = conn.cursor()
        
        for tactic in tactics:
            try:
                cursor.execute('''
                    INSERT INTO tactic_usage 
                    (tactic, goal_before, goal_after, context, theorem_name, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    tactic.tactic,
                    tactic.goal_before,
                    tactic.goal_after,
                    tactic.context,
                    tactic.theorem_name,
                    tactic.success
                ))
            except sqlite3.Error as e:
                logger.warning(f"Error storing tactic usage: {e}")
        
        conn.commit()
        conn.close()
    
    def _store_stats(self):
        """Store processing statistics."""
        conn = sqlite3.connect(self.output_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO processing_stats 
            (total_files, processed_files, total_theorems, total_definitions, total_tactics, processing_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            self.stats.total_files,
            self.stats.processed_files,
            self.stats.total_theorems,
            self.stats.total_definitions,
            self.stats.total_tactics,
            self.stats.processing_time
        ))
        
        conn.commit()
        conn.close()
    
    def get_training_problems(self, limit: Optional[int] = None, 
                            difficulty: Optional[ProblemDifficulty] = None) -> List[MathProblem]:
        """Extract training problems from processed data."""
        conn = sqlite3.connect(self.output_db)
        cursor = conn.cursor()
        
        query = "SELECT name, statement, proof, difficulty FROM theorems WHERE statement IS NOT NULL"
        params = []
        
        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty.value)
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        problems = []
        for name, statement, proof, diff_str in results:
            try:
                difficulty_enum = ProblemDifficulty(diff_str)
            except ValueError:
                difficulty_enum = ProblemDifficulty.MEDIUM
            
            problem = MathProblem(
                problem_id=name,
                statement=statement,
                proof=proof or "sorry",
                difficulty=difficulty_enum
            )
            problems.append(problem)
        
        conn.close()
        return problems
    
    def get_tactic_statistics(self) -> Dict[str, int]:
        """Get tactic usage statistics."""
        conn = sqlite3.connect(self.output_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT tactic, COUNT(*) FROM tactic_usage GROUP BY tactic ORDER BY COUNT(*) DESC")
        results = cursor.fetchall()
        
        conn.close()
        return dict(results)
    
    def export_to_json(self, output_file: str):
        """Export processed data to JSON format."""
        conn = sqlite3.connect(self.output_db)
        
        # Export theorems
        theorems_df = conn.execute("SELECT * FROM theorems").fetchall()
        definitions_df = conn.execute("SELECT * FROM definitions").fetchall()
        tactics_df = conn.execute("SELECT * FROM tactic_usage").fetchall()
        
        data = {
            "theorems": [dict(zip([desc[0] for desc in conn.execute("SELECT * FROM theorems").description], row)) 
                        for row in theorems_df],
            "definitions": [dict(zip([desc[0] for desc in conn.execute("SELECT * FROM definitions").description], row)) 
                           for row in definitions_df],
            "tactics": [dict(zip([desc[0] for desc in conn.execute("SELECT * FROM tactic_usage").description], row)) 
                       for row in tactics_df],
            "stats": dict(self.stats.__dict__)
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        conn.close()
        logger.info(f"Data exported to {output_file}")


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    mathlib_path = "/home/chorock/Projects/ATPLean/.lake/packages/mathlib"
    output_db = "mathlib_training_data.db"
    
    print("=== Mathlib Data Loader ===")
    
    # Initialize loader
    loader = MathlibDataLoader(mathlib_path, output_db)
    
    # Process files (start with small batch for testing)
    print("Processing Mathlib files...")
    stats = loader.process_all_files(max_workers=2, batch_size=50)
    
    print(f"Processing Statistics:")
    print(f"  Files processed: {stats.processed_files}/{stats.total_files}")
    print(f"  Theorems extracted: {stats.total_theorems}")
    print(f"  Definitions extracted: {stats.total_definitions}")
    print(f"  Tactic usages: {stats.total_tactics}")
    print(f"  Processing time: {stats.processing_time:.2f}s")
    print(f"  Error files: {len(stats.error_files)}")
    
    # Get training problems
    print("\nExtracting training problems...")
    problems = loader.get_training_problems(limit=1000)
    print(f"Extracted {len(problems)} training problems")
    
    # Show some examples
    if problems:
        print("\nExample problems:")
        for i, problem in enumerate(problems[:3]):
            print(f"{i+1}. {problem.problem_id}")
            print(f"   Statement: {problem.statement[:100]}...")
            print(f"   Proof: {problem.proof[:50] if problem.proof else 'None'}...")
            print(f"   Difficulty: {problem.difficulty}")
    
    # Tactic statistics
    tactic_stats = loader.get_tactic_statistics()
    print(f"\nMost common tactics:")
    for tactic, count in list(tactic_stats.items())[:10]:
        print(f"  {tactic}: {count}")
    
    # Export data
    loader.export_to_json("mathlib_data_export.json")
    
    print("Mathlib data loading completed!")