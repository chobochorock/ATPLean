"""
Definition Database for storing and managing mathematical definitions.
Supports word2vec comparison and retrieval of mathematical concepts.
"""

import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
from datetime import datetime

from lean_problem_reader import LeanReader
from lean_expression_parser import LeanExpressionParser


class DefinitionType(Enum):
    """Types of mathematical definitions."""
    INDUCTIVE = "inductive"
    FUNCTION = "function" 
    THEOREM = "theorem"
    AXIOM = "axiom"
    CONSTANT = "constant"
    STRUCTURE = "structure"
    CLASS = "class"
    INSTANCE = "instance"


@dataclass
class MathDefinition:
    """Represents a mathematical definition with metadata."""
    name: str
    definition_type: DefinitionType
    formal_statement: str
    informal_description: str = ""
    dependencies: List[str] = None
    parameters: List[str] = None
    return_type: str = ""
    proof_sketch: str = ""
    examples: List[str] = None
    category: str = "general"
    source_file: str = ""
    source_line: int = 0
    embedding: Optional[np.ndarray] = None
    created_at: str = ""
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = []
        if self.examples is None:
            self.examples = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def get_id(self) -> str:
        """Generate unique ID for this definition."""
        content = f"{self.name}:{self.formal_statement}:{self.definition_type.value}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['definition_type'] = self.definition_type.value
        data['embedding'] = self.embedding.tolist() if self.embedding is not None else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MathDefinition":
        """Create MathDefinition from dictionary."""
        # Handle enum conversion
        data['definition_type'] = DefinitionType(data['definition_type'])
        
        # Handle numpy array conversion
        if data.get('embedding') is not None:
            data['embedding'] = np.array(data['embedding'])
        
        return cls(**data)


class DefinitionDatabase:
    """
    Database for storing and retrieving mathematical definitions.
    Supports similarity search and dependency tracking.
    """
    
    def __init__(self, db_path: str = "math_definitions.db"):
        self.db_path = db_path
        self.expression_parser = LeanExpressionParser()
        self._init_database()
        
        # In-memory cache for fast access
        self._definition_cache: Dict[str, MathDefinition] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main definitions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS definitions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                definition_type TEXT NOT NULL,
                formal_statement TEXT NOT NULL,
                informal_description TEXT,
                return_type TEXT,
                proof_sketch TEXT,
                category TEXT,
                source_file TEXT,
                source_line INTEGER,
                embedding BLOB,
                created_at TEXT,
                UNIQUE(name, formal_statement)
            )
        ''')
        
        # Dependencies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dependencies (
                definition_id TEXT,
                dependency_name TEXT,
                FOREIGN KEY (definition_id) REFERENCES definitions (id),
                PRIMARY KEY (definition_id, dependency_name)
            )
        ''')
        
        # Parameters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameters (
                definition_id TEXT,
                parameter_name TEXT,
                parameter_type TEXT,
                position INTEGER,
                FOREIGN KEY (definition_id) REFERENCES definitions (id),
                PRIMARY KEY (definition_id, parameter_name)
            )
        ''')
        
        # Examples table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS examples (
                definition_id TEXT,
                example_text TEXT,
                example_type TEXT DEFAULT 'usage',
                FOREIGN KEY (definition_id) REFERENCES definitions (id)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_definitions_name ON definitions(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_definitions_type ON definitions(definition_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_definitions_category ON definitions(category)')
        
        conn.commit()
        conn.close()
    
    def add_definition(self, definition: MathDefinition) -> str:
        """
        Add a definition to the database.
        
        Args:
            definition: MathDefinition to add
            
        Returns:
            ID of the inserted definition
        """
        definition_id = definition.get_id()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert main definition
            embedding_bytes = definition.embedding.tobytes() if definition.embedding is not None else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO definitions 
                (id, name, definition_type, formal_statement, informal_description, 
                 return_type, proof_sketch, category, source_file, source_line, 
                 embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                definition_id, definition.name, definition.definition_type.value,
                definition.formal_statement, definition.informal_description,
                definition.return_type, definition.proof_sketch, definition.category,
                definition.source_file, definition.source_line, embedding_bytes,
                definition.created_at
            ))
            
            # Insert dependencies
            cursor.execute('DELETE FROM dependencies WHERE definition_id = ?', (definition_id,))
            for dep in definition.dependencies:
                cursor.execute(
                    'INSERT INTO dependencies (definition_id, dependency_name) VALUES (?, ?)',
                    (definition_id, dep)
                )
            
            # Insert parameters
            cursor.execute('DELETE FROM parameters WHERE definition_id = ?', (definition_id,))
            for i, param in enumerate(definition.parameters):
                param_name, param_type = param.split(':') if ':' in param else (param, '')
                cursor.execute(
                    'INSERT INTO parameters (definition_id, parameter_name, parameter_type, position) VALUES (?, ?, ?, ?)',
                    (definition_id, param_name.strip(), param_type.strip(), i)
                )
            
            # Insert examples
            cursor.execute('DELETE FROM examples WHERE definition_id = ?', (definition_id,))
            for example in definition.examples:
                cursor.execute(
                    'INSERT INTO examples (definition_id, example_text) VALUES (?, ?)',
                    (definition_id, example)
                )
            
            conn.commit()
            
            # Update cache
            self._definition_cache[definition_id] = definition
            if definition.embedding is not None:
                self._embedding_cache[definition_id] = definition.embedding
            
            return definition_id
            
        except sqlite3.Error as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_definition(self, definition_id: str) -> Optional[MathDefinition]:
        """Get definition by ID."""
        # Check cache first
        if definition_id in self._definition_cache:
            return self._definition_cache[definition_id]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get main definition
        cursor.execute('''
            SELECT id, name, definition_type, formal_statement, informal_description,
                   return_type, proof_sketch, category, source_file, source_line,
                   embedding, created_at
            FROM definitions WHERE id = ?
        ''', (definition_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        
        # Get dependencies
        cursor.execute(
            'SELECT dependency_name FROM dependencies WHERE definition_id = ?',
            (definition_id,)
        )
        dependencies = [row[0] for row in cursor.fetchall()]
        
        # Get parameters
        cursor.execute('''
            SELECT parameter_name, parameter_type 
            FROM parameters WHERE definition_id = ? 
            ORDER BY position
        ''', (definition_id,))
        parameters = [f"{name}:{ptype}" if ptype else name for name, ptype in cursor.fetchall()]
        
        # Get examples
        cursor.execute(
            'SELECT example_text FROM examples WHERE definition_id = ?',
            (definition_id,)
        )
        examples = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        # Reconstruct definition
        embedding = None
        if row[10]:  # embedding column
            embedding = np.frombuffer(row[10], dtype=np.float64)
        
        definition = MathDefinition(
            name=row[1],
            definition_type=DefinitionType(row[2]),
            formal_statement=row[3],
            informal_description=row[4] or "",
            dependencies=dependencies,
            parameters=parameters,
            return_type=row[5] or "",
            proof_sketch=row[6] or "",
            examples=examples,
            category=row[7] or "general",
            source_file=row[8] or "",
            source_line=row[9] or 0,
            embedding=embedding,
            created_at=row[11] or ""
        )
        
        # Cache the result
        self._definition_cache[definition_id] = definition
        if embedding is not None:
            self._embedding_cache[definition_id] = embedding
        
        return definition
    
    def search_definitions(self, query: str, limit: int = 10) -> List[Tuple[MathDefinition, float]]:
        """
        Search definitions by name or content.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of (definition, relevance_score) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple text search with relevance scoring
        cursor.execute('''
            SELECT id, name, formal_statement, informal_description,
                   (CASE 
                    WHEN name LIKE ? THEN 10
                    WHEN formal_statement LIKE ? THEN 5
                    WHEN informal_description LIKE ? THEN 3
                    ELSE 1
                    END) as relevance
            FROM definitions
            WHERE name LIKE ? OR formal_statement LIKE ? OR informal_description LIKE ?
            ORDER BY relevance DESC, name
            LIMIT ?
        ''', (f"%{query}%", f"%{query}%", f"%{query}%", 
              f"%{query}%", f"%{query}%", f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            definition = self.get_definition(row[0])
            if definition:
                relevance = row[4] / 10.0  # Normalize to 0-1
                results.append((definition, relevance))
        
        conn.close()
        return results
    
    def find_similar_definitions(self, embedding: np.ndarray, limit: int = 10, 
                               threshold: float = 0.7) -> List[Tuple[MathDefinition, float]]:
        """
        Find definitions similar to given embedding using cosine similarity.
        
        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of (definition, similarity_score) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all definitions with embeddings
        cursor.execute('SELECT id, embedding FROM definitions WHERE embedding IS NOT NULL')
        
        similarities = []
        for definition_id, embedding_bytes in cursor.fetchall():
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding, stored_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity >= threshold:
                similarities.append((definition_id, similarity))
        
        conn.close()
        
        # Sort by similarity and get definitions
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for definition_id, similarity in similarities[:limit]:
            definition = self.get_definition(definition_id)
            if definition:
                results.append((definition, similarity))
        
        return results
    
    def get_definitions_by_category(self, category: str) -> List[MathDefinition]:
        """Get all definitions in a specific category."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM definitions WHERE category = ?', (category,))
        definition_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return [self.get_definition(def_id) for def_id in definition_ids if self.get_definition(def_id)]
    
    def get_definition_dependencies(self, definition_id: str, recursive: bool = False) -> List[MathDefinition]:
        """
        Get dependencies for a definition.
        
        Args:
            definition_id: ID of the definition
            recursive: Whether to get transitive dependencies
            
        Returns:
            List of dependency definitions
        """
        if not recursive:
            definition = self.get_definition(definition_id)
            if not definition:
                return []
            
            dependencies = []
            for dep_name in definition.dependencies:
                # Find definition by name
                matches = self.search_definitions(dep_name, limit=1)
                if matches:
                    dependencies.append(matches[0][0])
            
            return dependencies
        
        # Recursive dependency resolution
        visited = set()
        dependencies = []
        
        def _get_deps_recursive(def_id: str):
            if def_id in visited:
                return
            visited.add(def_id)
            
            definition = self.get_definition(def_id)
            if not definition:
                return
            
            for dep_name in definition.dependencies:
                matches = self.search_definitions(dep_name, limit=1)
                if matches:
                    dep_def = matches[0][0]
                    dependencies.append(dep_def)
                    _get_deps_recursive(dep_def.get_id())
        
        _get_deps_recursive(definition_id)
        return dependencies
    
    def import_from_lean_file(self, file_path: str) -> List[str]:
        """
        Import definitions from a Lean file.
        
        Args:
            file_path: Path to Lean file
            
        Returns:
            List of imported definition IDs
        """
        reader = LeanReader(file_path)
        if not reader.read_file():
            raise ValueError(f"Could not read file: {file_path}")
        
        content = reader.get_all_content()
        imported_ids = []
        
        # Import inductive types
        for inductive in content.get("inductive_types", []):
            definition = self._create_definition_from_inductive(inductive, file_path)
            definition_id = self.add_definition(definition)
            imported_ids.append(definition_id)
        
        # Import functions
        for function in content.get("functions", []):
            definition = self._create_definition_from_function(function, file_path)
            definition_id = self.add_definition(definition)
            imported_ids.append(definition_id)
        
        # Import theorems
        for theorem in content.get("theorems", []):
            definition = self._create_definition_from_theorem(theorem, file_path)
            definition_id = self.add_definition(definition)
            imported_ids.append(definition_id)
        
        return imported_ids
    
    def _create_definition_from_inductive(self, inductive_data: Dict[str, Any], source_file: str) -> MathDefinition:
        """Create MathDefinition from inductive type data."""
        constructors = [c["name"] for c in inductive_data.get("constructors", [])]
        
        return MathDefinition(
            name=inductive_data["name"],
            definition_type=DefinitionType.INDUCTIVE,
            formal_statement=inductive_data.get("raw_definition", ""),
            informal_description=f"Inductive type with constructors: {', '.join(constructors)}",
            dependencies=self._extract_dependencies_from_text(inductive_data.get("raw_definition", "")),
            category="type_theory",
            source_file=source_file
        )
    
    def _create_definition_from_function(self, function_data: Dict[str, Any], source_file: str) -> MathDefinition:
        """Create MathDefinition from function data."""
        return MathDefinition(
            name=function_data["name"],
            definition_type=DefinitionType.FUNCTION,
            formal_statement=function_data.get("raw_definition", ""),
            informal_description=f"Function: {function_data['name']}",
            return_type=function_data.get("type", ""),
            dependencies=self._extract_dependencies_from_text(function_data.get("body", "")),
            category=self._categorize_function(function_data),
            source_file=source_file
        )
    
    def _create_definition_from_theorem(self, theorem_data: Dict[str, Any], source_file: str) -> MathDefinition:
        """Create MathDefinition from theorem data."""
        return MathDefinition(
            name=theorem_data["name"],
            definition_type=DefinitionType.THEOREM,
            formal_statement=theorem_data["statement"],
            informal_description=f"Theorem: {theorem_data['name']}",
            proof_sketch=theorem_data.get("proof", ""),
            parameters=self._extract_parameters(theorem_data.get("parameters", "")),
            dependencies=self._extract_dependencies_from_text(theorem_data["statement"]),
            category=self._categorize_theorem(theorem_data),
            source_file=source_file
        )
    
    def _extract_dependencies_from_text(self, text: str) -> List[str]:
        """Extract potential dependencies from text using pattern matching."""
        # Look for identifiers that might be dependencies
        dependencies = set()
        
        # Common patterns for Lean identifiers
        patterns = [
            r'\b([A-Z][a-zA-Z0-9_]*)\b',  # Type names
            r'\b([a-z][a-zA-Z0-9_]*)\b',  # Function/constant names
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            dependencies.update(matches)
        
        # Filter out common keywords and short names
        filtered_deps = []
        keywords = {'def', 'theorem', 'inductive', 'where', 'with', 'by', 'have', 'let', 'in', 'if', 'then', 'else'}
        
        for dep in dependencies:
            if len(dep) > 2 and dep.lower() not in keywords:
                filtered_deps.append(dep)
        
        return filtered_deps[:10]  # Limit to avoid noise
    
    def _extract_parameters(self, param_text: str) -> List[str]:
        """Extract parameters from parameter text."""
        if not param_text:
            return []
        
        # Simple parameter extraction
        params = []
        param_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^,)]+)'
        matches = re.findall(param_pattern, param_text)
        
        for name, ptype in matches:
            params.append(f"{name.strip()}:{ptype.strip()}")
        
        return params
    
    def _categorize_function(self, function_data: Dict[str, Any]) -> str:
        """Categorize function based on its definition."""
        name = function_data["name"].lower()
        body = function_data.get("body", "").lower()
        
        if "num" in name or "count" in name:
            return "counting"
        elif "vertex" in name or "edge" in name:
            return "graph_theory"
        elif "+" in body or "*" in body:
            return "arithmetic"
        else:
            return "general"
    
    def _categorize_theorem(self, theorem_data: Dict[str, Any]) -> str:
        """Categorize theorem based on its statement."""
        statement = theorem_data["statement"].lower()
        
        if "∀" in statement or "forall" in statement:
            return "universal"
        elif "=" in statement:
            return "equality"
        elif "tree" in statement:
            return "tree_theory"
        else:
            return "general"
    
    def export_to_json(self, filepath: str, category: Optional[str] = None) -> None:
        """Export definitions to JSON file."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute('SELECT id FROM definitions WHERE category = ?', (category,))
        else:
            cursor.execute('SELECT id FROM definitions')
        
        definition_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        definitions = []
        for def_id in definition_ids:
            definition = self.get_definition(def_id)
            if definition:
                definitions.append(definition.to_dict())
        
        with open(filepath, 'w') as f:
            json.dump(definitions, f, indent=2, default=str)
        
        print(f"Exported {len(definitions)} definitions to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic counts
        cursor.execute('SELECT COUNT(*) FROM definitions')
        total_definitions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM definitions WHERE embedding IS NOT NULL')
        definitions_with_embeddings = cursor.fetchone()[0]
        
        # By type
        cursor.execute('SELECT definition_type, COUNT(*) FROM definitions GROUP BY definition_type')
        by_type = dict(cursor.fetchall())
        
        # By category
        cursor.execute('SELECT category, COUNT(*) FROM definitions GROUP BY category')
        by_category = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_definitions": total_definitions,
            "definitions_with_embeddings": definitions_with_embeddings,
            "by_type": by_type,
            "by_category": by_category,
            "cache_size": len(self._definition_cache)
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== Definition Database Testing ===")
    
    # Initialize database
    db = DefinitionDatabase("/tmp/test_definitions.db")
    
    # Create sample definitions
    sample_definitions = [
        MathDefinition(
            name="MyTree",
            definition_type=DefinitionType.INDUCTIVE,
            formal_statement="inductive MyTree where | leaf : MyTree | branch : List MyTree → MyTree",
            informal_description="A tree data structure with leaves and branches",
            category="data_structures",
            examples=["leaf", "branch [leaf, leaf]"]
        ),
        MathDefinition(
            name="num_of_vertex",
            definition_type=DefinitionType.FUNCTION,
            formal_statement="def num_of_vertex : MyTree → ℕ",
            informal_description="Counts the number of vertices in a tree",
            dependencies=["MyTree"],
            return_type="ℕ",
            category="counting",
            examples=["num_of_vertex leaf = 1"]
        ),
        MathDefinition(
            name="vertex_eq_edge_plus_one",
            definition_type=DefinitionType.THEOREM,
            formal_statement="∀ t : MyTree, num_of_vertex t = num_of_edge t + 1",
            informal_description="In any tree, the number of vertices equals the number of edges plus one",
            dependencies=["MyTree", "num_of_vertex", "num_of_edge"],
            category="tree_theory",
            proof_sketch="by induction on tree structure"
        )
    ]
    
    # Add definitions to database
    print("Adding sample definitions...")
    for definition in sample_definitions:
        def_id = db.add_definition(definition)
        print(f"Added definition: {definition.name} (ID: {def_id})")
    
    # Test search functionality
    print("\n=== Testing Search ===")
    search_results = db.search_definitions("tree", limit=5)
    for definition, relevance in search_results:
        print(f"Found: {definition.name} (relevance: {relevance:.2f})")
    
    # Test category retrieval
    print("\n=== Testing Category Retrieval ===")
    tree_definitions = db.get_definitions_by_category("tree_theory")
    print(f"Tree theory definitions: {[d.name for d in tree_definitions]}")
    
    # Test dependency tracking
    print("\n=== Testing Dependencies ===")
    theorem_def = next((d for d in sample_definitions if d.name == "vertex_eq_edge_plus_one"), None)
    if theorem_def:
        theorem_id = theorem_def.get_id()
        deps = db.get_definition_dependencies(theorem_id)
        print(f"Dependencies for {theorem_def.name}: {[d.name for d in deps]}")
    
    # Show statistics
    print("\n=== Database Statistics ===")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test JSON export
    print("\n=== Testing Export ===")
    export_file = "/tmp/exported_definitions.json"
    db.export_to_json(export_file)
    
    print("Definition database testing completed!")