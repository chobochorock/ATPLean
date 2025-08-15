"""
Lean interaction system using lean_interact for direct Lean communication.
Replaces manual parsing with direct Lean server interaction.
"""

import re
from typing import List, Dict, Optional, Union, Any
from lean_problem_structure import LeanProblemStructure, StateNode
from lean_interact import LeanREPLConfig, LeanServer, ProofStep, Command
from lean_interact.interface import LeanError


class GenericTree:
    """
    Generic tree structure for parsing various tree-like Lean expressions.
    Not tied to any specific implementation like MyTree.
    """
    
    def __init__(self, node_type: str, children: List['GenericTree'] = None):
        self.node_type = node_type  # 'leaf', 'branch', etc.
        self.children = children or []
    
    def __str__(self):
        if self.node_type == 'leaf':
            return 'leaf'
        elif self.node_type == 'branch':
            children_str = ', '.join(str(child) for child in self.children)
            return f'branch [{children_str}]'
        else:
            return f'{self.node_type}({", ".join(str(child) for child in self.children)})'
    
    def count_nodes(self) -> int:
        """Count total nodes in the tree."""
        return 1 + sum(child.count_nodes() for child in self.children)
    
    def count_edges(self) -> int:
        """Count total edges in the tree."""
        return len(self.children) + sum(child.count_edges() for child in self.children)
    
    def height(self) -> int:
        """Calculate tree height."""
        if not self.children:
            return 1
        return 1 + max(child.height() for child in self.children)
    
    @staticmethod
    def leaf():
        """Create a leaf node."""
        return GenericTree('leaf')
    
    @staticmethod
    def branch(children: List['GenericTree']):
        """Create a branch node with given children."""
        return GenericTree('branch', children)


class LeanTreeParser:
    """
    Parses Lean tree expressions and converts them to Python GenericTree objects.

    Handles expressions like:
    - leaf
    - branch [leaf, leaf]
    - branch [leaf, branch [leaf]]
    """

    def __init__(self):
        self.debug = False

    def parse_tree_expression(self, expression: str) -> Optional[GenericTree]:
        """
        Parse a Lean tree expression and return a GenericTree object.

        Args:
            expression: Lean tree expression like "branch [leaf, branch [leaf]]"

        Returns:
            GenericTree object or None if parsing fails
        """
        expression = expression.strip()

        if self.debug:
            print(f"Parsing: {expression}")

        # Handle leaf case
        if expression == "leaf" or expression.endswith(".leaf"):
            return GenericTree.leaf()

        # Handle branch case
        if expression.startswith("branch") or expression.endswith("branch"):
            return self._parse_branch_expression(expression)

        # Try to handle constructor patterns with any prefix
        if "." in expression and ("leaf" in expression or "branch" in expression):
            clean_expr = expression.split(".")[-1]
            return self.parse_tree_expression(clean_expr)

        print(f"Warning: Could not parse tree expression: {expression}")
        return None

    def _parse_branch_expression(self, expression: str) -> Optional[GenericTree]:
        """Parse a branch expression like 'branch [leaf, branch [leaf]]'."""
        # Extract the list part
        match = re.search(r"branch\s*\[(.*)\]", expression)
        if not match:
            print(f"Warning: Invalid branch expression: {expression}")
            return None

        list_content = match.group(1).strip()

        if not list_content:
            # Empty branch
            return GenericTree.branch([])

        # Parse the children
        children = self._parse_tree_list(list_content)
        if children is None:
            return None

        return GenericTree.branch(children)

    def _parse_tree_list(self, list_content: str) -> Optional[List[GenericTree]]:
        """Parse a comma-separated list of tree expressions."""
        if not list_content.strip():
            return []

        # Split by commas, but be careful about nested brackets
        elements = self._split_respecting_brackets(list_content)

        children = []
        for element in elements:
            child = self.parse_tree_expression(element.strip())
            if child is None:
                print(f"Warning: Failed to parse child: {element}")
                return None
            children.append(child)

        return children

    def _split_respecting_brackets(self, content: str) -> List[str]:
        """Split by commas while respecting bracket nesting."""
        elements = []
        current_element = ""
        bracket_count = 0

        for char in content:
            if char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
            elif char == "," and bracket_count == 0:
                elements.append(current_element)
                current_element = ""
                continue

            current_element += char

        if current_element.strip():
            elements.append(current_element)

        return elements


class LeanInteractParser:
    """
    Main parser using lean_interact for direct Lean server communication.
    Replaces file parsing with direct Lean interaction.
    """

    def __init__(self, config: Optional[LeanREPLConfig] = None):
        self.config = config or LeanREPLConfig(verbose=False)
        self.server = LeanServer(self.config)
        self.tree_parser = LeanTreeParser()
        self.problem_structure = LeanProblemStructure()
        self.current_env = 0

    def load_lean_file(self, file_path: str) -> bool:
        """Load and process a Lean file through lean_interact."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Send the entire file content to Lean server
            response = self.server.run(Command(cmd=content, env=self.current_env))
            
            if isinstance(response, LeanError):
                print(f"Error loading file: {response.message}")
                return False
            
            # Update environment
            self.current_env = response.env
            
            # Extract information from the response
            self._process_lean_response(response, content)
            return True
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return False

    def _process_lean_response(self, response, content: str) -> None:
        """Process Lean server response and extract information."""
        # Extract definitions and theorems from content using simple text processing
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('inductive '):
                self._process_inductive_line(line)
            elif line.startswith('def '):
                self._process_definition_line(line)
            elif line.startswith('theorem ') or line.startswith('lemma '):
                self._process_theorem_line(line)
        
        # Process any sorries (incomplete proofs) from the response
        if hasattr(response, 'sorries') and response.sorries:
            for sorry in response.sorries:
                self._process_sorry(sorry)
    
    def _process_inductive_line(self, line: str) -> None:
        """Process an inductive type definition line."""
        # Extract inductive name
        match = re.match(r'inductive\s+(\w+)', line)
        if match:
            name = match.group(1)
            self.problem_structure.add_definition(f"inductive {name}")
    
    def _process_definition_line(self, line: str) -> None:
        """Process a definition line."""
        # Extract definition
        if ':' in line:
            self.problem_structure.add_definition(line)
    
    def _process_theorem_line(self, line: str) -> None:
        """Process a theorem or lemma line."""
        # Extract theorem statement
        match = re.match(r'(theorem|lemma)\s+(\w+)([^:]*):([^:=]+)', line)
        if match:
            theorem_type, name, params, statement = match.groups()
            theorem_text = f"{name}{params.strip()}: {statement.strip()}"
            self.problem_structure.add_theorem(theorem_text)
            
            # Create root state node
            root_state = StateNode("theorem_root", f"{theorem_type.title()}: {name}")
            self.problem_structure.add_proof_state(root_state)
    
    def _process_sorry(self, sorry) -> None:
        """Process a sorry (incomplete proof) from Lean server."""
        goal_text = getattr(sorry, 'goal', 'Unknown goal')
        state_node = StateNode("sorry_state", goal_text)
        self.problem_structure.add_proof_state(state_node)

    def create_theorem(self, theorem_statement: str) -> Optional[Dict[str, Any]]:
        """Create a new theorem using lean_interact."""
        try:
            # Add theorem with sorry to get the goals
            theorem_cmd = f"theorem temp_theorem : {theorem_statement} := sorry"
            response = self.server.run(Command(cmd=theorem_cmd, env=self.current_env))
            
            if isinstance(response, LeanError):
                return {"error": response.message}
            
            # Extract goals from sorries
            goals = []
            if hasattr(response, 'sorries') and response.sorries:
                goals = [sorry.goal for sorry in response.sorries]
            
            return {
                "success": True,
                "env": response.env,
                "goals": goals,
                "sorries": response.sorries if hasattr(response, 'sorries') else []
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def apply_tactic(self, tactic: str, proof_state: int) -> Optional[Dict[str, Any]]:
        """Apply a tactic to a proof state."""
        try:
            response = self.server.run(ProofStep(tactic=tactic, proof_state=proof_state))
            
            if isinstance(response, LeanError):
                return {"error": response.message}
            
            return {
                "success": True,
                "proof_state": response.proof_state,
                "goals": response.goals,
                "proof_status": response.proof_status
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_expression(self, expression: str) -> Optional[Dict[str, Any]]:
        """Evaluate a Lean expression."""
        try:
            eval_cmd = f"#eval {expression}"
            response = self.server.run(Command(cmd=eval_cmd, env=self.current_env))
            
            if isinstance(response, LeanError):
                return {"error": response.message}
            
            return {
                "success": True,
                "result": response,
                "env": response.env
            }
            
        except Exception as e:
            return {"error": str(e)}

    def extract_trees_from_content(self, content: str) -> List[GenericTree]:
        """Extract all tree expressions from arbitrary content."""
        trees = []

        # Find tree expressions using regex patterns
        tree_patterns = [
            r'leaf',
            r'branch\s*\[[^\]]*\]'
        ]
        
        for pattern in tree_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                tree = self.tree_parser.parse_tree_expression(match.group())
                if tree:
                    trees.append(tree)

        return trees

    def get_problem_structure(self) -> LeanProblemStructure:
        """Get the parsed problem structure."""
        return self.problem_structure

    def validate_parsed_content(self) -> Dict[str, Any]:
        """Validate the parsed content against Lean theorems."""
        validation_results = {
            "trees_valid": True,
            "tree_count": len(self.problem_structure.trees),
            "theorem_count": len(self.problem_structure.theorems),
            "definition_count": len(self.problem_structure.definitions),
            "validation_errors": [],
        }

        # Validate each tree
        for i, tree in enumerate(self.problem_structure.trees):
            # Generic validation - check if vertex = edge + 1 (for connected trees)
            vertices = tree.count_nodes()
            edges = tree.count_edges()
            if vertices != edges + 1:
                validation_results["trees_valid"] = False
                validation_results["validation_errors"].append(
                    f"Tree {i}: vertex-edge relation fails (vertices: {vertices}, edges: {edges})"
                )

        return validation_results


class LeanExpressionEvaluator:
    """
    Evaluates Lean expressions using Python equivalents.
    Useful for testing and validation.
    """

    def __init__(self):
        self.tree_parser = LeanTreeParser()

    def evaluate_tree_function(self, func_name: str, tree_expr: str) -> Optional[int]:
        """
        Evaluate a tree function like num_of_vertex or num_of_edge.

        Args:
            func_name: Function name (num_of_vertex, num_of_edge)
            tree_expr: Tree expression to evaluate

        Returns:
            Function result or None if evaluation fails
        """
        tree = self.tree_parser.parse_tree_expression(tree_expr)
        if tree is None:
            return None

        if func_name == "num_of_vertex":
            return tree.count_nodes()
        elif func_name == "num_of_edge":
            return tree.count_edges()
        elif func_name == "height":
            return tree.height()
        else:
            print(f"Unknown function: {func_name}")
            return None

    def evaluate_eval_command(self, eval_expr: str) -> Optional[Any]:
        """Evaluate a #eval command."""
        # Parse expressions like "num_of_vertex (branch [leaf, branch [leaf]])"
        func_match = re.match(r"(\w+)\s*\((.*)\)", eval_expr.strip())
        if func_match:
            func_name = func_match.group(1)
            tree_expr = func_match.group(2)
            return self.evaluate_tree_function(func_name, tree_expr)

        print(f"Could not parse eval expression: {eval_expr}")
        return None


# Integration and testing functions
def load_lean_file_with_interact(file_path: str, config: Optional[LeanREPLConfig] = None) -> LeanProblemStructure:
    """
    Convenience function to load a complete Lean file using lean_interact.

    Args:
        file_path: Path to the Lean file
        config: Optional LeanREPLConfig

    Returns:
        LeanProblemStructure containing all processed content
    """
    parser = LeanInteractParser(config)
    if parser.load_lean_file(file_path):
        return parser.get_problem_structure()
    else:
        return LeanProblemStructure()  # Empty structure

# Backward compatibility alias
def parse_lean_file(file_path: str) -> LeanProblemStructure:
    """Backward compatibility function - now uses lean_interact."""
    return load_lean_file_with_interact(file_path)


if __name__ == "__main__":
    # Test the lean_interact parser
    print("Testing Lean Interact Parser...")

    # Test tree parser (still works as before)
    tree_parser = LeanTreeParser()
    tree_parser.debug = True

    print("=== Testing Tree Parser ===")
    test_expressions = [
        "leaf",
        "branch [leaf]",
        "branch [leaf, leaf]",
        "branch [leaf, branch [leaf]]",
        "MyTree.leaf",
        "MyTree.branch [MyTree.leaf, MyTree.branch [MyTree.leaf]]",
    ]

    for expr in test_expressions:
        print(f"\nParsing: {expr}")
        tree = tree_parser.parse_tree_expression(expr)
        if tree:
            print(f"Result: {tree}")
            print(f"Vertices: {tree.count_nodes()}, Edges: {tree.count_edges()}")
            vertices = tree.count_nodes()
            edges = tree.count_edges()
            print(f"Relation holds: {vertices == edges + 1}")
        else:
            print("Failed to parse")

    # Test lean_interact parser
    print("\n=== Testing LeanInteract Parser ===")
    try:
        config = LeanREPLConfig(verbose=True)
        parser = LeanInteractParser(config)
        
        # Test creating a simple theorem
        print("Testing theorem creation...")
        result = parser.create_theorem("∀ n : ℕ, n + 0 = n")
        if result and "success" in result:
            print(f"  Theorem created successfully")
            print(f"  Goals: {result.get('goals', [])}")
            
            # Test applying a tactic if we have sorries
            if result.get('sorries'):
                proof_state = result['sorries'][0].pos
                print(f"  Testing tactic application on proof state {proof_state}...")
                tactic_result = parser.apply_tactic("simp", proof_state)
                if tactic_result and "success" in tactic_result:
                    print(f"    Tactic result: {tactic_result['proof_status']}")
                else:
                    print(f"    Tactic failed: {tactic_result}")
        else:
            print(f"  Theorem creation failed: {result}")
            
        # Test loading a file if Basic.lean exists
        print("Testing file loading...")
        basic_file = "../Basic.lean"
        try:
            if parser.load_lean_file(basic_file):
                structure = parser.get_problem_structure()
                summary = structure.get_summary()
                print("File loaded successfully:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
            else:
                print("Failed to load Basic.lean")
        except Exception as e:
            print(f"Error loading file: {e}")
            
    except Exception as e:
        print(f"LeanInteract test failed: {e}")
        print("Make sure lean_interact is properly installed and configured")

    # Test expression evaluator (unchanged)
    print("\n=== Testing Expression Evaluator ===")
    evaluator = LeanExpressionEvaluator()

    test_evals = [
        "num_of_vertex (branch [leaf, branch [leaf]])",
        "num_of_edge (branch [leaf, branch [leaf]])",
    ]

    for eval_expr in test_evals:
        result = evaluator.evaluate_eval_command(eval_expr)
        print(f"Eval: {eval_expr} = {result}")

