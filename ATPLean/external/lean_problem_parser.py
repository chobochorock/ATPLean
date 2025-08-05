"""
Parser for converting Lean mathematical expressions to Python structures.
Converts Lean tree expressions to Python MyTree objects and validates theorems.
"""

import re
from typing import List, Dict, Optional, Union, Any
from lean_problem_structure import LeanProblemStructure, StateNode
from lean_problem_reader import LeanReader


class LeanTreeParser:
    """
    Parses Lean tree expressions and converts them to Python MyTree objects.

    Handles expressions like:
    - leaf
    - branch [leaf, leaf]
    - branch [leaf, branch [leaf]]
    """

    def __init__(self):
        self.debug = False

    def parse_tree_expression(self, expression: str) -> Optional[MyTree]:
        """
        Parse a Lean tree expression and return a MyTree object.

        Args:
            expression: Lean tree expression like "branch [leaf, branch [leaf]]"

        Returns:
            MyTree object or None if parsing fails
        """
        expression = expression.strip()

        if self.debug:
            print(f"Parsing: {expression}")

        # Handle leaf case
        if expression == "leaf" or expression == "MyTree.leaf":
            return MyTree.leaf()

        # Handle branch case
        if expression.startswith("branch") or expression.startswith("MyTree.branch"):
            return self._parse_branch_expression(expression)

        # Try to handle constructor patterns
        if "MyTree." in expression:
            clean_expr = expression.replace("MyTree.", "")
            return self.parse_tree_expression(clean_expr)

        print(f"Warning: Could not parse tree expression: {expression}")
        return None

    def _parse_branch_expression(self, expression: str) -> Optional[MyTree]:
        """Parse a branch expression like 'branch [leaf, branch [leaf]]'."""
        # Extract the list part
        match = re.search(r"branch\s*\[(.*)\]", expression)
        if not match:
            print(f"Warning: Invalid branch expression: {expression}")
            return None

        list_content = match.group(1).strip()

        if not list_content:
            # Empty branch
            return MyTree.branch([])

        # Parse the children
        children = self._parse_tree_list(list_content)
        if children is None:
            return None

        return MyTree.branch(children)

    def _parse_tree_list(self, list_content: str) -> Optional[List[MyTree]]:
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


class LeanProblemParser:
    """
    Main parser that coordinates between reader and structure conversion.
    Converts entire Lean problems to Python structures.
    """

    def __init__(self, file_path: str):
        self.reader = LeanReader(file_path)
        self.tree_parser = LeanTreeParser()
        self.problem_structure = LeanProblemStructure()

    def parse_file(self) -> bool:
        """Parse the entire Lean file."""
        if not self.reader.read_file():
            return False

        # Extract all content
        content = self.reader.get_all_content()

        # Parse inductive types
        self._parse_inductive_types(content["inductive_types"])

        # Parse functions
        self._parse_functions(content["functions"])

        # Parse theorems
        self._parse_theorems(content["theorems"])

        # Parse eval commands to find example trees
        self._parse_eval_commands(content["eval_commands"])

        return True

    def _parse_inductive_types(self, inductives: List[Dict[str, Any]]) -> None:
        """Parse inductive type definitions."""
        for inductive in inductives:
            self.problem_structure.add_definition(
                f"inductive {inductive['name']} with constructors: "
                + ", ".join(c["name"] for c in inductive["constructors"])
            )

    def _parse_functions(self, functions: List[Dict[str, Any]]) -> None:
        """Parse function definitions."""
        for func in functions:
            self.problem_structure.add_definition(f"def {func['name']}: {func['type']}")

    def _parse_theorems(self, theorems: List[Dict[str, Any]]) -> None:
        """Parse theorem statements."""
        for theorem in theorems:
            theorem_text = f"{theorem['name']}: {theorem['statement']}"
            self.problem_structure.add_theorem(theorem_text)

            # Create proof state tree for this theorem
            root_state = StateNode("theorem_root", f"Theorem: {theorem['name']}")
            self.problem_structure.add_proof_state(root_state)

            # Parse proof structure if available
            if theorem["proof"]:
                self._parse_proof_structure(theorem["proof"], root_state)

    def _parse_proof_structure(self, proof: str, parent_state: StateNode) -> None:
        """Parse proof structure and create state nodes."""
        # Look for proof tactics and cases
        lines = proof.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("--"):
                continue

            # Create state nodes for different proof steps
            if line.startswith("match") or line.startswith("| "):
                case_state = StateNode(f"case_{i}", line, parent_state)
            elif line.startswith("by "):
                tactic_state = StateNode(f"tactic_{i}", line, parent_state)
            elif "simp" in line or "rw" in line or "ring" in line:
                step_state = StateNode(f"step_{i}", line, parent_state)

    def _parse_eval_commands(self, eval_commands: List[Dict[str, str]]) -> None:
        """Parse eval commands to extract example trees."""
        for eval_cmd in eval_commands:
            expression = eval_cmd["expression"]

            # Look for tree expressions in eval commands
            if "branch" in expression or "leaf" in expression:
                tree = self.tree_parser.parse_tree_expression(expression)
                if tree:
                    self.problem_structure.add_tree(tree)

    def extract_trees_from_content(self, content: str) -> List[MyTree]:
        """Extract all tree expressions from arbitrary content."""
        trees = []

        # Find all tree expressions
        tree_expressions = self.reader.find_tree_structures()

        for expr in tree_expressions:
            tree = self.tree_parser.parse_tree_expression(expr)
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
            if not tree.verify_vertex_edge_relation():
                validation_results["trees_valid"] = False
                validation_results["validation_errors"].append(
                    f"Tree {i}: vertex-edge relation fails"
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
            return tree.num_of_vertex()
        elif func_name == "num_of_edge":
            return tree.num_of_edge()
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
def parse_lean_file(file_path: str) -> LeanProblemStructure:
    """
    Convenience function to parse a complete Lean file.

    Args:
        file_path: Path to the Lean file

    Returns:
        LeanProblemStructure containing all parsed content
    """
    parser = LeanProblemParser(file_path)
    if parser.parse_file():
        return parser.get_problem_structure()
    else:
        return LeanProblemStructure()  # Empty structure


if __name__ == "__main__":
    # Test the parser with Basic.lean
    print("Testing Lean Problem Parser...")

    # Test tree parser
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
            print(f"Vertices: {tree.num_of_vertex()}, Edges: {tree.num_of_edge()}")
            print(f"Relation holds: {tree.verify_vertex_edge_relation()}")
        else:
            print("Failed to parse")

    # Test full file parser
    print("\n=== Testing Full File Parser ===")
    problem_structure = parse_lean_file("../Basic.lean")
    summary = problem_structure.get_summary()

    print("Problem Structure Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\nParsed Trees:")
    for i, tree in enumerate(problem_structure.trees):
        print(f"  Tree {i}: {tree}")
        print(f"    Vertices: {tree.num_of_vertex()}, Edges: {tree.num_of_edge()}")

    # Test evaluator
    print("\n=== Testing Expression Evaluator ===")
    evaluator = LeanExpressionEvaluator()

    test_evals = [
        "num_of_vertex (branch [leaf, branch [leaf]])",
        "num_of_edge (branch [leaf, branch [leaf]])",
    ]

    for eval_expr in test_evals:
        result = evaluator.evaluate_eval_command(eval_expr)
        print(f"Eval: {eval_expr} = {result}")

