"""
Tree structures for representing Lean problems and proofs.
Mirrors the MyTree inductive type from Basic.lean.
"""

from typing import List, Optional, Any
from enum import Enum


class StateNode:
    """
    Represents a node in the proof state tree for solving math problems.
    Uses hierarchical indexing (1.2.1.4) and tracks tactics applied.
    """

    def __init__(self, content: str, parent: Optional["StateNode"] = None, tactic: Optional[str] = None):
        self.content = content  # Current statement/proposition
        self.parent = parent
        self.children: List["StateNode"] = []
        self.tactic = tactic  # Tactic that led to this state
        self.index = self._generate_index()  # Hierarchical index like "1.2.1.4"

        # Add this node to parent's children if parent exists
        if parent:
            parent.children.append(self)
            # Regenerate index after adding to parent
            self.index = self._generate_index()

    def _generate_index(self) -> str:
        """Generate hierarchical index like '1.2.1.4'."""
        if self.parent is None:
            return "0"  # Root node
        
        # Get parent's index and this node's position among siblings
        parent_index = self.parent.index
        sibling_position = len(self.parent.children)  # 0-based, will be 1-based
        
        if parent_index == "0":
            return str(sibling_position)
        else:
            return f"{parent_index}.{sibling_position}"

    def add_child(self, content: str, tactic: Optional[str] = None) -> "StateNode":
        """Add a child node with given content and tactic."""
        child = StateNode(content, self, tactic)
        return child

    def add_child_node(self, child: "StateNode") -> None:
        """Add an existing child node."""
        child.parent = self
        self.children.append(child)
        child.index = child._generate_index()

    def get_depth(self) -> int:
        """Get the depth of this node in the tree."""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1

    def get_root(self) -> "StateNode":
        """Get the root node of the tree."""
        if self.parent is None:
            return self
        return self.parent.get_root()

    def get_path_to_root(self) -> List["StateNode"]:
        """Get the path from this node to the root."""
        path = [self]
        current = self
        while current.parent:
            current = current.parent
            path.append(current)
        return path

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def get_tactic_chain(self) -> List[str]:
        """Get the chain of tactics from root to this node."""
        path = self.get_path_to_root()
        tactics = []
        for node in reversed(path):
            if node.tactic:
                tactics.append(node.tactic)
        return tactics

    def find_by_index(self, index: str) -> Optional["StateNode"]:
        """Find a node by its hierarchical index."""
        if self.index == index:
            return self
        
        for child in self.children:
            result = child.find_by_index(index)
            if result:
                return result
        
        return None

    def visualize_tree(self, max_width: int = 80) -> str:
        """Generate a text-based tree visualization."""
        return self._visualize_recursive("", True, max_width)

    def _visualize_recursive(self, prefix: str, is_last: bool, max_width: int) -> str:
        """Recursive helper for tree visualization."""
        # Prepare content display
        content_display = self.content[:max_width - len(prefix) - 10]
        if len(self.content) > max_width - len(prefix) - 10:
            content_display += "..."
        
        # Prepare tactic display
        tactic_display = f" [{self.tactic}]" if self.tactic else ""
        
        # Current node line
        connector = "└── " if is_last else "├── "
        result = f"{prefix}{connector}[{self.index}] {content_display}{tactic_display}\n"
        
        # Children
        if self.children:
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(self.children):
                is_child_last = (i == len(self.children) - 1)
                result += child._visualize_recursive(new_prefix, is_child_last, max_width)
        
        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "index": self.index,
            "content": self.content,
            "tactic": self.tactic,
            "children": [child.to_dict() for child in self.children]
        }

    @classmethod
    def from_dict(cls, data: dict, parent: Optional["StateNode"] = None) -> "StateNode":
        """Create StateNode from dictionary."""
        node = cls(data["content"], parent, data.get("tactic"))
        for child_data in data.get("children", []):
            cls.from_dict(child_data, node)
        return node

    def __str__(self) -> str:
        tactic_str = f" [{self.tactic}]" if self.tactic else ""
        return f"StateNode[{self.index}]: {self.content[:50]}...{tactic_str}"

    def __repr__(self) -> str:
        return self.__str__()


class LeanProblemStructure:
    """
    Container for all structures related to a Lean problem.
    Includes trees, proofs, and state information.
    """

    def __init__(self):
        self.state_trees: List[StateNode] = []
        self.theorems: List[str] = []
        self.definitions: List[str] = []
        self.proof_states: List[StateNode] = []

    def add_state_tree(self, tree: StateNode) -> None:
        """Add a state tree to the problem structure."""
        self.state_trees.append(tree)

    def add_theorem(self, theorem: str) -> None:
        """Add a theorem statement."""
        self.theorems.append(theorem)

    def add_definition(self, definition: str) -> None:
        """Add a definition."""
        self.definitions.append(definition)

    def add_proof_state(self, state: StateNode) -> None:
        """Add a proof state node."""
        self.proof_states.append(state)

    def get_summary(self) -> dict:
        """Get a summary of the problem structure."""
        return {
            "num_state_trees": len(self.state_trees),
            "num_theorems": len(self.theorems),
            "num_definitions": len(self.definitions),
            "num_proof_states": len(self.proof_states),
        }


# Example usage and test cases
if __name__ == "__main__":
    # Test enhanced state nodes with hierarchical indexing and tactics
    root_state = StateNode("∀ t : MyTree, num_of_vertex t = num_of_edge t + 1")
    
    # Add children with tactics
    case1 = root_state.add_child("Case 1: t = leaf", "match t with")
    case2 = root_state.add_child("Case 2: t = branch children", "match t with")
    
    # Add sub-cases
    case1_goal = case1.add_child("num_of_vertex leaf = num_of_edge leaf + 1", "simp")
    case1_proof = case1_goal.add_child("1 = 0 + 1", "unfold")
    case1_proof.add_child("True", "ring")
    
    case2_goal = case2.add_child("num_of_vertex (branch children) = num_of_edge (branch children) + 1", "simp")
    case2_inductive = case2_goal.add_child("Use inductive hypothesis", "induction")

    print("=== Enhanced State Node Testing ===")
    print(f"Root: {root_state}")
    print(f"Case 1 index: {case1.index}")
    print(f"Case 2 index: {case2.index}")
    print(f"Nested proof index: {case1_proof.index}")
    
    print("\n=== Tree Visualization ===")
    print(root_state.visualize_tree())
    
    print("\n=== Tactic Chain Example ===")
    final_node = case1_proof.children[0] if case1_proof.children else case1_proof
    print(f"Tactics to reach {final_node.index}: {final_node.get_tactic_chain()}")
    
    print("\n=== Find by Index ===")
    found_node = root_state.find_by_index("1.1.1")
    print(f"Node at 1.1.1: {found_node}")
    
    print("\n=== JSON Serialization ===")
    import json
    tree_dict = root_state.to_dict()
    print(json.dumps(tree_dict, indent=2)[:300] + "...")

