"""
Tree structures for representing Lean problems and proofs.
Mirrors the MyTree inductive type from Basic.lean.
"""

from typing import List, Optional, Any
from enum import Enum


class StateNode:
    """
    Represents a node in the proof state tree.
    Used for tracking proof progress and theorem dependencies.
    """

    def __init__(self, index: str, content: str, parent: Optional["StateNode"] = None):
        self.index = index
        self.content = content
        self.parent = parent
        self.children: List["StateNode"] = []

        # Add this node to parent's children if parent exists
        if parent:
            parent.children.append(self)

    def add_child(self, child: "StateNode") -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)

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

    def __str__(self) -> str:
        return f"StateNode({self.index}: {self.content[:50]}...)"

    def __repr__(self) -> str:
        return self.__str__()


class LeanProblemStructure:
    """
    Container for all structures related to a Lean problem.
    Includes trees, proofs, and state information.
    """

    def __init__(self):
        self.trees: List[MyTree] = []
        self.theorems: List[str] = []
        self.definitions: List[str] = []
        self.proof_states: List[StateNode] = []

    def add_tree(self, tree: MyTree) -> None:
        """Add a tree to the problem structure."""
        self.trees.append(tree)

    def add_theorem(self, theorem: str) -> None:
        """Add a theorem statement."""
        self.theorems.append(theorem)

    def add_definition(self, definition: str) -> None:
        """Add a definition."""
        self.definitions.append(definition)

    def add_proof_state(self, state: StateNode) -> None:
        """Add a proof state node."""
        self.proof_states.append(state)

    def verify_all_trees(self) -> bool:
        """Verify the vertex-edge relation for all trees."""
        return all(tree.verify_vertex_edge_relation() for tree in self.trees)

    def get_summary(self) -> dict:
        """Get a summary of the problem structure."""
        return {
            "num_trees": len(self.trees),
            "num_theorems": len(self.theorems),
            "num_definitions": len(self.definitions),
            "num_proof_states": len(self.proof_states),
            "all_trees_valid": self.verify_all_trees(),
        }


# Example usage and test cases
if __name__ == "__main__":
    # Test state nodes
    root_state = StateNode("0", "Initial theorem statement")
    child1 = StateNode("1", "Case 1: leaf", root_state)
    child2 = StateNode("2", "Case 2: branch", root_state)

    print("Testing state nodes:")
    print(f"Root: {root_state}")
    print(f"Child 1 depth: {child1.get_depth()}")
    print(f"Is child2 leaf: {child2.is_leaf()}")

