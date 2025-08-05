"""
Expression tree parser for converting mathematical expressions to tree structures.
Converts expressions like 'a + b', '(x + y) * z' into tree representations.
"""

import re
from typing import List, Optional, Union, Dict, Any
from enum import Enum


class ExpressionNodeType(Enum):
    """Types of nodes in an expression tree."""
    VARIABLE = "variable"
    CONSTANT = "constant"
    OPERATOR = "operator"
    FUNCTION = "function"
    PARENTHESES = "parentheses"


class ExpressionNode:
    """
    Represents a node in an expression tree.
    Each node can be a variable, constant, operator, or function.
    """
    
    def __init__(self, value: str, node_type: ExpressionNodeType, children: Optional[List["ExpressionNode"]] = None):
        self.value = value
        self.node_type = node_type
        self.children = children or []
        self.parent: Optional["ExpressionNode"] = None
        
        # Set parent for children
        for child in self.children:
            child.parent = self
    
    def add_child(self, child: "ExpressionNode") -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def get_depth(self) -> int:
        """Get the depth of this node."""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1
    
    def visualize_tree(self, max_width: int = 80) -> str:
        """Generate a text-based tree visualization."""
        return self._visualize_recursive("", True, max_width)
    
    def _visualize_recursive(self, prefix: str, is_last: bool, max_width: int) -> str:
        """Recursive helper for tree visualization."""
        # Prepare content display
        type_display = f"[{self.node_type.value}]"
        content_display = f"{self.value} {type_display}"
        
        # Current node line
        connector = "└── " if is_last else "├── "
        result = f"{prefix}{connector}{content_display}\n"
        
        # Children
        if self.children:
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(self.children):
                is_child_last = (i == len(self.children) - 1)
                result += child._visualize_recursive(new_prefix, is_child_last, max_width)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "value": self.value,
            "type": self.node_type.value,
            "children": [child.to_dict() for child in self.children]
        }
    
    def to_lean_expression(self) -> str:
        """Convert back to Lean expression string."""
        if self.is_leaf():
            return self.value
        
        if self.node_type == ExpressionNodeType.OPERATOR:
            if len(self.children) == 2:  # Binary operator
                left = self.children[0].to_lean_expression()
                right = self.children[1].to_lean_expression()
                return f"({left} {self.value} {right})"
            elif len(self.children) == 1:  # Unary operator
                operand = self.children[0].to_lean_expression()
                return f"({self.value}{operand})"
        
        elif self.node_type == ExpressionNodeType.FUNCTION:
            args = ", ".join(child.to_lean_expression() for child in self.children)
            return f"{self.value}({args})"
        
        # Default: concatenate children
        return "(" + " ".join(child.to_lean_expression() for child in self.children) + ")"
    
    def __str__(self) -> str:
        return f"ExprNode({self.value}[{self.node_type.value}])"
    
    def __repr__(self) -> str:
        return self.__str__()


class LeanExpressionParser:
    """
    Parser for Lean mathematical expressions.
    Converts expressions like 'a + b', '(x + y) * z' into expression trees.
    """
    
    def __init__(self):
        self.debug = False
        
        # Operator precedence (higher number = higher precedence)
        self.precedence = {
            '∨': 1, '∧': 2,  # Logic operators
            '=': 3, '≠': 3, '<': 3, '>': 3, '≤': 3, '≥': 3,  # Comparison
            '+': 4, '-': 4,  # Addition/subtraction
            '*': 5, '/': 5, '%': 5,  # Multiplication/division
            '^': 6, '**': 6,  # Exponentiation
            '¬': 7, 'neg': 7  # Unary operators
        }
        
        # Operators that are right-associative
        self.right_associative = {'^', '**'}
    
    def tokenize(self, expression: str) -> List[str]:
        """Tokenize a mathematical expression."""
        # Remove extra whitespace
        expression = expression.strip()
        
        # Regex pattern for tokens
        pattern = r'(\d+\.?\d*|[a-zA-Z_][a-zA-Z0-9_]*|[+\-*/^=≠<>≤≥∨∧¬%()]|\*\*)'
        tokens = re.findall(pattern, expression)
        
        if self.debug:
            print(f"Tokens: {tokens}")
        
        return tokens
    
    def parse_expression(self, expression: str) -> Optional[ExpressionNode]:
        """
        Parse a mathematical expression into an expression tree.
        
        Args:
            expression: Mathematical expression like "a + b" or "(x + y) * z"
            
        Returns:
            ExpressionNode representing the root of the expression tree
        """
        if self.debug:
            print(f"Parsing expression: {expression}")
        
        tokens = self.tokenize(expression)
        if not tokens:
            return None
        
        # Convert to postfix notation using shunting yard algorithm
        postfix = self._to_postfix(tokens)
        if self.debug:
            print(f"Postfix: {postfix}")
        
        # Build expression tree from postfix
        return self._build_tree_from_postfix(postfix)
    
    def _to_postfix(self, tokens: List[str]) -> List[str]:
        """Convert infix tokens to postfix using shunting yard algorithm."""
        output = []
        operator_stack = []
        
        for token in tokens:
            if self._is_operand(token):
                output.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                # Pop operators until '('
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                if operator_stack:  # Remove '('
                    operator_stack.pop()
            elif self._is_operator(token):
                # Pop operators with higher or equal precedence
                while (operator_stack and 
                       operator_stack[-1] != '(' and
                       self._has_higher_precedence(operator_stack[-1], token)):
                    output.append(operator_stack.pop())
                operator_stack.append(token)
        
        # Pop remaining operators
        while operator_stack:
            if operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            else:
                operator_stack.pop()
        
        return output
    
    def _build_tree_from_postfix(self, postfix: List[str]) -> Optional[ExpressionNode]:
        """Build expression tree from postfix notation."""
        stack = []
        
        for token in postfix:
            if self._is_operand(token):
                node_type = ExpressionNodeType.CONSTANT if token.isdigit() or '.' in token else ExpressionNodeType.VARIABLE
                node = ExpressionNode(token, node_type)
                stack.append(node)
            elif self._is_operator(token):
                # Binary operators
                if token in ['+', '-', '*', '/', '^', '**', '=', '≠', '<', '>', '≤', '≥', '∨', '∧']:
                    if len(stack) >= 2:
                        right = stack.pop()
                        left = stack.pop()
                        node = ExpressionNode(token, ExpressionNodeType.OPERATOR, [left, right])
                        stack.append(node)
                # Unary operators
                elif token in ['¬', 'neg']:
                    if len(stack) >= 1:
                        operand = stack.pop()
                        node = ExpressionNode(token, ExpressionNodeType.OPERATOR, [operand])
                        stack.append(node)
        
        return stack[0] if stack else None
    
    def _is_operand(self, token: str) -> bool:
        """Check if token is an operand (variable or constant)."""
        return (token.isalnum() or 
                token.replace('.', '').isdigit() or
                token.replace('_', '').isalnum())
    
    def _is_operator(self, token: str) -> bool:
        """Check if token is an operator."""
        return token in self.precedence
    
    def _has_higher_precedence(self, op1: str, op2: str) -> bool:
        """Check if op1 has higher precedence than op2."""
        prec1 = self.precedence.get(op1, 0)
        prec2 = self.precedence.get(op2, 0)
        
        if op2 in self.right_associative:
            return prec1 > prec2
        else:
            return prec1 >= prec2
    
    def parse_lean_proposition(self, proposition: str) -> Optional[ExpressionNode]:
        """
        Parse a Lean proposition into an expression tree.
        Handles forall quantifiers and implications.
        """
        # Handle forall quantifiers
        if proposition.strip().startswith('∀') or proposition.strip().startswith('forall'):
            return self._parse_quantified_expression(proposition)
        
        # Handle implications
        if '→' in proposition:
            return self._parse_implication(proposition)
        
        # Regular expression
        return self.parse_expression(proposition)
    
    def _parse_quantified_expression(self, expr: str) -> Optional[ExpressionNode]:
        """Parse quantified expressions like '∀ x : Type, P(x)'."""
        # Simplified parsing - extract the main proposition
        if ',' in expr:
            parts = expr.split(',', 1)
            quantifier_part = parts[0].strip()
            proposition_part = parts[1].strip()
            
            # Create quantifier node
            quantifier_node = ExpressionNode("∀", ExpressionNodeType.OPERATOR)
            
            # Parse variable binding
            var_match = re.search(r'[∀forall]\s+([^:]+)', quantifier_part)
            if var_match:
                var_binding = var_match.group(1).strip()
                var_node = ExpressionNode(var_binding, ExpressionNodeType.VARIABLE)
                quantifier_node.add_child(var_node)
            
            # Parse proposition
            prop_node = self.parse_expression(proposition_part)
            if prop_node:
                quantifier_node.add_child(prop_node)
            
            return quantifier_node
        
        return self.parse_expression(expr)
    
    def _parse_implication(self, expr: str) -> Optional[ExpressionNode]:
        """Parse implications like 'P → Q'."""
        parts = expr.split('→', 1)
        if len(parts) == 2:
            antecedent = self.parse_expression(parts[0].strip())
            consequent = self.parse_expression(parts[1].strip())
            
            if antecedent and consequent:
                impl_node = ExpressionNode("→", ExpressionNodeType.OPERATOR, [antecedent, consequent])
                return impl_node
        
        return self.parse_expression(expr)


# Testing and example usage
if __name__ == "__main__":
    parser = LeanExpressionParser()
    parser.debug = True
    
    print("=== Expression Tree Parser Testing ===")
    
    test_expressions = [
        "a + b",
        "(x + y) * z",
        "a + b * c",
        "x = y + z",
        "∀ t : MyTree, num_of_vertex t = num_of_edge t + 1",
        "P → Q ∧ R",
        "a < b ∨ b < c",
        "¬(x = y)"
    ]
    
    for expr in test_expressions:
        print(f"\n--- Parsing: {expr} ---")
        tree = parser.parse_lean_proposition(expr)
        
        if tree:
            print("Expression Tree:")
            print(tree.visualize_tree())
            print(f"Back to Lean: {tree.to_lean_expression()}")
            
            print("JSON representation:")
            import json
            print(json.dumps(tree.to_dict(), indent=2))
        else:
            print("Failed to parse")
        
        print("-" * 50)