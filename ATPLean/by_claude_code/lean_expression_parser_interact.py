"""
Lean Expression Parser using lean_interact for direct Lean server communication.
Handles mathematical expressions and converts them to tree structures with Lean verification.
"""

import re
from typing import List, Optional, Union, Dict, Any
from enum import Enum
from lean_interact import LeanREPLConfig, LeanServer, Command
from lean_interact.interface import LeanError

# Import original parsing logic for backup
# from lean_expression_parser import LeanExpressionParser as OriginalLeanExpressionParser, ExpressionNode as OriginalExpressionNode


class ExpressionNodeType(Enum):
    """Types of nodes in an expression tree."""
    VARIABLE = "variable"
    CONSTANT = "constant"
    OPERATOR = "operator"
    FUNCTION = "function"
    PARENTHESES = "parentheses"
    QUANTIFIER = "quantifier"


class LeanVerifiedExpressionNode:
    """
    Enhanced expression node that can be verified through lean_interact.
    Represents a node in an expression tree with Lean verification capabilities.
    """
    
    def __init__(self, value: str, node_type: ExpressionNodeType, children: Optional[List["LeanVerifiedExpressionNode"]] = None, lean_verified: bool = False):
        self.value = value
        self.node_type = node_type
        self.children = children or []
        self.parent: Optional["LeanVerifiedExpressionNode"] = None
        self.lean_verified = lean_verified
        self.lean_type_info: Optional[str] = None
        
        # Set parent for children
        for child in self.children:
            child.parent = self
    
    def add_child(self, child: "LeanVerifiedExpressionNode") -> None:
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
        verification_status = "✓" if self.lean_verified else "?"
        content_display = f"{self.value} {type_display} {verification_status}"
        
        if self.lean_type_info:
            content_display += f" : {self.lean_type_info}"
        
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
            "lean_verified": self.lean_verified,
            "lean_type_info": self.lean_type_info,
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
        
        elif self.node_type == ExpressionNodeType.QUANTIFIER:
            if len(self.children) >= 2:
                var_binding = self.children[0].to_lean_expression()
                proposition = self.children[1].to_lean_expression()
                return f"{self.value} {var_binding}, {proposition}"
        
        # Default: concatenate children
        return "(" + " ".join(child.to_lean_expression() for child in self.children) + ")"
    
    def __str__(self) -> str:
        verification_status = "✓" if self.lean_verified else "?"
        return f"LeanExprNode({self.value}[{self.node_type.value}]{verification_status})"
    
    def __repr__(self) -> str:
        return self.__str__()


class LeanInteractExpressionParser:
    """
    Enhanced expression parser that uses lean_interact for verification.
    Converts expressions and verifies them through the Lean server.
    """
    
    def __init__(self, config: Optional[LeanREPLConfig] = None):
        self.config = config or LeanREPLConfig(verbose=False)
        self.server = LeanServer(self.config)
        self.current_env = 0
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
        
        # Verification cache
        self.verification_cache: Dict[str, Dict[str, Any]] = {}
    
    def tokenize(self, expression: str) -> List[str]:
        """Tokenize a mathematical expression."""
        # Remove extra whitespace
        expression = expression.strip()
        
        # Enhanced regex pattern for Lean expressions
        pattern = r'(∀|forall|∃|exists|\d+\.?\d*|[a-zA-Z_][a-zA-Z0-9_]*|[+\-*/^=≠<>≤≥∨∧¬%()→,:]|\*\*|\.)'
        tokens = re.findall(pattern, expression)
        
        if self.debug:
            print(f"Tokens: {tokens}")
        
        return tokens
    
    def parse_expression_with_verification(self, expression: str) -> Optional[LeanVerifiedExpressionNode]:
        """
        Parse and verify a mathematical expression through lean_interact.
        
        Args:
            expression: Mathematical expression like "a + b" or "∀ x : ℕ, x + 0 = x"
            
        Returns:
            LeanVerifiedExpressionNode with verification status
        """
        if self.debug:
            print(f"Parsing expression with verification: {expression}")
        
        # Check cache first
        if expression in self.verification_cache:
            cached = self.verification_cache[expression]
            if cached['success']:
                return self._build_verified_tree_from_cache(expression, cached)
        
        # Parse using original algorithm
        tree = self.parse_expression(expression)
        if tree is None:
            return None
        
        # Verify through lean_interact
        verification_result = self._verify_expression_with_lean(expression)
        
        # Update tree with verification information
        self._update_tree_verification(tree, verification_result)
        
        # Cache result
        self.verification_cache[expression] = verification_result
        
        return tree
    
    def parse_expression(self, expression: str) -> Optional[LeanVerifiedExpressionNode]:
        """
        Parse a mathematical expression into an expression tree (without verification).
        Falls back to original parsing algorithm.
        """
        if self.debug:
            print(f"Parsing expression: {expression}")
        
        tokens = self.tokenize(expression)
        if not tokens:
            return None
        
        # Handle special Lean constructs first
        if tokens[0] in ['∀', 'forall', '∃', 'exists']:
            return self._parse_quantified_expression(expression)
        
        if '→' in tokens:
            return self._parse_implication(expression)
        
        # Convert to postfix notation using shunting yard algorithm
        postfix = self._to_postfix(tokens)
        if self.debug:
            print(f"Postfix: {postfix}")
        
        # Build expression tree from postfix
        return self._build_tree_from_postfix(postfix)
    
    def _verify_expression_with_lean(self, expression: str) -> Dict[str, Any]:
        """Verify an expression through lean_interact."""
        try:
            # Try to check the expression as a type
            check_cmd = f"#check {expression}"
            response = self.server.run(Command(cmd=check_cmd, env=self.current_env))
            
            if isinstance(response, LeanError):
                # Try as a proposition
                prop_cmd = f"example : {expression} := sorry"
                prop_response = self.server.run(Command(cmd=prop_cmd, env=self.current_env))
                
                if isinstance(prop_response, LeanError):
                    return {
                        "success": False,
                        "error": response.message,
                        "type": "unknown"
                    }
                else:
                    return {
                        "success": True,
                        "type": "proposition",
                        "env": prop_response.env,
                        "has_goals": hasattr(prop_response, 'sorries') and bool(prop_response.sorries)
                    }
            else:
                return {
                    "success": True,
                    "type": "expression",
                    "env": response.env,
                    "type_info": str(response) if hasattr(response, '__str__') else "verified"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "type": "unknown"
            }
    
    def _update_tree_verification(self, tree: LeanVerifiedExpressionNode, verification_result: Dict[str, Any]) -> None:
        """Update tree nodes with verification information."""
        if verification_result.get('success', False):
            tree.lean_verified = True
            tree.lean_type_info = verification_result.get('type_info', verification_result.get('type'))
        
        # Recursively update children (simplified approach)
        for child in tree.children:
            if verification_result.get('success', False):
                child.lean_verified = True  # Optimistic verification for children
    
    def _build_verified_tree_from_cache(self, expression: str, cached_result: Dict[str, Any]) -> LeanVerifiedExpressionNode:
        """Build a verified tree from cached verification result."""
        # Simple implementation - create a single node with verification info
        node = LeanVerifiedExpressionNode(
            expression, 
            ExpressionNodeType.VARIABLE,  # Simplified
            lean_verified=True
        )
        node.lean_type_info = cached_result.get('type_info', cached_result.get('type'))
        return node
    
    # ===== Original parsing methods (preserved as backup) =====
    
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
    
    def _build_tree_from_postfix(self, postfix: List[str]) -> Optional[LeanVerifiedExpressionNode]:
        """Build expression tree from postfix notation."""
        stack = []
        
        for token in postfix:
            if self._is_operand(token):
                node_type = ExpressionNodeType.CONSTANT if token.isdigit() or '.' in token else ExpressionNodeType.VARIABLE
                node = LeanVerifiedExpressionNode(token, node_type)
                stack.append(node)
            elif self._is_operator(token):
                # Binary operators
                if token in ['+', '-', '*', '/', '^', '**', '=', '≠', '<', '>', '≤', '≥', '∨', '∧', '→']:
                    if len(stack) >= 2:
                        right = stack.pop()
                        left = stack.pop()
                        node = LeanVerifiedExpressionNode(token, ExpressionNodeType.OPERATOR, [left, right])
                        stack.append(node)
                # Unary operators
                elif token in ['¬', 'neg']:
                    if len(stack) >= 1:
                        operand = stack.pop()
                        node = LeanVerifiedExpressionNode(token, ExpressionNodeType.OPERATOR, [operand])
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
    
    def _parse_quantified_expression(self, expr: str) -> Optional[LeanVerifiedExpressionNode]:
        """Parse quantified expressions like '∀ x : Type, P(x)'."""
        # Simplified parsing - extract the main proposition
        if ',' in expr:
            parts = expr.split(',', 1)
            quantifier_part = parts[0].strip()
            proposition_part = parts[1].strip()
            
            # Determine quantifier type
            quantifier_symbol = "∀" if (quantifier_part.startswith('∀') or quantifier_part.startswith('forall')) else "∃"
            
            # Create quantifier node
            quantifier_node = LeanVerifiedExpressionNode(quantifier_symbol, ExpressionNodeType.QUANTIFIER)
            
            # Parse variable binding
            var_match = re.search(r'[∀∃forallexists]\s+([^:,]+)', quantifier_part)
            if var_match:
                var_binding = var_match.group(1).strip()
                var_node = LeanVerifiedExpressionNode(var_binding, ExpressionNodeType.VARIABLE)
                quantifier_node.add_child(var_node)
            
            # Parse proposition
            prop_node = self.parse_expression(proposition_part)
            if prop_node:
                quantifier_node.add_child(prop_node)
            
            return quantifier_node
        
        return self.parse_expression(expr)
    
    def _parse_implication(self, expr: str) -> Optional[LeanVerifiedExpressionNode]:
        """Parse implications like 'P → Q'."""
        parts = expr.split('→', 1)
        if len(parts) == 2:
            antecedent = self.parse_expression(parts[0].strip())
            consequent = self.parse_expression(parts[1].strip())
            
            if antecedent and consequent:
                impl_node = LeanVerifiedExpressionNode("→", ExpressionNodeType.OPERATOR, [antecedent, consequent])
                return impl_node
        
        return self.parse_expression(expr)
    
    # ===== lean_interact specific methods =====
    
    def evaluate_expression_with_lean(self, expression: str) -> Optional[Dict[str, Any]]:
        """Evaluate an expression through lean_interact."""
        try:
            eval_cmd = f"#eval {expression}"
            response = self.server.run(Command(cmd=eval_cmd, env=self.current_env))
            
            if isinstance(response, LeanError):
                return {"error": response.message, "success": False}
            
            return {
                "success": True,
                "result": str(response),
                "env": response.env
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_expression_type(self, expression: str) -> Optional[Dict[str, Any]]:
        """Get the type of an expression through lean_interact."""
        try:
            check_cmd = f"#check {expression}"
            response = self.server.run(Command(cmd=check_cmd, env=self.current_env))
            
            if isinstance(response, LeanError):
                return {"error": response.message, "success": False}
            
            return {
                "success": True,
                "type": str(response),
                "env": response.env
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def parse_lean_proposition(self, proposition: str) -> Optional[LeanVerifiedExpressionNode]:
        """
        Parse a Lean proposition with verification.
        Enhanced version that uses lean_interact for verification.
        """
        # Use the verification-enabled parser
        return self.parse_expression_with_verification(proposition)
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get statistics about verification results."""
        total = len(self.verification_cache)
        successful = sum(1 for result in self.verification_cache.values() if result.get('success', False))
        
        return {
            'total_expressions': total,
            'successfully_verified': successful,
            'verification_rate': successful / total if total > 0 else 0,
            'cache_size': total
        }


# Backward compatibility aliases
LeanExpressionParser = LeanInteractExpressionParser
ExpressionNode = LeanVerifiedExpressionNode


# Testing and example usage
if __name__ == "__main__":
    parser = LeanInteractExpressionParser()
    parser.debug = True
    
    print("=== Lean Interact Expression Parser Testing ===")
    
    test_expressions = [
        "a + b",
        "(x + y) * z",
        "a + b * c",
        "x = y + z",
        "∀ t : MyTree, num_of_vertex t = num_of_edge t + 1",
        "P → Q ∧ R",
        "a < b ∨ b < c",
        "¬(x = y)",
        "∀ n : ℕ, n + 0 = n"
    ]
    
    for expr in test_expressions:
        print(f"\n--- Parsing with verification: {expr} ---")
        tree = parser.parse_expression_with_verification(expr)
        
        if tree:
            print("Expression Tree:")
            print(tree.visualize_tree())
            print(f"Back to Lean: {tree.to_lean_expression()}")
            print(f"Verified: {tree.lean_verified}")
            
            if tree.lean_type_info:
                print(f"Type info: {tree.lean_type_info}")
            
            print("JSON representation:")
            import json
            print(json.dumps(tree.to_dict(), indent=2))
            
            # Test additional lean_interact features
            print("\n--- lean_interact Features ---")
            type_info = parser.get_expression_type(expr)
            if type_info and type_info.get('success'):
                print(f"Type check result: {type_info['type']}")
            else:
                print(f"Type check failed: {type_info}")
            
        else:
            print("Failed to parse")
        
        print("-" * 50)
    
    # Show verification statistics
    print("\n=== Verification Statistics ===")
    stats = parser.get_verification_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")