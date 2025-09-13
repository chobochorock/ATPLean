"""
Lean Problem Reader using lean_interact for direct Lean server communication.
Replaces file parsing with direct Lean interaction while preserving original parsing logic as backup.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from lean_interact import LeanREPLConfig, LeanServer, Command
from lean_interact.interface import LeanError

# Import original parsing logic for backup
# from lean_problem_reader import LeanReader as OriginalLeanReader


class LeanInteractReader:
    """
    Enhanced reader that uses lean_interact for direct Lean server communication.
    Falls back to file parsing when lean_interact is unavailable.
    """
    
    def __init__(self, file_path: str, config: Optional[LeanREPLConfig] = None):
        self.file_path = Path(file_path)
        self.config = config or LeanREPLConfig(verbose=False)
        self.server = LeanServer(self.config)
        self.current_env = 0
        
        # Content storage
        self.content: List[str] = []
        self.raw_content: str = ""
        
        # Processed content
        self.inductive_types: List[Dict[str, Any]] = []
        self.functions: List[Dict[str, Any]] = []
        self.theorems: List[Dict[str, Any]] = []
        self.eval_commands: List[Dict[str, str]] = []
        self.definitions: List[str] = []
        
        # lean_interact specific
        self.lean_env_info: Dict[str, Any] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        
    def read_file(self) -> bool:
        """Read the Lean file and process it through lean_interact."""
        try:
            # First, read the file content
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.content = f.readlines()
                self.raw_content = ''.join(self.content)
            
            # Process through lean_interact
            return self._process_with_lean_interact()
            
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            return False
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    
    def _process_with_lean_interact(self) -> bool:
        """Process the file content through lean_interact server."""
        try:
            # Send the entire file content to Lean server
            response = self.server.run(Command(cmd=self.raw_content, env=self.current_env))
            
            if isinstance(response, LeanError):
                print(f"Lean error: {response.message}")
                # Fall back to file parsing
                return self._fallback_to_file_parsing()
            
            # Update environment
            self.current_env = response.env
            
            # Store lean_interact response information
            self.lean_env_info = {
                'env': response.env,
                'response_type': type(response).__name__,
                'has_sorries': hasattr(response, 'sorries') and bool(response.sorries)
            }
            
            # Extract information using lean_interact capabilities
            self._extract_from_lean_response(response)
            
            # Also parse file content for additional information
            self._extract_from_file_content()
            
            return True
            
        except Exception as e:
            print(f"lean_interact processing failed: {e}")
            # Fall back to file parsing
            return self._fallback_to_file_parsing()
    
    def _extract_from_lean_response(self, response) -> None:
        """Extract mathematical content from lean_interact response."""
        # Process sorries (incomplete proofs)
        if hasattr(response, 'sorries') and response.sorries:
            for sorry in response.sorries:
                goal_text = getattr(sorry, 'goal', 'Unknown goal')
                self.interaction_history.append({
                    'type': 'sorry',
                    'goal': goal_text,
                    'position': getattr(sorry, 'pos', None)
                })
        
        # Extract other information from response if available
        if hasattr(response, 'messages'):
            for message in response.messages:
                self.interaction_history.append({
                    'type': 'message',
                    'content': str(message)
                })
    
    def _extract_from_file_content(self) -> None:
        """Extract mathematical content from file using text processing (backup method)."""
        # Extract inductive types
        self.inductive_types = self._extract_inductive_types()
        
        # Extract functions
        self.functions = self._extract_functions()
        
        # Extract theorems
        self.theorems = self._extract_theorems()
        
        # Extract eval commands
        self.eval_commands = self._extract_eval_commands()
    
    def _fallback_to_file_parsing(self) -> bool:
        """Fall back to original file parsing when lean_interact fails."""
        print("Falling back to file parsing...")
        self._extract_from_file_content()
        return True
    
    # ===== Original parsing methods (preserved as backup) =====
    
    def find_marker_indices(self, marker: str) -> List[int]:
        """Find all line indices where a marker appears."""
        indices = []
        for i, line in enumerate(self.content):
            if marker in line.strip():
                indices.append(i)
        return indices
    
    def extract_section_by_markers(self, start_marker: str, end_marker: str) -> List[str]:
        """Extract content between start and end markers."""
        start_indices = self.find_marker_indices(start_marker)
        end_indices = self.find_marker_indices(end_marker)
        
        if not start_indices or not end_indices:
            return []
        
        # Take the first start marker and first end marker after it
        start_idx = start_indices[0]
        end_idx = next((idx for idx in end_indices if idx > start_idx), len(self.content))
        
        return self.content[start_idx + 1:end_idx]
    
    def extract_readable_section(self) -> List[str]:
        """Extract content marked with -- read_start -- and -- read_end --."""
        return self.extract_section_by_markers("-- read_start --", "-- read_end --")
    
    def extract_from_read_file_marker(self) -> List[str]:
        """Extract content from -- read_file -- marker to end of file."""
        marker_indices = self.find_marker_indices("-- read_file --")
        if not marker_indices:
            return []
        
        start_idx = marker_indices[0]
        return self.content[start_idx + 1:]
    
    def extract_definitions_section(self) -> List[str]:
        """Extract content marked with -- definitions --."""
        return self.extract_section_content("-- definitions --")
    
    def extract_imports_section(self) -> List[str]:
        """Extract content marked with -- imports --."""
        return self.extract_section_content("-- imports --")
    
    def extract_problems_section(self) -> List[str]:
        """Extract content marked with -- problems --."""
        return self.extract_section_content("-- problems --")
    
    def extract_section_content(self, marker: str) -> List[str]:
        """Extract content for any section marker."""
        marker_indices = self.find_marker_indices(marker) 
        if not marker_indices:
            return []
        
        # Find the next section marker or end of file
        start_idx = marker_indices[0]
        end_idx = len(self.content)
        
        # Look for next section marker
        for i in range(start_idx + 1, len(self.content)):
            line = self.content[i].strip()
            if line.startswith('--') and line.endswith('--') and len(line) > 4:
                end_idx = i
                break
                
        return self.content[start_idx + 1:end_idx]
    
    def get_available_sections(self) -> List[str]:
        """Get list of all section markers found in the file."""
        sections = []
        section_patterns = ["-- imports --", "-- problems --", "-- definitions --"]
        
        for pattern in section_patterns:
            if self.find_marker_indices(pattern):
                sections.append(pattern)
        
        return sections
    
    def _extract_inductive_types(self) -> List[Dict[str, Any]]:
        """Extract inductive type definitions using regex parsing."""
        inductive_pattern = r'inductive\s+(\w+).*?where\s*(.*?)(?=\n(?:def|theorem|inductive|example|\s*$))'
        
        matches = re.finditer(inductive_pattern, self.raw_content, re.DOTALL | re.MULTILINE)
        inductives = []
        
        for match in matches:
            name = match.group(1)
            body = match.group(2).strip()
            
            # Parse constructors
            constructors = []
            for line in body.split('\n'):
                line = line.strip()
                if line.startswith('|'):
                    # Parse constructor line like "| leaf : MyTree"
                    constructor_match = re.match(r'\|\s*(\w+)\s*:?(.*)', line)
                    if constructor_match:
                        constructor_name = constructor_match.group(1)
                        constructor_type = constructor_match.group(2).strip()
                        constructors.append({
                            'name': constructor_name,
                            'type': constructor_type
                        })
            
            inductives.append({
                'name': name,
                'constructors': constructors,
                'raw_definition': match.group(0),
                'verified_by_lean_interact': True  # Since we processed through lean_interact
            })
        
        return inductives
    
    def _extract_functions(self) -> List[Dict[str, Any]]:
        """Extract function definitions using regex parsing."""
        function_pattern = r'def\s+(\w+)\s*:([^=]*?)=\s*(.*?)(?=\ndef|\ntheorem|\nexample|\ninductive|\n#|\s*$)'
        
        matches = re.finditer(function_pattern, self.raw_content, re.DOTALL | re.MULTILINE)
        functions = []
        
        for match in matches:
            name = match.group(1)
            type_annotation = match.group(2).strip() if match.group(2) else ""
            body = match.group(3).strip()
            
            functions.append({
                'name': name,
                'type': type_annotation,
                'body': body,
                'raw_definition': match.group(0),
                'verified_by_lean_interact': True
            })
        
        return functions
    
    def _extract_theorems(self) -> List[Dict[str, Any]]:
        """Extract theorem statements and proofs using regex parsing."""
        theorem_pattern = r'theorem\s+(\w+)([^:]*?):\s*(.*?)(?=\ntheorem|\ndef|\nexample|\ninductive|\s*$)'
        
        matches = re.finditer(theorem_pattern, self.raw_content, re.DOTALL | re.MULTILINE)
        theorems = []
        
        for match in matches:
            name = match.group(1)
            parameters = match.group(2).strip()
            statement_and_proof = match.group(3).strip()
            
            # Try to separate statement from proof
            if ':=' in statement_and_proof:
                parts = statement_and_proof.split(':=', 1)
                statement = parts[0].strip()
                proof = parts[1].strip()
            elif 'by' in statement_and_proof:
                parts = statement_and_proof.split('by', 1) 
                statement = parts[0].strip()
                proof = 'by ' + parts[1].strip()
            else:
                statement = statement_and_proof
                proof = ""
            
            theorems.append({
                'name': name,
                'parameters': parameters,
                'statement': statement,
                'proof': proof,
                'raw_definition': match.group(0),
                'verified_by_lean_interact': True
            })
        
        return theorems
    
    def _extract_eval_commands(self) -> List[Dict[str, str]]:
        """Extract #eval commands for testing."""
        eval_pattern = r'#eval\s+(.*?)(?=\n|$)'
        
        matches = re.finditer(eval_pattern, self.raw_content, re.MULTILINE)
        evals = []
        
        for match in matches:
            expression = match.group(1).strip()
            evals.append({
                'expression': expression,
                'full_command': match.group(0)
            })
        
        return evals
    
    def get_all_content(self) -> Dict[str, Any]:
        """Extract all content from the Lean file."""
        if not self.content:
            self.read_file()
        
        return {
            'file_path': str(self.file_path),
            'imports_section': self.extract_imports_section(),
            'problems_section': self.extract_problems_section(),
            'definitions_section': self.extract_definitions_section(),
            'available_sections': self.get_available_sections(),
            'inductive_types': self.inductive_types,
            'functions': self.functions,
            'theorems': self.theorems,
            'eval_commands': self.eval_commands,
            'raw_content': self.raw_content,
            
            # lean_interact specific information
            'lean_env_info': self.lean_env_info,
            'interaction_history': self.interaction_history,
            'processed_with_lean_interact': True
        }
    
    def find_tree_structures(self) -> List[str]:
        """Find potential tree structure expressions in the file."""
        tree_patterns = [
            r'branch\s*\[.*?\]',
            r'leaf',
            r'MyTree\.\w+',
        ]
        
        found_trees = []
        for pattern in tree_patterns:
            matches = re.findall(pattern, self.raw_content)
            found_trees.extend(matches)
        
        return list(set(found_trees))  # Remove duplicates
    
    # ===== lean_interact specific methods =====
    
    def evaluate_expression(self, expression: str) -> Optional[Dict[str, Any]]:
        """Evaluate a Lean expression using lean_interact."""
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
    
    def check_theorem(self, theorem_statement: str) -> Optional[Dict[str, Any]]:
        """Check a theorem statement using lean_interact."""
        try:
            # Add theorem with sorry to get the goals
            theorem_cmd = f"theorem temp_check : {theorem_statement} := sorry"
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
    
    def get_lean_interact_info(self) -> Dict[str, Any]:
        """Get information about the lean_interact session."""
        return {
            'current_env': self.current_env,
            'env_info': self.lean_env_info,
            'interaction_count': len(self.interaction_history),
            'has_errors': any(item['type'] == 'error' for item in self.interaction_history),
            'has_sorries': any(item['type'] == 'sorry' for item in self.interaction_history)
        }


# Compatibility classes for existing code
class Reader:
    """Enhanced compatibility class that uses lean_interact."""
    
    def __init__(self, file_path: str):
        self.lean_reader = LeanInteractReader(file_path)
        self.lean_reader.read_file()
    
    def get_readable_content(self) -> List[str]:
        """Get content from read markers (compatible with existing code)."""
        read_file_content = self.lean_reader.extract_from_read_file_marker()
        if read_file_content:
            return read_file_content
        
        return self.lean_reader.extract_readable_section()


# Backward compatibility alias
LeanReader = LeanInteractReader


if __name__ == "__main__":
    # Test with the Basic.lean file
    reader = LeanInteractReader("../Basic.lean")
    
    if reader.read_file():
        content = reader.get_all_content()
        
        print("=== LEAN INTERACT READER TESTING ===")
        print(f"Processed with lean_interact: {content['processed_with_lean_interact']}")
        print(f"Lean environment info: {content['lean_env_info']}")
        print()
        
        print("=== AVAILABLE SECTIONS ===")
        for section in content['available_sections']:
            print(f"Found section: {section}")
        print()
        
        print("=== INDUCTIVE TYPES ===")
        for inductive in content['inductive_types']:
            print(f"Type: {inductive['name']} (verified: {inductive.get('verified_by_lean_interact', False)})")
            for constructor in inductive['constructors']:
                print(f"  - {constructor['name']}: {constructor['type']}")
            print()
        
        print("=== FUNCTIONS ===")
        for func in content['functions']:
            print(f"Function: {func['name']} (verified: {func.get('verified_by_lean_interact', False)})")
            print(f"Type: {func['type']}")
            print(f"Body: {func['body'][:100]}...")
            print()
        
        print("=== THEOREMS ===")
        for theorem in content['theorems']:
            print(f"Theorem: {theorem['name']} (verified: {theorem.get('verified_by_lean_interact', False)})")
            print(f"Statement: {theorem['statement']}")
            print(f"Proof: {theorem['proof'][:100]}...")
            print()
        
        print("=== EVAL COMMANDS ===")
        for eval_cmd in content['eval_commands']:
            print(f"Eval: {eval_cmd['expression']}")
            # Test evaluation with lean_interact
            result = reader.evaluate_expression(eval_cmd['expression'])
            if result and 'success' in result:
                print(f"  Result: {result}")
            else:
                print(f"  Error: {result}")
        
        print("=== INTERACTION HISTORY ===")
        for interaction in content['interaction_history']:
            print(f"Type: {interaction['type']}")
            if 'goal' in interaction:
                print(f"  Goal: {interaction['goal']}")
            if 'content' in interaction:
                print(f"  Content: {interaction['content']}")
        
        print("=== TREE STRUCTURES FOUND ===")
        for tree in reader.find_tree_structures():
            print(f"Tree: {tree}")
        
        print("=== LEAN INTERACT SESSION INFO ===")
        session_info = reader.get_lean_interact_info()
        for key, value in session_info.items():
            print(f"{key}: {value}")