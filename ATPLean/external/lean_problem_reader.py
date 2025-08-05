"""
Reader for parsing Lean files and extracting mathematical content.
Recognizes definitions, theorems, proofs, and tree structures.
"""

import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class LeanReader:
    """
    Reads and parses Lean files to extract mathematical content.
    
    Recognizes special comment markers:
    - -- read_file --
    - -- read_start --
    - -- read_end --  
    - -- definitions --
    """
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content: List[str] = []
        self.raw_content: str = ""
        
    def read_file(self) -> bool:
        """Read the Lean file into memory."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.content = f.readlines()
                self.raw_content = ''.join(self.content)
            return True
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            return False
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    
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
    
    def extract_inductive_types(self) -> List[Dict[str, any]]:
        """Extract inductive type definitions."""
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
                'raw_definition': match.group(0)
            })
        
        return inductives
    
    def extract_functions(self) -> List[Dict[str, any]]:
        """Extract function definitions."""
        # Pattern for function definitions like "def name : type"
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
                'raw_definition': match.group(0)
            })
        
        return functions
    
    def extract_theorems(self) -> List[Dict[str, any]]:
        """Extract theorem statements and proofs."""
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
                'raw_definition': match.group(0)
            })
        
        return theorems
    
    def extract_eval_commands(self) -> List[Dict[str, str]]:
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
    
    def get_all_content(self) -> Dict[str, any]:
        """Extract all content from the Lean file."""
        if not self.content:
            self.read_file()
        
        return {
            'file_path': str(self.file_path),
            'imports_section': self.extract_imports_section(),
            'problems_section': self.extract_problems_section(),
            'definitions_section': self.extract_definitions_section(),
            'available_sections': self.get_available_sections(),
            'inductive_types': self.extract_inductive_types(),
            'functions': self.extract_functions(),
            'theorems': self.extract_theorems(),
            'eval_commands': self.extract_eval_commands(),
            'raw_content': self.raw_content
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


# Example usage and compatibility with existing lean_controller.py
class Reader:
    """Legacy compatibility class for existing code."""
    
    def __init__(self, file_path: str):
        self.lean_reader = LeanReader(file_path)
        self.lean_reader.read_file()
    
    def get_readable_content(self) -> List[str]:
        """Get content from read markers (compatible with lean_controller.py)."""
        read_file_content = self.lean_reader.extract_from_read_file_marker()
        if read_file_content:
            return read_file_content
        
        return self.lean_reader.extract_readable_section()


if __name__ == "__main__":
    # Test with the Basic.lean file
    reader = LeanReader("../Basic.lean")
    
    if reader.read_file():
        content = reader.get_all_content()
        
        print("=== AVAILABLE SECTIONS ===")
        for section in content['available_sections']:
            print(f"Found section: {section}")
        print()
        
        print("=== IMPORTS SECTION ===")
        imports = content['imports_section']
        if imports:
            for line in imports[:5]:  # Show first 5 lines
                print(f"  {line.strip()}")
        else:
            print("  No imports section found")
        print()
        
        print("=== PROBLEMS SECTION ===")
        problems = content['problems_section']
        if problems:
            for line in problems[:5]:  # Show first 5 lines
                print(f"  {line.strip()}")
        else:
            print("  No problems section found")
        print()
        
        print("=== DEFINITIONS SECTION ===")
        definitions = content['definitions_section']
        if definitions:
            for line in definitions[:5]:  # Show first 5 lines
                print(f"  {line.strip()}")
        else:
            print("  No definitions section found")
        print()
        
        print("=== INDUCTIVE TYPES ===")
        for inductive in content['inductive_types']:
            print(f"Type: {inductive['name']}")
            for constructor in inductive['constructors']:
                print(f"  - {constructor['name']}: {constructor['type']}")
            print()
        
        print("=== FUNCTIONS ===")
        for func in content['functions']:
            print(f"Function: {func['name']}")
            print(f"Type: {func['type']}")
            print(f"Body: {func['body'][:100]}...")
            print()
        
        print("=== THEOREMS ===")
        for theorem in content['theorems']:
            print(f"Theorem: {theorem['name']}")
            print(f"Statement: {theorem['statement']}")
            print(f"Proof: {theorem['proof'][:100]}...")
            print()
        
        print("=== EVAL COMMANDS ===")
        for eval_cmd in content['eval_commands']:
            print(f"Eval: {eval_cmd['expression']}")
        
        print("=== TREE STRUCTURES FOUND ===")
        for tree in reader.find_tree_structures():
            print(f"Tree: {tree}")