#!/usr/bin/env python3
"""
Simple test script for Mathlib data extraction functionality.
Tests basic parsing and data extraction without complex dependencies.
"""

import re
import sqlite3
from pathlib import Path
from typing import List, Tuple
import tempfile
import time


class SimpleLeanParser:
    """Simplified Lean parser for testing purposes."""
    
    def __init__(self):
        # Basic patterns for testing
        self.theorem_pattern = re.compile(r'^(theorem|lemma|example)\s+([a-zA-Z_][a-zA-Z0-9_\'\.]*).*?:\s*(.*?)(?:by|:=|\s*$)', re.MULTILINE)
        self.definition_pattern = re.compile(r'^(def|inductive|structure|class)\s+([a-zA-Z_][a-zA-Z0-9_\'\.]*).*?:?\s*(.*?)(?::=|\s*where|\s*$)', re.MULTILINE)
        self.tactic_pattern = re.compile(r'\b(simp|rw|ring|norm_num|linarith|omega|exact|apply|intro|intros|cases|induction|constructor|tauto|decide|assumption|rfl)\b')
    
    def parse_file(self, file_path: Path) -> Tuple[List[dict], List[dict], List[str]]:
        """Parse a Lean file and extract basic content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract theorems
            theorems = []
            for match in self.theorem_pattern.finditer(content):
                theorem = {
                    'type': match.group(1),
                    'name': match.group(2),
                    'statement': match.group(3).strip(),
                    'file': str(file_path)
                }
                theorems.append(theorem)
            
            # Extract definitions
            definitions = []
            for match in self.definition_pattern.finditer(content):
                definition = {
                    'type': match.group(1),
                    'name': match.group(2),
                    'content': match.group(3).strip(),
                    'file': str(file_path)
                }
                definitions.append(definition)
            
            # Extract tactics
            tactics = list(set(self.tactic_pattern.findall(content)))
            
            return theorems, definitions, tactics
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return [], [], []


def test_mathlib_extraction():
    """Test Mathlib data extraction."""
    print("🧪 Testing Mathlib Data Extraction")
    print("=" * 50)
    
    # Setup
    mathlib_path = Path("/home/chorock/Projects/ATPLean/.lake/packages/mathlib")
    parser = SimpleLeanParser()
    
    # Check if Mathlib exists
    if not mathlib_path.exists():
        print(f"❌ Mathlib path not found: {mathlib_path}")
        return
    
    print(f"✅ Mathlib path found: {mathlib_path}")
    
    # Find Lean files
    print("\n📁 Finding Lean files...")
    lean_files = []
    
    # Look in main directories
    for subdir in ["Mathlib", "Archive", "MathlibTest"]:
        dir_path = mathlib_path / subdir
        if dir_path.exists():
            files = list(dir_path.glob("**/*.lean"))
            lean_files.extend(files)
            print(f"   {subdir}: {len(files)} files")
    
    total_files = len(lean_files)
    print(f"✅ Found {total_files} total Lean files")
    
    if total_files == 0:
        print("❌ No Lean files found")
        return
    
    # Test parsing on sample files
    print(f"\n🔍 Testing parsing on sample files...")
    sample_size = min(10, total_files)
    sample_files = lean_files[:sample_size]
    
    total_theorems = 0
    total_definitions = 0
    total_tactics = set()
    parse_errors = 0
    
    start_time = time.time()
    
    for i, file_path in enumerate(sample_files):
        try:
            theorems, definitions, tactics = parser.parse_file(file_path)
            
            total_theorems += len(theorems)
            total_definitions += len(definitions)
            total_tactics.update(tactics)
            
            print(f"   File {i+1:2d}: {len(theorems):3d} theorems, {len(definitions):3d} definitions, {len(tactics):2d} tactics - {file_path.name}")
            
            # Show some examples
            if theorems:
                print(f"      Example theorem: {theorems[0]['name']}")
            if definitions:
                print(f"      Example definition: {definitions[0]['name']}")
            
        except Exception as e:
            parse_errors += 1
            print(f"   ❌ Error parsing {file_path.name}: {e}")
    
    parse_time = time.time() - start_time
    
    # Summary
    print(f"\n📊 Parsing Results:")
    print(f"   Files processed: {sample_size}")
    print(f"   Parse errors: {parse_errors}")
    print(f"   Total theorems: {total_theorems}")
    print(f"   Total definitions: {total_definitions}")
    print(f"   Unique tactics: {len(total_tactics)}")
    print(f"   Processing time: {parse_time:.2f}s")
    print(f"   Files per second: {sample_size/parse_time:.1f}")
    
    if total_tactics:
        print(f"\n🔧 Common tactics found: {', '.join(sorted(list(total_tactics))[:10])}")
    
    # Test database storage
    print(f"\n💾 Testing database storage...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
        
        # Create database
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE test_theorems (
                id INTEGER PRIMARY KEY,
                name TEXT,
                statement TEXT,
                file_path TEXT
            )
        ''')
        
        # Store some sample data
        sample_theorems = []
        for file_path in sample_files[:3]:
            theorems, _, _ = parser.parse_file(file_path)
            for theorem in theorems[:2]:  # First 2 theorems per file
                sample_theorems.append((
                    theorem['name'],
                    theorem['statement'],
                    theorem['file']
                ))
        
        cursor.executemany(
            'INSERT INTO test_theorems (name, statement, file_path) VALUES (?, ?, ?)',
            sample_theorems
        )
        
        conn.commit()
        
        # Test retrieval
        cursor.execute('SELECT COUNT(*) FROM test_theorems')
        count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"✅ Database test successful: stored {count} theorems")
        
        # Cleanup
        import os
        os.unlink(temp_db_path)
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    
    # Estimate full processing time
    if sample_size > 0 and parse_time > 0:
        estimated_full_time = (total_files / sample_size) * parse_time
        print(f"\n⏱️  Estimated full processing time: {estimated_full_time/60:.1f} minutes")
        estimated_theorems = (total_theorems / sample_size) * total_files
        print(f"📚 Estimated total theorems in Mathlib: {estimated_theorems:.0f}")
    
    print(f"\n✅ Mathlib extraction test completed successfully!")
    
    return {
        'total_files': total_files,
        'sample_size': sample_size,
        'theorems_found': total_theorems,
        'definitions_found': total_definitions,
        'tactics_found': len(total_tactics),
        'processing_time': parse_time,
        'estimated_full_time': estimated_full_time if 'estimated_full_time' in locals() else 0
    }


if __name__ == "__main__":
    results = test_mathlib_extraction()
    
    print(f"\n🎯 Test Summary:")
    print(f"   Success: {'✅' if results else '❌'}")
    if results:
        print(f"   Files available: {results['total_files']}")
        print(f"   Sample processed: {results['sample_size']}")
        print(f"   Theorems per file: {results['theorems_found']/max(results['sample_size'],1):.1f}")
        print(f"   Processing rate: {results['sample_size']/max(results['processing_time'],0.1):.1f} files/sec")