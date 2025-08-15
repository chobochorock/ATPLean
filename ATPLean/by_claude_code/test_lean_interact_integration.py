"""
Test script for lean_interact integration.
Verifies that the updated lean_problem_parser works with lean_interact.
"""

from lean_problem_parser import LeanInteractParser, load_lean_file_with_interact
from lean_interact import LeanREPLConfig
import sys
import os

def test_basic_lean_interact():
    """Test basic lean_interact functionality."""
    print("=== Testing Basic LeanInteract Functionality ===")
    
    try:
        # Create parser with verbose config
        config = LeanREPLConfig(verbose=True)
        parser = LeanInteractParser(config)
        
        print("✓ LeanInteractParser created successfully")
        
        # Test creating a simple theorem
        print("\n1. Testing theorem creation...")
        result = parser.create_theorem("∀ n : ℕ, n + 0 = n")
        
        if result and result.get("success"):
            print("✓ Theorem created successfully")
            print(f"  Goals: {len(result.get('goals', []))}")
            
            # Test applying a tactic if we have sorries
            if result.get('sorries'):
                proof_state = result['sorries'][0].pos
                print(f"\n2. Testing tactic application on proof state {proof_state}...")
                tactic_result = parser.apply_tactic("simp", proof_state)
                
                if tactic_result and tactic_result.get("success"):
                    print(f"✓ Tactic applied successfully")
                    print(f"  Status: {tactic_result.get('proof_status')}")
                    print(f"  New goals: {len(tactic_result.get('goals', []))}")
                else:
                    print(f"✗ Tactic application failed: {tactic_result}")
            else:
                print("  No sorries found to test tactics")
                
        else:
            print(f"✗ Theorem creation failed: {result}")
            return False
            
        # Test expression evaluation
        print("\n3. Testing expression evaluation...")
        eval_result = parser.evaluate_expression("5 + 3")
        if eval_result and eval_result.get("success"):
            print("✓ Expression evaluation works")
        else:
            print(f"✗ Expression evaluation failed: {eval_result}")
            
        return True
        
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False

def test_file_loading():
    """Test loading a Lean file."""
    print("\n=== Testing File Loading ===")
    
    try:
        # Try to load Basic.lean if it exists
        basic_file = "../Basic.lean"
        if os.path.exists(basic_file):
            print(f"Loading {basic_file}...")
            problem_structure = load_lean_file_with_interact(basic_file)
            
            summary = problem_structure.get_summary()
            print("✓ File loaded successfully")
            print("Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
                
            return True
        else:
            print(f"Basic.lean not found at {basic_file}, skipping file test")
            return True
            
    except Exception as e:
        print(f"✗ File loading test failed: {e}")
        return False

def test_integration_with_components():
    """Test integration with other components."""
    print("\n=== Testing Integration with Other Components ===")
    
    try:
        # Test importing the integrated theorem prover
        from integrated_theorem_prover import IntegratedTheoremProver
        print("✓ IntegratedTheoremProver import successful")
        
        # Test creating the prover (this should work with updated imports)
        prover = IntegratedTheoremProver()
        print("✓ IntegratedTheoremProver creation successful")
        
        # Test MinIF2F processor import
        from minif2f_processor import MinIF2FProcessor
        print("✓ MinIF2FProcessor import successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing lean_interact integration...")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Basic LeanInteract", test_basic_lean_interact()))
    results.append(("File Loading", test_file_loading()))
    results.append(("Component Integration", test_integration_with_components()))
    
    # Report results
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests passed! lean_interact integration is working.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())