# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATPLean is a Lean 4 project focused on mathematical proofs and theorem proving. The project demonstrates tree structures and mathematical theorems, particularly proving that the number of vertices in a tree equals the number of edges plus one.

## Core Architecture

### Lean 4 Structure
- **Main module**: `ATPLean.lean` - Entry point importing Basic
- **Core implementation**: `ATPLean/Basic.lean` - Contains mathematical definitions and proofs
- **Package configuration**: `lakefile.toml` - Lake build system configuration
- **Dependency management**: Uses Mathlib for mathematical foundations

### Python External Tools
- **Location**: `ATPLean/external/` directory
- **Purpose**: Python utilities for interacting with Lean code
- **Key files**:
  - `lean_controller.py` - Interfaces with Lean through lean-interact library
  - `pyproject.toml` - Python project dependencies (lean-interact, openai, transformers)
  - `main.py` - Main Python entry point

### Mathematical Components
The project implements:
- **MyTree inductive type**: Custom tree structure with leaf and branch constructors
- **Vertex counting**: `num_of_vertex` function for counting tree nodes
- **Edge counting**: `num_of_edge` function for counting tree edges  
- **Core theorem**: `vertex_eq_edge_plus_one` proving V = E + 1 for trees

## Development Commands

### Lean 4 Development
```bash
# Build the project
lake build

# Check proofs interactively
lake exe lean --run ATPLean/Basic.lean

# Update dependencies
lake update
```

### Python External Tools
```bash
# From ATPLean/external/ directory
uv run python lean_controller.py  # Run Lean interaction script
uv run python main.py            # Run main Python script

# Install dependencies
uv sync
```

## Key Implementation Details

### Tree Structure Definition
- Uses pattern matching for recursive tree operations
- Supports n-ary trees through List MyTree in branch constructor
- Implements fold-based counting algorithms

### Proof Strategy
- The main theorem uses structural induction on trees
- Base cases handle leaf and empty branch scenarios
- Inductive step uses recursive hypothesis and arithmetic simplification

### Code Parsing Markers
The Lean file uses special comments for external parsing:
- `-- read_file --` / `-- read_start --` / `-- read_end --` mark sections for Python processing

### Lean Version
- **Toolchain**: Lean 4 v4.22.0-rc3
- **Dependencies**: Mathlib (master branch)
- **Build system**: Lake

## File Organization Patterns

- Lean source files use `.lean` extension
- Mathematical proofs follow Mathlib conventions
- Python utilities are isolated in `external/` subdirectory
- Configuration files (`lakefile.toml`, `lean-toolchain`) at project root

## Testing and Validation

Use `#eval` commands in Lean files to test function behavior:
```lean
#eval num_of_vertex (branch [leaf, branch [leaf]])
#eval num_of_edge (branch [leaf, branch [leaf]])
```

Proofs can be verified using `lake build` which will report any proof errors.