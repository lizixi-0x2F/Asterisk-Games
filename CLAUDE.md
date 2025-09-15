# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codebase Overview

This repository contains the implementation of the **Adjacency-Structured Parallel Propagation (ASPP)** model, a mathematical framework for abstract reasoning tasks. The codebase implements the theoretical framework described in the mathematical paper `abs.md`.

### Core Architecture

- **ASPP Model**: Implements the mathematical framework from the paper with:
  - Graph-structured parallel propagation
  - Local update rules with FFN networks
  - Dynamic graph support
  - Theorem validation components

- **Main Components**:
  - `aspp_model/`: Core model implementation
    - `graph_structure.py`: Graph data structures (chain forward star storage)
    - `aspp_operator.py`: ASPP operator implementation (Definitions 1.3-1.4)
    - `ffn_update.py`: FFN-based local update rules (Definition 1.3)
    - `operator_stack.py`: Operator stacking for improved generalization
    - `rope_encoding.py`: Rotary Position Encoding implementation

- **Data**: Contains SRSD-Feynman datasets for testing and validation

### Development Setup

This is a PyTorch-based research codebase. Key dependencies:
- Python 3.8+
- PyTorch 1.9+
- Standard scientific Python stack (numpy, scipy)

### Key Mathematical Concepts Implemented

- **Definition 1.3**: ASPP Operator `Φ(H^(t); E) = H^(t+1)`
- **Definition 1.4**: K-Step Reasoning Evolution `Φ^(K)(H^(0); E)`
- **Theorem 2.1**: Universality (MPNN simulation)
- **Theorem 2.2**: Convergence under contraction mapping

### Usage Patterns

The model is designed for abstract reasoning tasks and can be extended for:
- Graph-based reasoning problems
- Structured prediction tasks
- Theoretical computer science experiments
- Neural-symbolic integration research

### File Structure Conventions

- Mathematical definitions are implemented as Python classes
- Theorem proofs are implemented as validation methods
- Graph structures use efficient chain forward star representation
- All components support dynamic graph switching