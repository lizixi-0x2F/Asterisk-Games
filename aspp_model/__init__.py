"""
ASPP (Adjacency-Structured Parallel Propagation) Model Implementation
Based on the mathematical framework defined in the paper.

This package implements the ASPP model with:
- Chain forward star graph storage
- FFN-based local update rules
- Parallel propagation operators
- Theorem validation experiments
- Operator stacking for improved generalization
"""

from .graph_structure import GraphStructure
from .aspp_operator import ASPPOperator
from .ffn_update import FFNUpdate

__version__ = "0.3.0"
__all__ = [
    'GraphStructure',
    'ASPPOperator', 
    'FFNUpdate',
    'PureGeneticProgramming'
]
