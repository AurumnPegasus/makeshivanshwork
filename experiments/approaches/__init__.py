"""
Memory Architecture Approaches

This module implements our candidate memory architectures,
each embodying different theoretical foundations.

APPROACH HIERARCHY (from simple to complex):

A. Flat Vector (Baseline)
   - Single embedding per message
   - Cosine similarity search
   - Theoretical basis: Dense retrieval

B. Multi-Vector
   - Content + Entity + Intent embeddings
   - Query-adaptive weighting
   - Theoretical basis: Multi-aspect representation

C. Vector + Graph
   - Vector search + entity co-occurrence graph
   - Combined ranking
   - Theoretical basis: Hybrid structured/unstructured retrieval

D. Hypergraph
   - Messages as hyperedges connecting entities
   - Jaccard + semantic scoring
   - Theoretical basis: Higher-order relationships

E. Hierarchical
   - Immediate → Working → Episodic → Semantic
   - Query-aware level routing
   - Theoretical basis: Human memory architecture

F-K. Advanced approaches building on above

KEY INNOVATIONS ACROSS ALL APPROACHES:

1. Multi-embedding representation (Novel)
   - No existing system does this
   - Preserves orthogonal semantic dimensions

2. Intent-aware retrieval (Novel)
   - Critical for safety (question ≠ command)
   - Prevents false actions

3. Compositional graph structure (Extends MAGMA)
   - Multiple graph types
   - Learned traversal policy

4. Uncertainty quantification (Novel)
   - Confidence tracking
   - Contradiction detection
"""

from .base import MemoryApproach, ApproachConfig
from .a_flat_vector import FlatVectorMemory
from .b_multi_vector import MultiVectorMemory
from .c_vector_graph import VectorGraphMemory
from .d_hypergraph import HypergraphMemory
from .e_hierarchical import HierarchicalMemory

__all__ = [
    "MemoryApproach",
    "ApproachConfig",
    "FlatVectorMemory",
    "MultiVectorMemory",
    "VectorGraphMemory",
    "HypergraphMemory",
    "HierarchicalMemory",
]

# Advanced approaches (conditional imports)
try:
    from .f_neural import NeuralMemory
    __all__.append("NeuralMemory")
except ImportError:
    pass

try:
    from .k_ultimate_hybrid import UltimateHybridMemory
    __all__.append("UltimateHybridMemory")
except ImportError:
    pass
