"""
Baseline Memory System Adapters

This module provides unified adapters for existing memory systems:
- Letta/MemGPT: Two-tier memory with core/recall/archival
- mem0: Hybrid vector + graph memory
- Supermemory: Brain-inspired with intelligent decay
- LangChain: Various conversation memory implementations
- Raw Context: Naive baseline (dump all history)
"""

from .base import MemorySystemAdapter, Memory, SearchResult
from .raw_context import RawContextAdapter
from .langchain_adapter import LangChainBufferAdapter, LangChainSummaryAdapter

__all__ = [
    "MemorySystemAdapter",
    "Memory",
    "SearchResult",
    "RawContextAdapter",
    "LangChainBufferAdapter",
    "LangChainSummaryAdapter",
]

# Conditional imports for optional dependencies
try:
    from .letta_adapter import LettaAdapter
    __all__.append("LettaAdapter")
except ImportError:
    pass

try:
    from .mem0_adapter import Mem0Adapter
    __all__.append("Mem0Adapter")
except ImportError:
    pass

try:
    from .supermemory_adapter import SupermemoryAdapter
    __all__.append("SupermemoryAdapter")
except ImportError:
    pass
