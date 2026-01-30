"""
Shared Components for Memory Architecture Research

These components implement the building blocks used across all memory approaches.
Each component is designed to be:
- Theoretically grounded
- Empirically validated
- Modular and composable
- Publication-ready

Key innovations:
1. Multi-embedding with Matryoshka Representation Learning (MRL)
2. Intent classification with compositional semantics
3. Entity extraction with coreference resolution
4. Graph operations with hyperbolic embeddings
"""

from .embeddings import (
    EmbeddingModel,
    GeminiEmbedding,
    MultiEmbedding,
    MatryoshkaEmbedding,
)
from .entity_extraction import (
    EntityExtractor,
    Entity,
    EntityType,
    extract_entities,
)
from .intent_classification import (
    IntentClassifier,
    Intent,
    IntentType,
    classify_intent,
)
from .graph_utils import (
    EntityGraph,
    HyperGraph,
    GraphNode,
    GraphEdge,
    HyperEdge,
)
from .summarization import (
    Summarizer,
    EpisodicSummarizer,
    summarize_session,
)
from .synthesis import (
    AnswerSynthesizer,
    synthesize_answer,
)

__all__ = [
    # Embeddings
    "EmbeddingModel",
    "GeminiEmbedding",
    "MultiEmbedding",
    "MatryoshkaEmbedding",
    # Entity extraction
    "EntityExtractor",
    "Entity",
    "EntityType",
    "extract_entities",
    # Intent classification
    "IntentClassifier",
    "Intent",
    "IntentType",
    "classify_intent",
    # Graph utilities
    "EntityGraph",
    "HyperGraph",
    "GraphNode",
    "GraphEdge",
    "HyperEdge",
    # Summarization
    "Summarizer",
    "EpisodicSummarizer",
    "summarize_session",
    # Synthesis
    "AnswerSynthesizer",
    "synthesize_answer",
]
