"""
Approach A: Flat Vector Memory

The simplest vector-based approach:
- Single embedding per message
- Cosine similarity search
- No structure

This serves as our baseline for vector approaches.
All other approaches should beat this.

Theoretical Basis:
- Dense retrieval (Karpukhin et al. 2020)
- Assumes semantic similarity = relevance
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Handle imports for both package and direct script execution
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from baselines.base import Memory, MemoryStats, MemoryType

from approaches.base import (
    ApproachConfig,
    EnhancedSearchResult,
    MemoryApproach,
    RetrievalPath,
)


class FlatVectorMemory(MemoryApproach):
    """
    Flat vector memory: single embedding per message.

    This is our simplest approach. Uses only content embedding
    and cosine similarity for retrieval.
    """

    def __init__(self, config: Optional[ApproachConfig] = None):
        super().__init__(config)

        # Storage
        self._memories: Dict[str, Memory] = {}
        self._embeddings: Dict[str, List[float]] = {}  # memory_id → embedding
        self._message_order: List[str] = []

        # Indices
        self._by_user: Dict[str, List[str]] = {}
        self._by_session: Dict[str, List[str]] = {}

    @property
    def name(self) -> str:
        return "flat_vector"

    @property
    def version(self) -> str:
        return "1.0.0"

    def add_message_enhanced(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entities: Optional[List[str]] = None,
        intent: Optional[str] = None,
        embeddings: Optional[Dict[str, List[float]]] = None,
        memory_id: Optional[str] = None,
    ) -> Memory:
        """Add message with pre-computed data"""
        actual_id = self.add_message(
            role, content, session_id, user_id, timestamp, metadata, memory_id=memory_id
        )
        memory = self._memories[actual_id]

        # Use pre-computed embedding if provided
        if embeddings and "content" in embeddings:
            self._embeddings[actual_id] = embeddings["content"]

        return memory

    def add_message(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Add a message with embedding"""
        memory_id = memory_id or str(uuid.uuid4())
        timestamp = timestamp or datetime.now()

        # Create memory
        memory = Memory(
            id=memory_id,
            content=content,
            role=role,
            timestamp=timestamp,
            session_id=session_id or "default",
            user_id=user_id or "default",
            memory_type=MemoryType.MESSAGE,
            metadata=metadata or {},
        )

        self._memories[memory_id] = memory
        self._message_order.append(memory_id)

        # Index by user and session
        user = user_id or "default"
        if user not in self._by_user:
            self._by_user[user] = []
        self._by_user[user].append(memory_id)

        session = session_id or "default"
        if session not in self._by_session:
            self._by_session[session] = []
        self._by_session[session].append(memory_id)

        # Generate and store embedding
        if self._embedding_model:
            embedding = self._embedding_model.embed_content(content)
            if embedding:
                self._embeddings[memory_id] = embedding

        return memory_id

    def search_enhanced(
        self,
        query: str,
        k: int = 10,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        query_intent: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        return_paths: bool = True,
    ) -> List[EnhancedSearchResult]:
        """Search using cosine similarity"""
        if not self._embedding_model:
            # Fallback to recent messages
            return self._get_recent(k, user_id, session_id)

        # Get query embedding
        query_embedding = self._embedding_model.embed_query_simple(query)
        if not query_embedding:
            return self._get_recent(k, user_id, session_id)

        # Calculate similarities
        results = []
        for memory_id, memory in self._memories.items():
            # Apply filters
            if user_id and memory.user_id != user_id:
                continue
            if session_id and memory.session_id != session_id:
                continue

            # Get embedding
            embedding = self._embeddings.get(memory_id)
            if embedding is None:
                continue

            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)

            if similarity >= self.config.similarity_threshold:
                results.append(
                    EnhancedSearchResult(
                        memory=memory,
                        final_score=similarity,
                        paths=[
                            RetrievalPath(
                                name="vector",
                                weight=similarity,
                                results=[],
                            )
                        ],
                        reasoning=f"Cosine similarity: {similarity:.3f}",
                    )
                )

        # Sort by score and return top k
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:k]

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        user_id: Optional[str] = None,
    ) -> str:
        """Get formatted context for LLM"""
        results = self.search(query, k=20, user_id=user_id)

        context_parts = []
        total_tokens = 0

        for result in results:
            memory = result.memory
            # Estimate tokens (rough approximation)
            entry = f"[{memory.timestamp.isoformat()}] {memory.role}: {memory.content}"
            estimated_tokens = len(entry) // 4

            if total_tokens + estimated_tokens > max_tokens:
                break

            context_parts.append(entry)
            total_tokens += estimated_tokens

        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clear all memories"""
        self._memories.clear()
        self._embeddings.clear()
        self._message_order.clear()
        self._by_user.clear()
        self._by_session.clear()

    def stats(self) -> MemoryStats:
        """Return memory statistics"""
        return MemoryStats(
            total_memories=len(self._memories),
            memory_by_type={MemoryType.MESSAGE.value: len(self._memories)},
            total_entities=0,
            total_relationships=0,
            index_size_bytes=sum(len(str(e)) * 4 for e in self._embeddings.values()),
        )

    def export_all(self, user_id: Optional[str] = None) -> List[Memory]:
        """Export all memories, optionally filtered by user."""
        if user_id:
            return [
                self._memories[mid]
                for mid in self._by_user.get(user_id, [])
                if mid in self._memories
            ]
        return list(self._memories.values())

    def _get_recent(
        self,
        k: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[EnhancedSearchResult]:
        """Get recent messages as fallback"""
        memory_ids = self._message_order[-k * 2 :]  # Get more to allow filtering

        results = []
        for memory_id in reversed(memory_ids):
            memory = self._memories.get(memory_id)
            if not memory:
                continue
            if user_id and memory.user_id != user_id:
                continue
            if session_id and memory.session_id != session_id:
                continue

            results.append(
                EnhancedSearchResult(
                    memory=memory,
                    final_score=1.0,
                    paths=[
                        RetrievalPath(
                            name="recency",
                            weight=1.0,
                            results=[],
                        )
                    ],
                    reasoning="Fallback: recent message",
                )
            )

            if len(results) >= k:
                break

        return results

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a_arr = np.array(a)
        b_arr = np.array(b)

        dot_product = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))
