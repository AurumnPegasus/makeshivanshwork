"""
Approach D: Hypergraph Memory

KEY INSIGHT: Messages create n-ary relationships, not pairwise.

"Meeting with Jerry about AI on Thursday" connects 4 entities simultaneously.
A pairwise graph creates 6 edges; a hypergraph creates 1 hyperedge.

Hypergraph advantages:
1. Preserves co-occurrence context (entities appeared together)
2. More efficient storage (O(n) vs O(n²) edges)
3. Better for "what did we discuss involving X AND Y?" queries
4. Natural message-as-hyperedge mapping

Theoretical Basis:
- Hypergraph Neural Networks (Feng et al. 2019)
- Higher-order link prediction (Benson et al. 2018)
- Used in MAGMA for complex relationship reasoning

Scoring:
- Jaccard: |query_entities ∩ hyperedge_entities| / |query_entities ∪ hyperedge_entities|
- Semantic: cosine(query_embedding, hyperedge_embedding)
- Combined: α * Jaccard + (1-α) * Semantic
"""

import sys
from pathlib import Path

# Handle imports for both package and direct script execution
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from baselines.base import Memory, MemoryStats, MemoryType
from components.graph_utils import HyperGraph

from approaches.base import (
    ApproachConfig,
    EnhancedSearchResult,
    MemoryApproach,
    RetrievalPath,
)


class HypergraphMemory(MemoryApproach):
    """
    Hypergraph-based memory for n-ary relationships.

    Each message becomes a hyperedge connecting all its entities.
    Retrieval uses combined Jaccard (structural) + semantic scoring.

    This is more powerful than pairwise graphs for:
    - Multi-entity queries ("Jerry AND AI AND Thursday")
    - Context preservation (entities appeared together)
    - Efficient storage and lookup
    """

    def __init__(self, config: Optional[ApproachConfig] = None):
        super().__init__(config)

        # Message storage
        self._memories: Dict[str, Memory] = {}
        self._embeddings: Dict[str, Dict[str, List[float]]] = {}
        self._message_order: List[str] = []

        # Indices
        self._by_user: Dict[str, List[str]] = {}
        self._by_session: Dict[str, List[str]] = {}

        # Hypergraph
        self._hypergraph = HyperGraph()

        # Message→Hyperedge mapping
        self._message_to_hyperedge: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "Hypergraph Memory"

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
        """Add message as hyperedge"""
        memory_id = memory_id or str(uuid.uuid4())
        timestamp = timestamp or datetime.now()

        # Create memory
        memory = Memory(
            id=memory_id,
            content=content,
            role=role,
            timestamp=timestamp,
            memory_type=MemoryType.MESSAGE,
            session_id=session_id,
            user_id=user_id,
            entities=entities or [],
            metadata={
                **(metadata or {}),
                "intent": intent,
            },
        )

        # Store embeddings
        if embeddings:
            self._embeddings[memory_id] = embeddings
        else:
            computed = self.compute_embeddings(content, entities, intent)
            self._embeddings[memory_id] = computed

        # Store memory
        self._memories[memory_id] = memory
        self._message_order.append(memory_id)

        # Index
        if user_id:
            if user_id not in self._by_user:
                self._by_user[user_id] = []
            self._by_user[user_id].append(memory_id)

        if session_id:
            if session_id not in self._by_session:
                self._by_session[session_id] = []
            self._by_session[session_id].append(memory_id)

        # Create hyperedge if entities present
        if entities:
            entity_ids = [e.lower() for e in entities]

            # Add entity nodes
            for entity, entity_id in zip(entities, entity_ids):
                self._hypergraph.add_node(entity_id, entity, "entity")

            # Get hyperedge embedding (use content embedding)
            he_embedding = embeddings.get("content") if embeddings else None
            if not he_embedding and "content" in self._embeddings.get(memory_id, {}):
                he_embedding = self._embeddings[memory_id]["content"]

            # Create hyperedge
            hyperedge_id = f"he_{memory_id}"
            self._hypergraph.add_hyperedge(
                hyperedge_id=hyperedge_id,
                node_ids=entity_ids,
                hyperedge_type="message",
                embedding=he_embedding,
                source_message_id=memory_id,
                timestamp=timestamp,
            )

            self._message_to_hyperedge[memory_id] = hyperedge_id

        return memory

    def search_enhanced(
        self,
        query: str,
        k: int = 10,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        return_paths: bool = True,
    ) -> List[EnhancedSearchResult]:
        """
        Hypergraph search with Jaccard + semantic scoring.

        For queries with entities:
        - Find hyperedges with overlapping entities
        - Score by Jaccard similarity + semantic similarity
        - Return associated messages

        For queries without entities:
        - Fall back to semantic search
        """
        import time

        start_time = time.time()

        # Get query embeddings
        query_embeddings, weights = self.compute_query_embeddings(query)

        # Extract query entities
        query_entities = []
        if self._entity_extractor:
            query_entities = [
                e.name.lower() for e in self._entity_extractor.extract(query)
            ]

        # Get query embedding for semantic matching
        query_embedding = None
        if self._embedding_model:
            result = self._embedding_model.embed(query, task_type="RETRIEVAL_QUERY")
            query_embedding = result.embedding

        # Get candidate memories
        if session_id and session_id in self._by_session:
            candidate_ids = set(self._by_session[session_id])
        elif user_id and user_id in self._by_user:
            candidate_ids = set(self._by_user[user_id])
        else:
            candidate_ids = set(self._message_order)

        # Score memories
        scored = []

        if query_entities:
            # Hypergraph search
            alpha = 0.4  # Balance Jaccard vs semantic

            hyperedge_results = self._hypergraph.search(
                query_node_ids=query_entities,
                query_embedding=query_embedding,
                top_k=k * 3,  # Get more for filtering
                alpha=alpha,
            )

            for hyperedge, he_score in hyperedge_results:
                memory_id = hyperedge.source_message_id
                if memory_id and memory_id in candidate_ids:
                    scored.append((memory_id, he_score, "hypergraph"))

        # Also do semantic search for memories without hyperedges
        if query_embedding:
            for memory_id in candidate_ids:
                if memory_id not in self._message_to_hyperedge:
                    # No hyperedge - use semantic only
                    if memory_id in self._embeddings:
                        mem_emb = self._embeddings[memory_id].get("content")
                        if mem_emb:
                            score = self.cosine_similarity(query_embedding, mem_emb)
                            if score >= self.config.similarity_threshold:
                                scored.append((memory_id, score, "semantic"))

        # Deduplicate (keep highest score)
        best_scores = {}
        for memory_id, score, source in scored:
            if memory_id not in best_scores or score > best_scores[memory_id][0]:
                best_scores[memory_id] = (score, source)

        # Sort by score
        sorted_results = sorted(
            best_scores.items(), key=lambda x: x[1][0], reverse=True
        )

        # Build results
        latency = (time.time() - start_time) * 1000
        results = []

        for memory_id, (score, source) in sorted_results[:k]:
            memory = self._memories[memory_id]

            paths = []
            if return_paths:
                paths.append(
                    RetrievalPath(
                        name=source,
                        weight=1.0,
                        results=[],
                        latency_ms=latency,
                        metadata={"score": score},
                    )
                )

            # Build reasoning
            if source == "hypergraph":
                he_id = self._message_to_hyperedge.get(memory_id)
                if he_id:
                    he = self._hypergraph.get_hyperedge(he_id)
                    if he:
                        overlap = len(set(query_entities) & he.node_ids)
                        reasoning = (
                            f"Hypergraph: {overlap} entity overlap, score={score:.3f}"
                        )
                    else:
                        reasoning = f"Hypergraph: score={score:.3f}"
                else:
                    reasoning = f"Hypergraph: score={score:.3f}"
            else:
                reasoning = f"Semantic: score={score:.3f}"

            results.append(
                EnhancedSearchResult(
                    memory=memory,
                    final_score=score,
                    paths=paths,
                    reasoning=reasoning,
                    confidence=score,
                )
            )

        return results

    def get_co_occurring_entities(
        self,
        entity: str,
        min_co_occurrence: int = 1,
    ) -> List[Tuple[str, int]]:
        """
        Get entities that frequently co-occur with given entity.

        This leverages hypergraph structure - entities in same hyperedges.
        """
        return self._hypergraph.get_related_nodes(
            entity.lower(), min_co_occurrence=min_co_occurrence
        )

    def find_messages_with_entities(
        self,
        entities: List[str],
        require_all: bool = True,
    ) -> List[Memory]:
        """
        Find messages containing specified entities.

        Args:
            entities: List of entities to search for
            require_all: If True, message must contain ALL entities

        Returns:
            List of matching memories
        """
        entity_ids = [e.lower() for e in entities]

        hyperedges = self._hypergraph.get_hyperedges_containing(
            entity_ids, require_all=require_all
        )

        memories = []
        for he in hyperedges:
            if he.source_message_id and he.source_message_id in self._memories:
                memories.append(self._memories[he.source_message_id])

        return memories

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear memories"""
        count = len(self._memories)

        self._memories.clear()
        self._embeddings.clear()
        self._message_order.clear()
        self._by_user.clear()
        self._by_session.clear()
        self._hypergraph = HyperGraph()
        self._message_to_hyperedge.clear()

        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        hg_stats = self._hypergraph.stats()

        timestamps = [m.timestamp for m in self._memories.values() if m.timestamp]

        return MemoryStats(
            total_memories=len(self._memories),
            total_entities=hg_stats["num_nodes"],
            total_relationships=hg_stats["num_hyperedges"],
            memory_size_bytes=sum(len(m.content) for m in self._memories.values()),
            oldest_memory=min(timestamps) if timestamps else None,
            newest_memory=max(timestamps) if timestamps else None,
            metadata={
                "hypergraph": hg_stats,
                "messages_with_hyperedges": len(self._message_to_hyperedge),
            },
        )

    def export_all(self, user_id: Optional[str] = None) -> List[Memory]:
        """Export all memories"""
        if user_id and user_id in self._by_user:
            return [
                self._memories[mid]
                for mid in self._by_user[user_id]
                if mid in self._memories
            ]
        return list(self._memories.values())

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get specific memory"""
        return self._memories.get(memory_id)
