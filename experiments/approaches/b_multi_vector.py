"""
Approach B: Multi-Vector Memory

KEY INNOVATION: Multiple embeddings per message.

Each message has:
- Content embedding: What was said (semantic meaning)
- Entity embedding: Who/what is mentioned
- Intent embedding: Purpose (command/question/info)

Query-adaptive weighting adjusts the contribution of each
embedding type based on the query type.

Theoretical Basis:
- Multi-aspect representation learning
- Query-dependent relevance (different queries need different aspects)

Why this is novel:
- No existing memory system uses multi-embedding
- MemGPT, mem0, Supermemory all use single embeddings
- This preserves orthogonal semantic dimensions

Expected improvement over Approach A:
- Better entity recall: "Who is Jerry?" → entity embedding dominates
- Better intent matching: "Did I ask to X?" → intent embedding helps
- More robust: query mismatch on one aspect doesn't tank relevance

Performance Optimizations:
- Vectorized similarity computation using NumPy (10-100x speedup)
- Batch processing of embeddings
- Efficient top-k selection using partial sort
"""

import sys
from pathlib import Path

# Handle imports for both package and direct script execution
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from baselines.base import Memory, MemoryStats, MemoryType
from utils.math import batch_cosine_similarity, top_k_indices

from approaches.base import (
    ApproachConfig,
    EnhancedSearchResult,
    MemoryApproach,
    RetrievalPath,
)


class MultiVectorMemory(MemoryApproach):
    """
    Multi-vector memory with query-adaptive weighting.

    Stores three embeddings per message:
    - content: Semantic meaning
    - entity: Named entities
    - intent: Purpose/function

    Adjusts weights based on query type for targeted retrieval.
    """

    def __init__(self, config: Optional[ApproachConfig] = None):
        super().__init__(config)

        # Storage
        self._memories: Dict[str, Memory] = {}
        self._embeddings: Dict[
            str, Dict[str, List[float]]
        ] = {}  # memory_id → {type: embedding}
        self._message_order: List[str] = []

        # Indices
        self._by_user: Dict[str, List[str]] = {}
        self._by_session: Dict[str, List[str]] = {}
        self._by_entity: Dict[str, List[str]] = {}  # entity → [memory_ids]

    @property
    def name(self) -> str:
        return "Multi-Vector Memory"

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
        """Add message with multi-embeddings"""
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
            # Compute embeddings
            computed = self.compute_embeddings(content, entities, intent)
            self._embeddings[memory_id] = computed

        # Store memory
        self._memories[memory_id] = memory
        self._message_order.append(memory_id)

        # Index by user
        if user_id:
            if user_id not in self._by_user:
                self._by_user[user_id] = []
            self._by_user[user_id].append(memory_id)

        # Index by session
        if session_id:
            if session_id not in self._by_session:
                self._by_session[session_id] = []
            self._by_session[session_id].append(memory_id)

        # Index by entity
        for entity in entities or []:
            entity_lower = entity.lower()
            if entity_lower not in self._by_entity:
                self._by_entity[entity_lower] = []
            self._by_entity[entity_lower].append(memory_id)

        # Check capacity and evict if needed
        self._memories, self._message_order, self._embeddings = (
            self._check_capacity_and_evict(
                self._memories, self._message_order, self._embeddings
            )
        )

        # Invalidate query cache since we added new data
        self.invalidate_cache()

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
        Multi-path search with adaptive weighting.

        Uses VECTORIZED similarity computation for 10-100x speedup.

        Paths:
        1. Content similarity
        2. Entity similarity (if query has entities)
        3. Intent similarity (if query has intent)

        Final score = weighted combination based on query type.
        """
        import time

        start_time = time.time()

        # Input validation
        if not query or not query.strip():
            return []
        if k <= 0:
            return []

        # Check query cache first
        cached = self._get_cached_results(query, k, user_id, session_id)
        if cached is not None:
            return cached
        k = min(k, 1000)  # Cap at reasonable limit

        # Get query embeddings and adaptive weights
        query_embeddings, weights = self.compute_query_embeddings(query)

        if not query_embeddings:
            return self._recency_search(k, user_id, session_id)

        # Get candidate memories
        if session_id and session_id in self._by_session:
            candidate_ids = list(self._by_session[session_id])
        elif user_id and user_id in self._by_user:
            candidate_ids = list(self._by_user[user_id])
        else:
            candidate_ids = list(self._message_order)

        # Also include entity-matched memories
        candidate_set = set(candidate_ids)
        if self._entity_extractor:
            query_entities = self._entity_extractor.extract(query)
            for entity in query_entities:
                entity_lower = entity.name.lower()
                if entity_lower in self._by_entity:
                    for mid in self._by_entity[entity_lower]:
                        if mid not in candidate_set:
                            candidate_ids.append(mid)
                            candidate_set.add(mid)

        # Filter to candidates with embeddings
        valid_candidates = [mid for mid in candidate_ids if mid in self._embeddings]

        if not valid_candidates:
            return self._recency_search(k, user_id, session_id)

        # === VECTORIZED SIMILARITY COMPUTATION ===
        # Build embedding matrices for batch computation
        embedding_types = ["content", "entity", "intent"]
        type_scores = {}

        for emb_type in embedding_types:
            if emb_type not in query_embeddings:
                continue

            # Gather all candidate embeddings for this type
            candidates_with_type = []
            candidate_indices = []
            embeddings_list = []

            for i, mid in enumerate(valid_candidates):
                if emb_type in self._embeddings[mid]:
                    candidates_with_type.append(mid)
                    candidate_indices.append(i)
                    embeddings_list.append(self._embeddings[mid][emb_type])

            if not embeddings_list:
                continue

            # Stack into matrix and compute batch similarity
            candidate_matrix = np.array(embeddings_list, dtype=np.float32)
            query_vec = query_embeddings[emb_type]

            # Vectorized cosine similarity (10-100x faster than loop)
            similarities = batch_cosine_similarity(query_vec, candidate_matrix)

            # Store scores indexed by candidate position
            type_scores[emb_type] = {
                valid_candidates[idx]: float(similarities[j])
                for j, idx in enumerate(candidate_indices)
            }

        # === COMBINE WEIGHTED SCORES ===
        final_scores = []
        path_details = {}

        for mid in valid_candidates:
            scores = {}
            weighted_sum = 0.0
            total_weight = 0.0

            for emb_type, weight in weights.items():
                if emb_type in type_scores and mid in type_scores[emb_type]:
                    score = type_scores[emb_type][mid]
                    scores[emb_type] = score
                    weighted_sum += weight * score
                    total_weight += weight

            if total_weight > 0:
                final_score = weighted_sum / total_weight
            else:
                final_score = 0.0

            if final_score >= self.config.similarity_threshold:
                final_scores.append((mid, final_score))
                path_details[mid] = scores

        # === EFFICIENT TOP-K SELECTION ===
        if not final_scores:
            return self._recency_search(k, user_id, session_id)

        # Use efficient top-k if many candidates
        if len(final_scores) > k * 2:
            scores_array = np.array([s for _, s in final_scores])
            top_indices = top_k_indices(
                scores_array, k, threshold=self.config.similarity_threshold
            )
            scored = [(final_scores[i][0], final_scores[i][1]) for i in top_indices]
        else:
            # For small sets, regular sort is fine
            final_scores.sort(key=lambda x: x[1], reverse=True)
            scored = final_scores[:k]

        # Build results
        latency = (time.time() - start_time) * 1000
        results = []

        for memory_id, final_score in scored:
            memory = self._memories[memory_id]
            scores = path_details.get(memory_id, {})

            # Build path information
            paths = []
            for emb_type, score in scores.items():
                paths.append(
                    RetrievalPath(
                        name=f"{emb_type}_similarity",
                        weight=weights.get(emb_type, 0.0),
                        results=[],
                        latency_ms=latency / len(scores) if scores else latency,
                        metadata={"score": score},
                    )
                )

            # Build reasoning
            reasoning_parts = [
                f"{emb_type}: {score:.3f} (w={weights.get(emb_type, 0):.2f})"
                for emb_type, score in scores.items()
            ]
            reasoning = (
                f"Multi-vector: {' + '.join(reasoning_parts)} = {final_score:.3f}"
            )

            results.append(
                EnhancedSearchResult(
                    memory=memory,
                    final_score=final_score,
                    paths=paths if return_paths else [],
                    reasoning=reasoning,
                    confidence=final_score,
                )
            )

        # Cache results for future queries
        self._cache_results(query, k, user_id, session_id, results)

        return results

    def _recency_search(
        self,
        k: int,
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> List[EnhancedSearchResult]:
        """Fallback to recency"""
        if session_id and session_id in self._by_session:
            ids = self._by_session[session_id]
        elif user_id and user_id in self._by_user:
            ids = self._by_user[user_id]
        else:
            ids = self._message_order

        recent_ids = ids[-k:] if len(ids) > k else ids
        recent_ids = list(reversed(recent_ids))

        results = []
        for i, memory_id in enumerate(recent_ids):
            if memory_id in self._memories:
                memory = self._memories[memory_id]
                score = 1.0 - (i * 0.05)

                results.append(
                    EnhancedSearchResult(
                        memory=memory,
                        final_score=score,
                        paths=[RetrievalPath(name="recency", weight=1.0, results=[])],
                        reasoning="Retrieved by recency (no embeddings)",
                        confidence=0.5,
                    )
                )

        return results

    def search_by_entity(
        self,
        entity: str,
        k: int = 10,
    ) -> List[EnhancedSearchResult]:
        """
        Direct entity lookup.

        Useful for "What do you know about X?" queries.
        """
        entity_lower = entity.lower()

        if entity_lower not in self._by_entity:
            return []

        memory_ids = self._by_entity[entity_lower]

        # Score by recency and entity count
        scored = []
        for memory_id in memory_ids:
            if memory_id not in self._memories:
                continue

            memory = self._memories[memory_id]

            # Score by entity match quality
            entity_count = sum(1 for e in memory.entities if entity_lower in e.lower())
            recency_idx = (
                self._message_order.index(memory_id)
                if memory_id in self._message_order
                else 0
            )
            recency_score = recency_idx / max(1, len(self._message_order))

            score = 0.5 * entity_count + 0.5 * recency_score
            scored.append((memory_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for memory_id, score in scored[:k]:
            memory = self._memories[memory_id]
            results.append(
                EnhancedSearchResult(
                    memory=memory,
                    final_score=score,
                    paths=[RetrievalPath(name="entity_lookup", weight=1.0, results=[])],
                    reasoning=f"Entity match for '{entity}'",
                    confidence=0.9,
                )
            )

        return results

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear memories"""
        if session_id and session_id in self._by_session:
            ids_to_remove = set(self._by_session[session_id])
            for mid in ids_to_remove:
                self._memories.pop(mid, None)
                self._embeddings.pop(mid, None)
            del self._by_session[session_id]
            # Clean entity index
            for entity, mids in self._by_entity.items():
                self._by_entity[entity] = [m for m in mids if m not in ids_to_remove]
            return len(ids_to_remove)

        if user_id and user_id in self._by_user:
            ids_to_remove = set(self._by_user[user_id])
            for mid in ids_to_remove:
                self._memories.pop(mid, None)
                self._embeddings.pop(mid, None)
            del self._by_user[user_id]
            for entity, mids in self._by_entity.items():
                self._by_entity[entity] = [m for m in mids if m not in ids_to_remove]
            return len(ids_to_remove)

        count = len(self._memories)
        self._memories.clear()
        self._embeddings.clear()
        self._message_order.clear()
        self._by_user.clear()
        self._by_session.clear()
        self._by_entity.clear()
        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        timestamps = [m.timestamp for m in self._memories.values() if m.timestamp]

        # Count embedding types
        emb_counts = {"content": 0, "entity": 0, "intent": 0}
        for embs in self._embeddings.values():
            for emb_type in emb_counts:
                if emb_type in embs:
                    emb_counts[emb_type] += 1

        return MemoryStats(
            total_memories=len(self._memories),
            total_entities=len(self._by_entity),
            total_relationships=0,
            memory_size_bytes=sum(len(m.content) for m in self._memories.values()),
            oldest_memory=min(timestamps) if timestamps else None,
            newest_memory=max(timestamps) if timestamps else None,
            metadata={
                "embedding_counts": emb_counts,
                "unique_entities": len(self._by_entity),
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
