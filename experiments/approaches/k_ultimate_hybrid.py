"""
Approach K: Ultimate Hybrid Memory (ASM v2)

Our moonshot: Combine the best of ALL approaches into one system.

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                    ULTIMATE MEMORY ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│  Multi-Embedding Layer (Novel)                                   │
│  ├── Content embedding (semantic meaning)                        │
│  ├── Entity embedding (who/what)                                 │
│  └── Intent embedding (command/question/info)                    │
├─────────────────────────────────────────────────────────────────┤
│  Multi-Structure Layer                                           │
│  ├── Hypergraph (n-ary entity relationships)                    │
│  ├── Temporal index (time-based retrieval)                      │
│  └── Entity graph (pairwise for path queries)                   │
├─────────────────────────────────────────────────────────────────┤
│  Memory Hierarchy (Human-Inspired)                               │
│  ├── Immediate: Last N messages (always in context)             │
│  ├── Working: Recent session key points                          │
│  ├── Episodic: Session summaries                                 │
│  └── Semantic: Facts and preferences                             │
├─────────────────────────────────────────────────────────────────┤
│  Intelligent Retrieval                                           │
│  ├── Intent-aware query routing                                  │
│  ├── Multi-path retrieval (vector + graph + temporal)           │
│  ├── Learned combination weights                                 │
│  └── Confidence scoring with uncertainty                         │
└─────────────────────────────────────────────────────────────────┘

COST OPTIMIZATION:
- Use free Gemini embeddings (gemini-embedding-001)
- Aggressive caching (embedding cache, result cache)
- Batch processing for embeddings
- Lazy computation (only compute what's needed)
- MRL (Matryoshka) for flexible dimensions

WHAT MAKES THIS NOVEL:
1. Multi-embedding per memory (nobody else does this)
2. Intent-aware retrieval (critical for safety)
3. Combined hypergraph + pairwise graph + hierarchy
4. Uncertainty quantification
5. Cost-efficient design

TARGET PERFORMANCE:
- Beat MAGMA (70% LoCoMo)
- <2% false action rate
- <$50 total experiment cost
"""

import sys
from pathlib import Path

# Handle imports for both package and direct script execution
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import logging
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from baselines.base import Memory, MemoryStats, MemoryType
from components.graph_utils import EntityGraph, HyperGraph

from approaches.base import (
    ApproachConfig,
    EnhancedSearchResult,
    MemoryApproach,
    RetrievalPath,
)

logger = logging.getLogger(__name__)


class UltimateHybridMemory(MemoryApproach):
    """
    Ultimate Hybrid Memory: The best of everything.

    Combines:
    - Multi-embedding (Approach B)
    - Entity + Hypergraph (Approaches C + D)
    - Memory hierarchy (Approach E)
    - Intent-aware retrieval (Novel)
    - Uncertainty tracking (Novel)

    Optimized for:
    - Accuracy (beat all benchmarks)
    - Cost (free embeddings, caching)
    - Generality (no domain-specific tuning)

    IMPORTANT: Call initialize() with embedding_model before using search.
    """

    # Configuration
    IMMEDIATE_SIZE = 10
    WORKING_WINDOW_HOURS = 24
    CACHE_SIZE = 1000

    def __init__(self, config: Optional[ApproachConfig] = None):
        super().__init__(config)

        # === Storage ===
        self._memories: Dict[str, Memory] = {}
        self._embeddings: Dict[str, Dict[str, List[float]]] = {}
        self._message_order: List[str] = []

        # === Indices ===
        self._by_user: Dict[str, List[str]] = {}
        self._by_session: Dict[str, List[str]] = {}
        self._by_entity: Dict[str, Set[str]] = {}
        self._by_date: Dict[str, List[str]] = {}  # date_str → memory_ids

        # === Multi-Structure ===
        self._entity_graph = EntityGraph()
        self._hypergraph = HyperGraph()

        # === Memory Hierarchy ===
        self._immediate: deque = deque(maxlen=self.IMMEDIATE_SIZE)
        self._facts: Dict[str, Dict[str, Any]] = {}
        self._fact_embeddings: Dict[str, List[float]] = {}

        # === Caching (Cost Optimization) ===
        self._query_cache: Dict[str, List[EnhancedSearchResult]] = {}
        self._embedding_cache: Dict[str, List[float]] = {}

        # === Statistics for Learning ===
        self._retrieval_stats: Dict[str, Dict[str, float]] = {}

    @property
    def name(self) -> str:
        return "Ultimate Hybrid Memory (ASM v2)"

    @property
    def version(self) -> str:
        return "2.0.0"

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
        """Add message with all indexing"""
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

        # === Store Embeddings ===
        if embeddings:
            self._embeddings[memory_id] = embeddings
        else:
            # Use cached embeddings if available
            cache_key = f"emb:{hash(content)}"
            if cache_key in self._embedding_cache:
                self._embeddings[memory_id] = {
                    "content": self._embedding_cache[cache_key]
                }
            elif self._embedding_model:
                # Only compute embeddings if model is available
                try:
                    computed = self.compute_embeddings(content, entities, intent)
                    self._embeddings[memory_id] = computed
                    if "content" in computed:
                        self._embedding_cache[cache_key] = computed["content"]
                except Exception as e:
                    logger.warning(f"Failed to compute embeddings: {e}")
                    self._embeddings[memory_id] = {}
            else:
                # No embedding model - store empty embeddings
                self._embeddings[memory_id] = {}

        # === Store Memory ===
        self._memories[memory_id] = memory
        self._message_order.append(memory_id)

        # === Index by User/Session/Date ===
        if user_id:
            if user_id not in self._by_user:
                self._by_user[user_id] = []
            self._by_user[user_id].append(memory_id)

        if session_id:
            if session_id not in self._by_session:
                self._by_session[session_id] = []
            self._by_session[session_id].append(memory_id)

        date_key = timestamp.strftime("%Y-%m-%d")
        if date_key not in self._by_date:
            self._by_date[date_key] = []
        self._by_date[date_key].append(memory_id)

        # === Immediate Memory ===
        self._immediate.append(memory)

        # === Update Structures ===
        if entities:
            self._update_structures(entities, memory_id, timestamp, embeddings)

        # === Extract Facts ===
        self._extract_facts(content, entities, memory_id)

        # === Invalidate Query Cache ===
        self._query_cache.clear()

        return memory

    def _update_structures(
        self,
        entities: List[str],
        memory_id: str,
        timestamp: datetime,
        embeddings: Optional[Dict[str, List[float]]],
    ):
        """Update entity graph and hypergraph"""
        entity_ids = [e.lower() for e in entities]

        # Index by entity
        for entity, entity_id in zip(entities, entity_ids):
            if entity_id not in self._by_entity:
                self._by_entity[entity_id] = set()
            self._by_entity[entity_id].add(memory_id)

            # Add to entity graph
            self._entity_graph.add_node(entity_id, entity, "entity")

        # Add pairwise edges to entity graph
        for i, e1 in enumerate(entity_ids):
            for e2 in entity_ids[i + 1 :]:
                self._entity_graph.add_edge(
                    e1, e2, "mentioned_with", bidirectional=True
                )

        # Add hyperedge
        he_embedding = None
        if embeddings and "content" in embeddings:
            he_embedding = embeddings["content"]

        self._hypergraph.add_hyperedge(
            hyperedge_id=f"he_{memory_id}",
            node_ids=entity_ids,
            embedding=he_embedding,
            source_message_id=memory_id,
            timestamp=timestamp,
        )

    def _extract_facts(
        self,
        content: str,
        entities: Optional[List[str]],
        source_id: str,
    ):
        """Extract semantic facts"""
        content_lower = content.lower()

        # Preference extraction
        if any(
            p in content_lower
            for p in ["i prefer", "i like", "i always", "i never", "i usually"]
        ):
            fact_id = f"pref_{uuid.uuid4().hex[:8]}"
            self._facts[fact_id] = {
                "type": "preference",
                "content": content,
                "source_ids": [source_id],
                "confidence": 0.9,
            }
            # Embed fact (use cache)
            cache_key = f"fact:{hash(content)}"
            if cache_key not in self._embedding_cache and self._embedding_model:
                result = self._embedding_model.embed(
                    content, task_type="RETRIEVAL_DOCUMENT"
                )
                self._embedding_cache[cache_key] = result.embedding
            if cache_key in self._embedding_cache:
                self._fact_embeddings[fact_id] = self._embedding_cache[cache_key]

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
        Multi-path intelligent search.

        1. Check intent safety
        2. Route query to appropriate paths
        3. Execute parallel retrieval
        4. Combine with learned weights
        5. Add confidence scores

        Note: Works with or without embedding model, but vector search
        requires embedding_model to be set via initialize().
        """
        import time

        start_time = time.time()

        # === Check Cache ===
        cache_key = f"{query}:{user_id}:{session_id}:{k}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        # === Intent Safety Check (graceful fallback if not configured) ===
        try:
            _, _, safety_confidence = self.check_intent_safety(query)
        except Exception as e:
            logger.debug(f"Intent safety check failed: {e}")
            safety_confidence = 0.5

        # === Determine Query Type and Route ===
        query_type, path_weights = self._analyze_query(query)

        # === Get Query Embedding (graceful fallback if not configured) ===
        query_embeddings = {}
        emb_weights = {"content": 1.0}

        if self._embedding_model:
            try:
                query_embeddings, emb_weights = self.compute_query_embeddings(query)
            except Exception as e:
                logger.warning(f"Failed to compute query embeddings: {e}")
                # Fall back to non-vector paths
                path_weights["vector"] = 0.0
                path_weights["semantic"] = 0.0
        else:
            # No embedding model - disable vector-based paths
            logger.debug("No embedding model configured - using non-vector paths only")
            path_weights["vector"] = 0.0
            path_weights["semantic"] = 0.0
            # Boost non-vector paths
            remaining = (
                path_weights.get("entity", 0.2)
                + path_weights.get("hypergraph", 0.1)
                + path_weights.get("temporal", 0.1)
                + path_weights.get("immediate", 0.1)
            )
            if remaining > 0:
                scale = 1.0 / remaining
                for key in ["entity", "hypergraph", "temporal", "immediate"]:
                    if key in path_weights:
                        path_weights[key] *= scale

        # === Extract Query Entities ===
        query_entities = []
        if self._entity_extractor:
            try:
                query_entities = [
                    e.name.lower() for e in self._entity_extractor.extract(query)
                ]
            except Exception as e:
                logger.debug(f"Entity extraction failed: {e}")

        # === Get Candidates ===
        if session_id and session_id in self._by_session:
            candidate_ids = set(self._by_session[session_id])
        elif user_id and user_id in self._by_user:
            candidate_ids = set(self._by_user[user_id])
        else:
            candidate_ids = set(self._message_order)

        # === Multi-Path Retrieval ===
        path_results = {}

        # Path 1: Vector Search
        if path_weights.get("vector", 0) > 0:
            path_results["vector"] = self._vector_search(
                query_embeddings, emb_weights, candidate_ids
            )

        # Path 2: Entity Graph
        if path_weights.get("entity", 0) > 0 and query_entities:
            path_results["entity"] = self._entity_search(query_entities, candidate_ids)

        # Path 3: Hypergraph
        if path_weights.get("hypergraph", 0) > 0 and query_entities:
            path_results["hypergraph"] = self._hypergraph_search(
                query_entities, query_embeddings.get("content"), candidate_ids
            )

        # Path 4: Temporal
        if path_weights.get("temporal", 0) > 0:
            path_results["temporal"] = self._temporal_search(query, candidate_ids)

        # Path 5: Semantic (Facts)
        if path_weights.get("semantic", 0) > 0:
            path_results["semantic"] = self._semantic_search(
                query_embeddings.get("content")
            )

        # Path 6: Immediate (Always include)
        path_results["immediate"] = self._immediate_search()

        # === Combine Results ===
        combined = self._combine_paths(path_results, path_weights)

        # === Build Final Results ===
        latency = (time.time() - start_time) * 1000
        results = []

        for memory_id, (final_score, contributing_paths) in list(combined.items())[:k]:
            memory = self._memories.get(memory_id)
            if not memory:
                # Check if it's a fact
                if memory_id in self._facts:
                    fact = self._facts[memory_id]
                    memory = Memory(
                        id=memory_id,
                        content=fact["content"],
                        role="system",
                        timestamp=datetime.now(),
                        memory_type=MemoryType.FACT,
                    )
                else:
                    continue

            paths = []
            if return_paths:
                for path_name, score in contributing_paths.items():
                    paths.append(
                        RetrievalPath(
                            name=path_name,
                            weight=path_weights.get(path_name, 0),
                            results=[],
                            latency_ms=latency / len(contributing_paths),
                            metadata={"score": score},
                        )
                    )

            # Build reasoning
            path_strs = [f"{p}:{s:.2f}" for p, s in contributing_paths.items()]
            reasoning = f"Paths: {', '.join(path_strs)} → {final_score:.3f}"

            # Calculate confidence
            confidence = min(1.0, final_score * safety_confidence)

            results.append(
                EnhancedSearchResult(
                    memory=memory,
                    final_score=final_score,
                    paths=paths,
                    reasoning=reasoning,
                    confidence=confidence,
                )
            )

        # === Cache Results ===
        if len(self._query_cache) < self.CACHE_SIZE:
            self._query_cache[cache_key] = results

        return results

    def _analyze_query(self, query: str) -> Tuple[str, Dict[str, float]]:
        """Analyze query and determine path weights"""
        query_lower = query.lower()

        # Default weights
        weights = {
            "vector": 0.4,
            "entity": 0.2,
            "hypergraph": 0.1,
            "temporal": 0.1,
            "semantic": 0.1,
            "immediate": 0.1,
        }

        # Adjust based on query type
        if any(kw in query_lower for kw in ["who is", "what is", "tell me about"]):
            # Entity-focused
            query_type = "entity"
            weights = {
                "vector": 0.2,
                "entity": 0.4,
                "hypergraph": 0.2,
                "temporal": 0.0,
                "semantic": 0.2,
                "immediate": 0.0,
            }

        elif any(kw in query_lower for kw in ["related to", "connection", "between"]):
            # Relationship-focused
            query_type = "relationship"
            weights = {
                "vector": 0.1,
                "entity": 0.3,
                "hypergraph": 0.4,
                "temporal": 0.0,
                "semantic": 0.1,
                "immediate": 0.1,
            }

        elif any(
            kw in query_lower for kw in ["yesterday", "last week", "today", "when"]
        ):
            # Temporal-focused
            query_type = "temporal"
            weights = {
                "vector": 0.3,
                "entity": 0.1,
                "hypergraph": 0.0,
                "temporal": 0.4,
                "semantic": 0.1,
                "immediate": 0.1,
            }

        elif any(kw in query_lower for kw in ["prefer", "always", "usually", "like"]):
            # Semantic/preference-focused
            query_type = "semantic"
            weights = {
                "vector": 0.2,
                "entity": 0.1,
                "hypergraph": 0.0,
                "temporal": 0.0,
                "semantic": 0.6,
                "immediate": 0.1,
            }

        elif any(kw in query_lower for kw in ["just said", "just now"]):
            # Immediate-focused
            query_type = "immediate"
            weights = {
                "vector": 0.0,
                "entity": 0.0,
                "hypergraph": 0.0,
                "temporal": 0.0,
                "semantic": 0.0,
                "immediate": 1.0,
            }

        else:
            query_type = "general"

        return query_type, weights

    def _vector_search(
        self,
        query_embeddings: Dict[str, List[float]],
        weights: Dict[str, float],
        candidate_ids: Set[str],
    ) -> Dict[str, float]:
        """Vector similarity search"""
        if not query_embeddings:
            return {}

        results = {}
        for memory_id in candidate_ids:
            if memory_id not in self._embeddings:
                continue

            score = self.weighted_similarity(
                query_embeddings, self._embeddings[memory_id], weights
            )

            if score >= self.config.similarity_threshold:
                results[memory_id] = score

        return results

    def _entity_search(
        self,
        query_entities: List[str],
        candidate_ids: Set[str],
    ) -> Dict[str, float]:
        """Entity-based search using graph"""
        results = {}

        for entity in query_entities:
            # Direct matches
            if entity in self._by_entity:
                for memory_id in self._by_entity[entity]:
                    if memory_id in candidate_ids:
                        results[memory_id] = results.get(memory_id, 0) + 0.5

            # Graph neighbors (1-hop)
            traversal = self._entity_graph.bfs(entity, max_depth=2)
            for node, depth, _ in traversal:
                if node.id in self._by_entity:
                    hop_score = 0.3 / depth
                    for memory_id in self._by_entity[node.id]:
                        if memory_id in candidate_ids:
                            results[memory_id] = results.get(memory_id, 0) + hop_score

        # Normalize
        if results:
            max_score = max(results.values())
            if max_score > 0:
                results = {k: v / max_score for k, v in results.items()}

        return results

    def _hypergraph_search(
        self,
        query_entities: List[str],
        query_embedding: Optional[List[float]],
        candidate_ids: Set[str],
    ) -> Dict[str, float]:
        """Hypergraph search with Jaccard + semantic"""
        if not query_entities:
            return {}

        he_results = self._hypergraph.search(
            query_node_ids=query_entities,
            query_embedding=query_embedding,
            top_k=50,
            alpha=0.5,
        )

        results = {}
        for he, score in he_results:
            if he.source_message_id and he.source_message_id in candidate_ids:
                results[he.source_message_id] = score

        return results

    def _temporal_search(
        self,
        query: str,
        candidate_ids: Set[str],
    ) -> Dict[str, float]:
        """Time-based search"""
        results = {}
        query_lower = query.lower()
        now = datetime.now()

        # Determine time range
        if "today" in query_lower:
            target_date = now.strftime("%Y-%m-%d")
            if target_date in self._by_date:
                for memory_id in self._by_date[target_date]:
                    if memory_id in candidate_ids:
                        results[memory_id] = 1.0

        elif "yesterday" in query_lower:
            target_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            if target_date in self._by_date:
                for memory_id in self._by_date[target_date]:
                    if memory_id in candidate_ids:
                        results[memory_id] = 1.0

        elif "last week" in query_lower:
            for i in range(7):
                target_date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
                if target_date in self._by_date:
                    score = 1.0 - (i * 0.1)
                    for memory_id in self._by_date[target_date]:
                        if memory_id in candidate_ids:
                            results[memory_id] = max(results.get(memory_id, 0), score)

        return results

    def _semantic_search(
        self,
        query_embedding: Optional[List[float]],
    ) -> Dict[str, float]:
        """Search facts/preferences"""
        if not query_embedding:
            return {}

        results = {}
        for fact_id, fact_emb in self._fact_embeddings.items():
            score = self.cosine_similarity(query_embedding, fact_emb)
            if score >= self.config.similarity_threshold:
                results[fact_id] = score

        return results

    def _immediate_search(self) -> Dict[str, float]:
        """Return immediate memory contents"""
        return {m.id: 1.0 for m in self._immediate}

    def _combine_paths(
        self,
        path_results: Dict[str, Dict[str, float]],
        path_weights: Dict[str, float],
    ) -> Dict[str, Tuple[float, Dict[str, float]]]:
        """Combine results from all paths"""
        combined = {}

        for path_name, results in path_results.items():
            weight = path_weights.get(path_name, 0)
            if weight <= 0:
                continue

            for memory_id, score in results.items():
                if memory_id not in combined:
                    combined[memory_id] = (0.0, {})

                current_score, contributing = combined[memory_id]
                weighted_score = score * weight
                new_score = current_score + weighted_score
                contributing[path_name] = score
                combined[memory_id] = (new_score, contributing)

        # Normalize and sort
        if combined:
            max_score = max(score for score, _ in combined.values())
            if max_score > 0:
                combined = {k: (v[0] / max_score, v[1]) for k, v in combined.items()}

        # Sort by score
        sorted_combined = dict(
            sorted(combined.items(), key=lambda x: x[1][0], reverse=True)
        )

        return sorted_combined

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear all memory"""
        count = len(self._memories)

        self._memories.clear()
        self._embeddings.clear()
        self._message_order.clear()
        self._by_user.clear()
        self._by_session.clear()
        self._by_entity.clear()
        self._by_date.clear()
        self._entity_graph = EntityGraph()
        self._hypergraph = HyperGraph()
        self._immediate.clear()
        self._facts.clear()
        self._fact_embeddings.clear()
        self._query_cache.clear()

        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        timestamps = [m.timestamp for m in self._memories.values() if m.timestamp]

        return MemoryStats(
            total_memories=len(self._memories),
            total_entities=len(self._by_entity),
            total_relationships=self._entity_graph.stats()["num_edges"],
            memory_size_bytes=sum(len(m.content) for m in self._memories.values()),
            oldest_memory=min(timestamps) if timestamps else None,
            newest_memory=max(timestamps) if timestamps else None,
            metadata={
                "hyperedges": self._hypergraph.stats()["num_hyperedges"],
                "facts": len(self._facts),
                "embedding_cache_size": len(self._embedding_cache),
                "query_cache_size": len(self._query_cache),
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
