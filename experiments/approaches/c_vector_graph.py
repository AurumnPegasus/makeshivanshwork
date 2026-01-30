"""
Approach C: Vector + Graph Memory

Combines vector search with entity co-occurrence graph:
- Vector search for semantic similarity
- Graph traversal for relationship queries

This extends Approach B with explicit structure.

Theoretical Basis:
- Hybrid retrieval (dense + sparse/structured)
- Knowledge graphs for relationship reasoning
- Similar to mem0's approach but with multi-embedding

Key Innovation:
- Graph edges have embeddings (relationship semantics)
- Multi-path retrieval: vector + graph
- Learned combination weights
"""

import sys
from pathlib import Path

# Handle imports for both package and direct script execution
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from baselines.base import (
    Entity,
    Memory,
    MemoryStats,
    MemoryType,
    Relationship,
)
from components.graph_utils import EntityGraph

from approaches.base import (
    ApproachConfig,
    EnhancedSearchResult,
    MemoryApproach,
    RetrievalPath,
)


class VectorGraphMemory(MemoryApproach):
    """
    Vector + Graph memory for hybrid retrieval.

    Combines:
    1. Multi-vector semantic search (from Approach B)
    2. Entity co-occurrence graph
    3. Graph traversal for relationship queries

    Graph captures:
    - Entity co-occurrence in messages
    - Explicit relationships (if extracted)
    - Temporal proximity
    """

    def __init__(self, config: Optional[ApproachConfig] = None):
        super().__init__(config)

        # Message storage (same as Approach B)
        self._memories: Dict[str, Memory] = {}
        self._embeddings: Dict[str, Dict[str, List[float]]] = {}
        self._message_order: List[str] = []

        # Indices
        self._by_user: Dict[str, List[str]] = {}
        self._by_session: Dict[str, List[str]] = {}
        self._by_entity: Dict[str, List[str]] = {}

        # Entity graph
        self._graph = EntityGraph()

        # Entity to memory mapping (for graph→memory lookup)
        self._entity_memories: Dict[str, Set[str]] = defaultdict(set)

    @property
    def name(self) -> str:
        return "Vector + Graph Memory"

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
        """Add message and update graph"""
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

        # Index by user/session
        if user_id:
            if user_id not in self._by_user:
                self._by_user[user_id] = []
            self._by_user[user_id].append(memory_id)

        if session_id:
            if session_id not in self._by_session:
                self._by_session[session_id] = []
            self._by_session[session_id].append(memory_id)

        # Update entity graph
        if entities:
            self._update_graph(entities, memory_id, timestamp)

        return memory

    def _update_graph(
        self,
        entities: List[str],
        memory_id: str,
        timestamp: datetime,
    ):
        """Update entity graph with co-occurrence"""
        # Add/update nodes
        for entity in entities:
            entity_lower = entity.lower()

            # Add to graph
            self._graph.add_node(
                node_id=entity_lower,
                name=entity,
                node_type="entity",
            )

            # Track entity→memory mapping
            self._entity_memories[entity_lower].add(memory_id)

            # Index for direct lookup
            if entity_lower not in self._by_entity:
                self._by_entity[entity_lower] = []
            self._by_entity[entity_lower].append(memory_id)

        # Add edges for co-occurrence
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                e1_lower = e1.lower()
                e2_lower = e2.lower()

                self._graph.add_edge(
                    source_id=e1_lower,
                    target_id=e2_lower,
                    edge_type="mentioned_with",
                    weight=1.0,
                    bidirectional=True,
                )

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
        Hybrid search: vector + graph.

        Paths:
        1. Multi-vector similarity (content, entity, intent)
        2. Graph traversal (if entities in query)

        Final score = weighted combination.
        """
        # Get query embeddings and weights
        query_embeddings, weights = self.compute_query_embeddings(query)

        # Get query entities
        query_entities = []
        if self._entity_extractor:
            query_entities = [
                e.name.lower() for e in self._entity_extractor.extract(query)
            ]

        # Detect if this is a relationship query
        query_lower = query.lower()
        is_relationship_query = any(
            kw in query_lower
            for kw in ["related to", "connection", "relationship", "between"]
        )

        # Adjust weights for relationship queries
        if is_relationship_query:
            graph_weight = 0.6
            vector_weight = 0.4
        else:
            graph_weight = 0.3
            vector_weight = 0.7

        # === Path 1: Vector search ===
        vector_results = self._vector_search(
            query_embeddings, weights, user_id, session_id
        )

        # === Path 2: Graph search ===
        graph_results = {}
        if query_entities:
            graph_results = self._graph_search(query_entities)

        # === Combine results ===
        all_memory_ids = set(vector_results.keys()) | set(graph_results.keys())

        combined = []
        for memory_id in all_memory_ids:
            vector_score = vector_results.get(memory_id, 0.0)
            graph_score = graph_results.get(memory_id, 0.0)

            final_score = vector_weight * vector_score + graph_weight * graph_score

            combined.append((memory_id, final_score, vector_score, graph_score))

        # Sort by final score
        combined.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []

        for memory_id, final_score, vector_score, graph_score in combined[:k]:
            memory = self._memories[memory_id]

            paths = []
            if return_paths:
                if vector_score > 0:
                    paths.append(
                        RetrievalPath(
                            name="vector",
                            weight=vector_weight,
                            results=[],
                            metadata={"score": vector_score},
                        )
                    )
                if graph_score > 0:
                    paths.append(
                        RetrievalPath(
                            name="graph",
                            weight=graph_weight,
                            results=[],
                            metadata={"score": graph_score},
                        )
                    )

            reasoning = f"Vector: {vector_score:.3f} + Graph: {graph_score:.3f} = {final_score:.3f}"

            results.append(
                EnhancedSearchResult(
                    memory=memory,
                    final_score=final_score,
                    paths=paths,
                    reasoning=reasoning,
                    confidence=final_score,
                )
            )

        return results

    def _vector_search(
        self,
        query_embeddings: Dict[str, List[float]],
        weights: Dict[str, float],
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> Dict[str, float]:
        """Vector search component"""
        if not query_embeddings:
            return {}

        # Get candidates
        if session_id and session_id in self._by_session:
            candidate_ids = self._by_session[session_id]
        elif user_id and user_id in self._by_user:
            candidate_ids = self._by_user[user_id]
        else:
            candidate_ids = self._message_order

        results = {}
        for memory_id in candidate_ids:
            if memory_id not in self._embeddings:
                continue

            memory_embeddings = self._embeddings[memory_id]
            score = self.weighted_similarity(
                query_embeddings, memory_embeddings, weights
            )

            if score >= self.config.similarity_threshold:
                results[memory_id] = score

        return results

    def _graph_search(
        self,
        query_entities: List[str],
    ) -> Dict[str, float]:
        """Graph traversal component"""
        results = {}

        for entity in query_entities:
            # Direct entity matches
            if entity in self._entity_memories:
                for memory_id in self._entity_memories[entity]:
                    results[memory_id] = results.get(memory_id, 0.0) + 0.5

            # Graph neighbors (1-hop)
            traversal = self._graph.bfs(entity, max_depth=2)
            for node, depth, path in traversal:
                if node.id in self._entity_memories:
                    # Score decays with depth
                    hop_score = 0.3 / depth
                    for memory_id in self._entity_memories[node.id]:
                        results[memory_id] = results.get(memory_id, 0.0) + hop_score

        # Normalize scores
        if results:
            max_score = max(results.values())
            if max_score > 0:
                results = {k: v / max_score for k, v in results.items()}

        return results

    def get_entities(
        self,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Entity]:
        """Get entities from graph"""
        entities = []

        for node_id in list(self._graph._nodes.keys())[:limit]:
            node = self._graph._nodes[node_id]
            entities.append(
                Entity(
                    name=node.name,
                    entity_type=node.node_type,
                    mentions=node.access_count,
                    first_seen=node.created_at,
                    last_seen=node.last_accessed,
                )
            )

        return entities

    def get_relationships(
        self,
        entity1: Optional[str] = None,
        entity2: Optional[str] = None,
        relation_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Relationship]:
        """Get relationships from graph"""
        relationships = []

        if entity1 and entity2:
            # Specific relationship
            edge = self._graph.get_edge(entity1.lower(), entity2.lower())
            if edge:
                relationships.append(
                    Relationship(
                        from_entity=edge.source_id,
                        to_entity=edge.target_id,
                        relation_type=edge.edge_type,
                        confidence=1.0,
                        count=edge.occurrence_count,
                    )
                )

            # Also check path
            path = self._graph.find_path(entity1.lower(), entity2.lower())
            if path and len(path) > 1:
                relationships.append(
                    Relationship(
                        from_entity=entity1.lower(),
                        to_entity=entity2.lower(),
                        relation_type="connected_via",
                        confidence=0.5,
                        count=1,
                        metadata={"path_length": len(path)},
                    )
                )

        elif entity1:
            # All relationships for entity1
            neighbors = self._graph.get_neighbors(entity1.lower())
            for node, edge in neighbors:
                relationships.append(
                    Relationship(
                        from_entity=edge.source_id,
                        to_entity=edge.target_id,
                        relation_type=edge.edge_type,
                        confidence=1.0,
                        count=edge.occurrence_count,
                    )
                )

        return relationships

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear memories"""
        # For simplicity, clear everything if user/session specified
        # A production implementation would be more surgical
        count = len(self._memories)

        self._memories.clear()
        self._embeddings.clear()
        self._message_order.clear()
        self._by_user.clear()
        self._by_session.clear()
        self._by_entity.clear()
        self._graph = EntityGraph()
        self._entity_memories.clear()

        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        graph_stats = self._graph.stats()

        timestamps = [m.timestamp for m in self._memories.values() if m.timestamp]

        return MemoryStats(
            total_memories=len(self._memories),
            total_entities=graph_stats["num_nodes"],
            total_relationships=graph_stats["num_edges"],
            memory_size_bytes=sum(len(m.content) for m in self._memories.values()),
            oldest_memory=min(timestamps) if timestamps else None,
            newest_memory=max(timestamps) if timestamps else None,
            metadata={
                "graph": graph_stats,
                "entity_memory_coverage": len(self._entity_memories),
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
