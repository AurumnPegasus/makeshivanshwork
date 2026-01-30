"""
Graph Utilities for Memory Architecture Research

This module implements graph structures for memory:

1. **Entity Graph** - Pairwise entity relationships
   - Nodes: Entities (people, topics, etc.)
   - Edges: Relationships with types and embeddings
   - Operations: BFS, DFS, shortest path, community detection

2. **HyperGraph** - Multi-way relationships
   - Hyperedges: Connect arbitrary sets of entities
   - Key insight: A message connects ALL its entities
   - Better than pairwise for complex relationships

3. **Temporal Graph** - Time-aware relationships
   - Edges have timestamps
   - Supports temporal queries
   - Decay-weighted traversal

Key Innovation: Hypergraph representation captures that
"Meeting with Jerry about AI on Thursday" creates a single
4-way relationship, not 6 pairwise edges.

Theoretical Foundation:
- Hypergraphs generalize graphs (edges → hyperedges)
- Better for modeling n-ary relationships
- Used in MAGMA for multi-hop reasoning

References:
- Feng et al. "Hypergraph Neural Networks" AAAI 2019
- Benson et al. "Simplicial closure and higher-order link prediction" PNAS 2018
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
import heapq
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """A node in the entity graph"""
    id: str
    name: str
    node_type: str = "entity"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Temporal tracking
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.id == other.id


@dataclass
class GraphEdge:
    """An edge in the entity graph"""
    source_id: str
    target_id: str
    edge_type: str = "related_to"
    weight: float = 1.0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Temporal tracking
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    occurrence_count: int = 1

    @property
    def id(self) -> str:
        return f"{self.source_id}→{self.edge_type}→{self.target_id}"


@dataclass
class HyperEdge:
    """
    A hyperedge connecting multiple nodes.

    Unlike regular edges that connect pairs, hyperedges connect
    arbitrary sets of nodes. This better represents complex relationships.
    """
    id: str
    node_ids: FrozenSet[str]
    hyperedge_type: str = "co_occurrence"
    embedding: Optional[List[float]] = None
    source_message_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Temporal tracking
    timestamp: Optional[datetime] = None
    weight: float = 1.0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, HyperEdge) and self.id == other.id

    def contains(self, node_id: str) -> bool:
        return node_id in self.node_ids

    def overlap(self, other: "HyperEdge") -> int:
        """Count of shared nodes"""
        return len(self.node_ids & other.node_ids)

    def jaccard_similarity(self, other: "HyperEdge") -> float:
        """Jaccard similarity between hyperedges"""
        intersection = len(self.node_ids & other.node_ids)
        union = len(self.node_ids | other.node_ids)
        return intersection / union if union > 0 else 0.0


class EntityGraph:
    """
    Entity graph with relationship tracking.

    Supports:
    - Multiple edge types (mentioned_with, works_at, etc.)
    - Edge embeddings for semantic similarity
    - Temporal decay for relevance scoring
    - Efficient traversal operations
    """

    def __init__(self):
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}

        # Adjacency lists for efficient traversal
        self._outgoing: Dict[str, List[str]] = defaultdict(list)  # node_id → [edge_ids]
        self._incoming: Dict[str, List[str]] = defaultdict(list)  # node_id → [edge_ids]

        # Type indices
        self._nodes_by_type: Dict[str, Set[str]] = defaultdict(set)
        self._edges_by_type: Dict[str, Set[str]] = defaultdict(set)

    def add_node(
        self,
        node_id: str,
        name: str,
        node_type: str = "entity",
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
    ) -> GraphNode:
        """Add or update a node"""
        if node_id in self._nodes:
            # Update existing
            node = self._nodes[node_id]
            node.last_accessed = datetime.now()
            node.access_count += 1
            if embedding:
                node.embedding = embedding
            if metadata:
                node.metadata.update(metadata)
        else:
            # Create new
            node = GraphNode(
                id=node_id,
                name=name,
                node_type=node_type,
                embedding=embedding,
                metadata=metadata or {},
                created_at=datetime.now(),
                last_accessed=datetime.now(),
            )
            self._nodes[node_id] = node
            self._nodes_by_type[node_type].add(node_id)

        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "related_to",
        weight: float = 1.0,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        bidirectional: bool = True,
    ) -> GraphEdge:
        """Add or update an edge"""
        edge_id = f"{source_id}→{edge_type}→{target_id}"

        if edge_id in self._edges:
            # Update existing
            edge = self._edges[edge_id]
            edge.weight += weight
            edge.occurrence_count += 1
            edge.last_updated = datetime.now()
            if embedding:
                edge.embedding = embedding
            if metadata:
                edge.metadata.update(metadata)
        else:
            # Create new
            edge = GraphEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                embedding=embedding,
                metadata=metadata or {},
                created_at=datetime.now(),
                last_updated=datetime.now(),
            )
            self._edges[edge_id] = edge
            self._outgoing[source_id].append(edge_id)
            self._incoming[target_id].append(edge_id)
            self._edges_by_type[edge_type].add(edge_id)

        # Add reverse edge if bidirectional
        if bidirectional:
            reverse_id = f"{target_id}→{edge_type}→{source_id}"
            if reverse_id not in self._edges:
                self.add_edge(
                    target_id, source_id, edge_type, weight,
                    embedding, metadata, bidirectional=False
                )

        return edge

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID"""
        return self._nodes.get(node_id)

    def get_edge(self, source_id: str, target_id: str, edge_type: str = None) -> Optional[GraphEdge]:
        """Get edge between nodes"""
        if edge_type:
            edge_id = f"{source_id}→{edge_type}→{target_id}"
            return self._edges.get(edge_id)

        # Find any edge between nodes
        for edge_id in self._outgoing.get(source_id, []):
            edge = self._edges[edge_id]
            if edge.target_id == target_id:
                return edge
        return None

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "both",  # "out", "in", "both"
    ) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get neighboring nodes with edges"""
        neighbors = []

        if direction in ("out", "both"):
            for edge_id in self._outgoing.get(node_id, []):
                edge = self._edges[edge_id]
                if edge_type is None or edge.edge_type == edge_type:
                    if edge.target_id in self._nodes:
                        neighbors.append((self._nodes[edge.target_id], edge))

        if direction in ("in", "both"):
            for edge_id in self._incoming.get(node_id, []):
                edge = self._edges[edge_id]
                if edge_type is None or edge.edge_type == edge_type:
                    if edge.source_id in self._nodes:
                        neighbors.append((self._nodes[edge.source_id], edge))

        return neighbors

    def bfs(
        self,
        start_id: str,
        max_depth: int = 3,
        edge_types: Optional[List[str]] = None,
        node_filter: Optional[Callable[[GraphNode], bool]] = None,
    ) -> List[Tuple[GraphNode, int, List[GraphEdge]]]:
        """
        Breadth-first search from start node.

        Returns: List of (node, depth, path_edges)
        """
        if start_id not in self._nodes:
            return []

        visited = {start_id}
        queue = [(start_id, 0, [])]  # (node_id, depth, path_edges)
        results = []

        while queue:
            current_id, depth, path = queue.pop(0)

            if depth > 0:
                node = self._nodes[current_id]
                if node_filter is None or node_filter(node):
                    results.append((node, depth, path))

            if depth < max_depth:
                for neighbor, edge in self.get_neighbors(current_id):
                    if edge_types and edge.edge_type not in edge_types:
                        continue
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        queue.append((neighbor.id, depth + 1, path + [edge]))

        return results

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> Optional[List[GraphEdge]]:
        """Find shortest path between nodes"""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        visited = {source_id}
        queue = [(source_id, [])]

        while queue:
            current_id, path = queue.pop(0)

            if current_id == target_id:
                return path

            if len(path) < max_depth:
                for neighbor, edge in self.get_neighbors(current_id):
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        queue.append((neighbor.id, path + [edge]))

        return None

    def get_subgraph(
        self,
        node_ids: List[str],
        include_edges_between: bool = True,
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Extract subgraph containing specified nodes"""
        nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]
        edges = []

        if include_edges_between:
            node_set = set(node_ids)
            for edge in self._edges.values():
                if edge.source_id in node_set and edge.target_id in node_set:
                    edges.append(edge)

        return nodes, edges

    def stats(self) -> Dict[str, Any]:
        """Graph statistics"""
        return {
            "num_nodes": len(self._nodes),
            "num_edges": len(self._edges),
            "node_types": {t: len(ids) for t, ids in self._nodes_by_type.items()},
            "edge_types": {t: len(ids) for t, ids in self._edges_by_type.items()},
        }


class HyperGraph:
    """
    Hypergraph for memory representation.

    Key insight: When we say "Meeting with Jerry about AI on Thursday",
    we're creating a 4-way relationship, not 6 separate pairs.
    Hypergraphs capture this naturally.

    Operations:
    - Add hyperedge (message → set of entities)
    - Query by entity (find all hyperedges containing entity)
    - Jaccard similarity between hyperedges
    - Combined semantic + structural scoring
    """

    def __init__(self):
        self._nodes: Dict[str, GraphNode] = {}
        self._hyperedges: Dict[str, HyperEdge] = {}

        # Index: node_id → [hyperedge_ids]
        self._node_to_hyperedges: Dict[str, Set[str]] = defaultdict(set)

    def add_node(
        self,
        node_id: str,
        name: str,
        node_type: str = "entity",
        embedding: Optional[List[float]] = None,
    ) -> GraphNode:
        """Add or get a node"""
        if node_id not in self._nodes:
            self._nodes[node_id] = GraphNode(
                id=node_id,
                name=name,
                node_type=node_type,
                embedding=embedding,
                created_at=datetime.now(),
            )
        return self._nodes[node_id]

    def add_hyperedge(
        self,
        hyperedge_id: str,
        node_ids: List[str],
        hyperedge_type: str = "co_occurrence",
        embedding: Optional[List[float]] = None,
        source_message_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        weight: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> HyperEdge:
        """Add a hyperedge connecting multiple nodes"""
        node_set = frozenset(node_ids)

        hyperedge = HyperEdge(
            id=hyperedge_id,
            node_ids=node_set,
            hyperedge_type=hyperedge_type,
            embedding=embedding,
            source_message_id=source_message_id,
            timestamp=timestamp or datetime.now(),
            weight=weight,
            metadata=metadata or {},
        )

        self._hyperedges[hyperedge_id] = hyperedge

        # Update index
        for node_id in node_ids:
            self._node_to_hyperedges[node_id].add(hyperedge_id)

        return hyperedge

    def get_hyperedge(self, hyperedge_id: str) -> Optional[HyperEdge]:
        """Get hyperedge by ID"""
        return self._hyperedges.get(hyperedge_id)

    def get_hyperedges_containing(
        self,
        node_ids: List[str],
        require_all: bool = False,
    ) -> List[HyperEdge]:
        """
        Get hyperedges containing specified nodes.

        Args:
            node_ids: Nodes to search for
            require_all: If True, require ALL nodes; if False, ANY node

        Returns:
            List of matching hyperedges
        """
        if not node_ids:
            return []

        if require_all:
            # Intersection of all node hyperedge sets
            sets = [self._node_to_hyperedges.get(nid, set()) for nid in node_ids]
            if not all(sets):
                return []
            common_ids = set.intersection(*sets)
            return [self._hyperedges[hid] for hid in common_ids]
        else:
            # Union of all node hyperedge sets
            all_ids = set()
            for nid in node_ids:
                all_ids.update(self._node_to_hyperedges.get(nid, set()))
            return [self._hyperedges[hid] for hid in all_ids]

    def search(
        self,
        query_node_ids: List[str],
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        alpha: float = 0.5,  # Weight for structural vs semantic
    ) -> List[Tuple[HyperEdge, float]]:
        """
        Search hyperedges by structure and/or semantics.

        Score = α * jaccard_score + (1-α) * semantic_score

        Args:
            query_node_ids: Entities in the query
            query_embedding: Query embedding for semantic matching
            top_k: Number of results
            alpha: Balance between structural (1) and semantic (0)

        Returns:
            List of (hyperedge, score) sorted by score
        """
        query_set = frozenset(query_node_ids)
        results = []

        for hyperedge in self._hyperedges.values():
            # Structural score: Jaccard similarity
            intersection = len(query_set & hyperedge.node_ids)
            union = len(query_set | hyperedge.node_ids)
            jaccard = intersection / union if union > 0 else 0.0

            # Semantic score: cosine similarity
            semantic = 0.0
            if query_embedding and hyperedge.embedding:
                q = np.array(query_embedding)
                h = np.array(hyperedge.embedding)
                semantic = float(np.dot(q, h) / (np.linalg.norm(q) * np.linalg.norm(h)))

            # Combined score
            score = alpha * jaccard + (1 - alpha) * semantic

            if score > 0:
                results.append((hyperedge, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_related_nodes(
        self,
        node_id: str,
        min_co_occurrence: int = 1,
    ) -> List[Tuple[str, int]]:
        """
        Get nodes that co-occur with given node.

        Returns list of (node_id, co_occurrence_count).
        """
        co_occurrences: Dict[str, int] = defaultdict(int)

        for he_id in self._node_to_hyperedges.get(node_id, set()):
            hyperedge = self._hyperedges[he_id]
            for other_id in hyperedge.node_ids:
                if other_id != node_id:
                    co_occurrences[other_id] += 1

        # Filter and sort
        results = [
            (nid, count) for nid, count in co_occurrences.items()
            if count >= min_co_occurrence
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def stats(self) -> Dict[str, Any]:
        """Hypergraph statistics"""
        hyperedge_sizes = [len(he.node_ids) for he in self._hyperedges.values()]
        return {
            "num_nodes": len(self._nodes),
            "num_hyperedges": len(self._hyperedges),
            "avg_hyperedge_size": np.mean(hyperedge_sizes) if hyperedge_sizes else 0,
            "max_hyperedge_size": max(hyperedge_sizes) if hyperedge_sizes else 0,
            "min_hyperedge_size": min(hyperedge_sizes) if hyperedge_sizes else 0,
        }


class TemporalGraph(EntityGraph):
    """
    Temporal-aware entity graph with decay.

    Extends EntityGraph with:
    - Time-aware edge weights
    - Decay functions for relevance
    - Temporal queries
    """

    def __init__(
        self,
        decay_rate: float = 0.1,  # Per day
        decay_type: str = "exponential",  # exponential, linear, none
    ):
        super().__init__()
        self._decay_rate = decay_rate
        self._decay_type = decay_type

    def _compute_decay(self, edge: GraphEdge, reference_time: datetime) -> float:
        """Compute temporal decay factor for edge"""
        if self._decay_type == "none" or not edge.last_updated:
            return 1.0

        days_old = (reference_time - edge.last_updated).days

        if self._decay_type == "exponential":
            return np.exp(-self._decay_rate * days_old)
        elif self._decay_type == "linear":
            return max(0, 1 - self._decay_rate * days_old)
        else:
            return 1.0

    def get_weighted_neighbors(
        self,
        node_id: str,
        reference_time: Optional[datetime] = None,
        edge_type: Optional[str] = None,
    ) -> List[Tuple[GraphNode, GraphEdge, float]]:
        """Get neighbors with decay-weighted scores"""
        reference_time = reference_time or datetime.now()
        results = []

        for neighbor, edge in self.get_neighbors(node_id, edge_type):
            decay = self._compute_decay(edge, reference_time)
            weight = edge.weight * decay
            results.append((neighbor, edge, weight))

        # Sort by weight
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def temporal_search(
        self,
        start_id: str,
        reference_time: Optional[datetime] = None,
        max_depth: int = 3,
        min_weight: float = 0.1,
    ) -> List[Tuple[GraphNode, float, List[GraphEdge]]]:
        """
        Time-aware graph search with decay weighting.

        Uses priority queue to explore highest-weight paths first.
        """
        if start_id not in self._nodes:
            return []

        reference_time = reference_time or datetime.now()
        visited = set()

        # Priority queue: (-weight, node_id, depth, path)
        heap = [(-1.0, start_id, 0, [])]
        results = []

        while heap:
            neg_weight, current_id, depth, path = heapq.heappop(heap)
            weight = -neg_weight

            if current_id in visited:
                continue
            visited.add(current_id)

            if depth > 0 and weight >= min_weight:
                results.append((self._nodes[current_id], weight, path))

            if depth < max_depth:
                for neighbor, edge, edge_weight in self.get_weighted_neighbors(
                    current_id, reference_time
                ):
                    if neighbor.id not in visited:
                        new_weight = weight * edge_weight
                        if new_weight >= min_weight:
                            heapq.heappush(heap, (
                                -new_weight,
                                neighbor.id,
                                depth + 1,
                                path + [edge]
                            ))

        return results
