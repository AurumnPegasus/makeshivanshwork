"""
Approach E: Hierarchical Memory

Human-inspired memory hierarchy:

LEVEL 0 - IMMEDIATE (Working Memory)
- Last N messages (raw)
- Always in context
- No retrieval needed
- Like CPU L1 cache

LEVEL 1 - WORKING (Recent Memory)
- Recent sessions' key points
- Fast vector retrieval
- Moderate detail
- Like CPU L2 cache

LEVEL 2 - EPISODIC (Session Memory)
- Session summaries
- Compressed but complete
- Temporal organization
- Like RAM

LEVEL 3 - SEMANTIC (Long-term Memory)
- Extracted facts
- Entity knowledge
- Preferences
- Like disk storage

Query routing:
- "What did I just say?" → Level 0
- "Earlier today..." → Level 1
- "Last week..." → Level 2
- "What's Jerry's background?" → Level 3

Theoretical Basis:
- Atkinson-Shiffrin memory model (1968)
- Complementary Learning Systems (McClelland et al. 1995)
- Working memory capacity (Miller's 7±2)

This is how human memory actually works - we're biomimetic.
"""

import sys
from pathlib import Path

# Handle imports for both package and direct script execution
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from baselines.base import Memory, MemoryStats, MemoryType
from components.summarization import Summarizer, Summary

from approaches.base import (
    ApproachConfig,
    EnhancedSearchResult,
    MemoryApproach,
    RetrievalPath,
)


class HierarchicalMemory(MemoryApproach):
    """
    Hierarchical memory with multiple levels.

    Inspired by human memory architecture:
    - Immediate: Always accessible (like sensory memory)
    - Working: Recent key points (like short-term memory)
    - Episodic: Session summaries (like episodic memory)
    - Semantic: Facts and knowledge (like semantic memory)

    Query routing based on temporal and semantic cues.
    """

    # Level configuration
    IMMEDIATE_SIZE = 10  # Last N messages always available
    WORKING_WINDOW_HOURS = 24  # Recent messages for working memory
    EPISODIC_SUMMARIZE_THRESHOLD = 20  # Summarize after N messages
    SEMANTIC_CONFIDENCE_THRESHOLD = 0.8  # Min confidence for facts

    def __init__(self, config: Optional[ApproachConfig] = None):
        super().__init__(config)

        # Level 0: Immediate (ring buffer)
        self._immediate: deque = deque(maxlen=self.IMMEDIATE_SIZE)

        # Level 1: Working memory (recent, detailed)
        self._working: Dict[str, Memory] = {}
        self._working_embeddings: Dict[str, Dict[str, List[float]]] = {}

        # Level 2: Episodic memory (session summaries)
        self._episodes: Dict[str, Summary] = {}
        self._episode_embeddings: Dict[str, List[float]] = {}

        # Level 3: Semantic memory (facts, preferences)
        self._facts: Dict[str, Dict[str, Any]] = {}
        self._fact_embeddings: Dict[str, List[float]] = {}

        # All messages (for export/stats)
        self._all_memories: Dict[str, Memory] = {}
        self._message_order: List[str] = []

        # Indices
        self._by_user: Dict[str, List[str]] = {}
        self._by_session: Dict[str, List[str]] = {}

        # Session tracking for episodic summarization
        self._current_session: Optional[str] = None
        self._session_messages: List[Memory] = []

        # Summarizer
        self._summarizer: Optional[Summarizer] = None

    @property
    def name(self) -> str:
        return "Hierarchical Memory"

    @property
    def version(self) -> str:
        return "1.0.0"

    def initialize(
        self,
        embedding_model=None,
        entity_extractor=None,
        intent_classifier=None,
        llm_client=None,
    ):
        """Initialize with optional LLM for summarization"""
        super().initialize(embedding_model, entity_extractor, intent_classifier)
        if llm_client:
            self._summarizer = Summarizer(llm_client=llm_client)

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
        """Add message to appropriate hierarchy levels"""
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

        # Compute embeddings
        if embeddings:
            mem_embeddings = embeddings
        else:
            mem_embeddings = self.compute_embeddings(content, entities, intent)

        # Store in master index
        self._all_memories[memory_id] = memory
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

        # === Level 0: Immediate ===
        self._immediate.append(memory)

        # === Level 1: Working ===
        self._working[memory_id] = memory
        self._working_embeddings[memory_id] = mem_embeddings

        # Prune old working memory
        self._prune_working_memory(timestamp)

        # === Level 2: Episodic ===
        # Track session for summarization
        if session_id != self._current_session:
            # Session changed - summarize previous
            if self._session_messages:
                self._create_episode()
            self._current_session = session_id
            self._session_messages = []

        self._session_messages.append(memory)

        # Check if we should summarize
        if len(self._session_messages) >= self.EPISODIC_SUMMARIZE_THRESHOLD:
            self._create_episode()
            self._session_messages = []

        # === Level 3: Semantic ===
        # Extract facts from content
        self._extract_and_store_facts(content, entities, memory_id)

        return memory

    def _prune_working_memory(self, current_time: datetime):
        """Remove old entries from working memory"""
        cutoff = current_time - timedelta(hours=self.WORKING_WINDOW_HOURS)

        to_remove = []
        for memory_id, memory in self._working.items():
            if memory.timestamp and memory.timestamp < cutoff:
                to_remove.append(memory_id)

        for memory_id in to_remove:
            del self._working[memory_id]
            self._working_embeddings.pop(memory_id, None)

    def _create_episode(self):
        """Create episodic summary from session messages"""
        if not self._session_messages:
            return

        episode_id = f"ep_{uuid.uuid4().hex[:8]}"

        if self._summarizer:
            # Use LLM summarization
            from ..components.summarization import Message as SumMsg

            msgs = [
                SumMsg(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    timestamp=m.timestamp,
                    entities=m.entities,
                )
                for m in self._session_messages
            ]
            summary = self._summarizer.summarize(msgs, method="hybrid")
        else:
            # Simple extractive summary
            key_points = [m.content for m in self._session_messages[:5]]
            summary = Summary(
                text="\n".join(f"- {kp}" for kp in key_points),
                key_points=key_points,
                source_message_ids=[m.id for m in self._session_messages],
                source_count=len(self._session_messages),
            )

        self._episodes[episode_id] = summary

        # Embed the summary
        if self._embedding_model and summary.text:
            result = self._embedding_model.embed(
                summary.text, task_type="RETRIEVAL_DOCUMENT"
            )
            self._episode_embeddings[episode_id] = result.embedding

    def _extract_and_store_facts(
        self,
        content: str,
        entities: Optional[List[str]],
        source_id: str,
    ):
        """Extract and store semantic facts"""
        # Simple fact extraction (could use LLM for better results)
        content_lower = content.lower()

        # Preference patterns
        if any(
            p in content_lower for p in ["i prefer", "i like", "i always", "i never"]
        ):
            fact_id = f"pref_{uuid.uuid4().hex[:8]}"
            self._facts[fact_id] = {
                "type": "preference",
                "content": content,
                "source_ids": [source_id],
                "confidence": 0.9,
            }

            if self._embedding_model:
                result = self._embedding_model.embed(
                    content, task_type="RETRIEVAL_DOCUMENT"
                )
                self._fact_embeddings[fact_id] = result.embedding

        # Fact patterns (X is Y, X works at Y)
        elif entities and any(
            p in content_lower for p in [" is ", " was ", " works ", " lives "]
        ):
            fact_id = f"fact_{uuid.uuid4().hex[:8]}"
            self._facts[fact_id] = {
                "type": "fact",
                "content": content,
                "entities": entities,
                "source_ids": [source_id],
                "confidence": 0.8,
            }

            if self._embedding_model:
                result = self._embedding_model.embed(
                    content, task_type="RETRIEVAL_DOCUMENT"
                )
                self._fact_embeddings[fact_id] = result.embedding

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
        Hierarchical search with query routing.

        Routes to appropriate level(s) based on query type.
        """
        import time

        start_time = time.time()

        # Determine query routing
        levels = self._route_query(query)

        # Get query embedding
        query_embedding = None
        if self._embedding_model:
            result = self._embedding_model.embed(query, task_type="RETRIEVAL_QUERY")
            query_embedding = result.embedding

        # Collect results from each level
        all_results = []

        for level, weight in levels:
            if level == 0:
                # Immediate memory - return as is
                for memory in self._immediate:
                    all_results.append((memory, 1.0, "immediate", weight))

            elif level == 1:
                # Working memory - vector search
                if query_embedding:
                    for memory_id, embeddings in self._working_embeddings.items():
                        if "content" in embeddings:
                            score = self.cosine_similarity(
                                query_embedding, embeddings["content"]
                            )
                            if score >= self.config.similarity_threshold:
                                memory = self._working[memory_id]
                                all_results.append((memory, score, "working", weight))

            elif level == 2:
                # Episodic memory - search summaries
                if query_embedding:
                    for ep_id, embedding in self._episode_embeddings.items():
                        score = self.cosine_similarity(query_embedding, embedding)
                        if score >= self.config.similarity_threshold:
                            episode = self._episodes[ep_id]
                            # Return source messages
                            for msg_id in episode.source_message_ids[:3]:
                                if msg_id in self._all_memories:
                                    all_results.append(
                                        (
                                            self._all_memories[msg_id],
                                            score * 0.9,
                                            "episodic",
                                            weight,
                                        )
                                    )

            elif level == 3:
                # Semantic memory - search facts
                if query_embedding:
                    for fact_id, embedding in self._fact_embeddings.items():
                        score = self.cosine_similarity(query_embedding, embedding)
                        if score >= self.config.similarity_threshold:
                            fact = self._facts[fact_id]
                            # Create synthetic memory for fact
                            fact_memory = Memory(
                                id=fact_id,
                                content=fact["content"],
                                role="system",
                                timestamp=datetime.now(),
                                memory_type=MemoryType.FACT,
                                metadata={"fact_type": fact["type"]},
                            )
                            all_results.append((fact_memory, score, "semantic", weight))

        # Score and deduplicate
        scored = {}
        for memory, score, level_name, level_weight in all_results:
            weighted_score = score * level_weight
            if memory.id not in scored or weighted_score > scored[memory.id][0]:
                scored[memory.id] = (weighted_score, memory, level_name)

        # Sort by score
        sorted_results = sorted(scored.values(), key=lambda x: x[0], reverse=True)

        # Build results
        latency = (time.time() - start_time) * 1000
        results = []

        for final_score, memory, level_name in sorted_results[:k]:
            paths = []
            if return_paths:
                paths.append(
                    RetrievalPath(
                        name=level_name,
                        weight=1.0,
                        results=[],
                        latency_ms=latency,
                    )
                )

            results.append(
                EnhancedSearchResult(
                    memory=memory,
                    final_score=final_score,
                    paths=paths,
                    reasoning=f"From {level_name} memory: {final_score:.3f}",
                    confidence=final_score,
                )
            )

        return results

    def _route_query(self, query: str) -> List[Tuple[int, float]]:
        """
        Route query to appropriate memory level(s).

        Returns list of (level, weight) tuples.
        """
        query_lower = query.lower()

        # Immediate (just said, just now)
        if any(kw in query_lower for kw in ["just said", "just now", "right now"]):
            return [(0, 1.0)]

        # Working (today, earlier, recently)
        if any(
            kw in query_lower for kw in ["today", "earlier", "recently", "a moment ago"]
        ):
            return [(0, 0.3), (1, 0.7)]

        # Episodic (last week, yesterday, last time)
        if any(
            kw in query_lower
            for kw in ["last week", "yesterday", "last time", "previous"]
        ):
            return [(1, 0.3), (2, 0.7)]

        # Semantic (facts, preferences, background)
        if any(
            kw in query_lower
            for kw in ["what is", "who is", "background", "prefer", "always"]
        ):
            return [(3, 0.6), (1, 0.4)]

        # Default: search all levels
        return [(0, 0.2), (1, 0.4), (2, 0.2), (3, 0.2)]

    def consolidate(self) -> Dict[str, Any]:
        """
        Memory consolidation (like sleep for humans).

        - Summarize remaining session messages
        - Extract new facts
        - Prune old working memory
        """
        # Summarize current session if any
        if self._session_messages:
            self._create_episode()
            self._session_messages = []

        # Prune working memory
        self._prune_working_memory(datetime.now())

        return {
            "episodes_created": len(self._episodes),
            "facts_stored": len(self._facts),
            "working_memory_size": len(self._working),
        }

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear all memory levels"""
        count = len(self._all_memories)

        self._immediate.clear()
        self._working.clear()
        self._working_embeddings.clear()
        self._episodes.clear()
        self._episode_embeddings.clear()
        self._facts.clear()
        self._fact_embeddings.clear()
        self._all_memories.clear()
        self._message_order.clear()
        self._by_user.clear()
        self._by_session.clear()
        self._session_messages.clear()

        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        timestamps = [m.timestamp for m in self._all_memories.values() if m.timestamp]

        return MemoryStats(
            total_memories=len(self._all_memories),
            total_entities=0,
            total_relationships=0,
            memory_size_bytes=sum(len(m.content) for m in self._all_memories.values()),
            oldest_memory=min(timestamps) if timestamps else None,
            newest_memory=max(timestamps) if timestamps else None,
            metadata={
                "immediate_size": len(self._immediate),
                "working_size": len(self._working),
                "episodes": len(self._episodes),
                "facts": len(self._facts),
            },
        )

    def export_all(self, user_id: Optional[str] = None) -> List[Memory]:
        """Export all memories"""
        if user_id and user_id in self._by_user:
            return [
                self._all_memories[mid]
                for mid in self._by_user[user_id]
                if mid in self._all_memories
            ]
        return list(self._all_memories.values())

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get specific memory"""
        return self._all_memories.get(memory_id)
