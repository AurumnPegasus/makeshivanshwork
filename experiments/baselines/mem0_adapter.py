"""
mem0 Adapter

Wrapper for mem0 - the popular hybrid vector + graph memory system.

Features:
- Hybrid vector + graph memory
- Memory scopes: User / Session / Agent
- Automatic entity extraction
- Rerankers for improved retrieval

Performance claims:
- 26% uplift over OpenAI memory on LOCOMO benchmark
- 91% latency reduction
- 90% token reduction

See: https://mem0.ai/

Updated 2025: Uses v2 API with required filters parameter.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import (
    Entity,
    Memory,
    MemoryStats,
    MemorySystemAdapter,
    MemoryType,
    Relationship,
    SearchResult,
)

logger = logging.getLogger(__name__)

try:
    from mem0 import Memory as Mem0Memory
    from mem0 import MemoryClient

    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    logger.warning("mem0 not installed. Run: pip install mem0ai")


class Mem0Adapter(MemorySystemAdapter):
    """
    Adapter for mem0 memory system.

    mem0 uses a hybrid approach:
    - Vector embeddings for semantic search
    - Graph memory for entity relationships
    - Automatic entity extraction

    Supports multiple memory scopes:
    - User memories (persist across sessions)
    - Session memories (single conversation)
    - Agent memories (shared across users)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_cloud: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize mem0 adapter.

        Args:
            api_key: mem0 cloud API key (if using cloud)
            use_cloud: Whether to use mem0 cloud vs local
            config: Custom mem0 configuration
        """
        if not MEM0_AVAILABLE:
            raise ImportError("mem0 not installed. Run: pip install mem0ai")

        self._use_cloud = use_cloud and api_key

        if self._use_cloud:
            self._memory = MemoryClient(api_key=api_key)
        else:
            # Local mem0 with default or custom config
            self._memory = Mem0Memory(config=config) if config else Mem0Memory()

        # Track our messages for stats
        self._messages: Dict[str, Memory] = {}
        self._message_order: List[str] = []
        self._default_user_id = "benchmark_user"

    @property
    def name(self) -> str:
        return "mem0" + (" (Cloud)" if self._use_cloud else " (Local)")

    @property
    def version(self) -> str:
        return "2.0.x"

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
        """Store a message in mem0"""
        memory_id = memory_id or str(uuid.uuid4())
        user_id = user_id or self._default_user_id

        # Store in our tracking
        memory = Memory(
            id=memory_id,
            content=content,
            role=role,
            timestamp=timestamp or datetime.now(),
            memory_type=MemoryType.MESSAGE,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )
        self._messages[memory_id] = memory
        self._message_order.append(memory_id)

        # Add to mem0
        try:
            # Format the message for mem0
            messages = [{"role": role, "content": content}]

            # mem0 cloud API
            if self._use_cloud:
                result = self._memory.add(
                    messages=messages,
                    user_id=user_id,
                    metadata={
                        "session_id": session_id,
                        "timestamp": timestamp.isoformat()
                        if timestamp
                        else datetime.now().isoformat(),
                        **(metadata or {}),
                    },
                )
            else:
                # Local mem0
                result = self._memory.add(
                    messages=messages,
                    user_id=user_id,
                    metadata=metadata,
                )

            # Store mem0's ID if available
            if result:
                if isinstance(result, dict):
                    if "results" in result and result["results"]:
                        first_result = result["results"][0]
                        if isinstance(first_result, dict) and "id" in first_result:
                            memory.metadata["mem0_id"] = first_result["id"]
                    elif "id" in result:
                        memory.metadata["mem0_id"] = result["id"]

        except Exception as e:
            logger.error(f"Failed to add message to mem0: {e}")

        return memory_id

    def search(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search mem0 memories"""
        user_id = user_id or self._default_user_id
        results = []

        try:
            # Build filters - required for v2 API
            search_filters = filters or {}
            search_filters["user_id"] = user_id

            # Search mem0
            search_results = self._memory.search(
                query=query,
                filters=search_filters,
                limit=k,
            )

            # Handle different response formats
            memories_list = []
            if isinstance(search_results, dict):
                memories_list = search_results.get("results", [])
            elif isinstance(search_results, list):
                memories_list = search_results

            for i, r in enumerate(memories_list):
                # Handle different result formats
                if isinstance(r, dict):
                    content = (
                        r.get("memory") or r.get("content") or r.get("text") or str(r)
                    )
                    score = r.get("score", 1.0 - (i * 0.1))
                    metadata = r.get("metadata", {})
                    created_at = r.get("created_at") or r.get("createdAt")
                    mem_id = r.get("id", f"mem0_{i}")
                else:
                    content = str(r)
                    score = 1.0 - (i * 0.1)
                    metadata = {}
                    created_at = None
                    mem_id = f"mem0_{i}"

                # Parse timestamp
                ts = datetime.now()
                if created_at:
                    try:
                        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        pass

                memory = Memory(
                    id=mem_id,
                    content=content,
                    role="assistant",
                    timestamp=ts,
                    memory_type=MemoryType.MESSAGE,
                    metadata=metadata,
                )

                results.append(
                    SearchResult(
                        memory=memory,
                        score=float(score) if score else 0.5,
                        source="mem0_vector",
                        explanation="From mem0 vector search",
                    )
                )

        except Exception as e:
            logger.error(f"Failed to search mem0: {e}")

        return results

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        user_id: Optional[str] = None,
    ) -> str:
        """Get formatted context from mem0"""
        user_id = user_id or self._default_user_id

        lines = ["## Relevant Memories"]

        # Search for relevant memories
        search_results = self.search(query, k=10, user_id=user_id)
        for result in search_results:
            lines.append(f"- {result.memory.content}")

        # Try to get entities if available
        entities = self.get_entities(query, user_id=user_id, limit=10)
        if entities:
            lines.append("\n## Known Entities")
            for entity in entities:
                lines.append(f"- {entity.name} ({entity.entity_type})")

        return "\n".join(lines)

    def get_entities(
        self,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Entity]:
        """Get entities from mem0's graph memory"""
        user_id = user_id or self._default_user_id
        entities = []

        # mem0 cloud doesn't expose entities directly in the same way
        # This would need to be done through their graph API if available
        return entities

    def get_relationships(
        self,
        entity1: Optional[str] = None,
        entity2: Optional[str] = None,
        relation_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Relationship]:
        """Get relationships from mem0's graph memory"""
        # mem0's graph features may not be directly exposed
        return []

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear memories"""
        user_id = user_id or self._default_user_id
        count = len(self._messages)

        try:
            # Delete from mem0 using delete_all
            self._memory.delete_all(user_id=user_id)
        except Exception as e:
            logger.error(f"Failed to clear mem0: {e}")

        # Clear local tracking
        self._messages.clear()
        self._message_order.clear()

        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        mem0_count = 0

        try:
            # Try to get count from mem0 with filters
            all_memories = self._memory.get_all(
                filters={"user_id": self._default_user_id}
            )
            if isinstance(all_memories, dict):
                results = all_memories.get("results", [])
                mem0_count = len(results)
            elif isinstance(all_memories, list):
                mem0_count = len(all_memories)
        except Exception as e:
            logger.debug(f"Could not get mem0 count: {e}")

        return MemoryStats(
            total_memories=len(self._messages) + mem0_count,
            total_entities=0,
            total_relationships=0,
            memory_size_bytes=sum(len(m.content) for m in self._messages.values()),
            oldest_memory=None,
            newest_memory=None,
            metadata={
                "use_cloud": self._use_cloud,
                "mem0_count": mem0_count,
            },
        )

    def consolidate(self) -> Dict[str, Any]:
        """mem0 handles consolidation automatically"""
        return {
            "status": "automatic",
            "message": "mem0 handles consolidation automatically",
        }
