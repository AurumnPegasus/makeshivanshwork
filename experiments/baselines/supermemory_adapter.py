"""
Supermemory Adapter

Wrapper for Supermemory - brain-inspired memory with intelligent decay.

Features:
- Brain-inspired memory engine with intelligent decay
- Dual-layer time-stamping: documentDate + eventDate
- Three-tier on Cloudflare edge
- WebGL graph visualization

Performance claims:
- State-of-the-art on LongMemEval benchmark
- Reliable recall, temporal reasoning, knowledge updates at scale

See: https://supermemory.ai/

Updated 2025: Uses official supermemory SDK v3.x.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import (
    Memory,
    MemoryStats,
    MemorySystemAdapter,
    MemoryType,
    SearchResult,
)

logger = logging.getLogger(__name__)

try:
    from supermemory import Supermemory

    SUPERMEMORY_AVAILABLE = True
    logger.info("Supermemory SDK loaded successfully")
except ImportError:
    SUPERMEMORY_AVAILABLE = False
    logger.warning("Supermemory SDK not installed. Run: pip install supermemory")


class SupermemoryAdapter(MemorySystemAdapter):
    """
    Adapter for Supermemory.

    Supermemory uses a brain-inspired approach:
    - Intelligent decay (forgetting unimportant memories)
    - Dual timestamps (document date vs event date)
    - Three-tier architecture on Cloudflare edge

    Updated for SDK v3.x which uses:
    - client.add() for adding memories
    - client.search.memories() for searching
    - client.memories.list() for listing
    - Container tags for organization
    """

    def __init__(
        self,
        api_key: str,
        container_tag: str = "benchmark",
    ):
        """
        Initialize Supermemory adapter.

        Args:
            api_key: Supermemory API key
            container_tag: Tag to organize memories (replaces container_id concept)
        """
        if not SUPERMEMORY_AVAILABLE:
            raise ImportError(
                "Supermemory SDK not installed. Run: pip install supermemory"
            )

        self._api_key = api_key
        self._container_tag = container_tag

        # Initialize client
        self._client = Supermemory(api_key=api_key)
        logger.info(
            f"Supermemory client initialized with container tag: {container_tag}"
        )

        # Track our messages for stats
        self._messages: Dict[str, Memory] = {}
        self._message_order: List[str] = []

    @property
    def name(self) -> str:
        return "Supermemory"

    @property
    def version(self) -> str:
        return "3.x"

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
        """Store a message in Supermemory"""
        memory_id = memory_id or str(uuid.uuid4())
        timestamp = timestamp or datetime.now()

        # Store in our tracking
        memory = Memory(
            id=memory_id,
            content=content,
            role=role,
            timestamp=timestamp,
            memory_type=MemoryType.MESSAGE,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )
        self._messages[memory_id] = memory
        self._message_order.append(memory_id)

        # Add to Supermemory using the new SDK
        try:
            # Build metadata
            mem_metadata = {
                "role": role,
                "local_id": memory_id,
                "timestamp": timestamp.isoformat(),
            }
            if session_id:
                mem_metadata["session_id"] = session_id
            if user_id:
                mem_metadata["user_id"] = user_id
            if metadata:
                mem_metadata.update(
                    {
                        k: v
                        for k, v in metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    }
                )

            result = self._client.add(
                content=content,
                container_tag=self._container_tag,
                custom_id=memory_id,
                metadata=mem_metadata,
            )

            # Store Supermemory's ID
            if result and hasattr(result, "id"):
                memory.metadata["supermemory_id"] = result.id

            logger.debug(f"Added memory to Supermemory: {result}")

        except Exception as e:
            logger.error(f"Failed to add memory to Supermemory: {e}")

        return memory_id

    def search(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search Supermemory"""
        results = []

        try:
            # Use the new search.memories() API
            search_result = self._client.search.memories(
                q=query,
                container_tag=self._container_tag,
                limit=k,
                rerank=True,  # Use reranking for better results
            )

            # Process results
            memories_list = (
                search_result.results if hasattr(search_result, "results") else []
            )

            for i, r in enumerate(memories_list):
                # Extract content and metadata
                content = ""
                score = 1.0 - (i * 0.1)
                metadata = {}
                mem_id = f"sm_{i}"
                ts = datetime.now()
                role = "assistant"

                if hasattr(r, "content"):
                    content = r.content or ""
                if hasattr(r, "score"):
                    score = r.score
                if hasattr(r, "id"):
                    mem_id = r.id
                if hasattr(r, "metadata") and r.metadata:
                    metadata = dict(r.metadata) if r.metadata else {}
                    role = metadata.get("role", "assistant")
                    if "timestamp" in metadata:
                        try:
                            ts = datetime.fromisoformat(
                                metadata["timestamp"].replace("Z", "+00:00")
                            )
                        except (ValueError, AttributeError):
                            pass
                if hasattr(r, "created_at") and r.created_at:
                    try:
                        ts = datetime.fromisoformat(
                            str(r.created_at).replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

                memory = Memory(
                    id=mem_id,
                    content=content,
                    role=role,
                    timestamp=ts,
                    memory_type=MemoryType.MESSAGE,
                    metadata=metadata,
                )

                results.append(
                    SearchResult(
                        memory=memory,
                        score=float(score),
                        source="supermemory",
                        explanation="From Supermemory search",
                    )
                )

        except Exception as e:
            logger.error(f"Failed to search Supermemory: {e}")

        return results

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        user_id: Optional[str] = None,
    ) -> str:
        """Get formatted context from Supermemory"""
        lines = ["## Relevant Memories"]

        # Search for relevant memories
        search_results = self.search(query, k=10, user_id=user_id)
        for result in search_results:
            role = result.memory.role
            content = result.memory.content
            if content:
                lines.append(f"[{role}]: {content}")

        return "\n".join(lines)

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear memories"""
        count = len(self._messages)

        # Delete all memories with our container tag
        try:
            # List all memories first
            memories = self._client.memories.list(
                container_tags=[self._container_tag],
                limit=1000,
            )

            # Delete each memory
            if hasattr(memories, "memories"):
                for mem in memories.memories:
                    if hasattr(mem, "id"):
                        try:
                            self._client.memories.delete(id=mem.id)
                        except Exception as e:
                            logger.debug(f"Failed to delete memory {mem.id}: {e}")

        except Exception as e:
            logger.error(f"Failed to clear Supermemory: {e}")

        # Clear local tracking
        self._messages.clear()
        self._message_order.clear()

        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        supermemory_count = 0

        try:
            # Get memory count from Supermemory
            memories = self._client.memories.list(
                container_tags=[self._container_tag],
                limit=1,  # Just to get pagination info
            )

            if hasattr(memories, "pagination"):
                supermemory_count = int(memories.pagination.total_items or 0)

        except Exception as e:
            logger.debug(f"Could not get Supermemory stats: {e}")

        return MemoryStats(
            total_memories=len(self._messages) + supermemory_count,
            total_entities=0,  # Supermemory doesn't expose entities directly
            total_relationships=0,
            memory_size_bytes=sum(len(m.content) for m in self._messages.values()),
            oldest_memory=None,
            newest_memory=None,
            metadata={
                "container_tag": self._container_tag,
                "supermemory_count": supermemory_count,
            },
        )

    def consolidate(self) -> Dict[str, Any]:
        """Supermemory handles decay automatically"""
        return {
            "status": "automatic",
            "message": "Supermemory uses intelligent decay automatically",
        }
