"""
Letta/MemGPT Adapter

Wrapper for Letta (formerly MemGPT) - the original agentic memory system.

Features:
- Two-tier memory: Main context + External context
- Core Memory: Always-accessible compressed facts
- Recall Memory: Searchable database for semantic search
- Archival Memory: Long-term storage for important info

See: https://docs.letta.com/concepts/memgpt/

Updated 2025: Uses letta_client SDK instead of deprecated letta package.
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
    SearchResult,
)

logger = logging.getLogger(__name__)

# Try the new letta_client SDK first
try:
    from letta import EmbeddingConfig, LLMConfig
    from letta_client import Letta

    LETTA_AVAILABLE = True
    logger.info("Letta client SDK loaded successfully")
except ImportError:
    # Fallback to old import
    try:
        from letta import EmbeddingConfig, Letta, LLMConfig

        LETTA_AVAILABLE = True
        logger.info("Letta legacy SDK loaded")
    except ImportError:
        LETTA_AVAILABLE = False
        logger.warning("Letta not installed. Run: pip install letta-client letta")


class LettaAdapter(MemorySystemAdapter):
    """
    Adapter for Letta/MemGPT memory system.

    Letta uses a sophisticated two-tier memory architecture:
    1. Main context: What's currently in the LLM context window
    2. External context: Searchable database of past memories

    The agent can use tools to:
    - Search recall memory (recent conversations)
    - Search archival memory (long-term storage)
    - Write to archival memory
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Letta adapter.

        Args:
            llm_model: Model for agent reasoning
            embedding_model: Model for embeddings
            base_url: Letta server URL (if using server mode)
            api_key: Letta API key (if using cloud)
        """
        if not LETTA_AVAILABLE:
            raise ImportError(
                "Letta not installed. Run: pip install letta-client letta"
            )

        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._api_key = api_key
        self._base_url = base_url

        # Initialize Letta client with new SDK
        if api_key:
            self._client = Letta(api_key=api_key, base_url=base_url)
        else:
            self._client = Letta(base_url=base_url)

        # Create an agent for memory operations
        self._agent = None
        self._initialize_agent()

        # Track our own messages for stats
        self._messages: Dict[str, Memory] = {}
        self._message_order: List[str] = []

    def _initialize_agent(self):
        """Create a Letta agent for memory operations"""
        try:
            # List existing agents first
            existing_agents = list(self._client.agents.list())

            # Look for an existing benchmark agent
            benchmark_agent = None
            for agent in existing_agents:
                if (
                    hasattr(agent, "name")
                    and agent.name
                    and agent.name.startswith("memory_benchmark_")
                ):
                    benchmark_agent = agent
                    break

            if benchmark_agent:
                self._agent = benchmark_agent
                logger.info(f"Reusing existing Letta agent: {self._agent.id}")
            else:
                # Create new agent
                self._agent = self._client.agents.create(
                    name=f"memory_benchmark_{uuid.uuid4().hex[:8]}",
                    include_base_tools=True,
                )
                logger.info(f"Created new Letta agent: {self._agent.id}")

        except Exception as e:
            logger.error(f"Failed to initialize Letta agent: {e}")
            self._agent = None

    @property
    def name(self) -> str:
        return "Letta/MemGPT"

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
        """Store a message via Letta"""
        memory_id = memory_id or str(uuid.uuid4())

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

        # Send to Letta agent
        if self._agent:
            try:
                # Use the messages API
                response = self._client.agents.messages.create(
                    agent_id=self._agent.id,
                    messages=[{"role": role, "content": content}],
                )
                # Store Letta's response ID if available
                if hasattr(response, "id"):
                    memory.metadata["letta_id"] = response.id
            except Exception as e:
                logger.error(f"Failed to send message to Letta: {e}")

        return memory_id

    def search(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search Letta's archival memory (passages)"""
        results = []

        if not self._agent:
            return results

        try:
            # Search passages using the new SDK API
            # First try semantic search if available
            try:
                search_results = self._client.agents.passages.search(
                    agent_id=self._agent.id,
                    query=query,
                    limit=k,
                )
                passages = list(search_results) if search_results else []
            except Exception:
                # Fall back to listing passages
                passages = list(
                    self._client.agents.passages.list(
                        agent_id=self._agent.id,
                        limit=k,
                    )
                )

            for i, passage in enumerate(passages):
                text = passage.text if hasattr(passage, "text") else str(passage)
                passage_id = passage.id if hasattr(passage, "id") else f"letta_{i}"

                memory = Memory(
                    id=passage_id,
                    content=text,
                    role="assistant",
                    timestamp=passage.created_at
                    if hasattr(passage, "created_at")
                    else datetime.now(),
                    memory_type=MemoryType.MESSAGE,
                )

                # Use score from search if available, otherwise compute overlap
                if hasattr(passage, "score") and passage.score is not None:
                    score = float(passage.score)
                else:
                    query_words = set(query.lower().split())
                    text_words = set(text.lower().split())
                    overlap = len(query_words & text_words)
                    score = overlap / max(len(query_words), 1)

                results.append(
                    SearchResult(
                        memory=memory,
                        score=max(score, 0.1),  # Minimum score
                        source="letta_passages",
                        explanation="From Letta passages memory",
                    )
                )

        except Exception as e:
            logger.error(f"Failed to search Letta: {e}")

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        user_id: Optional[str] = None,
    ) -> str:
        """Get context from Letta's memory"""
        if not self._agent:
            return ""

        try:
            # Get core memory
            core = self._client.agents.core_memory.retrieve(agent_id=self._agent.id)

            lines = ["## Core Memory"]

            if hasattr(core, "blocks"):
                for block in core.blocks:
                    if hasattr(block, "label") and hasattr(block, "value"):
                        lines.append(f"**{block.label}**: {block.value}")

            # Search for relevant memories
            lines.append("\n## Relevant Memories")
            search_results = self.search(query, k=5, user_id=user_id)
            for result in search_results:
                lines.append(f"- {result.memory.content}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Failed to get Letta context: {e}")
            return ""

    def get_entities(
        self,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Entity]:
        """
        Letta doesn't have explicit entity extraction.
        Return empty list.
        """
        return []

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear memories by deleting and recreating agent"""
        count = len(self._messages)

        # Delete agent
        if self._agent:
            try:
                self._client.agents.delete(self._agent.id)
            except Exception as e:
                logger.error(f"Failed to delete Letta agent: {e}")

        # Recreate
        self._agent = None
        self._initialize_agent()

        # Clear local tracking
        self._messages.clear()
        self._message_order.clear()

        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        passages_count = 0

        if self._agent:
            try:
                # Get passages count using new API
                passages = list(
                    self._client.agents.passages.list(
                        agent_id=self._agent.id,
                        limit=100,
                    )
                )
                passages_count = len(passages)
            except Exception:
                pass

        return MemoryStats(
            total_memories=len(self._messages) + passages_count,
            total_entities=0,
            total_relationships=0,
            memory_size_bytes=sum(len(m.content) for m in self._messages.values()),
            oldest_memory=None,
            newest_memory=None,
            metadata={
                "agent_id": self._agent.id if self._agent else None,
                "passages_count": passages_count,
            },
        )

    def consolidate(self) -> Dict[str, Any]:
        """
        Trigger Letta's memory consolidation.

        Note: Letta handles this automatically during conversations.
        """
        return {
            "status": "automatic",
            "message": "Letta handles consolidation automatically",
        }

    def __del__(self):
        """Cleanup: delete agent on destruction"""
        # Don't delete agent on destruction - let it persist for reuse
        pass
