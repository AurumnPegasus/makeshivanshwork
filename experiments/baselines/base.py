"""
Abstract Base Class for Memory System Adapters

All memory systems (baselines and our approaches) implement this interface
to ensure fair, comparable evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(Enum):
    """Types of memory entries"""

    MESSAGE = "message"
    FACT = "fact"
    ENTITY = "entity"
    EPISODE = "episode"
    SUMMARY = "summary"


class IntentType(Enum):
    """Intent classification for messages"""

    COMMAND = "command"
    QUESTION = "question"
    STATEMENT = "statement"
    EMOTIONAL = "emotional"
    UNKNOWN = "unknown"


@dataclass
class Memory:
    """A single memory entry with all metadata"""

    id: str
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: datetime
    memory_type: MemoryType = MemoryType.MESSAGE

    # Optional metadata
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    intent: Optional[IntentType] = None

    # Embeddings (may be None if not computed)
    content_embedding: Optional[List[float]] = None
    entity_embedding: Optional[List[float]] = None
    intent_embedding: Optional[List[float]] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "memory_type": self.memory_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "entities": self.entities,
            "intent": self.intent.value if self.intent else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            role=data["role"],
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(),
            memory_type=MemoryType(data.get("memory_type", "message")),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            entities=data.get("entities", []),
            intent=IntentType(data["intent"]) if data.get("intent") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """A search result with score and source information"""

    memory: Memory
    score: float  # Relevance score (0-1, higher is better)
    source: str  # Which retrieval path found this (e.g., "vector", "graph", "keyword")

    # Optional: explain why this was retrieved
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "source": self.source,
            "explanation": self.explanation,
        }


@dataclass
class Entity:
    """An extracted entity with type and relationships"""

    name: str
    entity_type: str  # person, organization, topic, date, location, etc.
    mentions: int = 1
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """A relationship between two entities"""

    from_entity: str
    to_entity: str
    relation_type: str  # mentioned_with, works_at, related_to, etc.
    confidence: float = 1.0
    count: int = 1
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Statistics about the memory system"""

    total_memories: int
    total_entities: int
    total_relationships: int
    memory_size_bytes: int
    oldest_memory: Optional[datetime]
    newest_memory: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemorySystemAdapter(ABC):
    """
    Abstract base class for all memory systems.

    All memory systems must implement these methods to be benchmarked.
    This ensures fair comparison across different architectures.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this memory system"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string for reproducibility"""
        pass

    # ==================== Core Operations ====================

    @abstractmethod
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
        """
        Store a new message in memory.

        Args:
            role: 'user' or 'assistant'
            content: The message content
            session_id: Optional session identifier
            user_id: Optional user identifier
            timestamp: When the message occurred (default: now)
            metadata: Additional metadata to store
            memory_id: Optional ID to use for this memory. If not provided,
                      a new UUID will be generated.

        Returns:
            The ID of the stored memory
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for relevant memories.

        Args:
            query: The search query
            k: Number of results to return
            user_id: Filter by user
            session_id: Filter by session
            filters: Additional filters (e.g., time range, memory type)

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        pass

    @abstractmethod
    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Get formatted context for an LLM prompt.

        This is the primary interface for using memory in generation.
        Should return a well-formatted string suitable for inclusion
        in an LLM system/user prompt.

        Args:
            query: The current query/context
            max_tokens: Maximum tokens for the context
            user_id: Filter by user

        Returns:
            Formatted context string
        """
        pass

    # ==================== Entity Operations ====================

    def get_entities(
        self,
        query: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Entity]:
        """
        Get entities related to a query or all entities.

        Override this method if your system supports entity extraction.

        Args:
            query: Optional query to filter entities
            user_id: Filter by user
            limit: Maximum entities to return

        Returns:
            List of Entity objects
        """
        return []  # Default: not supported

    def get_relationships(
        self,
        entity1: Optional[str] = None,
        entity2: Optional[str] = None,
        relation_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Relationship]:
        """
        Get relationships between entities.

        Override this method if your system supports relationship tracking.

        Args:
            entity1: First entity (optional)
            entity2: Second entity (optional)
            relation_type: Filter by relationship type
            user_id: Filter by user

        Returns:
            List of Relationship objects
        """
        return []  # Default: not supported

    # ==================== Memory Management ====================

    @abstractmethod
    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """
        Clear memories.

        Args:
            user_id: Only clear this user's memories
            session_id: Only clear this session's memories

        Returns:
            Number of memories deleted
        """
        pass

    def consolidate(self) -> Dict[str, Any]:
        """
        Run memory consolidation/maintenance.

        Override this method if your system has periodic maintenance tasks.

        Returns:
            Statistics about the consolidation
        """
        return {"status": "not_supported"}

    # ==================== Introspection ====================

    @abstractmethod
    def stats(self) -> MemoryStats:
        """
        Return statistics about the memory system.

        Returns:
            MemoryStats object with counts and sizes
        """
        pass

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a specific memory by ID.

        Override this method if your system supports direct memory access.

        Args:
            memory_id: The memory ID

        Returns:
            Memory object or None if not found
        """
        return None  # Default: not supported

    def export_all(
        self,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        """
        Export all memories for analysis.

        Override this method if your system supports bulk export.

        Args:
            user_id: Filter by user

        Returns:
            List of all Memory objects
        """
        return []  # Default: not supported

    # ==================== Batch Operations ====================

    def add_messages_batch(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Add multiple messages in batch.

        Default implementation calls add_message in a loop.
        Override for more efficient batch processing.

        Args:
            messages: List of message dicts with 'role', 'content', etc.

        Returns:
            List of memory IDs
        """
        ids = []
        for msg in messages:
            memory_id = self.add_message(
                role=msg["role"],
                content=msg["content"],
                session_id=msg.get("session_id"),
                user_id=msg.get("user_id"),
                timestamp=msg.get("timestamp"),
                metadata=msg.get("metadata"),
            )
            ids.append(memory_id)
        return ids

    # ==================== Context Methods ====================

    def __enter__(self):
        """Support context manager protocol"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up on exit"""
        pass

    def __repr__(self) -> str:
        return f"<{self.name} v{self.version}>"
