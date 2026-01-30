"""
Raw Context Baseline

The naive approach: store all messages and return recent N as context.
This is what most systems do without memory management.
"""

import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import (
    Memory,
    MemoryStats,
    MemorySystemAdapter,
    MemoryType,
    SearchResult,
)


class RawContextAdapter(MemorySystemAdapter):
    """
    Naive baseline: store all messages, return most recent as context.

    This represents the simplest possible approach with no intelligence.
    All other systems should beat this baseline significantly.
    """

    def __init__(
        self,
        max_context_messages: int = 50,
        max_storage: int = 10000,
    ):
        """
        Initialize raw context adapter.

        Args:
            max_context_messages: Max messages to include in context
            max_storage: Max messages to store before dropping oldest
        """
        self.max_context_messages = max_context_messages
        self.max_storage = max_storage

        # Storage
        self._messages: Dict[str, Memory] = {}
        self._message_order: deque = deque(maxlen=max_storage)
        self._by_user: Dict[str, List[str]] = {}
        self._by_session: Dict[str, List[str]] = {}

    @property
    def name(self) -> str:
        return "Raw Context Baseline"

    @property
    def version(self) -> str:
        return "1.0.0"

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
        """Store a message"""
        memory_id = memory_id or str(uuid.uuid4())

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

        # Store
        self._messages[memory_id] = memory
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

        return memory_id

    def search(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        'Search' by returning most recent messages.

        This is a terrible search - it just returns the most recent messages
        without any semantic understanding. This is the baseline to beat.
        """
        # Get relevant message IDs
        if session_id and session_id in self._by_session:
            ids = self._by_session[session_id]
        elif user_id and user_id in self._by_user:
            ids = self._by_user[user_id]
        else:
            ids = list(self._message_order)

        # Get most recent k
        recent_ids = ids[-k:] if len(ids) > k else ids
        recent_ids = list(reversed(recent_ids))  # Most recent first

        # Convert to search results with decreasing scores
        results = []
        for i, memory_id in enumerate(recent_ids):
            if memory_id in self._messages:
                memory = self._messages[memory_id]
                results.append(
                    SearchResult(
                        memory=memory,
                        score=1.0 - (i * 0.1),  # Decreasing by recency
                        source="recency",
                        explanation="Retrieved by recency (no semantic search)",
                    )
                )

        return results

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Get context by dumping recent messages.

        This is exactly what we're trying to improve upon.
        """
        # Get recent messages
        if user_id and user_id in self._by_user:
            ids = self._by_user[user_id]
        else:
            ids = list(self._message_order)

        # Take most recent N
        recent_ids = ids[-self.max_context_messages :]

        # Format as context
        lines = ["## Recent Conversation History"]
        total_chars = 0
        max_chars = max_tokens * 4  # Rough token estimate

        for memory_id in recent_ids:
            if memory_id in self._messages:
                memory = self._messages[memory_id]
                line = f"[{memory.role}]: {memory.content}"

                if total_chars + len(line) > max_chars:
                    break

                lines.append(line)
                total_chars += len(line)

        return "\n".join(lines)

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear memories"""
        count = 0

        if session_id and session_id in self._by_session:
            for memory_id in self._by_session[session_id]:
                if memory_id in self._messages:
                    del self._messages[memory_id]
                    count += 1
            del self._by_session[session_id]

        elif user_id and user_id in self._by_user:
            for memory_id in self._by_user[user_id]:
                if memory_id in self._messages:
                    del self._messages[memory_id]
                    count += 1
            del self._by_user[user_id]

        else:
            count = len(self._messages)
            self._messages.clear()
            self._message_order.clear()
            self._by_user.clear()
            self._by_session.clear()

        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        timestamps = [m.timestamp for m in self._messages.values() if m.timestamp]

        return MemoryStats(
            total_memories=len(self._messages),
            total_entities=0,  # No entity extraction
            total_relationships=0,  # No relationship tracking
            memory_size_bytes=sum(len(m.content) for m in self._messages.values()),
            oldest_memory=min(timestamps) if timestamps else None,
            newest_memory=max(timestamps) if timestamps else None,
            metadata={
                "max_context_messages": self.max_context_messages,
                "num_users": len(self._by_user),
                "num_sessions": len(self._by_session),
            },
        )

    def export_all(
        self,
        user_id: Optional[str] = None,
    ) -> List[Memory]:
        """Export all memories"""
        if user_id and user_id in self._by_user:
            return [
                self._messages[mid]
                for mid in self._by_user[user_id]
                if mid in self._messages
            ]
        return list(self._messages.values())

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get specific memory"""
        return self._messages.get(memory_id)
