"""
LangChain Memory Adapters

Wrappers around LangChain's conversation memory implementations:
- ConversationBufferMemory: Stores raw conversation history
- ConversationSummaryMemory: Summarizes conversation over time
- ConversationBufferWindowMemory: Sliding window of recent messages
"""

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

try:
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
    )
    from langchain_core.messages import AIMessage, HumanMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class LangChainBufferAdapter(MemorySystemAdapter):
    """
    Adapter for LangChain's ConversationBufferMemory.

    This stores the full conversation history in memory.
    Similar to raw context but uses LangChain's infrastructure.
    """

    def __init__(self, llm=None):
        """
        Initialize LangChain buffer memory.

        Args:
            llm: Optional LLM for summarization (not used by buffer)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not installed. Run: pip install langchain langchain-community"
            )

        self._memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="history",
        )
        self._messages: Dict[str, Memory] = {}
        self._message_order: List[str] = []

    @property
    def name(self) -> str:
        return "LangChain ConversationBufferMemory"

    @property
    def version(self) -> str:
        return "0.2.x"

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

        # Add to LangChain memory
        if role == "user":
            self._memory.chat_memory.add_user_message(content)
        else:
            self._memory.chat_memory.add_ai_message(content)

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
        LangChain buffer doesn't support search - return recent messages.
        """
        # Get messages from LangChain
        messages = self._memory.chat_memory.messages[-k:]

        results = []
        for i, msg in enumerate(reversed(messages)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            memory = Memory(
                id=f"lc_{i}",
                content=msg.content,
                role=role,
                timestamp=datetime.now(),
                memory_type=MemoryType.MESSAGE,
            )
            results.append(
                SearchResult(
                    memory=memory,
                    score=1.0 - (i * 0.1),
                    source="recency",
                    explanation="LangChain buffer (no search)",
                )
            )

        return results

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        user_id: Optional[str] = None,
    ) -> str:
        """Get context from LangChain memory"""
        # LangChain's load_memory_variables returns formatted context
        vars = self._memory.load_memory_variables({})
        history = vars.get("history", [])

        if isinstance(history, list):
            lines = []
            for msg in history:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                lines.append(f"[{role}]: {msg.content}")
            return "\n".join(lines)

        return str(history)

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear memories"""
        count = len(self._messages)
        self._memory.clear()
        self._messages.clear()
        self._message_order.clear()
        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        return MemoryStats(
            total_memories=len(self._messages),
            total_entities=0,
            total_relationships=0,
            memory_size_bytes=sum(len(m.content) for m in self._messages.values()),
            oldest_memory=None,
            newest_memory=None,
        )


class LangChainSummaryAdapter(MemorySystemAdapter):
    """
    Adapter for LangChain's ConversationSummaryMemory.

    This progressively summarizes the conversation to save context space.
    """

    def __init__(self, llm=None):
        """
        Initialize LangChain summary memory.

        Args:
            llm: LLM to use for summarization
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not installed. Run: pip install langchain langchain-community"
            )

        self._llm = llm
        self._messages: Dict[str, Memory] = {}
        self._message_order: List[str] = []

        # Will initialize memory when LLM is available
        self._memory = None
        if llm:
            self._memory = ConversationSummaryMemory(
                llm=llm,
                return_messages=True,
                memory_key="history",
            )

    @property
    def name(self) -> str:
        return "LangChain ConversationSummaryMemory"

    @property
    def version(self) -> str:
        return "0.2.x"

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
        self._messages[memory_id] = memory
        self._message_order.append(memory_id)

        # Add to LangChain memory if initialized
        if self._memory:
            if role == "user":
                self._memory.chat_memory.add_user_message(content)
            else:
                self._memory.chat_memory.add_ai_message(content)

        return memory_id

    def search(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Summary memory doesn't support search - return summary + recent"""
        results = []

        # Add summary as first result if available
        if self._memory:
            summary = self._memory.buffer
            if summary:
                results.append(
                    SearchResult(
                        memory=Memory(
                            id="summary",
                            content=summary,
                            role="system",
                            timestamp=datetime.now(),
                            memory_type=MemoryType.SUMMARY,
                        ),
                        score=1.0,
                        source="summary",
                        explanation="Conversation summary",
                    )
                )

        # Add recent messages
        recent_ids = self._message_order[-(k - 1) :]
        for i, mid in enumerate(reversed(recent_ids)):
            if mid in self._messages:
                results.append(
                    SearchResult(
                        memory=self._messages[mid],
                        score=0.9 - (i * 0.1),
                        source="recency",
                    )
                )

        return results

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        user_id: Optional[str] = None,
    ) -> str:
        """Get summarized context"""
        if self._memory:
            vars = self._memory.load_memory_variables({})
            return str(vars.get("history", ""))

        # Fallback to recent messages
        recent = self._message_order[-20:]
        lines = []
        for mid in recent:
            if mid in self._messages:
                m = self._messages[mid]
                lines.append(f"[{m.role}]: {m.content}")
        return "\n".join(lines)

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear memories"""
        count = len(self._messages)
        if self._memory:
            self._memory.clear()
        self._messages.clear()
        self._message_order.clear()
        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        return MemoryStats(
            total_memories=len(self._messages),
            total_entities=0,
            total_relationships=0,
            memory_size_bytes=sum(len(m.content) for m in self._messages.values()),
            oldest_memory=None,
            newest_memory=None,
            metadata={
                "has_summary": self._memory is not None and bool(self._memory.buffer),
            },
        )


class LangChainWindowAdapter(MemorySystemAdapter):
    """
    Adapter for LangChain's ConversationBufferWindowMemory.

    This keeps only the last k messages.
    """

    def __init__(self, k: int = 10):
        """
        Initialize LangChain window memory.

        Args:
            k: Number of messages to keep in window
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not installed. Run: pip install langchain langchain-community"
            )

        self._k = k
        self._memory = ConversationBufferWindowMemory(
            k=k,
            return_messages=True,
            memory_key="history",
        )
        self._messages: Dict[str, Memory] = {}
        self._message_order: List[str] = []

    @property
    def name(self) -> str:
        return f"LangChain WindowMemory (k={self._k})"

    @property
    def version(self) -> str:
        return "0.2.x"

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
        self._messages[memory_id] = memory
        self._message_order.append(memory_id)

        if role == "user":
            self._memory.chat_memory.add_user_message(content)
        else:
            self._memory.chat_memory.add_ai_message(content)

        return memory_id

    def search(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Return messages in window"""
        messages = self._memory.chat_memory.messages

        results = []
        for i, msg in enumerate(reversed(messages[-k:])):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            results.append(
                SearchResult(
                    memory=Memory(
                        id=f"window_{i}",
                        content=msg.content,
                        role=role,
                        timestamp=datetime.now(),
                        memory_type=MemoryType.MESSAGE,
                    ),
                    score=1.0 - (i * 0.1),
                    source="window",
                )
            )

        return results

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        user_id: Optional[str] = None,
    ) -> str:
        """Get window context"""
        vars = self._memory.load_memory_variables({})
        history = vars.get("history", [])

        if isinstance(history, list):
            lines = []
            for msg in history:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                lines.append(f"[{role}]: {msg.content}")
            return "\n".join(lines)

        return str(history)

    def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear memories"""
        count = len(self._messages)
        self._memory.clear()
        self._messages.clear()
        self._message_order.clear()
        return count

    def stats(self) -> MemoryStats:
        """Return statistics"""
        return MemoryStats(
            total_memories=len(self._messages),
            total_entities=0,
            total_relationships=0,
            memory_size_bytes=sum(len(m.content) for m in self._messages.values()),
            oldest_memory=None,
            newest_memory=None,
            metadata={"window_size": self._k},
        )
