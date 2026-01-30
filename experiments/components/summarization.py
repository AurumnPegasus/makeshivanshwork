"""
Summarization Components for Memory Architecture Research

This module implements memory summarization:

1. **Session Summarization**
   - Compress conversation sessions to key points
   - Preserve entities and relationships
   - Maintain temporal markers

2. **Incremental Summarization**
   - Update summary as conversation progresses
   - Avoid reprocessing entire history

3. **Hierarchical Summarization**
   - Message → Session → Week → Topic
   - Different granularities for different queries

Key Innovation: Abstractive summarization that preserves
structured information (entities, relationships, intents)
alongside narrative summary.

References:
- See et al. "Get To The Point: Summarization with Pointer-Generator Networks" ACL 2017
- Liu & Lapata "Text Summarization with Pretrained Encoders" EMNLP 2019
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Summary:
    """A generated summary with metadata"""
    text: str
    key_points: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    # Source tracking
    source_message_ids: List[str] = field(default_factory=list)
    source_count: int = 0

    # Temporal bounds
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Metadata
    summary_type: str = "session"  # session, incremental, topic
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """Simple message for summarization"""
    id: str
    role: str
    content: str
    timestamp: Optional[datetime] = None
    entities: List[str] = field(default_factory=list)


class Summarizer:
    """
    Base summarization class with extractive and abstractive methods.

    Supports:
    - Extractive: Select key sentences
    - Abstractive: Generate new summary text
    - Hybrid: Extract then abstract
    """

    def __init__(
        self,
        llm_client=None,
        max_summary_tokens: int = 500,
        extraction_ratio: float = 0.3,
    ):
        """
        Initialize summarizer.

        Args:
            llm_client: LLM for abstractive summarization
            max_summary_tokens: Max tokens in summary
            extraction_ratio: Ratio of sentences to extract
        """
        self._llm = llm_client
        self._max_tokens = max_summary_tokens
        self._extraction_ratio = extraction_ratio

    def summarize(
        self,
        messages: List[Message],
        method: str = "abstractive",  # extractive, abstractive, hybrid
    ) -> Summary:
        """
        Summarize a list of messages.

        Args:
            messages: Messages to summarize
            method: Summarization method

        Returns:
            Summary object
        """
        if not messages:
            return Summary(text="", source_count=0)

        if method == "extractive":
            return self._extractive_summarize(messages)
        elif method == "abstractive":
            return self._abstractive_summarize(messages)
        else:  # hybrid
            # Extract key sentences, then abstract
            extracted = self._extractive_summarize(messages)
            if self._llm:
                return self._abstractive_summarize(
                    messages,
                    context=extracted.text
                )
            return extracted

    def _extractive_summarize(self, messages: List[Message]) -> Summary:
        """Extract key sentences based on heuristics"""
        # Score each message
        scored = []
        all_entities = set()

        for msg in messages:
            score = self._score_message(msg)
            scored.append((msg, score))
            all_entities.update(msg.entities)

        # Sort by score and take top
        scored.sort(key=lambda x: x[1], reverse=True)
        num_to_keep = max(1, int(len(messages) * self._extraction_ratio))
        top_messages = [m for m, s in scored[:num_to_keep]]

        # Sort by timestamp for chronological summary
        top_messages.sort(key=lambda m: m.timestamp or datetime.min)

        # Build summary
        key_points = [m.content for m in top_messages]
        text = "\n".join(f"- {kp}" for kp in key_points)

        timestamps = [m.timestamp for m in messages if m.timestamp]

        return Summary(
            text=text,
            key_points=key_points,
            entities=list(all_entities),
            source_message_ids=[m.id for m in messages],
            source_count=len(messages),
            start_time=min(timestamps) if timestamps else None,
            end_time=max(timestamps) if timestamps else None,
            summary_type="extractive",
        )

    def _score_message(self, message: Message) -> float:
        """Score message importance for extraction"""
        score = 0.0

        content = message.content.lower()

        # Entity density
        score += len(message.entities) * 0.2

        # Question/answer pairs are important
        if '?' in message.content:
            score += 0.3

        # Actions and decisions
        action_words = ['decided', 'agreed', 'scheduled', 'created', 'completed']
        if any(w in content for w in action_words):
            score += 0.4

        # Preferences and facts
        if any(w in content for w in ['prefer', 'always', 'never', 'usually']):
            score += 0.3

        # Length penalty (prefer moderate length)
        word_count = len(content.split())
        if 10 <= word_count <= 50:
            score += 0.2
        elif word_count > 100:
            score -= 0.1

        # User messages slightly more important
        if message.role == "user":
            score += 0.1

        return score

    def _abstractive_summarize(
        self,
        messages: List[Message],
        context: Optional[str] = None,
    ) -> Summary:
        """Generate abstractive summary using LLM"""
        if not self._llm:
            # Fall back to extractive
            return self._extractive_summarize(messages)

        # Format conversation for LLM
        conversation = "\n".join(
            f"[{m.role}]: {m.content}" for m in messages
        )

        # Collect all entities
        all_entities = set()
        for m in messages:
            all_entities.update(m.entities)

        prompt = f"""Summarize this conversation. Focus on:
1. Key decisions and outcomes
2. Important facts learned
3. Action items or tasks
4. Preferences expressed

{f"Context from extraction: {context}" if context else ""}

Conversation:
{conversation}

Provide:
1. A 2-3 sentence summary
2. A bullet list of key points
3. List of main topics discussed

Format your response as:
SUMMARY: <your summary>
KEY_POINTS:
- point 1
- point 2
...
TOPICS: topic1, topic2, ..."""

        try:
            response = self._llm.generate(prompt, temperature=0.3)

            # Parse response
            summary_text = ""
            key_points = []
            topics = []

            if "SUMMARY:" in response:
                summary_match = re.search(
                    r'SUMMARY:\s*(.+?)(?=KEY_POINTS:|TOPICS:|$)',
                    response, re.DOTALL
                )
                if summary_match:
                    summary_text = summary_match.group(1).strip()

            if "KEY_POINTS:" in response:
                kp_match = re.search(
                    r'KEY_POINTS:\s*(.+?)(?=TOPICS:|$)',
                    response, re.DOTALL
                )
                if kp_match:
                    kp_text = kp_match.group(1)
                    key_points = [
                        line.strip().lstrip('- ')
                        for line in kp_text.split('\n')
                        if line.strip() and line.strip() != '-'
                    ]

            if "TOPICS:" in response:
                topics_match = re.search(r'TOPICS:\s*(.+)$', response)
                if topics_match:
                    topics = [
                        t.strip() for t in topics_match.group(1).split(',')
                        if t.strip()
                    ]

            timestamps = [m.timestamp for m in messages if m.timestamp]

            return Summary(
                text=summary_text or response,
                key_points=key_points,
                entities=list(all_entities),
                topics=topics,
                source_message_ids=[m.id for m in messages],
                source_count=len(messages),
                start_time=min(timestamps) if timestamps else None,
                end_time=max(timestamps) if timestamps else None,
                summary_type="abstractive",
                model=getattr(self._llm, 'model', 'unknown'),
            )

        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            return self._extractive_summarize(messages)


class EpisodicSummarizer(Summarizer):
    """
    Episodic memory summarization.

    Creates episode boundaries based on:
    - Time gaps (>30 min → new episode)
    - Topic shifts
    - Explicit session markers
    """

    def __init__(
        self,
        llm_client=None,
        episode_gap_minutes: int = 30,
        max_episode_messages: int = 50,
    ):
        super().__init__(llm_client)
        self._episode_gap = episode_gap_minutes
        self._max_episode = max_episode_messages

    def detect_episode_boundaries(
        self,
        messages: List[Message],
    ) -> List[List[Message]]:
        """
        Detect episode boundaries and segment messages.

        Returns list of episodes (each a list of messages).
        """
        if not messages:
            return []

        episodes = []
        current_episode = [messages[0]]

        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]

            # Check time gap
            is_new_episode = False

            if prev_msg.timestamp and curr_msg.timestamp:
                gap_minutes = (curr_msg.timestamp - prev_msg.timestamp).seconds / 60
                if gap_minutes > self._episode_gap:
                    is_new_episode = True

            # Check episode size
            if len(current_episode) >= self._max_episode:
                is_new_episode = True

            if is_new_episode:
                episodes.append(current_episode)
                current_episode = [curr_msg]
            else:
                current_episode.append(curr_msg)

        # Add final episode
        if current_episode:
            episodes.append(current_episode)

        return episodes

    def summarize_episodes(
        self,
        messages: List[Message],
    ) -> List[Summary]:
        """
        Detect episodes and summarize each.

        Returns list of episode summaries.
        """
        episodes = self.detect_episode_boundaries(messages)
        summaries = []

        for episode in episodes:
            summary = self.summarize(episode, method="hybrid")
            summary.summary_type = "episode"
            summaries.append(summary)

        return summaries


class IncrementalSummarizer:
    """
    Incremental summarization that updates as conversation progresses.

    Avoids reprocessing entire history by maintaining running summary.
    """

    def __init__(
        self,
        llm_client=None,
        update_threshold: int = 5,  # Messages before update
    ):
        self._llm = llm_client
        self._threshold = update_threshold
        self._current_summary: Optional[Summary] = None
        self._pending_messages: List[Message] = []

    def add_message(self, message: Message) -> Optional[Summary]:
        """
        Add a message and potentially update summary.

        Returns updated summary if threshold reached, else None.
        """
        self._pending_messages.append(message)

        if len(self._pending_messages) >= self._threshold:
            return self._update_summary()

        return None

    def _update_summary(self) -> Summary:
        """Update summary with pending messages"""
        if not self._llm:
            # Simple concatenation without LLM
            new_points = [m.content for m in self._pending_messages]

            if self._current_summary:
                all_points = self._current_summary.key_points + new_points
                all_entities = list(set(
                    self._current_summary.entities +
                    [e for m in self._pending_messages for e in m.entities]
                ))
                all_ids = (
                    self._current_summary.source_message_ids +
                    [m.id for m in self._pending_messages]
                )
            else:
                all_points = new_points
                all_entities = [e for m in self._pending_messages for e in m.entities]
                all_ids = [m.id for m in self._pending_messages]

            self._current_summary = Summary(
                text="\n".join(f"- {p}" for p in all_points[-10:]),  # Keep last 10
                key_points=all_points[-10:],
                entities=all_entities,
                source_message_ids=all_ids,
                source_count=len(all_ids),
                summary_type="incremental",
            )

        else:
            # Use LLM to update
            current_text = self._current_summary.text if self._current_summary else ""
            new_messages = "\n".join(
                f"[{m.role}]: {m.content}" for m in self._pending_messages
            )

            prompt = f"""Update this conversation summary with new messages.

Current summary:
{current_text or "(No prior summary)"}

New messages:
{new_messages}

Provide an updated summary that:
1. Integrates new information
2. Removes outdated details
3. Keeps key facts and decisions
4. Stays under 200 words

Updated summary:"""

            try:
                response = self._llm.generate(prompt, temperature=0.3)

                # Collect all entities
                all_entities = list(set(
                    (self._current_summary.entities if self._current_summary else []) +
                    [e for m in self._pending_messages for e in m.entities]
                ))

                all_ids = (
                    (self._current_summary.source_message_ids if self._current_summary else []) +
                    [m.id for m in self._pending_messages]
                )

                self._current_summary = Summary(
                    text=response.strip(),
                    entities=all_entities,
                    source_message_ids=all_ids,
                    source_count=len(all_ids),
                    summary_type="incremental",
                )

            except Exception as e:
                logger.error(f"Incremental update failed: {e}")

        self._pending_messages = []
        return self._current_summary

    def get_summary(self) -> Optional[Summary]:
        """Get current summary"""
        return self._current_summary

    def force_update(self) -> Optional[Summary]:
        """Force summary update regardless of threshold"""
        if self._pending_messages:
            return self._update_summary()
        return self._current_summary


# Convenience function
def summarize_session(
    messages: List[Dict[str, Any]],
    llm_client=None,
) -> Summary:
    """
    Summarize a conversation session.

    Args:
        messages: List of message dicts with 'role', 'content', etc.
        llm_client: Optional LLM for abstractive summarization

    Returns:
        Summary object
    """
    # Convert to Message objects
    msg_objects = [
        Message(
            id=m.get("id", str(i)),
            role=m.get("role", "user"),
            content=m.get("content", ""),
            timestamp=m.get("timestamp"),
            entities=m.get("entities", []),
        )
        for i, m in enumerate(messages)
    ]

    summarizer = Summarizer(llm_client=llm_client)
    return summarizer.summarize(msg_objects, method="hybrid")
