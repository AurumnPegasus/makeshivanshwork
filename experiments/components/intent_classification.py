"""
Intent Classification for Memory Architecture Research

CRITICAL COMPONENT: This is THE key innovation for safety.

Problem: "Why did you delete my task?" should NOT delete the task.
Current systems fail this - they interpret questions as commands.

Our Solution: Multi-level intent classification with compositional semantics.

Taxonomy:
1. COMMAND - Direct action request
   - imperative: "Delete the task"
   - polite_request: "Could you delete the task?"
   - indirect_command: "I need that task deleted"

2. QUESTION - Information seeking
   - factual: "What tasks do I have?"
   - explanation: "Why did you delete it?" (NOT a command!)
   - hypothetical: "What if you deleted it?"
   - rhetorical: "Who would delete that?"

3. STATEMENT - Information sharing
   - fact: "The meeting is at 3pm"
   - opinion: "I think we should reschedule"
   - preference: "I prefer morning meetings"

4. EMOTIONAL - Emotional expression
   - positive: "That's great!"
   - negative: "That's frustrating"
   - neutral: "Interesting"

Key Innovation: Compositional semantics distinguish:
- "Why did you X?" → question about past action (QUESTION.explanation)
- "Can you X?" → polite command (COMMAND.polite_request)
- "What if you X?" → hypothetical (QUESTION.hypothetical)

This is novel - existing systems use flat classification.

References:
- Braun et al. "Evaluating Natural Language Understanding Services" 2017
- Larson et al. "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction" EMNLP 2019
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Primary intent categories"""
    COMMAND = "command"
    QUESTION = "question"
    STATEMENT = "statement"
    EMOTIONAL = "emotional"
    UNKNOWN = "unknown"


class IntentSubtype(Enum):
    """Subtypes for fine-grained classification"""

    # Command subtypes
    IMPERATIVE = "imperative"
    POLITE_REQUEST = "polite_request"
    INDIRECT_COMMAND = "indirect_command"

    # Question subtypes
    FACTUAL = "factual"
    EXPLANATION = "explanation"
    HYPOTHETICAL = "hypothetical"
    RHETORICAL = "rhetorical"
    CONFIRMATION = "confirmation"

    # Statement subtypes
    FACT = "fact"
    OPINION = "opinion"
    PREFERENCE = "preference"
    UPDATE = "update"

    # Emotional subtypes
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

    # Unknown
    NONE = "none"


@dataclass
class Intent:
    """Classification result with confidence"""
    type: IntentType
    subtype: IntentSubtype
    confidence: float
    action_safe: bool  # True if safe to execute actions
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.type.value}.{self.subtype.value}"


@dataclass
class ClassificationResult:
    """Full classification result"""
    intent: Intent
    text: str
    tokens: List[str] = field(default_factory=list)
    alternatives: List[Intent] = field(default_factory=list)
    processing_time_ms: float = 0.0


class IntentClassifier:
    """
    Multi-level intent classifier with compositional semantics.

    Uses a hybrid approach:
    1. Rule-based patterns (fast, high precision for key cases)
    2. Feature-based classifier (moderate accuracy)
    3. LLM classifier (high accuracy, slower)

    Critical Safety: When in doubt, classify as QUESTION (non-executable).
    """

    # Patterns that indicate questions about past actions (NOT commands!)
    EXPLANATION_PATTERNS = [
        r'^why\s+(did|didn\'t|have|haven\'t|would|wouldn\'t)\s+(you|i|we)',
        r'^how\s+(did|could|would|can)\s+(you|i|we)',
        r'^what\s+(did|made|caused)\s+(you|it)',
        r'(why|how|what)\s+(happened|occurred)',
    ]

    # Patterns that indicate hypotheticals (NOT commands!)
    HYPOTHETICAL_PATTERNS = [
        r'^what\s+if\s+(you|i|we)',
        r'^what\s+would\s+happen\s+if',
        r'^how\s+would\s+(you|i|we)',
        r'^suppose\s+(you|i|we)',
        r'^imagine\s+if',
        r'^hypothetically',
    ]

    # Patterns that indicate actual commands
    COMMAND_PATTERNS = [
        r'^(please\s+)?(add|create|delete|remove|update|modify|change|set)\s+',
        r'^(please\s+)?(do|make|run|execute|perform|complete)\s+',
        r'^(please\s+)?(send|email|call|message|notify)\s+',
        r'^(i\s+)?(want|need|would\s+like)\s+(you\s+)?to\s+',
        r'^can\s+you\s+(please\s+)?(add|create|delete|remove|update|send)',
        r'^could\s+you\s+(please\s+)?(add|create|delete|remove|update|send)',
    ]

    # Patterns for factual questions
    FACTUAL_PATTERNS = [
        r'^what\s+(is|are|was|were)\s+',
        r'^who\s+(is|are|was|were)\s+',
        r'^where\s+(is|are|was|were)\s+',
        r'^when\s+(is|are|was|were|did)\s+',
        r'^how\s+many\s+',
        r'^how\s+much\s+',
        r'^do\s+(you|i|we)\s+have\s+',
        r'^is\s+there\s+',
        r'^are\s+there\s+',
    ]

    # Patterns for confirmation questions
    CONFIRMATION_PATTERNS = [
        r'^(is|are|was|were|do|did|does|has|have|can|could|will|would)\s+(it|this|that|the)',
        r'\?\s*$',  # Ends with question mark
        r'^right\?$',
        r'^correct\?$',
        r'^(is\s+that|does\s+that)\s+(right|correct|true)',
    ]

    # Emotional patterns
    EMOTIONAL_PATTERNS = {
        'positive': [r'^(great|awesome|excellent|wonderful|thanks|thank\s+you|perfect)'],
        'negative': [r'^(ugh|damn|darn|frustrating|annoying|terrible|awful)'],
    }

    def __init__(
        self,
        use_llm: bool = True,
        llm_client=None,
        confidence_threshold: float = 0.8,
    ):
        """
        Initialize intent classifier.

        Args:
            use_llm: Use LLM for complex cases
            llm_client: LLM client
            confidence_threshold: Below this, use LLM fallback
        """
        self._use_llm = use_llm
        self._llm = llm_client
        self._threshold = confidence_threshold

    def classify(self, text: str) -> Intent:
        """
        Classify intent of text.

        Returns Intent with type, subtype, and action_safe flag.
        """
        text_lower = text.lower().strip()

        # 1. Check explanation patterns first (CRITICAL for safety)
        for pattern in self.EXPLANATION_PATTERNS:
            if re.search(pattern, text_lower):
                return Intent(
                    type=IntentType.QUESTION,
                    subtype=IntentSubtype.EXPLANATION,
                    confidence=0.95,
                    action_safe=False,
                    reasoning=f"Matched explanation pattern: {pattern}",
                )

        # 2. Check hypothetical patterns (CRITICAL for safety)
        for pattern in self.HYPOTHETICAL_PATTERNS:
            if re.search(pattern, text_lower):
                return Intent(
                    type=IntentType.QUESTION,
                    subtype=IntentSubtype.HYPOTHETICAL,
                    confidence=0.95,
                    action_safe=False,
                    reasoning=f"Matched hypothetical pattern: {pattern}",
                )

        # 3. Check factual question patterns
        for pattern in self.FACTUAL_PATTERNS:
            if re.search(pattern, text_lower):
                return Intent(
                    type=IntentType.QUESTION,
                    subtype=IntentSubtype.FACTUAL,
                    confidence=0.85,
                    action_safe=False,
                    reasoning=f"Matched factual pattern: {pattern}",
                )

        # 4. Check command patterns
        for pattern in self.COMMAND_PATTERNS:
            if re.search(pattern, text_lower):
                # Determine command subtype
                if re.search(r'^(can|could|would)\s+you', text_lower):
                    subtype = IntentSubtype.POLITE_REQUEST
                elif re.search(r'^(i\s+)?(want|need|would)', text_lower):
                    subtype = IntentSubtype.INDIRECT_COMMAND
                else:
                    subtype = IntentSubtype.IMPERATIVE

                return Intent(
                    type=IntentType.COMMAND,
                    subtype=subtype,
                    confidence=0.9,
                    action_safe=True,
                    reasoning=f"Matched command pattern: {pattern}",
                )

        # 5. Check confirmation patterns
        for pattern in self.CONFIRMATION_PATTERNS:
            if re.search(pattern, text_lower):
                return Intent(
                    type=IntentType.QUESTION,
                    subtype=IntentSubtype.CONFIRMATION,
                    confidence=0.8,
                    action_safe=False,
                    reasoning=f"Matched confirmation pattern: {pattern}",
                )

        # 6. Check emotional patterns
        for sentiment, patterns in self.EMOTIONAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    subtype = {
                        'positive': IntentSubtype.POSITIVE,
                        'negative': IntentSubtype.NEGATIVE,
                    }.get(sentiment, IntentSubtype.NEUTRAL)

                    return Intent(
                        type=IntentType.EMOTIONAL,
                        subtype=subtype,
                        confidence=0.85,
                        action_safe=False,
                        reasoning=f"Matched emotional pattern: {pattern}",
                    )

        # 7. Feature-based heuristics
        intent = self._feature_classify(text, text_lower)
        if intent.confidence >= self._threshold:
            return intent

        # 8. LLM fallback for ambiguous cases
        if self._use_llm and self._llm and intent.confidence < self._threshold:
            llm_intent = self._llm_classify(text)
            if llm_intent.confidence > intent.confidence:
                return llm_intent

        return intent

    def _feature_classify(self, text: str, text_lower: str) -> Intent:
        """Feature-based classification"""

        # Ends with question mark
        if text.rstrip().endswith('?'):
            return Intent(
                type=IntentType.QUESTION,
                subtype=IntentSubtype.FACTUAL,
                confidence=0.7,
                action_safe=False,
                reasoning="Ends with question mark",
            )

        # Starts with imperative verb
        imperative_verbs = [
            'add', 'create', 'delete', 'remove', 'update', 'set',
            'send', 'email', 'call', 'schedule', 'remind', 'tell',
            'show', 'list', 'find', 'search', 'get', 'make',
        ]
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in imperative_verbs:
            return Intent(
                type=IntentType.COMMAND,
                subtype=IntentSubtype.IMPERATIVE,
                confidence=0.75,
                action_safe=True,
                reasoning=f"Starts with imperative verb: {first_word}",
            )

        # Contains preference indicators
        if re.search(r'\b(i\s+)?(prefer|like|want|need)\b', text_lower):
            if re.search(r'(you\s+to|please)', text_lower):
                return Intent(
                    type=IntentType.COMMAND,
                    subtype=IntentSubtype.INDIRECT_COMMAND,
                    confidence=0.7,
                    action_safe=True,
                    reasoning="Indirect command with preference",
                )
            else:
                return Intent(
                    type=IntentType.STATEMENT,
                    subtype=IntentSubtype.PREFERENCE,
                    confidence=0.75,
                    action_safe=False,
                    reasoning="Preference statement",
                )

        # Default: statement with low confidence
        return Intent(
            type=IntentType.STATEMENT,
            subtype=IntentSubtype.FACT,
            confidence=0.5,
            action_safe=False,  # Conservative: don't execute actions
            reasoning="Default classification (low confidence)",
        )

    def _llm_classify(self, text: str) -> Intent:
        """LLM-based classification for ambiguous cases"""
        if not self._llm:
            return Intent(
                type=IntentType.UNKNOWN,
                subtype=IntentSubtype.NONE,
                confidence=0.0,
                action_safe=False,
            )

        prompt = f"""Classify the intent of this message.

CRITICAL: Distinguish between:
- Questions ABOUT past actions (e.g., "Why did you delete that?") - these are QUESTIONS, not commands
- Hypothetical questions (e.g., "What if you added X?") - these are QUESTIONS, not commands
- Actual commands (e.g., "Delete that" or "Can you delete that?")

Message: "{text}"

Classify as one of:
1. COMMAND.imperative - Direct command ("Delete the task")
2. COMMAND.polite_request - Polite command ("Could you delete the task?")
3. COMMAND.indirect_command - Indirect command ("I need that deleted")
4. QUESTION.factual - Seeking information ("What tasks do I have?")
5. QUESTION.explanation - Asking why/how something happened ("Why did you delete it?")
6. QUESTION.hypothetical - Hypothetical scenario ("What if you deleted it?")
7. QUESTION.confirmation - Confirming understanding ("Is that right?")
8. STATEMENT.fact - Sharing information ("The meeting is at 3pm")
9. STATEMENT.preference - Expressing preference ("I prefer morning meetings")
10. EMOTIONAL.positive - Positive expression ("Great!")
11. EMOTIONAL.negative - Negative expression ("That's frustrating")

Return ONLY the classification (e.g., "QUESTION.explanation") and nothing else."""

        try:
            response = self._llm.generate(prompt, temperature=0.0)
            response = response.strip().upper()

            # Parse response
            parts = response.replace(".", "_").split("_")
            if len(parts) >= 2:
                type_str = parts[0].lower()
                subtype_str = "_".join(parts[1:]).lower()

                type_map = {
                    "command": IntentType.COMMAND,
                    "question": IntentType.QUESTION,
                    "statement": IntentType.STATEMENT,
                    "emotional": IntentType.EMOTIONAL,
                }

                subtype_map = {
                    "imperative": IntentSubtype.IMPERATIVE,
                    "polite_request": IntentSubtype.POLITE_REQUEST,
                    "indirect_command": IntentSubtype.INDIRECT_COMMAND,
                    "factual": IntentSubtype.FACTUAL,
                    "explanation": IntentSubtype.EXPLANATION,
                    "hypothetical": IntentSubtype.HYPOTHETICAL,
                    "confirmation": IntentSubtype.CONFIRMATION,
                    "fact": IntentSubtype.FACT,
                    "preference": IntentSubtype.PREFERENCE,
                    "positive": IntentSubtype.POSITIVE,
                    "negative": IntentSubtype.NEGATIVE,
                }

                intent_type = type_map.get(type_str, IntentType.UNKNOWN)
                intent_subtype = subtype_map.get(subtype_str, IntentSubtype.NONE)

                # Commands are action_safe, others are not
                action_safe = intent_type == IntentType.COMMAND

                return Intent(
                    type=intent_type,
                    subtype=intent_subtype,
                    confidence=0.9,
                    action_safe=action_safe,
                    reasoning=f"LLM classification: {response}",
                )

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        return Intent(
            type=IntentType.UNKNOWN,
            subtype=IntentSubtype.NONE,
            confidence=0.0,
            action_safe=False,
            reasoning="LLM classification failed",
        )

    def is_safe_to_execute(self, text: str) -> Tuple[bool, str]:
        """
        Check if it's safe to execute actions for this text.

        Returns (is_safe, reason).
        """
        intent = self.classify(text)

        if not intent.action_safe:
            return False, f"Intent classified as {intent.label} (not a command)"

        if intent.confidence < 0.7:
            return False, f"Low confidence ({intent.confidence:.2f}) - asking for clarification"

        return True, f"Safe to execute: {intent.label} with confidence {intent.confidence:.2f}"


# Convenience function
def classify_intent(
    text: str,
    classifier: Optional[IntentClassifier] = None,
) -> Intent:
    """
    Classify intent of text.

    Args:
        text: Input text
        classifier: Optional pre-configured classifier

    Returns:
        Intent classification
    """
    if classifier is None:
        classifier = IntentClassifier(use_llm=False)  # Fast default
    return classifier.classify(text)
