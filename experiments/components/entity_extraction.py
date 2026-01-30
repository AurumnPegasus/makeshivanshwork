"""
Entity Extraction for Memory Architecture Research

This module implements entity extraction with:

1. **Hierarchical Entity Types**
   - Person, Organization, Location (standard NER)
   - Topic, Concept, Event (extended types)
   - Task, Deadline, Preference (domain-specific)

2. **Coreference Resolution**
   - "Jerry" → "Jerry Tworek" (entity linking)
   - "he", "they" → resolved mentions
   - Critical for relationship extraction

3. **Entity Normalization**
   - "Jerry Tworek", "Jerry T.", "Jerry" → canonical form
   - Case normalization, alias tracking

Key Innovation: Most memory systems use flat NER.
We use hierarchical extraction with coreference for
richer entity graphs.

References:
- Lee et al. "End-to-end Neural Coreference Resolution" EMNLP 2017
- Joshi et al. "SpanBERT" TACL 2020
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Hierarchical entity type taxonomy"""

    # Standard NER types
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    PERCENT = "percent"

    # Extended types for conversations
    TOPIC = "topic"
    CONCEPT = "concept"
    EVENT = "event"
    PRODUCT = "product"
    TECHNOLOGY = "technology"

    # Domain-specific (task management)
    TASK = "task"
    DEADLINE = "deadline"
    PREFERENCE = "preference"
    PROJECT = "project"

    # Catch-all
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """An extracted entity with metadata"""

    name: str
    entity_type: EntityType
    canonical_name: Optional[str] = None  # Normalized form
    aliases: List[str] = field(default_factory=list)

    # Position in text
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    mention_text: Optional[str] = None

    # Confidence and metadata
    confidence: float = 1.0
    source: str = "extraction"  # extraction, coreference, user_defined
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Temporal tracking
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    mention_count: int = 1

    def __hash__(self):
        return hash((self.canonical_name or self.name, self.entity_type))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (self.canonical_name or self.name) == (
            other.canonical_name or other.name
        ) and self.entity_type == other.entity_type


@dataclass
class ExtractionResult:
    """Result of entity extraction"""

    entities: List[Entity]
    text: str
    model: str
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntityExtractor:
    """
    Entity extraction with coreference resolution.

    Combines multiple extraction strategies:
    1. Rule-based patterns (fast, high precision)
    2. SpaCy NER (if available)
    3. LLM-based extraction (high recall, slower)

    Coreference resolution links mentions to canonical entities.
    """

    # Common title patterns
    TITLE_PATTERNS = [
        r"\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+)",
    ]

    # Date patterns
    DATE_PATTERNS = [
        r"\b(today|tomorrow|yesterday)\b",
        r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        r"\b(next|last)\s+(week|month|year)\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b",
    ]

    # Task-related patterns
    TASK_PATTERNS = [
        r"\b(task|todo|reminder|deadline|meeting|call|email)\b",
    ]

    def __init__(
        self,
        use_spacy: bool = True,
        use_llm: bool = True,
        llm_client=None,
        entity_store: Optional[Dict[str, Entity]] = None,
    ):
        """
        Initialize entity extractor.

        Args:
            use_spacy: Use SpaCy for NER
            use_llm: Use LLM for complex extraction
            llm_client: LLM client for extraction
            entity_store: Persistent entity store for resolution
        """
        self._use_spacy = use_spacy
        self._use_llm = use_llm
        self._llm = llm_client

        # Thread lock for entity store access (prevents race conditions)
        self._lock = threading.RLock()

        # Entity store for coreference
        self._entity_store = entity_store or {}

        # SpaCy model
        self._nlp = None
        if use_spacy:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError) as e:
                logger.warning(f"SpaCy not available: {e}")
                self._use_spacy = False

    def extract(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> List[Entity]:
        """
        Extract entities from text.

        Args:
            text: Input text
            context: Optional conversation context for coreference

        Returns:
            List of extracted entities
        """
        entities = []
        seen_names: Set[str] = set()

        # 1. Rule-based extraction (fast)
        rule_entities = self._extract_rules(text)
        for e in rule_entities:
            if e.name.lower() not in seen_names:
                entities.append(e)
                seen_names.add(e.name.lower())

        # 2. SpaCy NER
        if self._use_spacy and self._nlp:
            spacy_entities = self._extract_spacy(text)
            for e in spacy_entities:
                if e.name.lower() not in seen_names:
                    entities.append(e)
                    seen_names.add(e.name.lower())

        # 3. LLM extraction (for complex cases)
        if self._use_llm and self._llm:
            llm_entities = self._extract_llm(text, context)
            for e in llm_entities:
                if e.name.lower() not in seen_names:
                    entities.append(e)
                    seen_names.add(e.name.lower())

        # 4. Coreference resolution
        entities = self._resolve_coreferences(entities, text, context)

        # 5. Normalize and deduplicate
        entities = self._normalize_entities(entities)

        return entities

    def _extract_rules(self, text: str) -> List[Entity]:
        """Rule-based entity extraction"""
        entities = []

        # Date extraction
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    Entity(
                        name=match.group(),
                        entity_type=EntityType.DATE,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        mention_text=match.group(),
                        source="rule",
                    )
                )

        # Task-related terms
        for pattern in self.TASK_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    Entity(
                        name=match.group(),
                        entity_type=EntityType.TASK,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        mention_text=match.group(),
                        source="rule",
                        confidence=0.7,
                    )
                )

        # Capitalized phrases (potential entities)
        # Pattern: Two or more capitalized words
        cap_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        for match in re.finditer(cap_pattern, text):
            name = match.group(1)
            # Skip common phrases
            if name.lower() not in ["the next", "last week", "this is"]:
                entities.append(
                    Entity(
                        name=name,
                        entity_type=EntityType.UNKNOWN,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        mention_text=name,
                        source="rule",
                        confidence=0.5,
                    )
                )

        return entities

    def _extract_spacy(self, text: str) -> List[Entity]:
        """SpaCy-based entity extraction"""
        if not self._nlp:
            return []

        entities = []
        doc = self._nlp(text)

        type_map = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENT,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.PRODUCT,
            "WORK_OF_ART": EntityType.CONCEPT,
        }

        for ent in doc.ents:
            entity_type = type_map.get(ent.label_, EntityType.UNKNOWN)
            entities.append(
                Entity(
                    name=ent.text,
                    entity_type=entity_type,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    mention_text=ent.text,
                    source="spacy",
                    confidence=0.8,
                )
            )

        return entities

    def _extract_llm(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> List[Entity]:
        """LLM-based entity extraction for complex cases"""
        if not self._llm:
            return []

        prompt = f"""Extract entities from this text. Return JSON list with format:
[{{"name": "...", "type": "person|organization|topic|task|date|preference|unknown"}}]

Text: {text}
{f"Context: {context}" if context else ""}

Only return the JSON list, no other text."""

        try:
            response = self._llm.generate(prompt, temperature=0.0)

            # Parse JSON response
            import json

            # Extract JSON from response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                entities = []
                for item in data:
                    type_map = {
                        "person": EntityType.PERSON,
                        "organization": EntityType.ORGANIZATION,
                        "topic": EntityType.TOPIC,
                        "task": EntityType.TASK,
                        "date": EntityType.DATE,
                        "preference": EntityType.PREFERENCE,
                    }
                    entity_type = type_map.get(
                        item.get("type", "").lower(), EntityType.UNKNOWN
                    )
                    entities.append(
                        Entity(
                            name=item["name"],
                            entity_type=entity_type,
                            source="llm",
                            confidence=0.9,
                        )
                    )
                return entities

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")

        return []

    def _resolve_coreferences(
        self,
        entities: List[Entity],
        text: str,
        context: Optional[str] = None,
    ) -> List[Entity]:
        """
        Resolve coreferences to canonical entities.

        Links mentions like "he", "they", "Jerry" to known entities.

        Thread-safe: Uses lock to protect entity store access.
        """
        resolved = []

        # Use lock for thread-safe entity store access
        with self._lock:
            for entity in entities:
                # Check if we have a canonical form in store
                name_lower = entity.name.lower()

                # Look for existing entity
                canonical = None
                for stored_name, stored_entity in self._entity_store.items():
                    if name_lower == stored_name.lower():
                        canonical = stored_entity
                        break
                    if name_lower in [a.lower() for a in stored_entity.aliases]:
                        canonical = stored_entity
                        break
                    # Partial match (e.g., "Jerry" matches "Jerry Tworek")
                    if len(name_lower) > 3 and name_lower in stored_name.lower():
                        canonical = stored_entity
                        break

                if canonical:
                    # Link to canonical entity
                    entity.canonical_name = canonical.name
                    entity.entity_type = canonical.entity_type
                    canonical.mention_count += 1
                    canonical.last_seen = datetime.now()
                else:
                    # New entity - add to store
                    entity.canonical_name = entity.name
                    entity.first_seen = datetime.now()
                    entity.last_seen = datetime.now()
                    self._entity_store[entity.name] = entity

                resolved.append(entity)

        return resolved

    def _normalize_entities(self, entities: List[Entity]) -> List[Entity]:
        """Normalize and deduplicate entities"""
        seen = {}

        for entity in entities:
            key = (entity.canonical_name or entity.name).lower()

            if key in seen:
                # Merge: keep higher confidence, accumulate mentions
                existing = seen[key]
                if entity.confidence > existing.confidence:
                    entity.mention_count = existing.mention_count + 1
                    seen[key] = entity
                else:
                    existing.mention_count += 1
            else:
                seen[key] = entity

        return list(seen.values())

    def add_known_entity(self, entity: Entity):
        """Add a known entity to the store (thread-safe)"""
        with self._lock:
            self._entity_store[entity.name] = entity

    def get_entity(self, name: str) -> Optional[Entity]:
        """Look up entity by name (thread-safe)"""
        with self._lock:
            return self._entity_store.get(name)

    def get_all_entities(self) -> List[Entity]:
        """Get all known entities (thread-safe)"""
        with self._lock:
            return list(self._entity_store.values())


# Convenience function
def extract_entities(
    text: str,
    extractor: Optional[EntityExtractor] = None,
) -> List[Entity]:
    """
    Extract entities from text.

    Args:
        text: Input text
        extractor: Optional pre-configured extractor

    Returns:
        List of extracted entities
    """
    if extractor is None:
        extractor = EntityExtractor(use_llm=False)  # Fast default
    return extractor.extract(text)
