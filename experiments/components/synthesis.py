"""
Answer Synthesis for Memory Architecture Research

This module implements answer generation from retrieved memories.

Key Innovation: Don't just retrieve and concatenate - SYNTHESIZE.

Traditional approach:
- Retrieve: "Jerry works at OpenAI"
- Retrieve: "OpenAI makes GPT"
- Return: "Jerry works at OpenAI. OpenAI makes GPT."

Our approach:
- Retrieve relevant memories
- Reason over them
- Synthesize: "Jerry works at OpenAI, the company that created GPT."

Features:
1. **Grounded Generation**
   - Every claim must cite a source memory
   - Detect and flag hallucinations

2. **Uncertainty-Aware**
   - "I'm confident that..." vs "I believe..."
   - Flag contradictions in memories

3. **Multi-Hop Synthesis**
   - Chain reasoning across memories
   - "A works at B" + "B is in C" → "A works in C"

References:
- Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" NeurIPS 2020
- Gao et al. "Enabling Large Language Models to Generate Text with Citations" EMNLP 2023
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MemorySource:
    """A memory used in synthesis"""

    id: str
    content: str
    relevance_score: float
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """A citation linking claim to source"""

    claim: str
    source_id: str
    source_content: str
    confidence: float = 1.0


@dataclass
class SynthesizedAnswer:
    """A synthesized answer with provenance"""

    answer: str
    citations: List[Citation] = field(default_factory=list)
    sources_used: List[MemorySource] = field(default_factory=list)

    # Quality indicators
    confidence: float = 1.0
    is_grounded: bool = True
    has_contradictions: bool = False
    contradiction_details: Optional[str] = None

    # Metadata
    synthesis_method: str = "llm"
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnswerSynthesizer:
    """
    Synthesize answers from retrieved memories.

    Implements:
    - Grounded generation with citations
    - Uncertainty quantification
    - Contradiction detection
    - Multi-hop reasoning
    """

    def __init__(
        self,
        llm_client=None,
        require_citations: bool = True,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize synthesizer.

        Args:
            llm_client: LLM for generation
            require_citations: Require all claims to be cited
            confidence_threshold: Below this, flag as uncertain
        """
        self._llm = llm_client
        self._require_citations = require_citations
        self._confidence_threshold = confidence_threshold

    def synthesize(
        self,
        query: str,
        memories: List[MemorySource],
        context: Optional[str] = None,
    ) -> SynthesizedAnswer:
        """
        Synthesize answer from memories.

        Args:
            query: User's question
            memories: Retrieved relevant memories
            context: Additional context

        Returns:
            SynthesizedAnswer with citations and confidence
        """
        if not memories:
            return SynthesizedAnswer(
                answer="I don't have any relevant memories to answer this question.",
                confidence=0.0,
                is_grounded=True,
            )

        if not self._llm:
            return self._extractive_synthesis(query, memories)

        return self._llm_synthesis(query, memories, context)

    def _extractive_synthesis(
        self,
        query: str,
        memories: List[MemorySource],
    ) -> SynthesizedAnswer:
        """Simple extractive synthesis without LLM"""
        # Sort by relevance
        sorted_memories = sorted(
            memories, key=lambda m: m.relevance_score, reverse=True
        )

        # Build answer from top memories
        lines = ["Based on my memories:"]
        citations = []

        for i, mem in enumerate(sorted_memories[:5]):
            lines.append(f"- {mem.content}")
            citations.append(
                Citation(
                    claim=mem.content,
                    source_id=mem.id,
                    source_content=mem.content,
                    confidence=mem.relevance_score,
                )
            )

        return SynthesizedAnswer(
            answer="\n".join(lines),
            citations=citations,
            sources_used=sorted_memories[:5],
            confidence=sorted_memories[0].relevance_score if sorted_memories else 0.0,
            is_grounded=True,
            synthesis_method="extractive",
        )

    def _llm_synthesis(
        self,
        query: str,
        memories: List[MemorySource],
        context: Optional[str] = None,
    ) -> SynthesizedAnswer:
        """LLM-based synthesis with citations"""
        # Format memories for LLM
        memory_text = "\n".join(
            f"[{i + 1}] {mem.content}" for i, mem in enumerate(memories)
        )

        prompt = f"""Answer the user's question based ONLY on these memories.

MEMORIES:
{memory_text}

{f"ADDITIONAL CONTEXT: {context}" if context else ""}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the provided memories
2. Cite sources using [1], [2], etc.
3. If memories contradict, note the contradiction
4. If information is incomplete, say so
5. Do NOT make up information not in memories

ANSWER:"""

        try:
            response = self._llm.generate(prompt, temperature=0.3)

            # Extract citations from response
            citations = self._extract_citations(response, memories)

            # Check for contradictions
            has_contradictions, contradiction_details = self._check_contradictions(
                memories
            )

            # Calculate confidence
            confidence = self._calculate_confidence(response, memories, citations)

            # Check grounding
            is_grounded = self._check_grounding(response, memories)

            return SynthesizedAnswer(
                answer=response,
                citations=citations,
                sources_used=memories,
                confidence=confidence,
                is_grounded=is_grounded,
                has_contradictions=has_contradictions,
                contradiction_details=contradiction_details,
                synthesis_method="llm",
            )

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return self._extractive_synthesis(query, memories)

    def _extract_citations(
        self,
        response: str,
        memories: List[MemorySource],
    ) -> List[Citation]:
        """Extract citations from response"""
        citations = []

        # Find citation patterns like [1], [2], etc.
        citation_pattern = r"\[(\d+)\]"
        matches = re.findall(citation_pattern, response)

        for match in set(matches):
            idx = int(match) - 1
            if 0 <= idx < len(memories):
                mem = memories[idx]

                # Find the claim associated with this citation
                # Look for text before the citation
                claim_pattern = rf"([^.!?]*)\[{match}\]"
                claim_match = re.search(claim_pattern, response)
                claim = claim_match.group(1).strip() if claim_match else ""

                citations.append(
                    Citation(
                        claim=claim,
                        source_id=mem.id,
                        source_content=mem.content,
                        confidence=mem.relevance_score,
                    )
                )

        return citations

    def _check_contradictions(
        self,
        memories: List[MemorySource],
    ) -> Tuple[bool, Optional[str]]:
        """Check for contradictions in memories"""
        if not self._llm or len(memories) < 2:
            return False, None

        # Simple heuristic: check for negation patterns
        contents = [m.content.lower() for m in memories]

        # Look for potential contradictions
        contradictions = []

        for i, content1 in enumerate(contents):
            for j, content2 in enumerate(contents[i + 1 :], i + 1):
                # Check for negation
                if any(neg in content1 for neg in ["not", "never", "don't", "doesn't"]):
                    # Check if similar topic
                    words1 = set(content1.split())
                    words2 = set(content2.split())
                    overlap = len(words1 & words2) / min(len(words1), len(words2))
                    if overlap > 0.3:
                        contradictions.append(
                            f"Potential conflict between memory {i + 1} and {j + 1}"
                        )

        if contradictions:
            return True, "; ".join(contradictions)

        return False, None

    def _calculate_confidence(
        self,
        response: str,
        memories: List[MemorySource],
        citations: List[Citation],
    ) -> float:
        """Calculate confidence score for answer"""
        confidence = 0.5  # Base confidence

        # Factor 1: Memory relevance scores
        if memories:
            avg_relevance = sum(m.relevance_score for m in memories) / len(memories)
            confidence += 0.2 * avg_relevance

        # Factor 2: Citation coverage
        if citations and memories:
            citation_ratio = len(citations) / max(1, len(memories))
            confidence += 0.2 * min(1.0, citation_ratio)

        # Factor 3: Uncertainty language
        uncertainty_phrases = [
            "i'm not sure",
            "i don't know",
            "may",
            "might",
            "possibly",
            "perhaps",
            "uncertain",
            "unclear",
        ]
        if any(phrase in response.lower() for phrase in uncertainty_phrases):
            confidence -= 0.1

        # Factor 4: Response length (very short might be incomplete)
        if len(response.split()) < 10:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _check_grounding(
        self,
        response: str,
        memories: List[MemorySource],
    ) -> bool:
        """Check if response is grounded in memories"""
        if not memories:
            return False

        # Simple check: key words from memories appear in response
        memory_words = set()
        for mem in memories:
            memory_words.update(
                word.lower() for word in mem.content.split() if len(word) > 3
            )

        response_words = set(word.lower() for word in response.split() if len(word) > 3)

        overlap = len(memory_words & response_words)
        overlap_ratio = overlap / max(1, len(memory_words))

        # If >30% of memory words appear in response, consider grounded
        return overlap_ratio > 0.3

    def multi_hop_synthesis(
        self,
        query: str,
        initial_memories: List[MemorySource],
        retrieval_fn,
        max_hops: int = 3,
    ) -> SynthesizedAnswer:
        """
        Multi-hop reasoning over memories.

        Iteratively retrieves more memories based on intermediate reasoning.

        Args:
            query: User's question
            initial_memories: First retrieval results
            retrieval_fn: Function to retrieve more memories
            max_hops: Maximum reasoning hops

        Returns:
            SynthesizedAnswer with reasoning chain
        """
        if not self._llm:
            return self.synthesize(query, initial_memories)

        all_memories = list(initial_memories)
        reasoning_chain = []

        for hop in range(max_hops):
            # Check if we have enough information
            check_prompt = f"""Do we have enough information to answer: "{query}"

Available information:
{chr(10).join(f"- {m.content}" for m in all_memories)}

If YES, say "SUFFICIENT"
If NO, say "NEED: <what additional information is needed>"""

            check_response = self._llm.generate(check_prompt, temperature=0.0)

            if "SUFFICIENT" in check_response.upper():
                reasoning_chain.append(
                    f"Hop {hop + 1}: Sufficient information gathered"
                )
                break

            # Extract what we need
            need_match = re.search(r"NEED:\s*(.+)", check_response, re.IGNORECASE)
            if need_match:
                next_query = need_match.group(1).strip()
                reasoning_chain.append(f"Hop {hop + 1}: Need - {next_query}")

                # Retrieve more memories
                new_memories = retrieval_fn(next_query)
                if new_memories:
                    # Add new unique memories
                    existing_ids = {m.id for m in all_memories}
                    for mem in new_memories:
                        if mem.id not in existing_ids:
                            all_memories.append(mem)
                            existing_ids.add(mem.id)

                    reasoning_chain.append(
                        f"Hop {hop + 1}: Retrieved {len(new_memories)} additional memories"
                    )
                else:
                    reasoning_chain.append(
                        f"Hop {hop + 1}: No additional memories found"
                    )
                    break
            else:
                break

        # Final synthesis
        answer = self.synthesize(query, all_memories)
        answer.reasoning_chain = reasoning_chain
        return answer


# Convenience function
def synthesize_answer(
    query: str,
    memories: List[Dict[str, Any]],
    llm_client=None,
) -> SynthesizedAnswer:
    """
    Synthesize answer from memories.

    Args:
        query: User's question
        memories: List of memory dicts
        llm_client: Optional LLM for generation

    Returns:
        SynthesizedAnswer
    """
    # Convert to MemorySource objects
    sources = [
        MemorySource(
            id=m.get("id", str(i)),
            content=m.get("content", ""),
            relevance_score=m.get("score", 1.0),
            timestamp=m.get("timestamp"),
            metadata=m.get("metadata", {}),
        )
        for i, m in enumerate(memories)
    ]

    synthesizer = AnswerSynthesizer(llm_client=llm_client)
    return synthesizer.synthesize(query, sources)
