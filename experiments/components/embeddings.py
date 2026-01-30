"""
Embedding Components for Memory Architecture Research

This module implements state-of-the-art embedding techniques:

1. **Multi-Embedding Representation**
   - Content embedding: Semantic meaning of the message
   - Entity embedding: Who/what is mentioned
   - Intent embedding: Purpose (command/question/info)

2. **Matryoshka Representation Learning (MRL)**
   - Variable-dimension embeddings from single model
   - Scale 3072 → 768 → 256 based on needs
   - Theoretical basis: Kusupati et al. (2022)

3. **Task-Specific Embedding**
   - RETRIEVAL_DOCUMENT for storage
   - RETRIEVAL_QUERY for search
   - Improves asymmetric retrieval

Key Innovation: No existing memory system uses multi-embedding.
We hypothesize this preserves distinct semantic dimensions that
single embeddings conflate, improving retrieval precision.

References:
- Kusupati et al. "Matryoshka Representation Learning" NeurIPS 2022
- Muennighoff et al. "MTEB: Massive Text Embedding Benchmark" EACL 2023
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for embeddings
Embedding = List[float]


@dataclass
class EmbeddingResult:
    """Result of an embedding operation"""

    embedding: Embedding
    model: str
    dimensions: int
    task_type: Optional[str] = None
    tokens_used: int = 0
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier"""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output embedding dimensions"""
        pass

    @abstractmethod
    def embed(
        self,
        text: str,
        task_type: Optional[str] = None,
    ) -> EmbeddingResult:
        """Generate embedding for a single text"""
        pass

    @abstractmethod
    def embed_batch(
        self,
        texts: List[str],
        task_type: Optional[str] = None,
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts"""
        pass

    def similarity(
        self,
        embedding1: Embedding,
        embedding2: Embedding,
    ) -> float:
        """Compute cosine similarity between embeddings"""
        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class EmbeddingCache:
    """
    Disk-based embedding cache for efficiency.

    Embeddings are expensive to compute. This cache:
    - Stores embeddings keyed by (text_hash, model, task_type)
    - Uses disk storage for persistence across runs
    - Supports TTL and size limits
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: int = 1000,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._index_path = self.cache_dir / "index.json"
        self._index: Dict[str, Dict] = self._load_index()

    def _load_index(self) -> Dict:
        """Load cache index from disk"""
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_index(self):
        """Save cache index to disk"""
        self._index_path.write_text(json.dumps(self._index))

    def _make_key(
        self,
        text: str,
        model: str,
        task_type: Optional[str],
    ) -> str:
        """Create cache key from inputs"""
        content = f"{model}:{task_type}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(
        self,
        text: str,
        model: str,
        task_type: Optional[str] = None,
    ) -> Optional[EmbeddingResult]:
        """Retrieve cached embedding"""
        key = self._make_key(text, model, task_type)

        if key not in self._index:
            return None

        cache_file = self.cache_dir / f"{key}.npy"
        if not cache_file.exists():
            del self._index[key]
            return None

        try:
            embedding = np.load(cache_file).tolist()
            meta = self._index[key]
            return EmbeddingResult(
                embedding=embedding,
                model=meta["model"],
                dimensions=meta["dimensions"],
                task_type=meta.get("task_type"),
                cached=True,
            )
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def _get_cache_size(self) -> int:
        """Get total size of cached embeddings in bytes"""
        total = 0
        for f in self.cache_dir.glob("*.npy"):
            try:
                total += f.stat().st_size
            except OSError:
                pass
        return total

    def _evict_oldest(self, target_size: int) -> None:
        """Evict oldest cache entries until under target size"""
        # Get all cache files sorted by modification time (oldest first)
        cache_files = []
        for f in self.cache_dir.glob("*.npy"):
            try:
                cache_files.append((f, f.stat().st_mtime, f.stat().st_size))
            except OSError:
                pass

        cache_files.sort(key=lambda x: x[1])  # Sort by mtime

        current_size = sum(f[2] for f in cache_files)
        evicted_count = 0

        for cache_file, _, file_size in cache_files:
            if current_size <= target_size:
                break

            # Remove file and index entry
            key = cache_file.stem
            try:
                cache_file.unlink()
                self._index.pop(key, None)
                current_size -= file_size
                evicted_count += 1
            except OSError as e:
                logger.warning(f"Failed to evict cache file {cache_file}: {e}")

        if evicted_count > 0:
            logger.info(f"Evicted {evicted_count} cache entries to meet size limit")
            self._save_index()

    def put(
        self,
        text: str,
        result: EmbeddingResult,
    ):
        """Store embedding in cache"""
        # Check cache size and evict if needed
        current_size = self._get_cache_size()
        if current_size > self.max_size_bytes:
            # Evict to 80% capacity
            target_size = int(self.max_size_bytes * 0.8)
            self._evict_oldest(target_size)

        key = self._make_key(text, result.model, result.task_type)

        # Save embedding
        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, np.array(result.embedding))

        # Update index with timestamp for LRU tracking
        self._index[key] = {
            "model": result.model,
            "dimensions": result.dimensions,
            "task_type": result.task_type,
            "created_at": time.time(),
        }
        self._save_index()


class GeminiEmbedding(EmbeddingModel):
    """
    Gemini embedding model with MRL support.

    Uses gemini-embedding-001:
    - 3072 dimensions (supports MRL scaling)
    - Free tier available
    - State-of-the-art on MTEB

    Task types:
    - RETRIEVAL_DOCUMENT: For storing documents
    - RETRIEVAL_QUERY: For search queries
    - SEMANTIC_SIMILARITY: For comparing texts
    - CLASSIFICATION: For text classification
    """

    TASK_TYPES = [
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/gemini-embedding-001",
        output_dimensions: int = 3072,
        cache: Optional[EmbeddingCache] = None,
    ):
        """
        Initialize Gemini embedding model.

        Args:
            api_key: Gemini API key (uses GEMINI_API_KEY env var if not provided)
            model: Model identifier
            output_dimensions: Output dimensions (MRL: 3072, 768, 256)
            cache: Optional embedding cache
        """
        import os

        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model
        self._dimensions = output_dimensions
        self._cache = cache

        # Initialize Gemini client
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._genai = genai
        except ImportError:
            raise ImportError("google-generativeai not installed")

    @property
    def name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(
        self,
        text: str,
        task_type: Optional[str] = "RETRIEVAL_DOCUMENT",
    ) -> EmbeddingResult:
        """Generate embedding for text"""
        # Check cache
        if self._cache:
            cached = self._cache.get(text, self._model, task_type)
            if cached:
                return cached

        # Generate embedding
        try:
            result = self._genai.embed_content(
                model=self._model,
                content=text,
                task_type=task_type,
                output_dimensionality=self._dimensions,
            )

            embedding_result = EmbeddingResult(
                embedding=result["embedding"],
                model=self._model,
                dimensions=len(result["embedding"]),
                task_type=task_type,
                cached=False,
            )

            # Cache result
            if self._cache:
                self._cache.put(text, embedding_result)

            return embedding_result

        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise

    def embed_batch(
        self,
        texts: List[str],
        task_type: Optional[str] = "RETRIEVAL_DOCUMENT",
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts"""
        results = []

        # Check cache for each text
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            if self._cache:
                cached = self._cache.get(text, self._model, task_type)
                if cached:
                    results.append((i, cached))
                    continue

            uncached_indices.append(i)
            uncached_texts.append(text)
            results.append((i, None))

        # Batch embed uncached texts
        if uncached_texts:
            try:
                batch_result = self._genai.embed_content(
                    model=self._model,
                    content=uncached_texts,
                    task_type=task_type,
                    output_dimensionality=self._dimensions,
                )

                embeddings = batch_result["embedding"]
                if isinstance(embeddings[0], float):
                    # Single result
                    embeddings = [embeddings]

                for idx, embedding in zip(uncached_indices, embeddings):
                    text = texts[idx]
                    embedding_result = EmbeddingResult(
                        embedding=embedding,
                        model=self._model,
                        dimensions=len(embedding),
                        task_type=task_type,
                        cached=False,
                    )

                    # Update results list
                    for j, (i, r) in enumerate(results):
                        if i == idx:
                            results[j] = (i, embedding_result)
                            break

                    # Cache result
                    if self._cache:
                        self._cache.put(text, embedding_result)

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                raise

        # Return in original order
        return [r for _, r in sorted(results, key=lambda x: x[0])]


class MatryoshkaEmbedding(EmbeddingModel):
    """
    Matryoshka Representation Learning (MRL) wrapper.

    MRL produces embeddings where prefixes are valid lower-dimensional
    representations. This allows:
    - Fast approximate search with 256d
    - Accurate reranking with 3072d
    - Memory/latency tradeoffs at runtime

    Theoretical basis:
    - First d dimensions capture most important features
    - Nested structure from training objective
    - No information loss, just resolution

    Reference: Kusupati et al. "Matryoshka Representation Learning" NeurIPS 2022
    """

    SCALES = [3072, 768, 256, 64]  # Supported dimensions

    def __init__(
        self,
        base_model: EmbeddingModel,
        default_dimensions: int = 768,
    ):
        """
        Initialize MRL wrapper.

        Args:
            base_model: Full-dimension embedding model
            default_dimensions: Default output dimensions
        """
        self._base = base_model
        self._default_dims = default_dimensions

        if default_dimensions not in self.SCALES:
            logger.warning(
                f"Dimensions {default_dimensions} not in standard MRL scales. "
                f"Using closest: {min(self.SCALES, key=lambda x: abs(x - default_dimensions))}"
            )

    @property
    def name(self) -> str:
        return f"{self._base.name}_mrl_{self._default_dims}d"

    @property
    def dimensions(self) -> int:
        return self._default_dims

    def embed(
        self,
        text: str,
        task_type: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> EmbeddingResult:
        """Generate MRL embedding at specified dimension"""
        dims = dimensions or self._default_dims

        # Get full embedding
        result = self._base.embed(text, task_type)

        # Truncate to desired dimensions
        truncated = result.embedding[:dims]

        # Renormalize (important for cosine similarity)
        norm = np.linalg.norm(truncated)
        if norm > 0:
            truncated = (np.array(truncated) / norm).tolist()

        return EmbeddingResult(
            embedding=truncated,
            model=f"{result.model}_mrl",
            dimensions=dims,
            task_type=task_type,
            metadata={"full_dimensions": result.dimensions},
        )

    def embed_batch(
        self,
        texts: List[str],
        task_type: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        """Generate MRL embeddings for batch"""
        dims = dimensions or self._default_dims
        results = self._base.embed_batch(texts, task_type)

        mrl_results = []
        for result in results:
            truncated = result.embedding[:dims]
            norm = np.linalg.norm(truncated)
            if norm > 0:
                truncated = (np.array(truncated) / norm).tolist()

            mrl_results.append(
                EmbeddingResult(
                    embedding=truncated,
                    model=f"{result.model}_mrl",
                    dimensions=dims,
                    task_type=task_type,
                    metadata={"full_dimensions": result.dimensions},
                )
            )

        return mrl_results

    def multi_scale_embed(
        self,
        text: str,
        task_type: Optional[str] = None,
        scales: Optional[List[int]] = None,
    ) -> Dict[int, EmbeddingResult]:
        """
        Generate embeddings at multiple scales.

        Useful for hierarchical retrieval:
        1. Fast filtering with 64d
        2. Candidate selection with 256d
        3. Final ranking with 768d or 3072d
        """
        scales = scales or self.SCALES
        result = self._base.embed(text, task_type)

        multi_scale = {}
        for dims in scales:
            if dims <= result.dimensions:
                truncated = result.embedding[:dims]
                norm = np.linalg.norm(truncated)
                if norm > 0:
                    truncated = (np.array(truncated) / norm).tolist()

                multi_scale[dims] = EmbeddingResult(
                    embedding=truncated,
                    model=f"{result.model}_mrl",
                    dimensions=dims,
                    task_type=task_type,
                )

        return multi_scale


class MultiEmbedding:
    """
    Multi-Embedding Representation for Memory

    Key Innovation: Represent each memory with multiple embeddings
    that capture orthogonal semantic dimensions:

    1. Content Embedding: What was said (semantic meaning)
    2. Entity Embedding: Who/what is mentioned (named entities)
    3. Intent Embedding: Why it was said (purpose/function)

    Theoretical Motivation:
    - Single embeddings conflate distinct dimensions
    - "Meeting with Jerry about AI" has:
      - Content: scheduling semantics
      - Entity: Jerry, AI
      - Intent: action request
    - Multi-embedding preserves these for targeted retrieval

    Query-Adaptive Weighting:
    - "Who is Jerry?" → Entity weight high
    - "What should I do?" → Content weight high
    - "Did I ask to X?" → Intent weight high

    This is novel - no existing memory system does this.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        entity_extractor=None,
        intent_classifier=None,
        entity_cache_size: int = 10000,
    ):
        """
        Initialize multi-embedding.

        Args:
            embedding_model: Base embedding model
            entity_extractor: Entity extraction component
            intent_classifier: Intent classification component
            entity_cache_size: Max number of entity embeddings to cache
        """
        self._model = embedding_model
        self._entity_extractor = entity_extractor
        self._intent_classifier = intent_classifier

        # Entity embedding cache: normalized_entity -> EmbeddingResult
        # Same entities appear across many messages, so caching saves API calls
        self._entity_cache: Dict[str, EmbeddingResult] = {}
        self._entity_cache_size = entity_cache_size

    def _get_cached_entity_embedding(self, entity: str) -> Optional[EmbeddingResult]:
        """Get cached entity embedding if available"""
        key = entity.lower().strip()
        return self._entity_cache.get(key)

    def _cache_entity_embedding(self, entity: str, result: EmbeddingResult) -> None:
        """Cache entity embedding for future use"""
        # Simple LRU: if at capacity, remove 10% oldest (by insertion order)
        if len(self._entity_cache) >= self._entity_cache_size:
            keys_to_remove = list(self._entity_cache.keys())[
                : self._entity_cache_size // 10
            ]
            for key in keys_to_remove:
                del self._entity_cache[key]

        key = entity.lower().strip()
        self._entity_cache[key] = result

    def embed_message(
        self,
        content: str,
        entities: Optional[List[str]] = None,
        intent: Optional[str] = None,
    ) -> Dict[str, EmbeddingResult]:
        """
        Generate multi-embedding for a message.

        Returns:
            Dict with 'content', 'entity', 'intent' embeddings
        """
        embeddings = {}

        # Content embedding (always)
        embeddings["content"] = self._model.embed(
            content,
            task_type="RETRIEVAL_DOCUMENT",
        )

        # Entity embedding
        if entities is None and self._entity_extractor:
            entities = self._entity_extractor.extract(content)

        if entities:
            # Check cache for each entity, only embed uncached ones
            entity_embeddings = []
            uncached_entities = []

            for entity in entities:
                cached = self._get_cached_entity_embedding(entity)
                if cached:
                    entity_embeddings.append(cached.embedding)
                else:
                    uncached_entities.append(entity)

            # Embed uncached entities
            if uncached_entities:
                for entity in uncached_entities:
                    result = self._model.embed(
                        entity,
                        task_type="RETRIEVAL_DOCUMENT",
                    )
                    entity_embeddings.append(result.embedding)
                    self._cache_entity_embedding(entity, result)

            # Average entity embeddings for combined representation
            if entity_embeddings:
                import numpy as np

                avg_embedding = np.mean(entity_embeddings, axis=0).tolist()
                embeddings["entity"] = EmbeddingResult(
                    embedding=avg_embedding,
                    model=self._model.name,
                    dimensions=len(avg_embedding),
                    task_type="RETRIEVAL_DOCUMENT",
                    cached=len(uncached_entities) == 0,  # True if all from cache
                )

        # Intent embedding
        if intent is None and self._intent_classifier:
            intent_result = self._intent_classifier.classify(content)
            intent = intent_result.label if intent_result else None

        if intent:
            # Embed the intent label + relevant content
            intent_text = f"[{intent}] {content}"
            embeddings["intent"] = self._model.embed(
                intent_text,
                task_type="RETRIEVAL_DOCUMENT",
            )

        return embeddings

    def embed_query(
        self,
        query: str,
        query_type: Optional[str] = None,
    ) -> Tuple[Dict[str, EmbeddingResult], Dict[str, float]]:
        """
        Generate query embedding with adaptive weights.

        Returns:
            - Embeddings dict
            - Weights dict for combining scores
        """
        embeddings = {}
        weights = {"content": 0.6, "entity": 0.2, "intent": 0.2}  # Defaults

        # Query embedding
        embeddings["content"] = self._model.embed(
            query,
            task_type="RETRIEVAL_QUERY",
        )

        # Detect query type and adjust weights
        query_lower = query.lower()

        # Entity-focused queries
        if any(
            kw in query_lower
            for kw in ["who is", "what is", "tell me about", "know about"]
        ):
            weights = {"content": 0.3, "entity": 0.5, "intent": 0.2}

            if self._entity_extractor:
                entities = self._entity_extractor.extract(query)
                if entities:
                    entity_text = " ".join(entities)
                    embeddings["entity"] = self._model.embed(
                        entity_text,
                        task_type="RETRIEVAL_QUERY",
                    )

        # Temporal queries
        elif any(
            kw in query_lower
            for kw in ["when did", "last time", "yesterday", "last week"]
        ):
            weights = {"content": 0.7, "entity": 0.2, "intent": 0.1}

        # Intent-focused queries
        elif any(
            kw in query_lower for kw in ["did i ask", "did i want", "was i trying"]
        ):
            weights = {"content": 0.3, "entity": 0.2, "intent": 0.5}

            if self._intent_classifier:
                embeddings["intent"] = self._model.embed(
                    query,
                    task_type="RETRIEVAL_QUERY",
                )

        # Relationship queries
        elif any(
            kw in query_lower for kw in ["related to", "connection", "relationship"]
        ):
            weights = {"content": 0.3, "entity": 0.5, "intent": 0.2}

        return embeddings, weights

    def compute_similarity(
        self,
        query_embeddings: Dict[str, EmbeddingResult],
        memory_embeddings: Dict[str, EmbeddingResult],
        weights: Dict[str, float],
    ) -> float:
        """
        Compute weighted multi-embedding similarity.

        score = Σ weight_i * cosine(query_i, memory_i)
        """
        total_score = 0.0
        total_weight = 0.0

        for key, weight in weights.items():
            if key in query_embeddings and key in memory_embeddings:
                query_emb = np.array(query_embeddings[key].embedding)
                memory_emb = np.array(memory_embeddings[key].embedding)

                # Cosine similarity
                similarity = float(
                    np.dot(query_emb, memory_emb)
                    / (np.linalg.norm(query_emb) * np.linalg.norm(memory_emb))
                )

                total_score += weight * similarity
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    # Compatibility methods for simple embedding interface
    def embed(
        self,
        text: str,
        task_type: Optional[str] = None,
    ) -> EmbeddingResult:
        """
        Delegate to underlying model (compatibility method).
        """
        return self._model.embed(text, task_type=task_type)

    def embed_content(self, content: str) -> Optional[List[float]]:
        """
        Simple content embedding (compatibility method).
        Returns just the embedding vector.
        """
        try:
            result = self._model.embed(content, task_type="RETRIEVAL_DOCUMENT")
            return result.embedding if result else None
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def embed_query_simple(self, query: str) -> Optional[List[float]]:
        """
        Simple query embedding (compatibility method).
        Returns just the embedding vector.
        """
        try:
            result = self._model.embed(query, task_type="RETRIEVAL_QUERY")
            return result.embedding if result else None
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return None
