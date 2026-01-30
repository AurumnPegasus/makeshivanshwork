"""
Consolidated Math Utilities for Memory Architecture Research.

This module provides optimized, vectorized similarity functions used
across the codebase. All similarity computations should use these
functions to ensure consistency and performance.

Key Features:
- Zero-norm handling (prevents NaN/division by zero)
- Vectorized batch operations (10-100x faster than loops)
- Consistent behavior across all approaches
"""

from typing import Dict, List, Tuple, Union

import numpy as np

# Type aliases
Embedding = Union[List[float], np.ndarray]
EmbeddingMatrix = np.ndarray


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Cosine similarity in range [-1, 1], or 0.0 if either vector has zero norm

    Example:
        >>> cosine_similarity([1, 0, 0], [1, 0, 0])
        1.0
        >>> cosine_similarity([1, 0, 0], [0, 1, 0])
        0.0
    """
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def batch_cosine_similarity(
    query: Embedding,
    candidates: EmbeddingMatrix,
    normalized: bool = False,
) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple candidates.

    This is the vectorized version that provides 10-100x speedup over
    computing similarities in a loop.

    Args:
        query: Query embedding vector (1D array of shape [dim])
        candidates: Matrix of candidate embeddings (2D array of shape [n, dim])
        normalized: If True, assumes embeddings are already L2-normalized

    Returns:
        Array of similarities of shape [n], one per candidate

    Example:
        >>> query = [1, 0, 0]
        >>> candidates = [[1, 0, 0], [0, 1, 0], [0.707, 0.707, 0]]
        >>> batch_cosine_similarity(query, candidates)
        array([1.0, 0.0, 0.707...])
    """
    query_arr = np.asarray(query, dtype=np.float32)
    candidates_arr = np.asarray(candidates, dtype=np.float32)

    if candidates_arr.ndim == 1:
        # Single candidate - wrap in 2D
        candidates_arr = candidates_arr.reshape(1, -1)

    if len(candidates_arr) == 0:
        return np.array([], dtype=np.float32)

    if not normalized:
        # Normalize query
        query_norm = np.linalg.norm(query_arr)
        if query_norm == 0:
            return np.zeros(len(candidates_arr), dtype=np.float32)
        query_arr = query_arr / query_norm

        # Normalize candidates (handle zero-norm rows)
        candidate_norms = np.linalg.norm(candidates_arr, axis=1, keepdims=True)
        # Replace zero norms with 1 to avoid division by zero (result will be 0 anyway)
        candidate_norms = np.where(candidate_norms == 0, 1, candidate_norms)
        candidates_arr = candidates_arr / candidate_norms

    # Compute dot products (all at once)
    similarities = candidates_arr @ query_arr

    return similarities


def batch_pairwise_similarity(
    queries: EmbeddingMatrix,
    candidates: EmbeddingMatrix,
    normalized: bool = False,
) -> np.ndarray:
    """
    Compute pairwise cosine similarities between two sets of embeddings.

    Args:
        queries: Matrix of query embeddings (shape [m, dim])
        candidates: Matrix of candidate embeddings (shape [n, dim])
        normalized: If True, assumes embeddings are already L2-normalized

    Returns:
        Similarity matrix of shape [m, n] where result[i, j] is
        the similarity between queries[i] and candidates[j]
    """
    queries_arr = np.asarray(queries, dtype=np.float32)
    candidates_arr = np.asarray(candidates, dtype=np.float32)

    if not normalized:
        # Normalize queries
        query_norms = np.linalg.norm(queries_arr, axis=1, keepdims=True)
        query_norms = np.where(query_norms == 0, 1, query_norms)
        queries_arr = queries_arr / query_norms

        # Normalize candidates
        candidate_norms = np.linalg.norm(candidates_arr, axis=1, keepdims=True)
        candidate_norms = np.where(candidate_norms == 0, 1, candidate_norms)
        candidates_arr = candidates_arr / candidate_norms

    # Compute all pairwise similarities
    return queries_arr @ candidates_arr.T


def weighted_multi_similarity(
    query_embeddings: Dict[str, Embedding],
    memory_embeddings: Dict[str, Embedding],
    weights: Dict[str, float],
) -> float:
    """
    Compute weighted similarity across multiple embedding types.

    This is the core function for multi-embedding retrieval, combining
    content, entity, and intent embeddings with adaptive weights.

    Args:
        query_embeddings: Dict mapping embedding type to query embedding
        memory_embeddings: Dict mapping embedding type to memory embedding
        weights: Dict mapping embedding type to weight (should sum to ~1.0)

    Returns:
        Weighted average similarity score

    Example:
        >>> query_embs = {"content": [1, 0], "entity": [0, 1]}
        >>> memory_embs = {"content": [1, 0], "entity": [1, 0]}
        >>> weights = {"content": 0.6, "entity": 0.4}
        >>> weighted_multi_similarity(query_embs, memory_embs, weights)
        0.6  # 0.6 * 1.0 + 0.4 * 0.0
    """
    total_score = 0.0
    total_weight = 0.0

    for key, weight in weights.items():
        if key in query_embeddings and key in memory_embeddings:
            sim = cosine_similarity(query_embeddings[key], memory_embeddings[key])
            total_score += weight * sim
            total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0


def batch_weighted_multi_similarity(
    query_embeddings: Dict[str, Embedding],
    candidate_embeddings: Dict[str, EmbeddingMatrix],
    weights: Dict[str, float],
    candidate_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute weighted multi-similarity for multiple candidates at once.

    This is the vectorized version for searching across many memories,
    providing 10-100x speedup over sequential computation.

    Args:
        query_embeddings: Dict mapping embedding type to query embedding
        candidate_embeddings: Dict mapping embedding type to matrix of
            candidate embeddings (shape [n, dim])
        weights: Dict mapping embedding type to weight
        candidate_ids: List of candidate memory IDs (length n)

    Returns:
        Tuple of (scores array, valid_ids list) where scores are the
        weighted similarities for candidates that have embeddings

    Example:
        >>> query_embs = {"content": [1, 0, 0]}
        >>> candidate_embs = {"content": np.array([[1, 0, 0], [0, 1, 0]])}
        >>> weights = {"content": 1.0}
        >>> scores, ids = batch_weighted_multi_similarity(
        ...     query_embs, candidate_embs, weights, ["m1", "m2"]
        ... )
    """
    if not candidate_ids:
        return np.array([]), []

    # Compute similarity for each embedding type
    type_scores = {}

    for emb_type, weight in weights.items():
        if emb_type not in query_embeddings or emb_type not in candidate_embeddings:
            continue

        query_emb = query_embeddings[emb_type]
        cand_matrix = candidate_embeddings[emb_type]

        if len(cand_matrix) == 0:
            continue

        # Batch similarity computation
        similarities = batch_cosine_similarity(query_emb, cand_matrix)
        type_scores[emb_type] = similarities * weight

    if not type_scores:
        return np.zeros(len(candidate_ids)), candidate_ids

    # Combine weighted scores
    total_weight = sum(weights.get(t, 0) for t in type_scores.keys())
    combined_scores = (
        sum(type_scores.values()) / total_weight
        if total_weight > 0
        else np.zeros(len(candidate_ids))
    )

    return combined_scores, candidate_ids


def top_k_indices(scores: np.ndarray, k: int, threshold: float = 0.0) -> np.ndarray:
    """
    Get indices of top-k scores above threshold.

    Uses partial sort for efficiency when k << n.

    Args:
        scores: Array of similarity scores
        k: Number of top results to return
        threshold: Minimum score threshold

    Returns:
        Array of indices sorted by score (descending)
    """
    if len(scores) == 0:
        return np.array([], dtype=np.int64)

    # Apply threshold
    valid_mask = scores >= threshold
    valid_indices = np.where(valid_mask)[0]
    valid_scores = scores[valid_mask]

    if len(valid_scores) == 0:
        return np.array([], dtype=np.int64)

    # Get top-k
    k = min(k, len(valid_scores))

    if k >= len(valid_scores):
        # Return all sorted
        sorted_order = np.argsort(valid_scores)[::-1]
        return valid_indices[sorted_order]

    # Use argpartition for efficiency (O(n) vs O(n log n))
    partition_idx = np.argpartition(valid_scores, -k)[-k:]
    top_k_scores = valid_scores[partition_idx]
    sorted_order = np.argsort(top_k_scores)[::-1]

    return valid_indices[partition_idx[sorted_order]]


def normalize_embeddings(embeddings: EmbeddingMatrix) -> EmbeddingMatrix:
    """
    L2-normalize a matrix of embeddings.

    Useful for pre-normalizing embeddings to speed up later similarity
    computations (can then skip normalization step).

    Args:
        embeddings: Matrix of embeddings (shape [n, dim])

    Returns:
        Normalized embeddings (same shape)
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms
