"""
Configuration for Memory Architecture Research Benchmark

Copy this file to config.py and fill in your API keys.
Or create a .env file with your keys.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Load .env file if it exists
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env_path)
    except ImportError:
        # Fallback: manual .env parsing
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SEED_DATA_DIR = DATA_DIR / "seed_conversations"
TEST_CASES_DIR = DATA_DIR / "test_cases"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
RESULTS_DIR = BASE_DIR / "results"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""

    # Primary: Gemini embedding (free, SOTA)
    gemini_model: str = "models/gemini-embedding-001"
    gemini_dimensions: int = 3072  # MRL: can scale to 768, 256

    # Fallback: Gemini older model
    gemini_fallback_model: str = "models/text-embedding-004"
    gemini_fallback_dimensions: int = 768

    # Alternative: OpenAI (higher accuracy but costs)
    openai_model: str = "text-embedding-3-small"
    openai_dimensions: int = 1536

    # Batch settings
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class LLMConfig:
    """Configuration for LLM calls"""

    # Gemini for extraction/synthesis
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.1

    # OpenAI fallback
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1

    # Judge models (use multiple for averaging)
    judge_models: list = field(
        default_factory=lambda: ["gemini-2.0-flash", "gpt-4o-mini", "gemini-1.5-flash"]
    )


@dataclass
class BaselineConfig:
    """Configuration for baseline memory systems"""

    # Letta/MemGPT
    letta_api_key: Optional[str] = None
    letta_base_url: str = "http://localhost:8283"

    # mem0
    mem0_api_key: Optional[str] = None

    # Supermemory
    supermemory_api_key: Optional[str] = None

    # LangChain (uses Gemini/OpenAI keys)


@dataclass
class QueryWeightConfig:
    """
    Centralized configuration for query-adaptive embedding weights.

    These weights determine how content, entity, and intent embeddings
    are combined based on query type. Having them in one place ensures
    consistency across all approaches.
    """

    # Default weights for general queries
    default: dict = field(
        default_factory=lambda: {
            "content": 0.6,
            "entity": 0.2,
            "intent": 0.2,
        }
    )

    # Entity-focused queries: "Who is X?", "What is Y?", "Tell me about Z"
    entity_focused: dict = field(
        default_factory=lambda: {
            "content": 0.3,
            "entity": 0.5,
            "intent": 0.2,
        }
    )

    # Temporal queries: "When did...", "Yesterday...", "Last week..."
    temporal: dict = field(
        default_factory=lambda: {
            "content": 0.8,
            "entity": 0.1,
            "intent": 0.1,
        }
    )

    # Intent/action queries: "Did I ask to...", "Did I want..."
    intent_focused: dict = field(
        default_factory=lambda: {
            "content": 0.3,
            "entity": 0.2,
            "intent": 0.5,
        }
    )

    # Relationship queries: "How is X related to Y?"
    relationship: dict = field(
        default_factory=lambda: {
            "content": 0.3,
            "entity": 0.5,
            "intent": 0.2,
        }
    )

    # Keywords for detecting query types
    entity_keywords: list = field(
        default_factory=lambda: ["who is", "what is", "tell me about", "know about"]
    )
    temporal_keywords: list = field(
        default_factory=lambda: [
            "when",
            "yesterday",
            "last week",
            "last month",
            "today",
            "tomorrow",
        ]
    )
    intent_keywords: list = field(
        default_factory=lambda: ["did i ask", "did i want", "was i trying", "should i"]
    )
    relationship_keywords: list = field(
        default_factory=lambda: ["related to", "connection", "relationship", "between"]
    )

    def get_weights_for_query(self, query: str) -> dict:
        """
        Determine appropriate weights based on query content.

        Args:
            query: The search query string

        Returns:
            Dict of embedding type -> weight
        """
        query_lower = query.lower()

        # Check for entity-focused queries
        if any(kw in query_lower for kw in self.entity_keywords):
            return self.entity_focused.copy()

        # Check for temporal queries
        if any(kw in query_lower for kw in self.temporal_keywords):
            return self.temporal.copy()

        # Check for intent-focused queries
        if any(kw in query_lower for kw in self.intent_keywords):
            return self.intent_focused.copy()

        # Check for relationship queries
        if any(kw in query_lower for kw in self.relationship_keywords):
            return self.relationship.copy()

        # Default weights
        return self.default.copy()


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""

    # Retrieval metrics
    recall_k_values: list = field(default_factory=lambda: [1, 3, 5, 10, 20])

    # LLM judge settings
    num_judges: int = 3
    judge_temperature: float = 0.0

    # Answer quality scale
    quality_min: int = 1
    quality_max: int = 5

    # Safety thresholds
    max_false_action_rate: float = 0.02  # 2%
    max_hallucination_rate: float = 0.05  # 5%

    # Latency thresholds (ms)
    latency_p50_target: int = 300
    latency_p95_target: int = 1000
    latency_p99_target: int = 2000


@dataclass
class Config:
    """Main configuration class"""

    # API Keys (from environment or set directly)
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )

    # Sub-configs
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    query_weights: QueryWeightConfig = field(default_factory=QueryWeightConfig)

    # Database (for approaches that need persistence)
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL", "postgresql://localhost:5432/memory_benchmark"
        )
    )

    # Experiment settings
    random_seed: int = 42
    verbose: bool = True
    cache_embeddings: bool = True
    embedding_cache_dir: Path = field(
        default_factory=lambda: BASE_DIR / ".embedding_cache"
    )

    def validate(self) -> list[str]:
        """Validate configuration, return list of warnings"""
        warnings = []

        if not self.gemini_api_key:
            warnings.append("GEMINI_API_KEY not set - Gemini embeddings will fail")

        if not self.openai_api_key:
            warnings.append("OPENAI_API_KEY not set - OpenAI fallback unavailable")

        return warnings

    def __post_init__(self):
        """Create necessary directories"""
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()

# Validate on import
_warnings = config.validate()
if _warnings:
    import warnings as _w

    for w in _warnings:
        _w.warn(w, UserWarning)
