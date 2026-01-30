#!/usr/bin/env python3
"""
Memory Architecture Research Benchmark Runner

This is the main entry point for running the benchmark.

USAGE:
    # Run full benchmark
    python run_benchmark.py

    # Run specific systems only
    python run_benchmark.py --systems raw_context multi_vector

    # Run with minimal test cases (for testing)
    python run_benchmark.py --quick

    # Run with custom output directory
    python run_benchmark.py --output results/my_experiment

COST ESTIMATE:
- Embeddings: FREE (Gemini gemini-embedding-001)
- LLM Judge: ~$5 for 100 samples
- Total: <$50 for full benchmark
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from approaches.a_flat_vector import FlatVectorMemory
from approaches.b_multi_vector import MultiVectorMemory
from approaches.c_vector_graph import VectorGraphMemory
from approaches.d_hypergraph import HypergraphMemory
from approaches.e_hierarchical import HierarchicalMemory
from approaches.k_ultimate_hybrid import UltimateHybridMemory
from baselines.langchain_adapter import LangChainBufferAdapter
from baselines.raw_context import RawContextAdapter
from config import DATA_DIR, RESULTS_DIR, config
from evaluation.run_eval import Evaluator, run_full_evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_all_systems() -> Dict[str, Any]:
    """Get all available memory systems"""
    systems = {
        # Baselines
        "raw_context": RawContextAdapter(),

        # Our approaches
        "flat_vector": FlatVectorMemory(),
        "multi_vector": MultiVectorMemory(),
        "vector_graph": VectorGraphMemory(),
        "hypergraph": HypergraphMemory(),
        "hierarchical": HierarchicalMemory(),
        "ultimate_hybrid": UltimateHybridMemory(),
    }

    # Get API keys from environment
    import os
    letta_key = os.environ.get("LETTA_API_KEY", "")
    mem0_key = os.environ.get("MEM0_API_KEY", "")
    supermemory_key = os.environ.get("SUPERMEMORY_API_KEY", "")

    # Try to add Letta baseline
    if letta_key:
        try:
            from baselines.letta_adapter import LettaAdapter
            systems["letta"] = LettaAdapter(api_key=letta_key)
            logger.info("Letta baseline added")
        except ImportError as e:
            logger.warning(f"Letta not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize Letta: {e}")
    else:
        logger.warning("LETTA_API_KEY not set - skipping Letta baseline")

    # Try to add mem0 baseline
    if mem0_key:
        try:
            from baselines.mem0_adapter import Mem0Adapter
            systems["mem0"] = Mem0Adapter(api_key=mem0_key, use_cloud=True)
            logger.info("mem0 baseline added")
        except ImportError as e:
            logger.warning(f"mem0 not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize mem0: {e}")
    else:
        logger.warning("MEM0_API_KEY not set - skipping mem0 baseline")

    # Try to add Supermemory baseline
    if supermemory_key:
        try:
            from baselines.supermemory_adapter import SupermemoryAdapter
            systems["supermemory"] = SupermemoryAdapter(api_key=supermemory_key)
            logger.info("Supermemory baseline added")
        except ImportError as e:
            logger.warning(f"Supermemory not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize Supermemory: {e}")
    else:
        logger.warning("SUPERMEMORY_API_KEY not set - skipping Supermemory baseline")

    return systems


def initialize_systems(
    systems: Dict[str, Any],
    embedding_model=None,
    entity_extractor=None,
    intent_classifier=None,
):
    """Initialize systems with components"""
    for name, system in systems.items():
        if hasattr(system, "initialize"):
            system.initialize(
                embedding_model=embedding_model,
                entity_extractor=entity_extractor,
                intent_classifier=intent_classifier,
            )
            logger.info(f"Initialized {name}")


def load_test_data(quick: bool = False):
    """Load seed data and test cases"""
    seed_dir = DATA_DIR / "seed_conversations"
    test_dir = DATA_DIR / "test_cases"

    # Load messages
    messages = []
    for path in seed_dir.glob("*.json"):
        with open(path) as f:
            data = json.load(f)
            if "messages" in data:
                messages.extend(data["messages"])

    # Load test cases
    test_cases = []
    for path in test_dir.glob("*.json"):
        if path.name == "all_test_cases.json":
            continue  # Skip combined file
        with open(path) as f:
            test_cases.extend(json.load(f))

    if quick:
        # Reduce for quick testing
        messages = messages[:100]
        test_cases = test_cases[:50]

    logger.info(f"Loaded {len(messages)} messages, {len(test_cases)} test cases")
    return messages, test_cases


def print_results_summary(results: List[Any]):
    """Print summary of results"""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    def get_recall(r):
        """Safely get recall@5 from result"""
        try:
            if hasattr(r, "retrieval") and r.retrieval:
                return r.retrieval.recall_at_5
            elif isinstance(r, dict) and "retrieval" in r:
                return r["retrieval"].get("recall@5", 0)
        except Exception:
            pass
        return 0

    # Sort by recall@5
    sorted_results = sorted(results, key=get_recall, reverse=True)

    print(f"\n{'System':<25} {'R@5':<8} {'MRR':<8} {'FAR':<8} {'P50ms':<8}")
    print("-" * 57)

    for r in sorted_results:
        name = (
            r.system_name
            if hasattr(r, "system_name")
            else str(r.get("system_name", "?"))
        )
        r5 = (
            r.retrieval.recall_at_5
            if hasattr(r, "retrieval")
            else r.get("retrieval", {}).get("recall@5", 0)
        )
        mrr = (
            r.retrieval.mrr
            if hasattr(r, "retrieval")
            else r.get("retrieval", {}).get("mrr", 0)
        )
        far = (
            r.false_action_rate
            if hasattr(r, "false_action_rate")
            else r.get("false_action_rate", 0)
        )
        p50 = (
            r.latency_p50_ms
            if hasattr(r, "latency_p50_ms")
            else r.get("latency_p50_ms", 0)
        )

        print(f"{name:<25} {r5:<8.3f} {mrr:<8.3f} {far:<8.3f} {p50:<8.1f}")

    # Highlight winner
    if sorted_results:
        winner = sorted_results[0]
        name = (
            winner.system_name
            if hasattr(winner, "system_name")
            else winner.get("system_name", "?")
        )
        print(f"\n🏆 WINNER: {name}")


def estimate_cost(num_messages: int, num_test_cases: int) -> float:
    """Estimate total cost for benchmark"""
    # Embeddings: Free with Gemini
    embedding_cost = 0.0

    # LLM Judge: ~$0.0003/1K tokens, ~500 tokens per evaluation
    judge_samples = min(100, num_test_cases)
    judge_tokens = judge_samples * 500
    judge_cost = (judge_tokens / 1000) * 0.0003

    # Synthesis: ~100 calls, ~200 tokens each
    synthesis_cost = (100 * 200 / 1000) * 0.0003

    total = embedding_cost + judge_cost + synthesis_cost

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Memory Architecture Research Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Systems to benchmark (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with reduced data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge evaluation (faster, cheaper)",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Only estimate cost, don't run",
    )

    args = parser.parse_args()

    # Load data
    messages, test_cases = load_test_data(quick=args.quick)

    # Cost estimate
    estimated_cost = estimate_cost(len(messages), len(test_cases))
    logger.info(f"Estimated cost: ${estimated_cost:.2f}")

    if args.estimate_cost:
        print(f"\nEstimated benchmark cost: ${estimated_cost:.2f}")
        print(f"  - Messages: {len(messages)}")
        print(f"  - Test cases: {len(test_cases)}")
        print(f"  - Embeddings: FREE (Gemini)")
        print(f"  - LLM Judge: ~${estimated_cost:.2f}")
        return

    # Get systems
    all_systems = get_all_systems()

    if args.systems:
        systems = {k: v for k, v in all_systems.items() if k in args.systems}
        if not systems:
            logger.error(f"No matching systems. Available: {list(all_systems.keys())}")
            return
    else:
        systems = all_systems

    logger.info(f"Benchmarking {len(systems)} systems: {list(systems.keys())}")

    # Initialize embedding model if API key is available
    embedding_model = None
    entity_extractor = None
    intent_classifier = None

    if config.gemini_api_key:
        logger.info("Initializing Gemini embedding model...")
        try:
            from components.embeddings import GeminiEmbedding, MultiEmbedding

            # First create the base Gemini embedding model
            base_model = GeminiEmbedding(api_key=config.gemini_api_key)
            # Then wrap it in MultiEmbedding
            embedding_model = MultiEmbedding(embedding_model=base_model)
            logger.info("Embedding model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
            import traceback

            traceback.print_exc()

    if config.gemini_api_key:
        try:
            from components.entity_extraction import EntityExtractor

            entity_extractor = EntityExtractor(use_llm=False)  # Rule-based for speed
            logger.info("Entity extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize entity extractor: {e}")

        try:
            from components.intent_classification import IntentClassifier

            intent_classifier = IntentClassifier(use_llm=False)  # Rule-based for speed
            logger.info("Intent classifier initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize intent classifier: {e}")

    # Initialize systems with components
    initialize_systems(
        systems,
        embedding_model=embedding_model,
        entity_extractor=entity_extractor,
        intent_classifier=intent_classifier,
    )

    # Create evaluator
    llm_client = None  # Would need to configure with API key
    evaluator = Evaluator(
        llm_client=None if args.no_judge else llm_client,
        llm_judge_sample_size=50 if args.quick else 100,
    )

    # Run evaluation
    results = []
    for name, system in systems.items():
        logger.info(f"Evaluating {name}...")
        start = time.time()

        try:
            result = evaluator.evaluate_system(system, messages, test_cases)
            results.append(result)
            logger.info(f"  Completed in {time.time() - start:.1f}s")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

    # Save results
    output_dir = args.output or RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"benchmark_{timestamp}.json"

    evaluator.save_results(results, output_path)

    # Print summary
    print_results_summary(results)

    print(f"\nResults saved to: {output_path}")
    print(f"Total cost: ~${estimated_cost:.2f}")


if __name__ == "__main__":
    main()
