# Memory Architecture Research Benchmark

## Ultimate Goal

**Discover the most ROBUST, SCALABLE, GENERALIZED, and BEST-PERFORMING memory system for AI.**

This is not just about building a memory system for makearjowork.com. We are conducting fundamental research to answer:

> What is the optimal architecture for AI memory?

The winner of this benchmark should be:
- **Robust**: Works across domains, handles edge cases, resists adversarial inputs
- **Scalable**: Performs well from 100 to 100,000+ messages
- **Generalized**: Not overfit to one use case or benchmark
- **Best-performing**: Beats all existing systems on all metrics

---

## Directory Structure

```
experiments/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── config.py                    # API keys, model configs
│
├── data/
│   ├── seed_conversations/      # Synthetic conversation datasets
│   │   ├── personal_assistant.json    # 500+ messages
│   │   ├── technical_support.json     # 500+ messages
│   │   ├── knowledge_worker.json      # 500+ messages
│   │   └── mixed_domain.json          # 1000+ messages
│   │
│   ├── test_cases/              # Evaluation queries (310 total)
│   │   ├── entity_recall.json         # 50 cases
│   │   ├── temporal_recall.json       # 50 cases
│   │   ├── relationship_queries.json  # 50 cases
│   │   ├── preference_recall.json     # 30 cases
│   │   ├── question_vs_command.json   # 50 cases (CRITICAL!)
│   │   ├── multi_hop_reasoning.json   # 30 cases
│   │   ├── contradiction_detection.json # 20 cases
│   │   └── long_range_recall.json     # 30 cases
│   │
│   └── ground_truth/            # Human-annotated correct answers
│
├── baselines/                   # Existing systems to compare against
│   ├── base.py                  # Abstract adapter interface
│   ├── letta_adapter.py         # Wrapper for MemGPT/Letta
│   ├── mem0_adapter.py          # Wrapper for mem0
│   ├── supermemory_adapter.py   # Wrapper for Supermemory
│   ├── langchain_adapter.py     # LangChain memory wrappers
│   └── raw_context.py           # Naive baseline (dump all history)
│
├── approaches/                  # Our candidate implementations
│   ├── base.py                  # Abstract base class
│   ├── a_flat_vector.py         # Single embedding
│   ├── b_multi_vector.py        # Content/entity/intent embeddings
│   ├── c_vector_graph.py        # Vector + entity graph
│   ├── d_hypergraph.py          # Hyperedge memory
│   ├── e_hierarchical.py        # Multi-level memory hierarchy
│   ├── f_neural.py              # DNC-style learned retrieval
│   ├── g_hybrid_neural.py       # Neural + explicit structure
│   ├── h_temporal_attention.py  # Attention over past with decay
│   ├── i_predictive.py          # Next-query prediction
│   ├── j_uncertainty_aware.py   # Bayesian confidence tracking
│   └── k_ultimate_hybrid.py     # ASM v2 - our moonshot
│
├── components/                  # Shared utilities
│   ├── embeddings.py            # Gemini/OpenAI embedding functions
│   ├── entity_extraction.py     # NER via Gemini/spaCy
│   ├── intent_classification.py # Command vs question vs info
│   ├── graph_utils.py           # Graph/hypergraph operations
│   ├── summarization.py         # Episodic summary generation
│   └── synthesis.py             # Answer generation from memories
│
├── evaluation/
│   ├── metrics.py               # Recall, precision, F1, latency
│   ├── llm_judge.py             # LLM-as-judge for answer quality
│   ├── action_detector.py       # Detect unintended tool calls
│   └── run_eval.py              # Main evaluation script
│
├── analysis/
│   ├── visualize_results.py     # Generate charts/tables
│   ├── ablation_study.py        # Test component contributions
│   ├── error_analysis.py        # Categorize failure modes
│   └── statistical_tests.py     # Significance testing
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_baseline_comparison.ipynb
    ├── 03_approach_development.ipynb
    └── 04_results_analysis.ipynb
```

---

## Quick Start

```bash
# 1. Setup
cd experiments
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp config.example.py config.py
# Edit config.py with your API keys

# 3. Generate data (if not already present)
python data/generate_seed_data.py
python data/generate_test_cases.py

# 4. Run evaluation
python evaluation/run_eval.py --systems all --test-cases all

# 5. View results
python analysis/visualize_results.py
```

---

## Benchmark Scale

### Test Categories (310 total)

| Category | # Cases | Description |
|----------|---------|-------------|
| Entity Recall | 50 | "What do you know about X?" |
| Temporal Recall | 50 | "What did we discuss last week about Y?" |
| Relationship Queries | 50 | "How is X related to Y?" |
| Preference Recall | 30 | "What do I prefer for Z?" |
| Question vs Command | 50 | "Why did you X?" should NOT trigger X |
| Multi-hop Reasoning | 30 | "What did Person A say about Topic B in Context C?" |
| Contradiction Detection | 20 | "I said X before, now saying not-X" |
| Long Range Recall | 30 | Info from >100 messages ago |

### Seed Data (2500+ messages)

| Dataset | Messages | Sessions | Entities | Domains |
|---------|----------|----------|----------|---------|
| Personal Assistant | 500 | 50 | 80+ | Tasks, scheduling, notes |
| Technical Support | 500 | 40 | 60+ | Debugging, deployment |
| Knowledge Worker | 500 | 45 | 100+ | Research, writing |
| Mixed Domain | 1000 | 80 | 150+ | All of above |

---

## Metrics

### Retrieval Quality
- Recall@k (k=1,3,5,10,20)
- Precision@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

### Answer Quality (LLM-judged, 3 evaluators)
- Correctness (1-5)
- Completeness (1-5)
- Relevance (1-5)
- Coherence (1-5)

### Safety/Robustness
- False Action Rate (questions triggering commands)
- Hallucination Rate (answers not grounded in memory)
- Contradiction Handling (detects vs. ignores)

### Efficiency
- Latency: P50, P95, P99
- Memory Usage
- API Calls per Query
- Tokens Consumed

---

## Systems Under Evaluation

### Baselines
- **Letta/MemGPT**: Two-tier memory with core/recall/archival
- **mem0**: Hybrid vector + graph, 41K GitHub stars
- **Supermemory**: Brain-inspired with intelligent decay
- **LangChain**: ConversationBufferMemory, ConversationSummaryMemory
- **Raw Context**: Naive baseline (our current implementation)

### Our Approaches
- **A: Flat Vector**: Single embedding per message
- **B: Multi-Vector**: Content + Entity + Intent embeddings
- **C: Vector + Graph**: Vector search + entity co-occurrence graph
- **D: Hypergraph**: Messages as hyperedges connecting entities
- **E: Hierarchical**: Immediate → Working → Episodic → Semantic
- **F: Neural**: DNC-style differentiable addressing
- **G: Hybrid Neural**: Neural scoring + explicit structure
- **H: Temporal Attention**: Attention over history with decay
- **I: Predictive**: Next-query pre-fetching
- **J: Uncertainty-Aware**: Bayesian confidence tracking
- **K: Ultimate Hybrid (ASM v2)**: Everything combined

---

## Target Performance

| Metric | Target | Rationale |
|--------|--------|-----------|
| Recall@5 | > 0.85 | Beat all baselines |
| Answer Quality | > 4.5/5 | Near-perfect synthesis |
| False Action Rate | < 2% | Critical for safety |
| Latency P50 | < 300ms | Competitive UX |
| LoCoMo | > 75% | Beat MAGMA (70%) |
| LongMemEval | > 70% | Beat MAGMA (61.2%) |

---

## Key Innovations

1. **Multi-embedding per message** (content, entity, intent) - nobody else does this
2. **Intent-aware retrieval** (question ≠ command) - critical for safety
3. **Multi-graph architecture** with learned traversal - extends MAGMA
4. **Predictive pre-fetching** - near-zero latency
5. **Continuous learning from feedback** - gets better over time
6. **Proactive memory surfacing** - anticipates needs

---

## References

### Papers
- [MAGMA](https://arxiv.org/abs/2601.03236) - Multi-graph architecture (SOTA)
- [LongMemEval](https://arxiv.org/pdf/2410.10813) - ICLR 2025 benchmark
- [LoCoMo](https://snap-research.github.io/locomo/) - Snap Research benchmark
- [A-MEM](https://arxiv.org/abs/2502.12110) - NeurIPS 2025, Zettelkasten
- [H-MEM](https://www.alphaxiv.org/overview/2507.22925v1) - Hierarchical memory

### Products
- [Letta/MemGPT](https://docs.letta.com/) - Two-tier memory
- [mem0](https://mem0.ai/) - Graph memory
- [Supermemory](https://supermemory.ai/) - Brain-inspired decay
