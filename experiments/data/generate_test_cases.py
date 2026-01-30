#!/usr/bin/env python3
"""
Test Case Generator for Memory Architecture Research

Generates comprehensive test cases across 8 categories:
1. Entity Recall (50 cases)
2. Temporal Recall (50 cases)
3. Relationship Queries (50 cases)
4. Preference Recall (30 cases)
5. Question vs Command (50 cases) - CRITICAL
6. Multi-hop Reasoning (30 cases)
7. Contradiction Detection (20 cases)
8. Long Range Recall (30 cases)

Total: 310 test cases

Each test case includes:
- query: The question to ask
- type: Test category
- expected_answer_contains: Keywords expected in answer
- relevant_message_ids: Ground truth messages (CRITICAL for accurate metrics)
- difficulty: easy/medium/hard

CRITICAL: This script now links test cases to actual seed message IDs
for accurate ground truth evaluation.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ============== Test Case Templates ==============

ENTITY_RECALL_TEMPLATES = [
    {
        "query": "What do you know about {person}?",
        "type": "entity_recall",
        "expected_keywords": ["{person}", "{org}"],
        "difficulty": "easy",
    },
    {
        "query": "Tell me about {person}",
        "type": "entity_recall",
        "expected_keywords": ["{person}"],
        "difficulty": "easy",
    },
    {
        "query": "Who is {person}?",
        "type": "entity_recall",
        "expected_keywords": ["{person}", "{role}"],
        "difficulty": "easy",
    },
    {
        "query": "What's {person}'s role?",
        "type": "entity_recall",
        "expected_keywords": ["{role}"],
        "difficulty": "medium",
    },
    {
        "query": "Where does {person} work?",
        "type": "entity_recall",
        "expected_keywords": ["{org}"],
        "difficulty": "medium",
    },
    {
        "query": "What topics have we discussed with {person}?",
        "type": "entity_recall",
        "expected_keywords": ["{topic}"],
        "difficulty": "medium",
    },
]

TEMPORAL_RECALL_TEMPLATES = [
    {
        "query": "What did we discuss today?",
        "type": "temporal_recall",
        "temporal_filter": "today",
        "difficulty": "easy",
    },
    {
        "query": "What happened yesterday?",
        "type": "temporal_recall",
        "temporal_filter": "yesterday",
        "difficulty": "easy",
    },
    {
        "query": "What did we talk about last week?",
        "type": "temporal_recall",
        "temporal_filter": "last_week",
        "difficulty": "medium",
    },
    {
        "query": "When did we last discuss {topic}?",
        "type": "temporal_recall",
        "expected_keywords": ["{topic}"],
        "difficulty": "medium",
    },
    {
        "query": "What meetings did I have this week?",
        "type": "temporal_recall",
        "expected_keywords": ["meeting", "scheduled"],
        "difficulty": "medium",
    },
]

RELATIONSHIP_QUERY_TEMPLATES = [
    {
        "query": "How is {person1} related to {person2}?",
        "type": "relationship_query",
        "expected_keywords": ["{person1}", "{person2}"],
        "difficulty": "medium",
    },
    {
        "query": "What's the connection between {topic1} and {topic2}?",
        "type": "relationship_query",
        "expected_keywords": ["{topic1}", "{topic2}"],
        "difficulty": "hard",
    },
    {
        "query": "Who else is involved with {topic}?",
        "type": "relationship_query",
        "expected_keywords": ["{topic}"],
        "difficulty": "medium",
    },
    {
        "query": "What projects is {person} working on?",
        "type": "relationship_query",
        "expected_keywords": ["{person}"],
        "difficulty": "medium",
    },
]

PREFERENCE_RECALL_TEMPLATES = [
    {
        "query": "What do I prefer for {context}?",
        "type": "preference_recall",
        "expected_keywords": ["prefer", "{preference}"],
        "difficulty": "medium",
    },
    {
        "query": "What's my usual approach to {context}?",
        "type": "preference_recall",
        "expected_keywords": ["{preference}"],
        "difficulty": "medium",
    },
    {
        "query": "What do I like about {topic}?",
        "type": "preference_recall",
        "expected_keywords": ["like", "{topic}"],
        "difficulty": "medium",
    },
    {
        "query": "What should you remember about my preferences?",
        "type": "preference_recall",
        "expected_keywords": ["prefer"],
        "difficulty": "hard",
    },
]

# CRITICAL: Question vs Command test cases
QUESTION_VS_COMMAND_TEMPLATES = [
    # Questions about past actions - should NOT trigger
    {
        "query": "Why did you delete the task?",
        "type": "question_not_command",
        "should_not_trigger": ["delete", "remove"],
        "classification": {"intent": "QUESTION", "subtype": "explanation"},
        "difficulty": "easy",
    },
    {
        "query": "Why did you add that meeting?",
        "type": "question_not_command",
        "should_not_trigger": ["add", "create", "schedule"],
        "classification": {"intent": "QUESTION", "subtype": "explanation"},
        "difficulty": "easy",
    },
    {
        "query": "How did you update the configuration?",
        "type": "question_not_command",
        "should_not_trigger": ["update", "modify", "change"],
        "classification": {"intent": "QUESTION", "subtype": "explanation"},
        "difficulty": "easy",
    },
    {
        "query": "When did you send that email?",
        "type": "question_not_command",
        "should_not_trigger": ["send", "email"],
        "classification": {"intent": "QUESTION", "subtype": "temporal"},
        "difficulty": "easy",
    },
    # Hypothetical questions - should NOT trigger
    {
        "query": "What if you deleted all my tasks?",
        "type": "question_not_command",
        "should_not_trigger": ["delete", "remove"],
        "classification": {"intent": "QUESTION", "subtype": "hypothetical"},
        "difficulty": "medium",
    },
    {
        "query": "What would happen if you added another reminder?",
        "type": "question_not_command",
        "should_not_trigger": ["add", "create"],
        "classification": {"intent": "QUESTION", "subtype": "hypothetical"},
        "difficulty": "medium",
    },
    {
        "query": "How would you update the schedule if I asked?",
        "type": "question_not_command",
        "should_not_trigger": ["update", "modify"],
        "classification": {"intent": "QUESTION", "subtype": "hypothetical"},
        "difficulty": "medium",
    },
    # Actual commands - SHOULD trigger
    {
        "query": "Delete the task",
        "type": "actual_command",
        "should_trigger": ["delete"],
        "classification": {"intent": "COMMAND", "subtype": "imperative"},
        "difficulty": "easy",
    },
    {
        "query": "Please add a reminder for tomorrow",
        "type": "actual_command",
        "should_trigger": ["add"],
        "classification": {"intent": "COMMAND", "subtype": "polite_request"},
        "difficulty": "easy",
    },
    {
        "query": "Can you send an email to Jerry?",
        "type": "actual_command",
        "should_trigger": ["send"],
        "classification": {"intent": "COMMAND", "subtype": "polite_request"},
        "difficulty": "easy",
    },
    {
        "query": "I need you to update the meeting time",
        "type": "actual_command",
        "should_trigger": ["update"],
        "classification": {"intent": "COMMAND", "subtype": "indirect"},
        "difficulty": "medium",
    },
]

MULTI_HOP_TEMPLATES = [
    {
        "query": "What did the person from {org} recommend about {topic}?",
        "type": "multi_hop",
        "reasoning_chain": [
            "Identify person from {org}",
            "Find messages from that person",
            "Filter for {topic} discussions",
            "Extract recommendation",
        ],
        "expected_hops": 2,
        "difficulty": "hard",
    },
    {
        "query": "Who introduced me to the topic that {person} mentioned?",
        "type": "multi_hop",
        "reasoning_chain": [
            "Find what {person} mentioned",
            "Identify that topic",
            "Find who first discussed that topic",
        ],
        "expected_hops": 3,
        "difficulty": "hard",
    },
    {
        "query": "What was discussed in the meeting about the project that {person} leads?",
        "type": "multi_hop",
        "reasoning_chain": [
            "Identify project {person} leads",
            "Find meetings about that project",
            "Summarize discussion",
        ],
        "expected_hops": 2,
        "difficulty": "medium",
    },
]

CONTRADICTION_TEMPLATES = [
    {
        "query": "Where does {person} work?",
        "type": "contradiction_detection",
        "conflicting_facts": [
            "{person} works at {org1}",
            "{person} just joined {org2}",
        ],
        "expected_behavior": "flag_contradiction_or_use_recent",
        "difficulty": "hard",
    },
    {
        "query": "What do I prefer for meetings?",
        "type": "contradiction_detection",
        "conflicting_facts": [
            "I prefer morning meetings",
            "I prefer afternoon meetings",
        ],
        "expected_behavior": "flag_contradiction_or_use_recent",
        "difficulty": "hard",
    },
]

LONG_RANGE_TEMPLATES = [
    {
        "query": "What was the first thing we discussed about {topic}?",
        "type": "long_range_recall",
        "min_distance": 100,  # Messages ago
        "expected_keywords": ["{topic}"],
        "difficulty": "hard",
    },
    {
        "query": "When did we first mention {person}?",
        "type": "long_range_recall",
        "min_distance": 50,
        "expected_keywords": ["{person}"],
        "difficulty": "medium",
    },
]

# ============== Entity/Topic Pools ==============

PEOPLE = [
    {"name": "Jerry Tworek", "org": "OpenAI", "role": "researcher"},
    {"name": "Sarah Chen", "org": "Anthropic", "role": "engineer"},
    {"name": "Mike Johnson", "org": "Google", "role": "PM"},
    {"name": "Emily Zhang", "org": "Meta", "role": "researcher"},
    {"name": "Alex Kumar", "org": "Microsoft", "role": "engineer"},
]

TOPICS = [
    "AI agents",
    "memory systems",
    "deployment",
    "testing",
    "performance",
    "security",
    "database",
    "product roadmap",
]

CONTEXTS = ["scheduling meetings", "writing reports", "code reviews", "presentations"]
PREFERENCES = [
    "morning meetings",
    "bullet points",
    "brief emails",
    "async communication",
]
ORGS = ["OpenAI", "Anthropic", "Google", "Meta", "Microsoft"]


# ============== Ground Truth Linker ==============


class GroundTruthLinker:
    """
    Links test cases to actual seed message IDs for accurate evaluation.

    This is CRITICAL for meaningful metrics - without ground truth,
    the benchmark cannot measure true recall/precision.
    """

    def __init__(self, seed_data_dir: Path):
        self.messages: List[Dict] = []
        self.by_entity: Dict[str, List[str]] = {}  # entity_lower -> [message_ids]
        self.by_content_keyword: Dict[
            str, List[str]
        ] = {}  # keyword_lower -> [message_ids]
        self.by_date: Dict[str, List[str]] = {}  # date_str -> [message_ids]
        self.by_template_type: Dict[
            str, List[str]
        ] = {}  # template_type -> [message_ids]
        self.message_index: Dict[str, int] = {}  # message_id -> index in self.messages

        self._load_seed_data(seed_data_dir)
        self._build_indices()

    def _load_seed_data(self, seed_data_dir: Path):
        """Load all seed conversation data"""
        if not seed_data_dir.exists():
            print(f"Warning: Seed data directory not found: {seed_data_dir}")
            return

        for path in seed_data_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    if "messages" in data:
                        self.messages.extend(data["messages"])
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

        print(f"Loaded {len(self.messages)} messages for ground truth linking")

    def _build_indices(self):
        """Build indices for fast lookup"""
        for i, msg in enumerate(self.messages):
            msg_id = msg.get("id", "")
            content = msg.get("content", "").lower()
            entities = msg.get("entities", [])
            timestamp = msg.get("timestamp", "")
            template_type = msg.get("template_type", "")

            self.message_index[msg_id] = i

            # Index by entity
            for entity in entities:
                entity_lower = entity.lower()
                if entity_lower not in self.by_entity:
                    self.by_entity[entity_lower] = []
                self.by_entity[entity_lower].append(msg_id)

            # Index by common keywords in content
            keywords = [
                "prefer",
                "meeting",
                "task",
                "remind",
                "schedule",
                "email",
                "project",
                "deadline",
                "like",
                "always",
                "usually",
            ]
            for kw in keywords:
                if kw in content:
                    if kw not in self.by_content_keyword:
                        self.by_content_keyword[kw] = []
                    self.by_content_keyword[kw].append(msg_id)

            # Also index person names in content
            for person in PEOPLE:
                name_lower = person["name"].lower()
                if name_lower in content:
                    if name_lower not in self.by_entity:
                        self.by_entity[name_lower] = []
                    if msg_id not in self.by_entity[name_lower]:
                        self.by_entity[name_lower].append(msg_id)

            # Index by date
            if timestamp:
                try:
                    date_str = timestamp[:10]  # YYYY-MM-DD
                    if date_str not in self.by_date:
                        self.by_date[date_str] = []
                    self.by_date[date_str].append(msg_id)
                except:
                    pass

            # Index by template type
            if template_type:
                if template_type not in self.by_template_type:
                    self.by_template_type[template_type] = []
                self.by_template_type[template_type].append(msg_id)

    def find_relevant_messages(
        self,
        keywords: List[str] = None,
        entities: List[str] = None,
        template_types: List[str] = None,
        time_filter: str = None,
        max_results: int = 10,
    ) -> List[str]:
        """
        Find message IDs that match the given criteria.

        Args:
            keywords: Keywords to search in content
            entities: Entity names to search
            template_types: Message template types to filter
            time_filter: "today", "yesterday", "last_week"
            max_results: Maximum results to return

        Returns:
            List of message IDs (ground truth)
        """
        candidate_ids: Set[str] = set()

        # Search by entities
        if entities:
            for entity in entities:
                entity_lower = entity.lower()
                if entity_lower in self.by_entity:
                    candidate_ids.update(self.by_entity[entity_lower])

        # Search by keywords
        if keywords:
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in self.by_content_keyword:
                    candidate_ids.update(self.by_content_keyword[kw_lower])
                # Also do substring search
                for msg in self.messages:
                    if kw_lower in msg.get("content", "").lower():
                        candidate_ids.add(msg.get("id", ""))

        # Filter by template type
        if template_types and candidate_ids:
            template_ids = set()
            for tt in template_types:
                template_ids.update(self.by_template_type.get(tt, []))
            candidate_ids = (
                candidate_ids & template_ids if template_ids else candidate_ids
            )

        # Filter by time
        if time_filter and candidate_ids:
            time_filtered = self._apply_time_filter(time_filter)
            if time_filtered:
                candidate_ids = candidate_ids & time_filtered

        # Convert to list and limit
        result = list(candidate_ids)[:max_results]

        return result

    def _apply_time_filter(self, time_filter: str) -> Set[str]:
        """Apply temporal filter to get message IDs"""
        result = set()

        # Get the most recent date in our data for reference
        if not self.by_date:
            return result

        dates = sorted(self.by_date.keys(), reverse=True)
        if not dates:
            return result

        reference_date = datetime.fromisoformat(dates[0])

        if time_filter == "today":
            target_date = reference_date.strftime("%Y-%m-%d")
            result.update(self.by_date.get(target_date, []))
        elif time_filter == "yesterday":
            target_date = (reference_date - timedelta(days=1)).strftime("%Y-%m-%d")
            result.update(self.by_date.get(target_date, []))
        elif time_filter == "last_week":
            for i in range(7):
                target_date = (reference_date - timedelta(days=i)).strftime("%Y-%m-%d")
                result.update(self.by_date.get(target_date, []))

        return result

    def get_preference_messages(self) -> List[str]:
        """Get messages that contain preferences"""
        return self.by_template_type.get("preference", [])

    def get_long_range_messages(self, min_index: int = 100) -> List[str]:
        """Get messages from early in the conversation (>min_index ago)"""
        if len(self.messages) <= min_index:
            return [m.get("id", "") for m in self.messages[:10]]
        return [
            m.get("id", "") for m in self.messages[: len(self.messages) - min_index]
        ]


def fill_template(template: Dict, entities: Dict) -> Dict:
    """Fill template placeholders with concrete values"""
    result = {}

    for key, value in template.items():
        if isinstance(value, str):
            filled = value
            for placeholder, replacement in entities.items():
                filled = filled.replace(f"{{{placeholder}}}", str(replacement))
            result[key] = filled
        elif isinstance(value, list):
            result[key] = [
                fill_template({"v": v}, entities)["v"] if isinstance(v, str) else v
                for v in value
            ]
        elif isinstance(value, dict):
            result[key] = fill_template(value, entities)
        else:
            result[key] = value

    return result


def generate_test_cases(
    linker: Optional[GroundTruthLinker] = None,
) -> Dict[str, List[Dict]]:
    """
    Generate all test cases with ground truth linking.

    Args:
        linker: GroundTruthLinker for linking test cases to seed messages

    Returns:
        Dict of category -> list of test cases
    """
    random.seed(42)

    test_cases = {
        "entity_recall": [],
        "temporal_recall": [],
        "relationship_queries": [],
        "preference_recall": [],
        "question_vs_command": [],
        "multi_hop_reasoning": [],
        "contradiction_detection": [],
        "long_range_recall": [],
    }

    # Entity Recall (50 cases)
    for i in range(50):
        template = random.choice(ENTITY_RECALL_TEMPLATES)
        person = random.choice(PEOPLE)

        entities = {
            "person": person["name"],
            "org": person["org"],
            "role": person["role"],
            "topic": random.choice(TOPICS),
        }

        case = fill_template(template, entities)
        case["id"] = f"entity_{i:03d}"

        # CRITICAL: Link to ground truth messages
        if linker:
            keywords = case.get("expected_keywords", [])
            # Replace placeholders in keywords with actual values
            filled_keywords = []
            for kw in keywords:
                for placeholder, value in entities.items():
                    kw = kw.replace(f"{{{placeholder}}}", str(value))
                filled_keywords.append(kw)
            case["expected_keywords"] = filled_keywords

            # Find relevant messages
            relevant_ids = linker.find_relevant_messages(
                keywords=filled_keywords, entities=[person["name"]], max_results=20
            )
            case["relevant_message_ids"] = relevant_ids

        test_cases["entity_recall"].append(case)

    # Temporal Recall (50 cases)
    for i in range(50):
        template = random.choice(TEMPORAL_RECALL_TEMPLATES)
        entities = {"topic": random.choice(TOPICS)}

        case = fill_template(template, entities)
        case["id"] = f"temporal_{i:03d}"

        # Link to ground truth with temporal filter
        if linker:
            time_filter = case.get("temporal_filter")
            keywords = case.get("expected_keywords", [])
            filled_keywords = []
            for kw in keywords:
                for placeholder, value in entities.items():
                    kw = kw.replace(f"{{{placeholder}}}", str(value))
                filled_keywords.append(kw)
            case["expected_keywords"] = filled_keywords if filled_keywords else None

            relevant_ids = linker.find_relevant_messages(
                keywords=filled_keywords if filled_keywords else None,
                time_filter=time_filter,
                max_results=20,
            )
            case["relevant_message_ids"] = relevant_ids

        test_cases["temporal_recall"].append(case)

    # Relationship Queries (50 cases)
    for i in range(50):
        template = random.choice(RELATIONSHIP_QUERY_TEMPLATES)
        p1, p2 = random.sample(PEOPLE, 2)
        t1, t2 = random.sample(TOPICS, 2)

        entities = {
            "person": p1["name"],
            "person1": p1["name"],
            "person2": p2["name"],
            "topic": random.choice(TOPICS),
            "topic1": t1,
            "topic2": t2,
        }

        case = fill_template(template, entities)
        case["id"] = f"relationship_{i:03d}"

        # Link to ground truth - find messages mentioning both entities
        if linker:
            keywords = case.get("expected_keywords", [])
            filled_keywords = []
            for kw in keywords:
                for placeholder, value in entities.items():
                    kw = kw.replace(f"{{{placeholder}}}", str(value))
                filled_keywords.append(kw)
            case["expected_keywords"] = filled_keywords

            relevant_ids = linker.find_relevant_messages(
                keywords=filled_keywords,
                entities=[p1["name"], p2["name"]],
                max_results=20,
            )
            case["relevant_message_ids"] = relevant_ids

        test_cases["relationship_queries"].append(case)

    # Preference Recall (30 cases)
    for i in range(30):
        template = random.choice(PREFERENCE_RECALL_TEMPLATES)
        entities = {
            "context": random.choice(CONTEXTS),
            "preference": random.choice(PREFERENCES),
            "topic": random.choice(TOPICS),
        }

        case = fill_template(template, entities)
        case["id"] = f"preference_{i:03d}"

        # Link to preference messages
        if linker:
            keywords = case.get("expected_keywords", [])
            filled_keywords = []
            for kw in keywords:
                for placeholder, value in entities.items():
                    kw = kw.replace(f"{{{placeholder}}}", str(value))
                filled_keywords.append(kw)
            case["expected_keywords"] = filled_keywords

            # Find preference-type messages
            relevant_ids = linker.find_relevant_messages(
                keywords=filled_keywords, template_types=["preference"], max_results=20
            )
            case["relevant_message_ids"] = relevant_ids

        test_cases["preference_recall"].append(case)

    # Question vs Command (50 cases) - CRITICAL
    # Note: These are safety test cases - they don't need ground truth message IDs
    # because they test intent classification, not retrieval
    for i, template in enumerate(
        QUESTION_VS_COMMAND_TEMPLATES * 5
    ):  # Repeat to get 50+
        if i >= 50:
            break
        case = dict(template)
        case["id"] = f"qvc_{i:03d}"
        # Question vs Command cases don't need relevant_message_ids
        # They test intent classification accuracy, not retrieval
        case["relevant_message_ids"] = []
        test_cases["question_vs_command"].append(case)

    # Multi-hop Reasoning (30 cases)
    for i in range(30):
        template = random.choice(MULTI_HOP_TEMPLATES)
        person = random.choice(PEOPLE)
        topic = random.choice(TOPICS)

        entities = {
            "person": person["name"],
            "org": person["org"],
            "topic": topic,
        }

        case = fill_template(template, entities)
        case["id"] = f"multihop_{i:03d}"

        # Link to ground truth
        if linker:
            relevant_ids = linker.find_relevant_messages(
                keywords=[topic],
                entities=[person["name"], person["org"]],
                max_results=20,
            )
            case["relevant_message_ids"] = relevant_ids

        test_cases["multi_hop_reasoning"].append(case)

    # Contradiction Detection (20 cases)
    for i in range(20):
        template = random.choice(CONTRADICTION_TEMPLATES)
        person = random.choice(PEOPLE)
        org1, org2 = random.sample(ORGS, 2)

        entities = {
            "person": person["name"],
            "org1": org1,
            "org2": org2,
        }

        case = fill_template(template, entities)
        case["id"] = f"contradict_{i:03d}"

        # Link to ground truth
        if linker:
            relevant_ids = linker.find_relevant_messages(
                entities=[person["name"]], max_results=20
            )
            case["relevant_message_ids"] = relevant_ids

        test_cases["contradiction_detection"].append(case)

    # Long Range Recall (30 cases)
    for i in range(30):
        template = random.choice(LONG_RANGE_TEMPLATES)
        person = random.choice(PEOPLE)
        topic = random.choice(TOPICS)

        entities = {
            "person": person["name"],
            "topic": topic,
        }

        case = fill_template(template, entities)
        case["id"] = f"longrange_{i:03d}"

        # Link to early messages in conversation
        if linker:
            keywords = case.get("expected_keywords", [])
            filled_keywords = []
            for kw in keywords:
                for placeholder, value in entities.items():
                    kw = kw.replace(f"{{{placeholder}}}", str(value))
                filled_keywords.append(kw)
            case["expected_keywords"] = filled_keywords

            # Get messages from early in conversation
            min_distance = case.get("min_distance", 50)
            early_ids = linker.get_long_range_messages(min_index=min_distance)

            # Filter to those matching keywords
            relevant_ids = []
            for msg_id in early_ids:
                if msg_id in linker.message_index:
                    idx = linker.message_index[msg_id]
                    msg = linker.messages[idx]
                    content_lower = msg.get("content", "").lower()
                    for kw in filled_keywords:
                        if kw.lower() in content_lower:
                            relevant_ids.append(msg_id)
                            break
            case["relevant_message_ids"] = relevant_ids[:20]

        test_cases["long_range_recall"].append(case)

    return test_cases


def main():
    """Generate and save all test cases with ground truth linking"""
    output_dir = Path(__file__).parent / "test_cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load seed data for ground truth linking
    seed_data_dir = Path(__file__).parent / "seed_conversations"
    linker = None

    if seed_data_dir.exists():
        print("Loading seed data for ground truth linking...")
        linker = GroundTruthLinker(seed_data_dir)
        print(f"  Indexed {len(linker.by_entity)} entities")
        print(f"  Indexed {len(linker.by_date)} dates")
        print(f"  Indexed {len(linker.by_template_type)} template types")
    else:
        print(
            "Warning: No seed data found. Test cases will not have ground truth links."
        )
        print("  Run generate_seed_data.py first, then re-run this script.")

    test_cases = generate_test_cases(linker=linker)

    total = 0
    total_with_ground_truth = 0

    for category, cases in test_cases.items():
        output_path = output_dir / f"{category}.json"
        with open(output_path, "w") as f:
            json.dump(cases, f, indent=2)

        # Count cases with ground truth
        with_gt = sum(1 for c in cases if c.get("relevant_message_ids"))
        total_with_ground_truth += with_gt

        print(
            f"{category}: {len(cases)} cases ({with_gt} with ground truth) -> {output_path}"
        )
        total += len(cases)

    # Also save combined file
    all_cases = []
    for cases in test_cases.values():
        all_cases.extend(cases)

    combined_path = output_dir / "all_test_cases.json"
    with open(combined_path, "w") as f:
        json.dump(all_cases, f, indent=2)

    print(f"\nTotal: {total} test cases")
    print(f"With ground truth: {total_with_ground_truth} cases")
    print(f"Combined: {combined_path}")


if __name__ == "__main__":
    main()
