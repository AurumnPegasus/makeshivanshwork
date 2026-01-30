#!/usr/bin/env python3
"""
Seed Data Generator for Memory Architecture Research

Generates realistic, diverse conversation datasets for benchmarking.

DESIGN PRINCIPLES:
1. SCALABLE: Can generate 100 to 100,000+ messages
2. DIVERSE: Multiple domains, personas, conversation types
3. REALISTIC: Natural language patterns, entity relationships
4. TRACEABLE: Every message has ground truth metadata

DATASETS:
- personal_assistant: Task management, scheduling, notes
- technical_support: Debugging, deployment, errors
- knowledge_worker: Research, writing, brainstorming
- mixed_domain: All of the above interleaved

Each dataset includes:
- Messages with role, content, timestamp, session_id
- Ground truth entities extracted
- Ground truth relationships
- Temporal coherence (sessions span time realistically)
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============== Entity Pools ==============

PEOPLE = [
    {"name": "Jerry Tworek", "org": "OpenAI", "role": "researcher", "topics": ["AI agents", "reinforcement learning"]},
    {"name": "Dharsha Udayakumar", "org": "T-Hub", "role": "partner", "topics": ["startups", "incubation"]},
    {"name": "Sarah Chen", "org": "Anthropic", "role": "engineer", "topics": ["safety", "alignment"]},
    {"name": "Mike Johnson", "org": "Google", "role": "PM", "topics": ["product", "strategy"]},
    {"name": "Emily Zhang", "org": "Meta", "role": "researcher", "topics": ["NLP", "transformers"]},
    {"name": "Alex Kumar", "org": "Microsoft", "role": "engineer", "topics": ["Azure", "cloud"]},
    {"name": "Lisa Park", "org": "Amazon", "role": "architect", "topics": ["AWS", "infrastructure"]},
    {"name": "David Wilson", "org": "Stripe", "role": "founder", "topics": ["payments", "fintech"]},
    {"name": "Maria Garcia", "org": "Figma", "role": "designer", "topics": ["UI/UX", "design systems"]},
    {"name": "James Lee", "org": "Notion", "role": "engineer", "topics": ["productivity", "collaboration"]},
    {"name": "Anna Schmidt", "org": "OpenAI", "role": "policy", "topics": ["AI governance", "regulation"]},
    {"name": "Tom Brown", "org": "Scale AI", "role": "founder", "topics": ["data labeling", "ML ops"]},
    {"name": "Rachel Kim", "org": "Databricks", "role": "data scientist", "topics": ["analytics", "Spark"]},
    {"name": "Chris Martinez", "org": "Snowflake", "role": "architect", "topics": ["data warehouse", "SQL"]},
    {"name": "Jennifer Liu", "org": "Vercel", "role": "developer advocate", "topics": ["Next.js", "frontend"]},
]

TOPICS = [
    "AI agents", "memory systems", "retrieval", "embeddings", "transformers",
    "deployment", "CI/CD", "testing", "debugging", "performance",
    "fundraising", "pitch deck", "investors", "valuation", "term sheet",
    "product roadmap", "user research", "A/B testing", "metrics", "growth",
    "hiring", "interviews", "culture", "remote work", "compensation",
    "customer feedback", "support tickets", "SLA", "onboarding", "churn",
    "security", "authentication", "encryption", "compliance", "GDPR",
    "database", "PostgreSQL", "Redis", "MongoDB", "migrations",
]

PROJECTS = [
    "Project Alpha", "Memory System", "Dashboard v2", "API Gateway",
    "Mobile App", "Analytics Platform", "Auth Service", "Payment Integration",
    "Search Engine", "Recommendation System", "Notification Service",
]

LOCATIONS = [
    "San Francisco", "New York", "Seattle", "Austin", "Boston",
    "London", "Berlin", "Singapore", "Tokyo", "Toronto",
]

# ============== Message Templates ==============

PERSONAL_ASSISTANT_TEMPLATES = {
    "task_create": [
        "Add a task to {action} by {date}",
        "Remind me to {action} {date}",
        "I need to {action}. Can you track that?",
        "Create a reminder for {action}",
        "Don't let me forget to {action}",
    ],
    "task_query": [
        "What tasks do I have for today?",
        "Do I have any deadlines this week?",
        "What's on my agenda?",
        "Show me my pending tasks",
        "What should I focus on?",
    ],
    "meeting": [
        "Schedule a meeting with {person} about {topic} for {date}",
        "I'm meeting {person} {date} to discuss {topic}",
        "Set up a call with {person}",
        "{person} wants to meet about {topic}",
        "Block time for {person} meeting on {date}",
    ],
    "info_share": [
        "{person} mentioned that {fact}",
        "FYI, {person} is {status}",
        "Just learned that {person} {action}",
        "{person} told me about {topic}",
        "Heads up: {person} said {fact}",
    ],
    "preference": [
        "I prefer {preference} when {context}",
        "Remember that I always {preference}",
        "I like to {preference}",
        "Note: I don't like {anti_preference}",
        "My usual approach is to {preference}",
    ],
    "question_about_memory": [
        "What do you know about {topic}?",
        "What did we discuss about {topic}?",
        "Tell me about {person}",
        "When did we last talk about {topic}?",
        "What's {person}'s background?",
    ],
}

TECHNICAL_SUPPORT_TEMPLATES = {
    "error_report": [
        "I'm getting an error: {error}",
        "The {component} is failing with {error}",
        "Help! {component} is broken",
        "Why am I seeing {error}?",
        "Can you debug this issue: {error}",
    ],
    "deployment": [
        "Deploy the {component} to {environment}",
        "Push the latest {component} changes",
        "Roll back {component} to previous version",
        "What's the status of {component} deployment?",
        "The {environment} deployment is stuck",
    ],
    "debugging": [
        "How do I fix {error}?",
        "What's causing {component} to fail?",
        "The logs show {error}. What does that mean?",
        "Is {component} compatible with {dependency}?",
        "Why is {component} so slow?",
    ],
    "config": [
        "Set {config_key} to {config_value}",
        "Update the {component} configuration",
        "What's the current value of {config_key}?",
        "The {config_key} setting seems wrong",
        "Reset {component} to defaults",
    ],
}

KNOWLEDGE_WORKER_TEMPLATES = {
    "research": [
        "What's the latest on {topic}?",
        "Find papers about {topic}",
        "Who are the key players in {topic}?",
        "Summarize the state of {topic}",
        "How does {topic} compare to {topic2}?",
    ],
    "writing": [
        "Help me write about {topic}",
        "Draft an email to {person} about {topic}",
        "Review this paragraph about {topic}",
        "How should I structure the {document} section?",
        "Make this more concise: {text}",
    ],
    "brainstorm": [
        "Ideas for {topic}?",
        "How might we improve {component}?",
        "What are the risks of {approach}?",
        "Pros and cons of {approach}?",
        "Alternative approaches to {problem}?",
    ],
    "analysis": [
        "Analyze the {metric} trends",
        "Why did {metric} change last week?",
        "Compare {metric} across {segments}",
        "What's driving {metric}?",
        "Forecast {metric} for next quarter",
    ],
}

# ============== Generators ==============

@dataclass
class GeneratedMessage:
    """A generated message with metadata"""
    content: str
    role: str
    session_id: str
    user_id: str = "benchmark_user"
    timestamp: Optional[datetime] = None
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    template_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "content": self.content,
            "role": self.role,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "entities": self.entities,
            "topics": self.topics,
            "template_type": self.template_type,
            "metadata": self.metadata,
        }


class ConversationGenerator:
    """Generates realistic conversation sessions"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.people = PEOPLE
        self.topics = TOPICS
        self.projects = PROJECTS

    def _random_person(self) -> Dict:
        return random.choice(self.people)

    def _random_topic(self) -> str:
        return random.choice(self.topics)

    def _random_date(self, relative: bool = True) -> str:
        if relative:
            options = ["today", "tomorrow", "Thursday", "next week", "Friday", "end of month"]
            return random.choice(options)
        return (datetime.now() + timedelta(days=random.randint(1, 30))).strftime("%B %d")

    def _fill_template(self, template: str, context: Dict) -> Tuple[str, List[str], List[str]]:
        """Fill template and extract entities/topics"""
        entities = []
        topics = []

        # Fill person
        if "{person}" in template:
            person = context.get("person") or self._random_person()
            template = template.replace("{person}", person["name"])
            entities.append(person["name"])
            entities.append(person["org"])
            topics.extend(person.get("topics", []))

        # Fill topic
        if "{topic}" in template:
            topic = context.get("topic") or self._random_topic()
            template = template.replace("{topic}", topic)
            topics.append(topic)

        if "{topic2}" in template:
            topic2 = context.get("topic2") or self._random_topic()
            template = template.replace("{topic2}", topic2)
            topics.append(topic2)

        # Fill date
        if "{date}" in template:
            date = context.get("date") or self._random_date()
            template = template.replace("{date}", date)
            if date not in ["today", "tomorrow", "next week"]:
                entities.append(date)

        # Fill action
        if "{action}" in template:
            actions = [
                "follow up with the team",
                "review the proposal",
                "send the report",
                "update the documentation",
                "prepare the presentation",
                "schedule the demo",
                "complete the code review",
            ]
            template = template.replace("{action}", random.choice(actions))

        # Fill other placeholders
        placeholders = {
            "{fact}": ["they're launching next month", "the project is ahead of schedule", "funding is confirmed"],
            "{status}": ["on vacation", "at a conference", "working remotely"],
            "{preference}": ["have meetings in the morning", "use bullet points", "keep emails brief"],
            "{context}": ["scheduling meetings", "writing reports", "reviewing code"],
            "{anti_preference}": ["long meetings", "verbose documentation", "unnecessary meetings"],
            "{error}": ["connection timeout", "authentication failed", "null pointer exception", "out of memory"],
            "{component}": random.choice(self.projects),
            "{environment}": random.choice(["production", "staging", "dev"]),
            "{dependency}": ["Node 18", "Python 3.11", "React 18"],
            "{config_key}": ["MAX_CONNECTIONS", "TIMEOUT_MS", "LOG_LEVEL"],
            "{config_value}": ["100", "5000", "DEBUG"],
            "{document}": random.choice(["introduction", "methodology", "results", "conclusion"]),
            "{text}": "the implementation details of the system",
            "{metric}": random.choice(["conversion rate", "latency", "user engagement", "revenue"]),
            "{segments}": random.choice(["regions", "user types", "devices"]),
            "{approach}": random.choice(["microservices", "monolith", "serverless"]),
            "{problem}": random.choice(["scaling", "latency", "cost"]),
        }

        for placeholder, options in placeholders.items():
            if placeholder in template:
                value = random.choice(options) if isinstance(options, list) else options
                template = template.replace(placeholder, str(value))

        return template, list(set(entities)), list(set(topics))

    def generate_session(
        self,
        domain: str,
        session_id: str,
        start_time: datetime,
        num_turns: int = 10,
    ) -> List[GeneratedMessage]:
        """Generate a coherent conversation session"""
        messages = []

        # Select templates based on domain
        if domain == "personal_assistant":
            templates = PERSONAL_ASSISTANT_TEMPLATES
        elif domain == "technical_support":
            templates = TECHNICAL_SUPPORT_TEMPLATES
        elif domain == "knowledge_worker":
            templates = KNOWLEDGE_WORKER_TEMPLATES
        else:
            templates = {**PERSONAL_ASSISTANT_TEMPLATES, **TECHNICAL_SUPPORT_TEMPLATES, **KNOWLEDGE_WORKER_TEMPLATES}

        # Select session focus (2-3 entities that appear throughout)
        session_people = random.sample(self.people, min(3, len(self.people)))
        session_topics = random.sample(self.topics, min(3, len(self.topics)))

        current_time = start_time

        for turn in range(num_turns):
            # User message
            template_type = random.choice(list(templates.keys()))
            template = random.choice(templates[template_type])

            context = {
                "person": random.choice(session_people) if random.random() > 0.3 else None,
                "topic": random.choice(session_topics) if random.random() > 0.3 else None,
            }

            content, entities, topics = self._fill_template(template, context)

            user_msg = GeneratedMessage(
                content=content,
                role="user",
                session_id=session_id,
                timestamp=current_time,
                entities=entities,
                topics=topics,
                template_type=template_type,
            )
            messages.append(user_msg)

            # Advance time (1-5 minutes between messages in a session)
            current_time += timedelta(minutes=random.randint(1, 5))

            # Assistant response
            response = self._generate_response(template_type, entities, topics)
            assistant_msg = GeneratedMessage(
                content=response,
                role="assistant",
                session_id=session_id,
                timestamp=current_time,
                entities=entities,
                topics=topics,
                template_type=f"{template_type}_response",
            )
            messages.append(assistant_msg)

            current_time += timedelta(minutes=random.randint(1, 3))

        return messages

    def _generate_response(
        self,
        template_type: str,
        entities: List[str],
        topics: List[str],
    ) -> str:
        """Generate a realistic assistant response"""
        responses = {
            "task_create": [
                "I've added that to your tasks.",
                "Done! I'll remind you.",
                "Task created. Anything else?",
                "Got it, added to your list.",
            ],
            "task_query": [
                f"You have 3 tasks pending. The most urgent is related to {topics[0] if topics else 'your project'}.",
                "Here are your tasks for today: 1) Review code 2) Team meeting 3) Update docs",
                "You have 2 deadlines this week.",
            ],
            "meeting": [
                f"Meeting scheduled with {entities[0] if entities else 'the team'}.",
                "I've blocked that time on your calendar.",
                "Done! I'll send a reminder beforehand.",
            ],
            "info_share": [
                "Thanks for letting me know. I'll remember that.",
                "Noted. That's useful context.",
                "Got it, I'll keep that in mind.",
            ],
            "preference": [
                "I'll remember your preference.",
                "Noted! I'll apply that going forward.",
                "Thanks, I'll keep that in mind.",
            ],
            "question_about_memory": [
                f"Based on our conversations, here's what I know about {topics[0] if topics else 'that topic'}...",
                f"You mentioned {entities[0] if entities else 'this'} several times. Let me summarize...",
                "Looking at our history, I found several relevant discussions.",
            ],
            "error_report": [
                "I see the issue. Let me help you debug.",
                "That error usually means... Let me check.",
                "I've seen this before. Try...",
            ],
            "deployment": [
                "Deployment initiated. I'll notify you when complete.",
                "The deployment is in progress.",
                "Rollback complete. Previous version restored.",
            ],
            "debugging": [
                "The issue is likely caused by... Here's how to fix it.",
                "Looking at the logs, I suggest...",
                "Try checking the configuration for...",
            ],
            "config": [
                "Configuration updated.",
                "Current value is: 1000",
                "Settings have been reset.",
            ],
            "research": [
                f"Here's what I found about {topics[0] if topics else 'that topic'}...",
                "The latest research suggests...",
                "Key papers on this include...",
            ],
            "writing": [
                "Here's a draft for you to review...",
                "I've restructured it as follows...",
                "Here's a more concise version...",
            ],
            "brainstorm": [
                "Here are some ideas: 1) ... 2) ... 3) ...",
                "The main risks are... The opportunities are...",
                "Pros: ... Cons: ...",
            ],
            "analysis": [
                "The trend shows... The main driver is...",
                "Comparing across segments, I see...",
                "Based on the data, I forecast...",
            ],
        }

        default_responses = [
            "I understand. How can I help further?",
            "Got it. Let me know if you need anything else.",
            "Thanks for the context. Anything else?",
        ]

        return random.choice(responses.get(template_type, default_responses))


def generate_dataset(
    name: str,
    domain: str,
    num_sessions: int,
    messages_per_session: int,
    start_date: Optional[datetime] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a complete dataset"""
    generator = ConversationGenerator(seed=seed)
    all_messages = []

    start_date = start_date or datetime.now() - timedelta(days=30)

    for s in range(num_sessions):
        session_id = f"{name}_session_{s}"

        # Sessions are spread across the date range
        session_start = start_date + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(8, 20),
        )

        messages = generator.generate_session(
            domain=domain,
            session_id=session_id,
            start_time=session_start,
            num_turns=messages_per_session // 2,  # Each turn is 2 messages
        )

        all_messages.extend(messages)

    # Sort by timestamp
    all_messages.sort(key=lambda m: m.timestamp or datetime.min)

    # Collect all entities and topics
    all_entities = set()
    all_topics = set()
    for msg in all_messages:
        all_entities.update(msg.entities)
        all_topics.update(msg.topics)

    return {
        "name": name,
        "domain": domain,
        "num_sessions": num_sessions,
        "total_messages": len(all_messages),
        "unique_entities": len(all_entities),
        "unique_topics": len(all_topics),
        "date_range": {
            "start": all_messages[0].timestamp.isoformat() if all_messages else None,
            "end": all_messages[-1].timestamp.isoformat() if all_messages else None,
        },
        "messages": [msg.to_dict() for msg in all_messages],
        "entity_list": list(all_entities),
        "topic_list": list(all_topics),
    }


def main():
    """Generate all datasets"""
    output_dir = Path(__file__).parent / "seed_conversations"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("personal_assistant", "personal_assistant", 50, 10),
        ("technical_support", "technical_support", 40, 12),
        ("knowledge_worker", "knowledge_worker", 45, 11),
        ("mixed_domain", "mixed", 80, 12),
    ]

    total_messages = 0
    total_entities = set()

    for name, domain, num_sessions, msgs_per_session in datasets:
        print(f"Generating {name}...")

        data = generate_dataset(
            name=name,
            domain=domain,
            num_sessions=num_sessions,
            messages_per_session=msgs_per_session,
            seed=42 + hash(name) % 1000,
        )

        output_path = output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  - {data['total_messages']} messages")
        print(f"  - {data['unique_entities']} unique entities")
        print(f"  - Saved to {output_path}")

        total_messages += data["total_messages"]
        total_entities.update(data["entity_list"])

    print(f"\nTotal: {total_messages} messages, {len(total_entities)} unique entities")


if __name__ == "__main__":
    main()
