"""Curated evaluation question set.

Deliberately small (see Phase 5 stack approval) — RAGAS-style metrics
fire multiple LLM calls per sample (claim decomposition, per-claim
verification, hypothetical question generation), and Ollama has
demonstrated real fragility under load during this project's own
development (see Phase 4's request-backlog incident). A large,
auto-generated eval set would be slow to run and harder to sanity-check
by eye than a small, deliberately curated one covering all six pillars.
"""

EVAL_QUESTIONS: list[str] = [
    "How do I set cost budgets and alerts?",
    "What are the design principles for cost optimization?",
    "How should I handle service quotas to improve reliability?",
    "What is the recommended approach for disaster recovery?",
    "How do I secure access to AWS resources using IAM?",
    "What are common anti-patterns in security best practices?",
    "How can I monitor operational health of a workload?",
    "What practices help automate operational procedures?",
    "How do I select the right compute resources for performance?",
    "What is the difference between vertical and horizontal scaling?",
    "How can I reduce the carbon footprint of my workload?",
    "What are the best practices for sustainable resource usage?",
]
