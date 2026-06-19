from typing import Any, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str
    rewritten_question: str
    vectorstore: Any
    llm: Any
    chat_history: list[dict[str, str]]
    conversation_memory: list[dict[str, str]]
    memory_context: str
    candidate_docs: list[Document]
    relevant_docs: list[Document]
    relevance_score: float
    web_results: list[dict[str, Any]]
    final_context: str
    context_sources: list[str]
    research_plan: str
    research_evidence: list[dict[str, Any]]
    verification_report: str
    confidence_score: float
    consistency_passed: bool
    research_iterations: int
    sources: list[str]
    answer: str

