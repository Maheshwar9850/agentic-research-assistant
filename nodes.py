import json
import os
import re
from hashlib import sha256
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.documents import Document

from state import RAGState


UNKNOWN_ANSWER = "I don't know from the uploaded documents."
RELEVANCE_THRESHOLD = 0.7
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 5
DEEP_RESEARCH_WEB_RESULTS = 8
DEEP_RESEARCH_PDF_RESULTS = 10
MAX_MEMORY_MESSAGES = 12
MIN_RESEARCH_CONFIDENCE = 0.7
MAX_RESEARCH_ITERATIONS = 2


def trim_memory(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return messages[-MAX_MEMORY_MESSAGES:]


def append_memory_message(
    messages: list[dict[str, str]],
    role: str,
    message: str,
) -> list[dict[str, str]]:
    if messages and messages[-1].get("role") == role and messages[-1].get(
        "message"
    ) == message:
        return trim_memory(messages)

    return trim_memory([*messages, {"role": role, "message": message}])


def format_memory_context(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "No previous conversation."

    return "\n".join(
        f"{item['role'].title()}: {item['message']}" for item in trim_memory(messages)
    )


def initialize_memory(state: RAGState) -> RAGState:
    memory = state["conversation_memory"] or state["chat_history"]
    memory = append_memory_message(memory, "user", state["question"])

    return {
        **state,
        "conversation_memory": memory,
        "memory_context": format_memory_context(memory),
    }


def build_query_rewrite_prompt(
    question: str,
    memory_context: str,
) -> str:
    return f"""
Rewrite the user's question into a concise search query for retrieving relevant PDF chunks.
Use the recent chat only to resolve references like "it", "that", or "the previous topic".
Do not answer the question.
Do not add facts that are not present in the user's wording or recent chat.
Return only the rewritten search query.

Recent chat:
{memory_context}

User question:
{question}
"""


def build_research_plan_prompt(question: str, memory_context: str) -> str:
    return f"""
Create a concise research plan for answering the user's question.
Break the work into focused search and evidence-gathering steps.
Use the recent chat only to resolve follow-up references.
Return the plan as short bullets.

Recent chat:
{memory_context}

Research question:
{question}
"""


def build_relevance_grader_prompt(
    question: str,
    rewritten_question: str,
    relevant_docs: list[Document],
) -> str:
    context_blocks = []
    for index, doc in enumerate(relevant_docs, start=1):
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "unknown")
        context_blocks.append(
            f"[Document {index}: {source}, page {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are grading whether retrieved PDF chunks are relevant to a user's question.
Return a single numeric score between 0 and 1.

Scoring guide:
0.0 means the retrieved text is unrelated.
0.5 means the text is partially related but likely insufficient.
1.0 means the text directly contains enough information to answer.

Do not explain your score.
Return only the number.

Original question:
{question}

Optimized retrieval query:
{rewritten_question}

Retrieved PDF chunks:
{context}
"""


def parse_relevance_score(raw_score: str) -> float:
    match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", raw_score)
    if not match:
        return 0.0

    score = float(match.group(0))
    return max(0.0, min(score, 1.0))


def format_sources(documents: list[Document]) -> list[str]:
    seen = set()
    sources = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "unknown")
        label = f"{source}, page {page}"
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return sources


def keyword_overlap_score(question: str, document: Document) -> int:
    question_terms = set(re.findall(r"\w+", question.lower()))
    document_terms = set(re.findall(r"\w+", document.page_content.lower()))
    if not question_terms:
        return 0
    return len(question_terms & document_terms)


def rerank_documents(
    question: str,
    documents: list[Document],
    top_k: int = 4,
) -> list[Document]:
    scored_docs = sorted(
        documents,
        key=lambda doc: keyword_overlap_score(question, doc),
        reverse=True,
    )
    return scored_docs[:top_k]


def rewrite_query(state: RAGState) -> RAGState:
    prompt = build_query_rewrite_prompt(state["question"], state["memory_context"])
    response = state["llm"].invoke(prompt)
    rewritten_question = response.content.strip()

    if not rewritten_question:
        rewritten_question = state["question"]

    return {
        **state,
        "rewritten_question": rewritten_question,
    }


def create_research_plan(state: RAGState) -> RAGState:
    prompt = build_research_plan_prompt(state["question"], state["memory_context"])
    response = state["llm"].invoke(prompt)
    research_plan = response.content.strip()

    return {
        **state,
        "research_plan": research_plan,
    }


def build_prompt(
    user_question: str,
    final_context: str,
    memory_context: str,
) -> str:
    return f"""
You are a helpful question-answering assistant.
Answer the user's question using only the provided fused context.
If the answer is not present in the context, say: "{UNKNOWN_ANSWER}"
Include short source citations when you use context facts.

Recent chat:
{memory_context}

Question:
{user_question}

Fused context:
{final_context}
"""


def retrieve_documents(state: RAGState) -> RAGState:
    retrieval_query = state["rewritten_question"] or state["question"]
    candidate_docs = state["vectorstore"].similarity_search(retrieval_query, k=8)
    relevant_docs = rerank_documents(retrieval_query, candidate_docs, top_k=4)

    return {
        **state,
        "candidate_docs": candidate_docs,
        "relevant_docs": relevant_docs,
        "sources": format_sources(relevant_docs),
    }


def retrieve_research_pdfs(state: RAGState) -> RAGState:
    retrieval_query = state["rewritten_question"] or state["question"]
    candidate_docs = state["vectorstore"].similarity_search(
        retrieval_query,
        k=DEEP_RESEARCH_PDF_RESULTS,
    )
    relevant_docs = rerank_documents(
        retrieval_query,
        candidate_docs,
        top_k=DEEP_RESEARCH_PDF_RESULTS,
    )

    return {
        **state,
        "candidate_docs": candidate_docs,
        "relevant_docs": relevant_docs,
        "sources": format_sources(relevant_docs),
    }


def grade_retrieved_documents(state: RAGState) -> RAGState:
    if not state["relevant_docs"]:
        return {
            **state,
            "relevance_score": 0.0,
        }

    prompt = build_relevance_grader_prompt(
        state["question"],
        state["rewritten_question"],
        state["relevant_docs"],
    )
    response = state["llm"].invoke(prompt)
    relevance_score = parse_relevance_score(response.content)

    return {
        **state,
        "relevance_score": relevance_score,
    }


def route_after_grading(state: RAGState) -> str:
    if state["relevance_score"] > RELEVANCE_THRESHOLD:
        return "context_fusion"

    return "web_search"


def generate_answer(state: RAGState) -> RAGState:
    prompt = build_prompt(
        state["question"],
        state["final_context"],
        state["memory_context"],
    )
    response = state["llm"].invoke(prompt)
    answer = format_answer_with_sources(response.content, state["context_sources"])

    return {
        **state,
        "answer": answer,
    }


def update_memory(state: RAGState) -> RAGState:
    memory = append_memory_message(
        state["conversation_memory"],
        "assistant",
        state["answer"],
    )

    return {
        **state,
        "conversation_memory": memory,
        "memory_context": format_memory_context(memory),
    }


def get_tavily_api_key() -> str | None:
    return os.getenv("TAVILY_API_KEY") or os.getenv("tavily_api_key")


def search_tavily(query: str, max_results: int = TAVILY_MAX_RESULTS) -> list[dict]:
    api_key = get_tavily_api_key()
    if not api_key:
        raise ValueError(
            "Missing Tavily API key. Set TAVILY_API_KEY in your environment."
        )

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": max_results,
    }
    request = Request(
        TAVILY_SEARCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=20) as response:
        response_body = response.read().decode("utf-8")

    data = json.loads(response_body)
    return data.get("results", [])


def normalize_tavily_results(results: list[dict]) -> list[dict]:
    normalized_results = []

    for result in results:
        normalized_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": float(result.get("score", 0.0) or 0.0),
            }
        )

    return normalized_results


def search_web_for_research(state: RAGState) -> RAGState:
    retrieval_query = state["rewritten_question"] or state["question"]

    try:
        raw_results = search_tavily(
            retrieval_query,
            max_results=DEEP_RESEARCH_WEB_RESULTS,
        )
        web_results = normalize_tavily_results(raw_results)
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        web_results = []

    return {
        **state,
        "web_results": web_results,
    }


def format_web_results(results: list[dict]) -> str:
    if not results:
        return "No relevant web results were returned."

    lines = ["Top web results:"]
    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled result"
        url = result.get("url") or "No URL"
        content = result.get("content") or "No summary available."
        score = result.get("score", 0.0)
        lines.append(
            f"{index}. {title}\n"
            f"   URL: {url}\n"
            f"   Relevance: {score:.2f}\n"
            f"   Summary: {content}"
        )

    return "\n\n".join(lines)


def normalize_for_dedup(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def content_fingerprint(value: str) -> str:
    normalized = normalize_for_dedup(value)
    return sha256(normalized.encode("utf-8")).hexdigest()


def build_pdf_context_items(documents: list[Document]) -> list[dict]:
    context_items = []

    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "unknown")
        chunk_index = doc.metadata.get("chunk_index", index)
        source_label = f"{source}, page {page}"

        context_items.append(
            {
                "type": "pdf",
                "dedup_key": content_fingerprint(doc.page_content),
                "source": source_label,
                "content": (
                    f"[PDF {index}: {source_label}, chunk {chunk_index}]\n"
                    f"{doc.page_content}"
                ),
            }
        )

    return context_items


def build_web_context_items(results: list[dict]) -> list[dict]:
    context_items = []

    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled result"
        url = result.get("url") or ""
        content = result.get("content") or ""
        score = result.get("score", 0.0)
        dedup_source = url or f"{title} {content}"

        context_items.append(
            {
                "type": "web",
                "dedup_key": content_fingerprint(dedup_source),
                "source": f"{title} ({url})" if url else title,
                "content": (
                    f"[Web {index}: {title}]\n"
                    f"URL: {url or 'No URL'}\n"
                    f"Relevance: {score:.2f}\n"
                    f"Summary: {content or 'No summary available.'}"
                ),
            }
        )

    return context_items


def deduplicate_sources(sources: list[str]) -> list[str]:
    seen = set()
    deduplicated_sources = []

    for source in sources:
        normalized = normalize_for_dedup(source)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        deduplicated_sources.append(source)

    return deduplicated_sources


def remove_existing_sources_section(answer: str) -> str:
    return re.split(r"\n\s*Sources\s*:\s*", answer, maxsplit=1, flags=re.IGNORECASE)[
        0
    ].strip()


def format_sources_section(sources: list[str]) -> str:
    deduplicated_sources = deduplicate_sources(sources)
    if not deduplicated_sources:
        deduplicated_sources = ["No sources available"]

    source_lines = "\n".join(f"* {source}" for source in deduplicated_sources)
    return f"Sources:\n\n{source_lines}"


def format_answer_with_sources(answer: str, sources: list[str]) -> str:
    clean_answer = remove_existing_sources_section(answer)
    if not clean_answer:
        clean_answer = UNKNOWN_ANSWER

    return f"{clean_answer}\n\n{format_sources_section(sources)}"


def context_fusion(state: RAGState) -> RAGState:
    context_items = [
        *build_pdf_context_items(state["relevant_docs"]),
        *build_web_context_items(state["web_results"]),
    ]

    seen_keys = set()
    deduplicated_items = []
    context_sources = []

    for item in context_items:
        if item["dedup_key"] in seen_keys:
            continue

        seen_keys.add(item["dedup_key"])
        deduplicated_items.append(item["content"])
        context_sources.append(item["source"])

    final_context = "\n\n".join(deduplicated_items)
    if not final_context:
        final_context = "No relevant PDF chunks or web search results were found."

    return {
        **state,
        "final_context": final_context,
        "context_sources": deduplicate_sources(context_sources),
    }


def gather_evidence(state: RAGState) -> RAGState:
    context_items = [
        *build_pdf_context_items(state["relevant_docs"]),
        *build_web_context_items(state["web_results"]),
    ]

    seen_keys = set()
    research_evidence = []
    context_blocks = []
    context_sources = []

    for item in context_items:
        if item["dedup_key"] in seen_keys:
            continue

        seen_keys.add(item["dedup_key"])
        research_evidence.append(
            {
                "type": item["type"],
                "source": item["source"],
                "content": item["content"],
            }
        )
        context_blocks.append(item["content"])
        context_sources.append(item["source"])

    final_context = "\n\n".join(context_blocks)
    if not final_context:
        final_context = "No relevant PDF chunks or web search results were found."

    return {
        **state,
        "research_evidence": research_evidence,
        "final_context": final_context,
        "context_sources": deduplicate_sources(context_sources),
    }


def build_research_report_prompt(
    question: str,
    research_plan: str,
    verification_report: str,
    confidence_score: float,
    final_context: str,
    memory_context: str,
) -> str:
    return f"""
You are a deep research assistant.
Generate a well-structured research report using only the evidence in the final context.
Do not invent facts. If evidence is missing, state the limitation clearly.

Your report must use exactly these sections:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Sources

Recent chat:
{memory_context}

Research question:
{question}

Research plan:
{research_plan}

Verification confidence:
{confidence_score:.2f}

Verification notes:
{verification_report}

Final evidence context:
{final_context}
"""


def remove_existing_numbered_sources_section(report: str) -> str:
    return re.split(
        r"\n\s*(?:4\.\s*)?Sources\s*:?\s*",
        report,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


def format_numbered_research_sources(sources: list[str]) -> str:
    deduplicated_sources = deduplicate_sources(sources)
    if not deduplicated_sources:
        deduplicated_sources = ["No sources available"]

    source_lines = "\n".join(f"* {source}" for source in deduplicated_sources)
    return f"4. Sources\n\n{source_lines}"


def format_research_report_with_sources(report: str, sources: list[str]) -> str:
    clean_report = remove_existing_numbered_sources_section(report)
    if not clean_report:
        clean_report = (
            "1. Executive Summary\n\n"
            "I could not generate a report from the available evidence.\n\n"
            "2. Key Findings\n\n"
            "* No supported findings were available.\n\n"
            "3. Detailed Analysis\n\n"
            "The available context did not contain enough information."
        )

    return f"{clean_report}\n\n{format_numbered_research_sources(sources)}"


def generate_research_report(state: RAGState) -> RAGState:
    prompt = build_research_report_prompt(
        state["question"],
        state["research_plan"],
        state["verification_report"],
        state["confidence_score"],
        state["final_context"],
        state["memory_context"],
    )
    response = state["llm"].invoke(prompt)
    report = format_research_report_with_sources(
        response.content,
        state["context_sources"],
    )

    return {
        **state,
        "answer": report,
    }


def research_agent(state: RAGState) -> RAGState:
    next_state = state

    if not next_state["research_plan"]:
        next_state = create_research_plan(next_state)

    next_state = retrieve_research_pdfs(next_state)
    next_state = search_web_for_research(next_state)
    next_state = gather_evidence(next_state)

    return {
        **next_state,
        "research_iterations": next_state["research_iterations"] + 1,
    }


def build_verification_prompt(
    question: str,
    research_plan: str,
    final_context: str,
    context_sources: list[str],
) -> str:
    sources = "\n".join(f"* {source}" for source in context_sources)

    return f"""
You are a verification agent.
Check whether the gathered evidence is consistent, sufficient, and relevant to the research question.
Return JSON only with these keys:
- confidence_score: number between 0 and 1
- consistency_passed: boolean
- verification_report: short explanation of strengths, conflicts, and gaps

Research question:
{question}

Research plan:
{research_plan}

Sources:
{sources}

Gathered evidence:
{final_context}
"""


def parse_verification_response(raw_response: str) -> dict:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        if not json_match:
            score = parse_relevance_score(raw_response)
            return {
                "confidence_score": score,
                "consistency_passed": score >= MIN_RESEARCH_CONFIDENCE,
                "verification_report": raw_response.strip(),
            }
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            score = parse_relevance_score(raw_response)
            return {
                "confidence_score": score,
                "consistency_passed": score >= MIN_RESEARCH_CONFIDENCE,
                "verification_report": raw_response.strip(),
            }

    confidence_score = float(data.get("confidence_score", 0.0) or 0.0)
    confidence_score = max(0.0, min(confidence_score, 1.0))

    return {
        "confidence_score": confidence_score,
        "consistency_passed": bool(data.get("consistency_passed", False)),
        "verification_report": data.get("verification_report", "").strip(),
    }


def verification_agent(state: RAGState) -> RAGState:
    if not state["research_evidence"]:
        return {
            **state,
            "confidence_score": 0.0,
            "consistency_passed": False,
            "verification_report": "No evidence was gathered for verification.",
        }

    prompt = build_verification_prompt(
        state["question"],
        state["research_plan"],
        state["final_context"],
        state["context_sources"],
    )
    response = state["llm"].invoke(prompt)
    verification = parse_verification_response(response.content)

    return {
        **state,
        "confidence_score": verification["confidence_score"],
        "consistency_passed": verification["consistency_passed"],
        "verification_report": verification["verification_report"],
    }


def route_after_verification(state: RAGState) -> str:
    if state["confidence_score"] >= MIN_RESEARCH_CONFIDENCE and state[
        "consistency_passed"
    ]:
        return "report_agent"

    if state["research_iterations"] < MAX_RESEARCH_ITERATIONS:
        return "research_agent"

    return "report_agent"


def report_agent(state: RAGState) -> RAGState:
    return generate_research_report(state)


def web_search(state: RAGState) -> RAGState:
    retrieval_query = state["rewritten_question"] or state["question"]

    try:
        raw_results = search_tavily(retrieval_query)
        web_results = normalize_tavily_results(raw_results)
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {
            **state,
            "web_results": [],
            "answer": (
                f"Retrieved PDF relevance score: {state['relevance_score']:.2f}.\n\n"
                "The retrieved PDF context was not relevant enough, so I routed this "
                f"to Tavily web search. Tavily search is not available: {exc}"
            ),
        }

    return {
        **state,
        "web_results": web_results,
    }
