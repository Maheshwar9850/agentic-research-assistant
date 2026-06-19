from langgraph.graph import END, START, StateGraph

from nodes import (
    context_fusion,
    generate_answer,
    grade_retrieved_documents,
    initialize_memory,
    report_agent,
    retrieve_documents,
    research_agent,
    route_after_grading,
    route_after_verification,
    rewrite_query,
    update_memory,
    verification_agent,
    web_search,
)
from state import RAGState


def build_rag_graph():
    graph_builder = StateGraph(RAGState)

    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node("initialize_memory", initialize_memory)
    graph_builder.add_node("retrieve_documents", retrieve_documents)
    graph_builder.add_node("grade_retrieved_documents", grade_retrieved_documents)
    graph_builder.add_node("context_fusion", context_fusion)
    graph_builder.add_node("generate_answer", generate_answer)
    graph_builder.add_node("update_memory", update_memory)
    graph_builder.add_node("web_search", web_search)

    graph_builder.add_edge(START, "initialize_memory")
    graph_builder.add_edge("initialize_memory", "rewrite_query")
    graph_builder.add_edge("rewrite_query", "retrieve_documents")
    graph_builder.add_edge("retrieve_documents", "grade_retrieved_documents")
    graph_builder.add_conditional_edges(
        "grade_retrieved_documents",
        route_after_grading,
        {
            "context_fusion": "context_fusion",
            "web_search": "web_search",
        },
    )
    graph_builder.add_edge("web_search", "context_fusion")
    graph_builder.add_edge("context_fusion", "generate_answer")
    graph_builder.add_edge("generate_answer", "update_memory")
    graph_builder.add_edge("update_memory", END)

    return graph_builder.compile()


rag_graph = build_rag_graph()


def build_deep_research_graph():
    graph_builder = StateGraph(RAGState)

    graph_builder.add_node("initialize_memory", initialize_memory)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node("research_agent", research_agent)
    graph_builder.add_node("verification_agent", verification_agent)
    graph_builder.add_node("report_agent", report_agent)
    graph_builder.add_node("update_memory", update_memory)

    graph_builder.add_edge(START, "initialize_memory")
    graph_builder.add_edge("initialize_memory", "rewrite_query")
    graph_builder.add_edge("rewrite_query", "research_agent")
    graph_builder.add_edge("research_agent", "verification_agent")
    graph_builder.add_conditional_edges(
        "verification_agent",
        route_after_verification,
        {
            "research_agent": "research_agent",
            "report_agent": "report_agent",
        },
    )
    graph_builder.add_edge("report_agent", "update_memory")
    graph_builder.add_edge("update_memory", END)

    return graph_builder.compile()


deep_research_graph = build_deep_research_graph()


def build_initial_state(question, vectorstore, llm, chat_history, conversation_memory=None):
    return {
        "question": question,
        "rewritten_question": "",
        "vectorstore": vectorstore,
        "llm": llm,
        "chat_history": chat_history,
        "conversation_memory": conversation_memory or [],
        "memory_context": "",
        "candidate_docs": [],
        "relevant_docs": [],
        "relevance_score": 0.0,
        "web_results": [],
        "final_context": "",
        "context_sources": [],
        "research_plan": "",
        "research_evidence": [],
        "verification_report": "",
        "confidence_score": 0.0,
        "consistency_passed": False,
        "research_iterations": 0,
        "sources": [],
        "answer": "",
    }


def run_rag_graph(question, vectorstore, llm, chat_history, conversation_memory=None):
    initial_state = build_initial_state(
        question,
        vectorstore,
        llm,
        chat_history,
        conversation_memory,
    )
    return rag_graph.invoke(initial_state)


def run_deep_research_graph(
    question,
    vectorstore,
    llm,
    chat_history,
    conversation_memory=None,
):
    initial_state = build_initial_state(
        question,
        vectorstore,
        llm,
        chat_history,
        conversation_memory,
    )
    return deep_research_graph.invoke(initial_state)
