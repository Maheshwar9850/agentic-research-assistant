import os
import re
import uuid
from collections import Counter
from html import escape

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from graph import run_deep_research_graph, run_rag_graph
from llm_factory import get_llm
from templates import bot_template, css, user_template


load_dotenv()
GOOGLE_API_KEY = os.getenv("google_api_key")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

SMALL_TALK_RESPONSES = {
    "hi": "Hi! Upload and process your PDFs, then ask me anything about them.",
    "hello": "Hello! I can help you search and summarize your uploaded PDFs.",
    "hey": "Hey! Send me a question about your documents whenever you are ready.",
    "thanks": "You are welcome!",
    "thank you": "You are welcome!",
    "bye": "Goodbye! Come back whenever you want to explore another PDF.",
}


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def render_message(role, message, trace_data=None, sources=None):
    template = user_template if role == "user" else bot_template
    safe_message = escape(message).replace("\n", "<br>")
    st.write(template.replace("{{MSG}}", safe_message), unsafe_allow_html=True)
    
    if role == "bot":
        if sources and isinstance(sources, list) and len(sources) > 0 and sources[0] != "No sources available":
            st.markdown("**Sources:**")
            cols = st.columns(min(len(sources), 3))
            for idx, source in enumerate(sources):
                with cols[idx % 3]:
                    st.info(source, icon="📄")
                    
        if trace_data:
            with st.expander("Workflow Trace", expanded=False):
                st.markdown(f"**Rewritten Query:** `{trace_data.get('rewritten_question')}`")
                st.markdown(f"**Documents Retrieved:** `{trace_data.get('relevant_docs_count')}`")
                st.markdown(f"**Relevance Score:** `{trace_data.get('relevance_score')}`")
                st.markdown(f"**Web Search Triggered:** `{'Yes' if trace_data.get('web_results_count') > 0 else 'No'}`")


def add_message(role, message, trace_data=None, sources=None):
    st.session_state.chat_history.append({
        "role": role, 
        "message": message,
        "trace_data": trace_data,
        "sources": sources
    })


def is_small_talk(user_question):
    normalized = re.sub(r"[^a-zA-Z\s]", "", user_question).strip().lower()
    small_talk_phrases = set(SMALL_TALK_RESPONSES)
    return normalized in small_talk_phrases or any(
        normalized.startswith(f"{phrase} ") for phrase in small_talk_phrases
    )


def get_small_talk_response(user_question):
    normalized = re.sub(r"[^a-zA-Z\s]", "", user_question).strip().lower()
    for phrase, response in SMALL_TALK_RESPONSES.items():
        if normalized == phrase or normalized.startswith(f"{phrase} "):
            return response

    return SMALL_TALK_RESPONSES.get(
        normalized,
        "Hi! Upload and process your PDFs, then ask a question about them.",
    )


def get_pdf_documents(pdf_docs):
    documents = []
    file_stats = []

    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        file_name = pdf.name
        extracted_pages = 0

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if not page_text:
                continue

            extracted_pages += 1
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": file_name,
                        "page": page_number,
                    },
                )
            )

        file_stats.append(
            {
                "name": file_name,
                "pages": len(reader.pages),
                "extracted_pages": extracted_pages,
            }
        )

    return documents, file_stats


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    chunk_counts = Counter()
    for chunk in chunks:
        key = (chunk.metadata.get("source"), chunk.metadata.get("page"))
        chunk_counts[key] += 1
        chunk.metadata["chunk_index"] = chunk_counts[key]

    return chunks


def get_vectorstore(text_chunks):
    collection_name = st.session_state.collection_name
    client = chromadb.EphemeralClient()

    return Chroma.from_documents(
        documents=text_chunks,
        embedding=get_embeddings(),
        client=client,
        collection_name=collection_name,
    )


def get_conversation_chain(vectorstore):
    return {"llm": get_llm(), "vectorstore": vectorstore}


def delete_chroma_collection(client=None, collection_name=None):
    collection_name = collection_name or st.session_state.get("collection_name")
    if not collection_name:
        return

    if client is None:
        return

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


def answer_from_documents(user_question):
    vectorstore = st.session_state.conversation["vectorstore"]
    llm = st.session_state.conversation["llm"]
    
    is_deep = st.session_state.get("deep_research_mode", False)
    graph_runner = run_deep_research_graph if is_deep else run_rag_graph

    final_state = graph_runner(
        question=user_question,
        vectorstore=vectorstore,
        llm=llm,
        chat_history=st.session_state.chat_history,
        conversation_memory=st.session_state.graph_memory,
    )
    st.session_state.graph_memory = final_state["conversation_memory"]
    return final_state


def handle_userinput(user_question):
    add_message("user", user_question)

    if is_small_talk(user_question):
        add_message("bot", get_small_talk_response(user_question))
        return

    if st.session_state.conversation is None:
        add_message(
            "bot",
            "Please upload and process your PDFs first, then I can answer questions from them.",
        )
        return

    is_deep = st.session_state.get("deep_research_mode", False)
    with st.spinner("Deep Researching..." if is_deep else "Searching your PDFs..."):
        final_state = answer_from_documents(user_question)
        
    raw_answer = final_state.get("answer", "")
    
    import re
    clean_answer = re.split(r"\n\s*(?:4\.\s*)?Sources\s*:?\s*", raw_answer, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    trace_data = {
        "rewritten_question": final_state.get("rewritten_question", ""),
        "relevant_docs_count": len(final_state.get("relevant_docs", [])),
        "relevance_score": final_state.get("relevance_score", 0.0),
        "web_results_count": len(final_state.get("web_results", [])),
    }
    
    st.session_state.last_relevance_score = trace_data["relevance_score"]
    if trace_data["web_results_count"] > 0:
        st.session_state.web_search_count += 1
        
    st.session_state.current_provider = final_state["llm"].current_provider
    st.session_state.fallback_count = final_state["llm"].fallback_count
        
    add_message("bot", clean_answer, trace_data=trace_data, sources=final_state.get("sources", []))


def reset_documents():
    conversation = st.session_state.get("conversation")
    vectorstore = conversation.get("vectorstore") if conversation else None

    if vectorstore is not None and hasattr(vectorstore, "_client"):
        delete_chroma_collection(
            vectorstore._client,
            collection_name=st.session_state.collection_name,
        )
    else:
        delete_chroma_collection()

    st.session_state.conversation = None
    st.session_state.graph_memory = []
    st.session_state.file_stats = []
    st.session_state.chunk_count = 0
    st.session_state.index_status = "No documents indexed"


def initialize_session_state():
    defaults = {
        "conversation": None,
        "chat_history": [],
        "graph_memory": [],
        "file_stats": [],
        "chunk_count": 0,
        "index_status": "No documents indexed",
        "last_relevance_score": 0.0,
        "web_search_count": 0,
        "deep_research_mode": False,
        "current_provider": "Gemini",
        "fallback_count": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []
    if not isinstance(st.session_state.graph_memory, list):
        st.session_state.graph_memory = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = f"pdf_collection_{st.session_state.session_id}"


def render_document_status():
    st.caption(st.session_state.index_status)

    if st.session_state.file_stats:
        st.write("Indexed files")
        for file_info in st.session_state.file_stats:
            st.write(
                f"- {file_info['name']} "
                f"({file_info['extracted_pages']}/{file_info['pages']} pages with text)"
            )
        st.metric("Chunks indexed", st.session_state.chunk_count)


def main():
    st.set_page_config(page_title="Agentic Research Assistant", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    initialize_session_state()

    st.header("Agentic Research Assistant :books:")

    for item in st.session_state.chat_history:
        render_message(item["role"], item["message"], item.get("trace_data"), item.get("sources"))

    user_question = st.chat_input("Ask a question about your documents")
    if user_question:
        handle_userinput(user_question)
        st.rerun()

    with st.sidebar:
        st.title("Agent Status")
        
        st.toggle("Deep Research Mode", key="deep_research_mode")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🤖 Current LLM", st.session_state.current_provider)
            st.metric("Memory Entries", len(st.session_state.graph_memory))
            st.metric("Docs Indexed", len(st.session_state.file_stats))
            st.metric("Web Searches", st.session_state.web_search_count)
        with col2:
            st.metric("🔁 Fallbacks", st.session_state.fallback_count)
            st.metric("Current Mode", "Deep Research" if st.session_state.deep_research_mode else "Chat")
            st.metric("Chunks Indexed", st.session_state.chunk_count)
            st.metric("Last Relevance", f"{st.session_state.last_relevance_score:.2f}")

        st.divider()

        st.subheader("Document Management")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Upload at least one PDF first.")
            else:
                with st.spinner("Processing PDFs..."):
                    documents, file_stats = get_pdf_documents(pdf_docs)
                    text_chunks = get_text_chunks(documents)
                    if not text_chunks:
                        st.warning("No readable text was found in the uploaded PDFs.")
                        return

                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.file_stats = file_stats
                    st.session_state.chunk_count = len(text_chunks)
                    st.session_state.index_status = "Documents indexed and ready"
                st.success("Documents processed successfully.")

        if st.button("Clear documents / reset index"):
            reset_documents()
            st.success("Document index cleared.")

        render_document_status()


if __name__ == "__main__":
    main()
