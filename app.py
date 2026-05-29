import os
import re
import shutil
import tempfile
import uuid
from collections import Counter
from html import escape
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from templates import bot_template, css, user_template


load_dotenv()
GOOGLE_API_KEY = os.getenv("google_api_key")
CHROMA_BASE_PATH = Path(tempfile.gettempdir()) / "multi_pdf_chatbot_chroma"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

SMALL_TALK_RESPONSES = {
    "hi": "Hi! Upload and process your PDFs, then ask me anything about them.",
    "hello": "Hello! I can help you search and summarize your uploaded PDFs.",
    "hey": "Hey! Send me a question about your documents whenever you are ready.",
    "thanks": "You are welcome!",
    "thank you": "You are welcome!",
    "bye": "Goodbye! Come back whenever you want to explore another PDF.",
}


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def render_message(role, message):
    template = user_template if role == "user" else bot_template
    safe_message = escape(message).replace("\n", "<br>")
    st.write(template.replace("{{MSG}}", safe_message), unsafe_allow_html=True)


def add_message(role, message):
    st.session_state.chat_history.append({"role": role, "message": message})


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
    chroma_path = st.session_state.chroma_path
    collection_name = st.session_state.collection_name
    delete_chroma_collection(chroma_path=chroma_path, collection_name=collection_name)
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)

    return Chroma.from_documents(
        documents=text_chunks,
        embedding=get_embeddings(),
        client=client,
        collection_name=collection_name,
    )


def get_conversation_chain(vectorstore):
    return {"llm": get_llm(), "vectorstore": vectorstore}


def delete_chroma_collection(client=None, chroma_path=None, collection_name=None):
    collection_name = collection_name or st.session_state.get("collection_name")
    chroma_path = chroma_path or st.session_state.get("chroma_path")
    if not collection_name or not chroma_path:
        return

    client = client or chromadb.PersistentClient(path=chroma_path)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    shutil.rmtree(chroma_path, ignore_errors=True)


def format_sources(documents):
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


def keyword_overlap_score(question, document):
    question_terms = set(re.findall(r"\w+", question.lower()))
    document_terms = set(re.findall(r"\w+", document.page_content.lower()))
    if not question_terms:
        return 0
    return len(question_terms & document_terms)


def rerank_documents(question, documents, top_k=4):
    scored_docs = sorted(
        documents,
        key=lambda doc: keyword_overlap_score(question, doc),
        reverse=True,
    )
    return scored_docs[:top_k]


def build_prompt(user_question, relevant_docs):
    context_blocks = []
    for index, doc in enumerate(relevant_docs, start=1):
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "unknown")
        context_blocks.append(
            f"[Source {index}: {source}, page {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_blocks)
    history = "\n".join(
        f"{item['role'].title()}: {item['message']}"
        for item in st.session_state.chat_history[-6:]
    )

    return f"""
You are a helpful PDF question-answering assistant.
Answer the user's question using only the provided PDF context.
If the answer is not present in the context, say: "I don't know from the uploaded documents."
Include short source citations using the filename and page number when you use document facts.

Recent chat:
{history}

Question:
{user_question}

PDF context:
{context}
"""


def answer_from_documents(user_question):
    vectorstore = st.session_state.conversation["vectorstore"]
    candidate_docs = vectorstore.similarity_search(user_question, k=8)
    relevant_docs = rerank_documents(user_question, candidate_docs, top_k=4)

    prompt = build_prompt(user_question, relevant_docs)
    response = st.session_state.conversation["llm"].invoke(prompt)
    answer = response.content

    sources = format_sources(relevant_docs)
    if sources and "I don't know from the uploaded documents." not in answer:
        answer += "\n\nSources: " + "; ".join(sources)

    return answer


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

    with st.spinner("Searching your PDFs..."):
        answer = answer_from_documents(user_question)
    add_message("bot", answer)


def reset_documents():
    conversation = st.session_state.get("conversation")
    vectorstore = conversation.get("vectorstore") if conversation else None

    if vectorstore is not None and hasattr(vectorstore, "_client"):
        delete_chroma_collection(
            vectorstore._client,
            chroma_path=st.session_state.chroma_path,
            collection_name=st.session_state.collection_name,
        )
    else:
        delete_chroma_collection()

    st.session_state.conversation = None
    st.session_state.file_stats = []
    st.session_state.chunk_count = 0
    st.session_state.index_status = "No documents indexed"


def initialize_session_state():
    defaults = {
        "conversation": None,
        "chat_history": [],
        "file_stats": [],
        "chunk_count": 0,
        "index_status": "No documents indexed",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = f"pdf_collection_{st.session_state.session_id}"
    if "chroma_path" not in st.session_state:
        st.session_state.chroma_path = str(CHROMA_BASE_PATH / st.session_state.session_id)


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
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    initialize_session_state()

    st.header("Chat with multiple PDFs :books:")

    for item in st.session_state.chat_history:
        render_message(item["role"], item["message"])

    user_question = st.chat_input("Ask a question about your documents")
    if user_question:
        handle_userinput(user_question)
        st.rerun()

    with st.sidebar:
        st.subheader("Your documents")
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
