# Multi-PDF Chatbot

A Streamlit RAG app for asking grounded questions across multiple PDF files.

## What it does

- Upload and process multiple PDFs.
- Split extracted text with recursive chunking and overlap.
- Store chunk metadata including filename, page number, and chunk index.
- Retrieve candidates with HuggingFace embeddings and ChromaDB.
- Rerank retrieved chunks with a lightweight keyword-overlap pass.
- Answer with Gemini using strict document-grounding instructions.
- Show source citations for document-based answers.
- Handle greetings like `hi`, `hello`, and `thanks` without requiring PDFs.
- Show indexing status, uploaded files, page counts, and chunk count.

## Resume-safe bullets

- Built a RAG-based multi-PDF QA chatbot using LangChain, ChromaDB, HuggingFace embeddings, Gemini, and Streamlit.
- Improved retrieval relevance using recursive chunking with overlap and source-grounded context selection.
- Added metadata-aware document retrieval with page-level citations to reduce unsupported answers.
- Implemented conversational fallbacks for greetings, empty uploads, and out-of-context questions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file with:

```bash
google_api_key=your_google_api_key
```

## Optional benchmark

Edit `benchmark.py` and replace `EVAL_QUESTIONS` with questions from your own PDFs plus the expected source filename/page.

```bash
python benchmark.py
```
