<div align="center">
  
# 🤖 Agentic Research Assistant

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-State_Machines-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4.svg)](https://deepmind.google/technologies/gemini/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_Fallback-F55036.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An autonomous, multi-agent AI research assistant built with LangGraph.** <br>
*Seamlessly fuses PDF vector retrieval, live web search, and cross-provider LLM failover.*

---

![Demo GIF Placeholder](docs/demo.gif)

</div>

## 📖 Overview

The **Agentic Research Assistant** is an advanced Retrieval-Augmented Generation (RAG) system. Unlike standard linear chatbots, this system utilizes **LangGraph** to construct a dynamic, multi-agent state machine. It evaluates its own retrieval quality (Corrective RAG), searches the live web via Tavily when local PDFs are insufficient, and features an autonomous **Deep Research Mode** that verifies its own evidence loops before generating a structured report.

To ensure 100% uptime, the system employs a custom **Duck-Typed Failover Wrapper** that gracefully intercepts API rate limits (429s) from Google Gemini and instantly reroutes queries to Groq's Llama-3.3 model without disrupting the LangGraph pipeline.

## ✨ Key Features

- **Agentic Routing (CRAG):** Evaluates retrieved document relevance; intelligently routes to web search if local data is insufficient.
- **Deep Research Mode:** A multi-step autonomous loop (Plan → Search → Verify → Report) that prevents AI hallucinations.
- **Cross-Provider Failover:** Zero-downtime architecture automatically intercepts Gemini quota exhaustion and fails over to Groq.
- **Hybrid Context Fusion:** Normalizes and deduplicates PDF chunks and web search results using SHA-256 fingerprinting.
- **Observability UI:** Streamlit dashboard featuring workflow traces, fallback metrics, indexed chunk counts, and source cards.
- **Conversational Memory:** Maintains a sliding-window chat history for contextual follow-up queries.

## 🏗️ Architecture

The system utilizes two compiled LangGraph state machines sharing a unified `TypedDict` state.

```text
USER QUERY
   │
   ▼
[ Initialize Memory ]
   │
   ▼
[ Rewrite Query ] ───────┐
   │                     │ (If Deep Research Mode)
   ▼                     ▼
[ Retrieve PDFs ]   [ Research Agent ] (Plans & retrieves Web + PDF)
   │                     │
   ▼                     ▼
[ Grade Relevance ] [ Verification Agent ] ─────────┐
   │                     │                          │
   ├─> (Relevance < 0.7) │ (Confidence < 0.7)       │
   │      ▼              └───────────LOOP───────────┘
   │  [ Web Search ]     │
   │      │              │ (Confidence ≥ 0.7)
   ▼      ▼              ▼
[ Context Fusion ]  [ Report Generation Agent ]
   │                     │
   ▼                     │
[ Generate Answer ]      │
   │                     │
   ▼                     ▼
[ Update Memory ] <──────┘
   │
   ▼
FINAL ANSWER
```

### 🔁 LLM Failover Architecture
Nodes interact purely with a `FailoverLLMWrapper` which intercepts calls:
```text
State["llm"].invoke(prompt)
   │
   ├─► Try: Gemini 2.5 Flash
   │
   └─► Except (429/Quota) ─► Update State Metrics ─► Fallback: Groq (Llama-3.3-70b)
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- API Keys for Google Gemini, Groq, and Tavily

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/agentic-research-assistant.git
cd agentic-research-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY="your_gemini_key"
GROQ_API_KEY="your_groq_key"
TAVILY_API_KEY="your_tavily_key"
```

### 4. Run the Application
```bash
python -m streamlit run app.py
```

## 📂 Project Structure
```text
├── app.py                # Streamlit UI, PDF ingestion, and state syncing
├── graph.py              # LangGraph compilation (RAG & Deep Research)
├── nodes.py              # Core logic: retrieval, grading, search, agents
├── llm_factory.py        # Duck-typed LLM failover wrapper implementation
├── state.py              # RAGState TypedDict schema
├── templates.py          # HTML/CSS UI components
├── requirements.txt      # Python dependencies
└── .env                  # Secrets configuration
```

## 🧠 Why This Project Stands Out

### Agentic Routing
Instead of a naive vector search, this system grades its own retrieval. If the PDF chunks don't contain the answer (score < 0.7), the system autonomously routes the execution path to query the live internet via Tavily.

### Deep Research & Verification
Deep Research mode doesn't just search—it plans. An orchestrator LLM writes a research plan, gathers evidence, and passes it to a Verification Agent. If the evidence is weak, the agent autonomously loops back to the search phase up to two times before generating a final report.

### Resilient Failover Architecture
Free-tier API limits often crash prototype apps. This architecture uses an interceptor pattern to catch `429 Resource Exhausted` errors from Gemini and instantly routes the exact same prompt to Groq. It ensures zero downtime and updates the Streamlit frontend with active failover metrics.

### Hybrid Knowledge Retrieval
Fusing PDF vectors and web search results often leads to context window pollution. This project normalizes text and uses `SHA-256` cryptographic hashing to deduplicate overlapping facts before injecting them into the prompt.

## 🔮 Future Improvements
- **Persistent Vector Storage:** Migrate from Ephemeral ChromaDB to a managed instance (e.g., Pinecone/Weaviate) to persist documents across restarts.
- **Asynchronous Queues:** Decouple LangGraph execution from Streamlit using Celery/Redis to prevent blocking the UI during heavy Deep Research loops.
- **Semantic Chunking:** Upgrade from recursive character splitting to NLP-based semantic chunking for higher retrieval accuracy.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
