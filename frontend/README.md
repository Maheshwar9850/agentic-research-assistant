# Agentic Research Assistant Frontend

Modern React frontend for the Multi-PDF Agentic Research Assistant.

## Run

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Backend API Contract

Set `VITE_API_BASE_URL` in `.env` if your API is not on `http://localhost:8000`.

Expected endpoints:

```text
POST /api/documents
Content-Type: multipart/form-data
Body: files[]
Response: { session_id, index_status, chunk_count, files }

POST /api/chat
Content-Type: application/json
Body: { session_id, message, research_mode }
Response: { answer, sources, confidence_score, mode }

GET /api/history?session_id=...
Response: { messages }

POST /api/reset
Content-Type: application/json
Body: { session_id }
Response: { ok: true }
```
