import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  Brain,
  FileText,
  History,
  Loader2,
  Menu,
  MessageSquareText,
  PanelLeftClose,
  RotateCcw,
  Send,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import {
  fetchHistory,
  normalizeSources,
  resetSession,
  sendMessage,
  uploadDocuments,
} from "./api";
import type { ChatMessage, Source, UploadedFileInfo } from "./types";

function createMessage(
  role: ChatMessage["role"],
  content: string,
  extras: Partial<ChatMessage> = {},
): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    createdAt: new Date().toISOString(),
    ...extras,
  };
}

function formatTime(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function SourceList({ sources }: { sources: Source[] }) {
  if (!sources.length) {
    return null;
  }

  return (
    <div className="source-list" aria-label="Sources">
      <p>Sources</p>
      <div className="source-chips">
        {sources.map((source, index) =>
          source.url ? (
            <a
              className="source-chip"
              href={source.url}
              target="_blank"
              rel="noreferrer"
              key={`${source.label}-${index}`}
            >
              {source.type === "web" ? "Web" : "PDF"} · {source.label}
            </a>
          ) : (
            <span className="source-chip" key={`${source.label}-${index}`}>
              {source.type === "web" ? "Web" : "PDF"} · {source.label}
            </span>
          ),
        )}
      </div>
    </div>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <article className={`chat-bubble ${isUser ? "user" : "assistant"}`}>
      <div className="avatar" aria-hidden="true">
        {isUser ? <MessageSquareText size={18} /> : <Bot size={18} />}
      </div>
      <div className="bubble-body">
        <div className="bubble-meta">
          <span>{isUser ? "You" : "Assistant"}</span>
          {message.mode === "research" && (
            <span className="mode-pill">
              <Brain size={13} />
              Research
            </span>
          )}
          {typeof message.confidenceScore === "number" && (
            <span className="confidence-pill">
              <ShieldCheck size={13} />
              {Math.round(message.confidenceScore * 100)}%
            </span>
          )}
          <time>{formatTime(message.createdAt)}</time>
        </div>
        <div className="message-content">{message.content}</div>
        <SourceList sources={message.sources || []} />
      </div>
    </article>
  );
}

function UploadPanel({
  files,
  indexedFiles,
  chunkCount,
  indexStatus,
  isUploading,
  onFilesChange,
  onUpload,
}: {
  files: File[];
  indexedFiles: UploadedFileInfo[];
  chunkCount: number;
  indexStatus: string;
  isUploading: boolean;
  onFilesChange: (files: File[]) => void;
  onUpload: () => void;
}) {
  return (
    <section className="panel-section">
      <div className="section-heading">
        <Upload size={18} />
        <h2>PDF Upload</h2>
      </div>

      <label className="dropzone">
        <input
          type="file"
          accept="application/pdf"
          multiple
          onChange={(event) =>
            onFilesChange(Array.from(event.target.files || []))
          }
        />
        <FileText size={28} />
        <span>Choose PDFs</span>
        <small>{files.length ? `${files.length} selected` : "PDF only"}</small>
      </label>

      {!!files.length && (
        <div className="file-list">
          {files.map((file) => (
            <div className="file-row" key={`${file.name}-${file.size}`}>
              <FileText size={15} />
              <span>{file.name}</span>
            </div>
          ))}
        </div>
      )}

      <button
        className="primary-button"
        disabled={!files.length || isUploading}
        onClick={onUpload}
      >
        {isUploading ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
        Process PDFs
      </button>

      <div className="index-card">
        <span>{indexStatus}</span>
        <strong>{chunkCount} chunks</strong>
      </div>

      {!!indexedFiles.length && (
        <div className="indexed-list">
          {indexedFiles.map((file) => (
            <div className="indexed-file" key={file.name}>
              <span>{file.name}</span>
              {typeof file.extracted_pages === "number" &&
                typeof file.pages === "number" && (
                  <small>
                    {file.extracted_pages}/{file.pages} pages
                  </small>
                )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function HistoryPanel({
  messages,
  onJump,
}: {
  messages: ChatMessage[];
  onJump: (id: string) => void;
}) {
  const assistantMessages = messages.filter((message) => message.role === "assistant");

  return (
    <section className="panel-section">
      <div className="section-heading">
        <History size={18} />
        <h2>Chat History</h2>
      </div>

      <div className="history-list">
        {assistantMessages.length ? (
          assistantMessages.slice(-8).map((message) => (
            <button
              className="history-item"
              key={message.id}
              onClick={() => onJump(message.id)}
            >
              <span>{message.content}</span>
              <small>{formatTime(message.createdAt)}</small>
            </button>
          ))
        ) : (
          <p className="empty-note">No answers yet.</p>
        )}
      </div>
    </section>
  );
}

function ResearchToggle({
  enabled,
  onChange,
}: {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
}) {
  return (
    <button
      type="button"
      className={`research-toggle ${enabled ? "active" : ""}`}
      onClick={() => onChange(!enabled)}
      aria-pressed={enabled}
    >
      <Brain size={18} />
      <span>Research Mode</span>
      <span className="toggle-track">
        <span className="toggle-thumb" />
      </span>
    </button>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(
    localStorage.getItem("agentic-session-id"),
  );
  const [messages, setMessages] = useState<ChatMessage[]>([
    createMessage(
      "assistant",
      "Upload PDFs, then ask a question. Turn on Research Mode for deeper multi-source reports.",
      { sources: [] },
    ),
  ]);
  const [files, setFiles] = useState<File[]>([]);
  const [indexedFiles, setIndexedFiles] = useState<UploadedFileInfo[]>([]);
  const [chunkCount, setChunkCount] = useState(0);
  const [indexStatus, setIndexStatus] = useState("No documents indexed");
  const [researchMode, setResearchMode] = useState(false);
  const [input, setInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const canSend = input.trim().length > 0 && !isSending;

  const activeSources = useMemo(() => {
    return messages
      .flatMap((message) => message.sources || [])
      .filter((source, index, sources) => {
        return sources.findIndex((item) => item.label === source.label) === index;
      });
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    localStorage.setItem("agentic-session-id", sessionId);
    fetchHistory(sessionId)
      .then((history) => {
        if (history.length) {
          setMessages(history);
        }
      })
      .catch(() => {
        // History is optional; the active chat still works without it.
      });
  }, [sessionId]);

  async function handleUpload() {
    if (!files.length) {
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const response = await uploadDocuments(files);
      setSessionId(response.session_id);
      setIndexedFiles(response.files || []);
      setChunkCount(response.chunk_count || 0);
      setIndexStatus(response.index_status || "Documents indexed");
      setFiles([]);
      setMessages((current) => [
        ...current,
        createMessage(
          "system",
          `Indexed ${response.files?.length || 0} document(s).`,
        ),
      ]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleSend() {
    if (!canSend) {
      return;
    }

    const userText = input.trim();
    setInput("");
    setError(null);
    setIsSending(true);

    const userMessage = createMessage("user", userText, {
      mode: researchMode ? "research" : "standard",
    });
    setMessages((current) => [...current, userMessage]);

    try {
      const response = await sendMessage({
        sessionId,
        message: userText,
        researchMode,
      });
      const assistantMessage = createMessage("assistant", response.answer, {
        sources: normalizeSources(response.sources),
        confidenceScore: response.confidence_score,
        mode: response.mode || (researchMode ? "research" : "standard"),
      });
      setMessages((current) => [...current, assistantMessage]);
    } catch (caught) {
      const message =
        caught instanceof Error
          ? caught.message
          : "Could not reach the research backend.";
      setError(message);
      setMessages((current) => [
        ...current,
        createMessage("assistant", message, { sources: [] }),
      ]);
    } finally {
      setIsSending(false);
    }
  }

  async function handleReset() {
    setError(null);

    try {
      await resetSession(sessionId);
    } catch {
      // Reset local state even if the backend session has already expired.
    }

    localStorage.removeItem("agentic-session-id");
    setSessionId(null);
    setMessages([
      createMessage(
        "assistant",
        "Session reset. Upload PDFs to begin a new research workspace.",
      ),
    ]);
    setIndexedFiles([]);
    setChunkCount(0);
    setIndexStatus("No documents indexed");
  }

  function jumpToMessage(id: string) {
    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }

  return (
    <main className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="brand">
          <div className="brand-mark">
            <Brain size={22} />
          </div>
          <div>
            <h1>Agentic Research</h1>
            <p>PDF + web evidence assistant</p>
          </div>
        </div>

        <UploadPanel
          files={files}
          indexedFiles={indexedFiles}
          chunkCount={chunkCount}
          indexStatus={indexStatus}
          isUploading={isUploading}
          onFilesChange={setFiles}
          onUpload={handleUpload}
        />

        <HistoryPanel messages={messages} onJump={jumpToMessage} />

        {!!activeSources.length && (
          <section className="panel-section">
            <div className="section-heading">
              <ShieldCheck size={18} />
              <h2>Sources</h2>
            </div>
            <SourceList sources={activeSources} />
          </section>
        )}
      </aside>

      <section className="chat-area">
        <header className="topbar">
          <button
            className="icon-button"
            onClick={() => setSidebarOpen((value) => !value)}
            title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
          >
            {sidebarOpen ? <PanelLeftClose size={20} /> : <Menu size={20} />}
          </button>

          <div className="topbar-title">
            <h2>Research Workspace</h2>
            <p>{sessionId ? `Session ${sessionId}` : "No active session"}</p>
          </div>

          <div className="topbar-actions">
            <ResearchToggle enabled={researchMode} onChange={setResearchMode} />
            <button className="ghost-button" onClick={handleReset}>
              <RotateCcw size={17} />
              Reset
            </button>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            <span>{error}</span>
            <button onClick={() => setError(null)} title="Dismiss">
              <X size={16} />
            </button>
          </div>
        )}

        <div className="messages">
          {messages.map((message) => (
            <div id={message.id} key={message.id}>
              <ChatBubble message={message} />
            </div>
          ))}

          {isSending && (
            <article className="chat-bubble assistant">
              <div className="avatar" aria-hidden="true">
                <Bot size={18} />
              </div>
              <div className="bubble-body">
                <div className="typing-row">
                  <Loader2 className="spin" size={18} />
                  <span>
                    {researchMode
                      ? "Research agents are gathering evidence..."
                      : "Searching your workspace..."}
                  </span>
                </div>
              </div>
            </article>
          )}

          <div ref={bottomRef} />
        </div>

        <footer className="composer">
          <div className="composer-box">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={
                researchMode
                  ? "Ask for a deep research report..."
                  : "Ask a question about your PDFs..."
              }
              rows={1}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void handleSend();
                }
              }}
            />
            <button
              className="send-button"
              disabled={!canSend}
              onClick={() => void handleSend()}
              title="Send"
            >
              {isSending ? <Loader2 className="spin" size={19} /> : <Send size={19} />}
            </button>
          </div>
        </footer>
      </section>
    </main>
  );
}
