import type {
  ChatMessage,
  ChatResponse,
  HistoryResponse,
  Source,
  UploadResponse,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function normalizeSources(sources?: Array<string | Source>): Source[] {
  if (!sources) {
    return [];
  }

  return sources.map((source) => {
    if (typeof source !== "string") {
      return source;
    }

    const urlMatch = source.match(/https?:\/\/[^\s)]+/);
    return {
      label: source,
      url: urlMatch?.[0],
      type: urlMatch ? "web" : "pdf",
    };
  });
}

export async function uploadDocuments(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE_URL}/api/documents`, {
    method: "POST",
    body: formData,
  });

  return parseResponse<UploadResponse>(response);
}

export async function sendMessage(params: {
  sessionId: string | null;
  message: string;
  researchMode: boolean;
}): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: params.sessionId,
      message: params.message,
      research_mode: params.researchMode,
    }),
  });

  return parseResponse<ChatResponse>(response);
}

export async function fetchHistory(
  sessionId: string,
): Promise<ChatMessage[]> {
  const url = new URL(`${API_BASE_URL}/api/history`);
  url.searchParams.set("session_id", sessionId);

  const response = await fetch(url);
  const payload = await parseResponse<HistoryResponse>(response);
  return payload.messages;
}

export async function resetSession(sessionId: string | null): Promise<void> {
  if (!sessionId) {
    return;
  }

  const response = await fetch(`${API_BASE_URL}/api/reset`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ session_id: sessionId }),
  });

  await parseResponse<{ ok: boolean }>(response);
}
