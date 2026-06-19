export type ChatRole = "user" | "assistant" | "system";

export type Source = {
  label: string;
  url?: string;
  type?: "pdf" | "web" | "unknown";
};

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  sources?: Source[];
  confidenceScore?: number;
  mode?: "standard" | "research";
  createdAt: string;
};

export type UploadedFileInfo = {
  name: string;
  pages?: number;
  extracted_pages?: number;
};

export type UploadResponse = {
  session_id: string;
  index_status: string;
  chunk_count: number;
  files: UploadedFileInfo[];
};

export type ChatResponse = {
  answer: string;
  sources?: Array<string | Source>;
  confidence_score?: number;
  mode?: "standard" | "research";
};

export type HistoryResponse = {
  messages: ChatMessage[];
};
