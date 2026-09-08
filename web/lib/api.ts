export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, { cache: "no-store", ...init });
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new Error("ARGUS returned an unreadable response. Check the local API service.");
  }
  if (!response.ok) {
    const detail = typeof data === "object" && data !== null && "detail" in data ? data.detail : null;
    throw new Error(typeof detail === "string" ? detail : `ARGUS could not complete this request (HTTP ${response.status}).`);
  }
  return data as T;
}

export function errorMessage(error: unknown): string {
  if (error instanceof TypeError) return "The connection to ARGUS was lost. Check the local services and try again.";
  return error instanceof Error ? error.message : "The request could not be completed.";
}

export function safeUrl(value?: string | null): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

export function displayDate(value?: string | null): string {
  if (!value) return "Not recorded";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
