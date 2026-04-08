import { authStore } from "@/state/authStore";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export interface ApiError extends Error {
  status?: number;
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = authStore.getState().accessToken;
  const headers = new Headers(init.headers || {});
  headers.set("Content-Type", headers.get("Content-Type") || "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers, credentials: "include" });

  if (response.status === 401 && authStore.getState().refreshToken) {
    const refreshed = await refreshToken();
    if (refreshed) {
      return apiFetch<T>(path, init);
    }
  }

  if (!response.ok) {
    const err: ApiError = new Error(await response.text());
    err.status = response.status;
    throw err;
  }

  return response.json() as Promise<T>;
}

export async function login(email: string, password: string): Promise<{ access_token: string; refresh_token: string; role: string; }> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ username: email, password, role: "CLINICIAN" }),
  });
  if (!response.ok) throw new Error("Login failed");
  const payload = await response.json();
  const role = String(payload?.user?.role ?? "CLINICIAN").toLowerCase();
  // Backend auth is cookie-based; keep lightweight client tokens for route guards.
  return { access_token: "cookie-session", refresh_token: "cookie-session", role };
}

export async function refreshToken(): Promise<boolean> {
  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
  });

  if (!response.ok) return false;
  const payload = await response.json();
  const role = String(payload?.user?.role ?? authStore.getState().role ?? "CLINICIAN").toLowerCase();
  authStore.getState().setTokens("cookie-session", "cookie-session", role);
  return true;
}

export function streamUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}
