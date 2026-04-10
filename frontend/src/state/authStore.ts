import { create } from "zustand";

export type UserRole = "clinician" | "researcher" | "admin" | null;

export interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  role: UserRole;
  setTokens: (access: string, refresh: string, role: string) => void;
  clear: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  accessToken: null,
  refreshToken: null,
  role: null,
  setTokens: (access, refresh, role) =>
    set({
      accessToken: access,
      refreshToken: refresh,
      role: (role as UserRole) || "researcher",
    }),
  clear: () => {
    if (typeof window !== "undefined") {
      localStorage.removeItem("ns_logged_in");
    }
    set({ accessToken: null, refreshToken: null, role: null });
  },
}));

export const authStore = useAuthStore;
