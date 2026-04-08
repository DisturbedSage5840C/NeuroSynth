import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import App from "@/app/App";
import { queryClient } from "@/lib/queryClient";
import { useAuthStore } from "@/state/authStore";
import "./index.css";

// Seed demo auth so protected routes render in local sandbox environments.
if (!useAuthStore.getState().accessToken) {
  useAuthStore.getState().setTokens("demo-token", "demo-refresh", "clinician");
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </React.StrictMode>
);
