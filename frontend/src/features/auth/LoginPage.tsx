import { useState } from "react";
import { useNavigate } from "react-router";
import { login } from "@/lib/api";
import { useAuthStore } from "@/state/authStore";
import { Brain, Loader2 } from "lucide-react";

export function LoginPage(): JSX.Element {
  const navigate = useNavigate();
  const setTokens = useAuthStore((s) => s.setTokens);
  const [username, setUsername] = useState("clinician@neurosynth.local");
  const [password, setPassword] = useState("neurosynth");
  const [role, setRole] = useState("CLINICIAN");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const onSubmit = async () => {
    setLoading(true);
    setError("");
    try {
      const payload = await login(username, password, role);
      setTokens(payload.access_token, payload.refresh_token, payload.role);
      localStorage.setItem("ns_logged_in", "true");
      navigate("/");
    } catch {
      setError("Invalid credentials. Use: clinician@neurosynth.local / neurosynth");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-4">
      <div className="w-full max-w-sm space-y-5 rounded-xl border border-border bg-card p-6">

        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-primary/15 flex items-center justify-center">
            <Brain size={18} className="text-primary" />
          </div>
          <div>
            <div className="font-semibold text-foreground" style={{ fontSize: 16 }}>NeuroSynth</div>
            <div className="text-muted-foreground" style={{ fontSize: 11 }}>Clinical AI Platform</div>
          </div>
        </div>

        <div className="space-y-3">
          <div>
            <label className="block text-xs text-muted-foreground mb-1.5">Username</label>
            <input
              className="w-full rounded-lg border border-border bg-[var(--input-background)] px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary/30 transition-colors"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onSubmit()}
              autoComplete="username"
            />
          </div>

          <div>
            <label className="block text-xs text-muted-foreground mb-1.5">Password</label>
            <input
              type="password"
              className="w-full rounded-lg border border-border bg-[var(--input-background)] px-3 py-2 text-sm text-foreground focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary/30 transition-colors"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onSubmit()}
              autoComplete="current-password"
            />
          </div>

          <div>
            <label className="block text-xs text-muted-foreground mb-1.5">Role</label>
            <select
              className="w-full rounded-lg border border-border bg-[var(--input-background)] px-3 py-2 text-sm text-foreground focus:outline-none focus:border-primary transition-colors"
              value={role}
              onChange={(e) => setRole(e.target.value)}
            >
              <option value="CLINICIAN">Clinician</option>
              <option value="RESEARCHER">Researcher</option>
              <option value="ADMIN">Admin</option>
            </select>
          </div>
        </div>

        {error && (
          <p className="text-xs rounded-lg px-3 py-2" style={{ color: "var(--risk-critical)", background: "var(--risk-critical-bg)" }}>
            {error}
          </p>
        )}

        <button
          onClick={onSubmit}
          disabled={loading}
          className="w-full rounded-lg px-4 py-2.5 text-sm font-medium flex items-center justify-center gap-2 transition-opacity disabled:opacity-60"
          style={{ background: "var(--primary)", color: "var(--primary-foreground)" }}
        >
          {loading ? <><Loader2 size={14} className="animate-spin" /> Signing in...</> : "Sign in"}
        </button>

        <p className="text-center text-xs text-muted-foreground">
          Demo credentials pre-filled above
        </p>
      </div>
    </div>
  );
}
