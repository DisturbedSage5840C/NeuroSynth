import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { login } from "@/lib/api";
import { useAuthStore } from "@/state/authStore";
import { Loader2 } from "lucide-react";
import { Link } from "react-router";

import "./auth-experience.css";

const ROLES = ["CLINICIAN", "RESEARCHER", "ADMIN"] as const;

export function LoginPage(): JSX.Element {
  const navigate = useNavigate();
  const setTokens = useAuthStore((s) => s.setTokens);
  const hasRole = useAuthStore((s) => s.role);
  const [username, setUsername] = useState("clinician@neurosynth.local");
  const [password, setPassword] = useState("neurosynth");
  const [role, setRole] = useState<(typeof ROLES)[number]>("CLINICIAN");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (hasRole || localStorage.getItem("ns_logged_in") === "true") {
      navigate("/app", { replace: true });
    }
  }, [hasRole, navigate]);

  const onSubmit = async () => {
    setLoading(true);
    setError("");
    try {
      const payload = await login(username, password, role);
      setTokens(payload.access_token, payload.refresh_token, payload.role);
      localStorage.setItem("ns_logged_in", "true");
      navigate("/app", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message || "Login failed" : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ns-terminal min-h-screen flex items-center justify-center px-4 py-10">
      <div className="ns-terminal-scanlines" aria-hidden />

      <div className="ns-term-login relative z-10">
        <Link to="/" className="ns-term-back">‹ back</Link>

        <div className="ns-term-login-head">
          <span className="ns-term-prompt">neurosynth login</span>
          <h1 className="ns-term-wordmark ns-term-wordmark-sm mt-2">NeuroSynth</h1>
        </div>

        <label className="ns-term-label" htmlFor="ns-user">username</label>
        <input
          id="ns-user"
          className="ns-term-input"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSubmit()}
          autoComplete="username"
        />

        <label className="ns-term-label" htmlFor="ns-pass">password</label>
        <input
          id="ns-pass"
          type="password"
          className="ns-term-input"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSubmit()}
          autoComplete="current-password"
        />

        <span className="ns-term-label">role</span>
        <div className="ns-term-segment" role="radiogroup" aria-label="role">
          {ROLES.map((r) => (
            <button
              key={r}
              type="button"
              role="radio"
              aria-checked={role === r ? "true" : "false"}
              className={`ns-term-segment-btn${role === r ? " is-active" : ""}`}
              onClick={() => setRole(r)}
            >
              {r}
            </button>
          ))}
        </div>

        {error && <p className="ns-term-error">{error}</p>}

        <button
          type="button"
          onClick={onSubmit}
          disabled={loading || !username.trim() || !password}
          className="ns-term-submit"
        >
          {loading ? (
            <>
              <Loader2 size={14} className="animate-spin" /> authenticating…
            </>
          ) : (
            "Authenticate ▸"
          )}
        </button>

        <div className="ns-term-creds">
          <div className="ns-term-creds-title">demo accounts (use matching role)</div>
          <div>clinician@neurosynth.local · neurosynth · CLINICIAN</div>
          <div>researcher@neurosynth.local · neurosynth · RESEARCHER</div>
          <div>admin@neurosynth.local · neurosynth · ADMIN</div>
        </div>
      </div>
    </div>
  );
}
