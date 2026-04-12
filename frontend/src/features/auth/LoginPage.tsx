import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { login } from "@/lib/api";
import { useAuthStore } from "@/state/authStore";
import { ArrowLeft, Brain, Loader2, ShieldCheck } from "lucide-react";
import { Link } from "react-router";

import "./auth-experience.css";

export function LoginPage(): JSX.Element {
  const navigate = useNavigate();
  const setTokens = useAuthStore((s) => s.setTokens);
  const hasRole = useAuthStore((s) => s.role);
  const [username, setUsername] = useState("clinician@neurosynth.local");
  const [password, setPassword] = useState("neurosynth");
  const [role, setRole] = useState("CLINICIAN");
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
      if (err instanceof Error) {
        setError(err.message || "Login failed");
      } else {
        setError("Login failed");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ns-login-page min-h-screen overflow-hidden px-4 py-8">
      <div className="ns-login-bg-grid" />
      <div className="ns-login-orb ns-login-orb-one" />
      <div className="ns-login-orb ns-login-orb-two" />

      <div className="relative z-10 mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-5xl items-center justify-center">
        <div className="ns-login-shell grid w-full overflow-hidden md:grid-cols-[1.25fr_1fr]">
          <section className="hidden border-r border-white/10 p-8 md:block">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/8 px-3 py-1 text-xs text-blue-100/90">
              <ShieldCheck size={14} />
              Verified Access Layer
            </div>
            <h1 className="ns-brand-title mt-5 text-5xl font-semibold leading-tight">NeuroSynth</h1>
            <p className="mt-4 max-w-md text-sm text-slate-200/90">
              Sign in with the correct account and role. Role escalation is blocked in demo mode and live mode.
            </p>

            <div className="ns-login-credentials mt-7 rounded-xl p-4 text-xs text-slate-200/90">
              <div className="mb-2 font-semibold text-slate-100">Demo accounts</div>
              <div>clinician@neurosynth.local / neurosynth / CLINICIAN</div>
              <div>researcher@neurosynth.local / neurosynth / RESEARCHER</div>
              <div>admin@neurosynth.local / neurosynth / ADMIN</div>
            </div>
          </section>

          <section className="p-6 md:p-8">
            <Link
              to="/"
              className="mb-5 inline-flex items-center gap-2 text-xs text-slate-300 hover:text-white"
            >
              <ArrowLeft size={13} /> Back to landing
            </Link>

            <div className="mb-5 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-white/20 bg-white/10">
                <Brain size={18} className="text-cyan-200" />
              </div>
              <div>
                <div className="text-base font-semibold text-slate-100">Secure Login</div>
                <div className="text-xs text-slate-400">Clinical AI Platform</div>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <label className="mb-1.5 block text-xs text-slate-300">Username</label>
                <input
                  className="ns-login-input w-full rounded-lg px-3 py-2 text-sm placeholder:text-slate-500"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && onSubmit()}
                  autoComplete="username"
                />
              </div>

              <div>
                <label className="mb-1.5 block text-xs text-slate-300">Password</label>
                <input
                  type="password"
                  className="ns-login-input w-full rounded-lg px-3 py-2 text-sm"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && onSubmit()}
                  autoComplete="current-password"
                />
              </div>

              <div>
                <label className="mb-1.5 block text-xs text-slate-300">Role</label>
                <select
                  className="ns-login-input w-full rounded-lg px-3 py-2 text-sm"
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
              <p className="mt-4 rounded-lg border border-rose-400/40 bg-rose-500/15 px-3 py-2 text-xs text-rose-200">
                {error}
              </p>
            )}

            <button
              onClick={onSubmit}
              disabled={loading || !username.trim() || !password}
              className="ns-login-button mt-5 flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold disabled:opacity-60"
            >
              {loading ? (
                <>
                  <Loader2 size={14} className="animate-spin" /> Signing in...
                </>
              ) : (
                "Sign in"
              )}
            </button>

            <p className="mt-3 text-center text-xs text-slate-400">Use matching role for each account</p>
          </section>
        </div>
      </div>
    </div>
  );
}
