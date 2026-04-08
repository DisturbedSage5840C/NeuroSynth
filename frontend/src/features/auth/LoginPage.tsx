import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "@/lib/api";
import { useAuthStore } from "@/state/authStore";

export function LoginPage(): JSX.Element {
  const navigate = useNavigate();
  const setTokens = useAuthStore((s) => s.setTokens);
  const [email, setEmail] = useState("clinician@neurosynth.local");
  const [password, setPassword] = useState("neurosynth");
  const [error, setError] = useState("");

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      const payload = await login(email, password);
      setTokens(payload.access_token, payload.refresh_token, payload.role);
      navigate("/");
    } catch {
      setError("Invalid credentials");
    }
  };

  return (
    <form className="mx-auto w-full max-w-md space-y-3 rounded-panel border border-line bg-elevated p-4" onSubmit={onSubmit}>
      <h2 className="text-lg font-semibold">Clinical Login</h2>
      <label className="block text-sm">Email
        <input className="mt-1 w-full rounded border border-line bg-surface px-2 py-2" value={email} onChange={(e) => setEmail(e.target.value)} />
      </label>
      <label className="block text-sm">Password
        <input className="mt-1 w-full rounded border border-line bg-surface px-2 py-2" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
      </label>
      {error ? <p className="text-danger">{error}</p> : null}
      <button type="submit" className="w-full rounded bg-primary px-3 py-2 font-semibold text-surface">Sign in</button>
    </form>
  );
}
