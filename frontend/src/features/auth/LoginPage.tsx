import { useState } from "react";
import { useNavigate } from "react-router";
import { login } from "@/lib/api";
import { useAuthStore } from "@/state/authStore";

export function LoginPage(): JSX.Element {
  const navigate = useNavigate();
  const setTokens = useAuthStore((s) => s.setTokens);
  const [username, setUsername] = useState("clinician@neurosynth.local");
  const [password, setPassword] = useState("neurosynth");
  const [role, setRole] = useState("CLINICIAN");
  const [error, setError] = useState("");

  const onSubmit = async () => {
    setError("");
    try {
      const payload = await login(username, password, role);
      setTokens(payload.access_token, payload.refresh_token, payload.role);
      localStorage.setItem("ns_logged_in", "true");
      navigate("/");
    } catch {
      setError("Invalid credentials");
    }
  };

  return (
    <div className="mx-auto w-full max-w-md space-y-3 rounded-panel border border-line bg-elevated p-4">
      <h2 className="text-lg font-semibold">Clinical Login</h2>
      <label className="block text-sm">Username
        <input className="mt-1 w-full rounded border border-line bg-surface px-2 py-2" value={username} onChange={(e) => setUsername(e.target.value)} />
      </label>
      <label className="block text-sm">Password
        <input className="mt-1 w-full rounded border border-line bg-surface px-2 py-2" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
      </label>
      <label className="block text-sm">Role
        <select className="mt-1 w-full rounded border border-line bg-surface px-2 py-2" value={role} onChange={(e) => setRole(e.target.value)}>
          <option value="CLINICIAN">Clinician</option>
          <option value="RESEARCHER">Researcher</option>
          <option value="ADMIN">Admin</option>
        </select>
      </label>
      {error ? <p className="text-danger">{error}</p> : null}
      <button onClick={onSubmit} className="w-full rounded bg-primary px-3 py-2 font-semibold text-surface">Sign in</button>
    </div>
  );
}
