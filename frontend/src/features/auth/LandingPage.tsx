import { Link } from "react-router";

import "./auth-experience.css";

const STATS: Array<{ label: string; value: string }> = [
  { label: "neurological diseases", value: "8" },
  { label: "clinical biomarkers", value: "54" },
  { label: "forecast horizon", value: "48 mo" },
  { label: "validation AUC gate", value: "0.92" },
];

const BOOT_LINES: string[] = [
  "$ neurosynth --boot",
  "[ok] model registry        loaded (7-model ensemble)",
  "[ok] conformal predictor   calibrated  α=0.05",
  "[ok] xai engine            SHAP · LIME · counterfactuals",
  "[ok] clinical reports      SOAP · ICD-10 · FHIR R4",
  "[ok] drift monitor         online",
  "ready ▸ awaiting clinician session",
];

export function LandingPage(): JSX.Element {
  return (
    <div className="ns-terminal min-h-screen">
      <div className="ns-terminal-scanlines" aria-hidden />

      <main className="relative z-10 mx-auto grid min-h-screen w-full max-w-6xl items-center gap-10 px-6 py-12 md:grid-cols-2">
        {/* Left — wordmark + identity */}
        <section>
          <div className="ns-term-chip">CLINICAL INTELLIGENCE TERMINAL</div>
          <h1 className="ns-term-wordmark mt-5">NeuroSynth</h1>
          <p className="ns-term-subhead mt-4 max-w-md">
            Neurological risk analysis with explainable AI, 48-month trajectory
            forecasting, and clinician-ready decision support.
          </p>

          <div className="ns-term-stats mt-8">
            {STATS.map((s) => (
              <div key={s.label} className="ns-term-stat">
                <div className="ns-term-stat-value">{s.value}</div>
                <div className="ns-term-stat-label">{s.label}</div>
              </div>
            ))}
          </div>

          <div className="mt-9">
            <Link to="/login" className="ns-term-cta">
              Enter clinical workspace ▸
            </Link>
          </div>
          <div className="ns-term-foot mt-6">
            v3.0 · Research use only · HIPAA-aware · not a diagnostic device
          </div>
        </section>

        {/* Right — live boot readout panel */}
        <section className="ns-term-panel" aria-hidden>
          <div className="ns-term-panel-bar">
            <span className="ns-term-dot" />
            <span className="ns-term-dot" />
            <span className="ns-term-dot" />
            <span className="ns-term-panel-title">neurosynth · session</span>
          </div>
          <pre className="ns-term-readout">
            {BOOT_LINES.map((line) => (
              <div key={line} className={line.startsWith("[ok]") ? "ns-term-ok" : undefined}>
                {line}
              </div>
            ))}
          </pre>
        </section>
      </main>
    </div>
  );
}
