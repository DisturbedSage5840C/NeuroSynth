import { Link } from "react-router";
import { Brain, Sparkles, ShieldCheck } from "lucide-react";

import "./auth-experience.css";

export function LandingPage(): JSX.Element {
  return (
    <div className="ns-landing min-h-screen overflow-hidden">
      <div className="ns-landing-bg-grid" />
      <div className="ns-landing-orb ns-landing-orb-one" />
      <div className="ns-landing-orb ns-landing-orb-two" />
      <div className="ns-landing-orb ns-landing-orb-three" />

      <div className="ns-landing-cube-wrap" aria-hidden>
        <div className="ns-landing-cube">
          <span className="face front" />
          <span className="face back" />
          <span className="face left" />
          <span className="face right" />
          <span className="face top" />
          <span className="face bottom" />
        </div>
      </div>

      <main className="relative z-10 mx-auto flex min-h-screen w-full max-w-6xl items-center justify-center px-5 py-10">
        <section className="ns-glass-panel w-full max-w-3xl p-7 md:p-10">
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/8 px-3 py-1 text-xs text-blue-100/90">
            <Sparkles size={14} />
            Clinical Intelligence Platform
          </div>

          <h1 className="ns-brand-title mb-3 text-5xl font-semibold leading-[0.95] tracking-tight md:text-7xl">
            NeuroSynth
          </h1>

          <p className="max-w-2xl text-sm text-slate-200/90 md:text-base">
            Neurological risk analytics with explainable AI, trajectory forecasting, and clinician-ready decision support.
          </p>

          <div className="mt-7 flex flex-wrap gap-3 text-xs text-slate-200/90">
            <span className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-white/6 px-3 py-1">
              <Brain size={13} /> Multi-model inference
            </span>
            <span className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-white/6 px-3 py-1">
              <ShieldCheck size={13} /> Role-secured access
            </span>
          </div>

          <div className="mt-9">
            <Link
              to="/login"
              className="ns-login-cta inline-flex items-center justify-center rounded-xl px-6 py-3 text-sm font-semibold"
            >
              Login
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
