import React, { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Brain,
  FlaskConical,
  Gauge,
  LineChart as LineChartIcon,
  Network,
  ShieldCheck
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const tabs = ["Dashboard", "Patient Analysis", "Causal Map", "Research"];

const biomarkerConfig = [
  { key: "age", label: "Age", min: 50, max: 100, step: 1 },
  { key: "educ", label: "EDUC", min: 0, max: 25, step: 1 },
  { key: "ses", label: "SES", min: 1, max: 5, step: 1 },
  { key: "mmse", label: "MMSE", min: 0, max: 30, step: 1 },
  { key: "cdr", label: "CDR", min: 0, max: 3, step: 0.5 },
  { key: "etiv", label: "eTIV", min: 900, max: 2000, step: 1 },
  { key: "nwbv", label: "nWBV", min: 0.6, max: 0.9, step: 0.001 },
  { key: "asf", label: "ASF", min: 0.8, max: 1.8, step: 0.001 }
];

const initialPatient = {
  age: 74,
  educ: 14,
  ses: 2,
  mmse: 24,
  cdr: 0.5,
  etiv: 1450,
  nwbv: 0.72,
  asf: 1.2
};

function RiskGauge({ probability }) {
  const value = Math.max(0, Math.min(1, probability || 0));
  const angle = Math.PI * value;
  const radius = 80;
  const cx = 90;
  const cy = 90;
  const x = cx - radius * Math.cos(angle);
  const y = cy - radius * Math.sin(angle);

  const color = value > 0.85 ? "#ef4444" : value > 0.7 ? "#f97316" : value > 0.5 ? "#facc15" : "#22c55e";

  return (
    <svg viewBox="0 0 180 110" className="w-full max-w-[220px]">
      <path d="M 10 90 A 80 80 0 0 1 170 90" stroke="#1f2937" strokeWidth="18" fill="none" />
      <path
        d="M 10 90 A 80 80 0 0 1 170 90"
        stroke={color}
        strokeWidth="18"
        fill="none"
        strokeDasharray={`${value * 251.2} 251.2`}
      />
      <line x1={cx} y1={cy} x2={x} y2={y} stroke="#e2e8f0" strokeWidth="4" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="5" fill="#e2e8f0" />
      <text x="90" y="106" fill="#cbd5e1" textAnchor="middle" fontSize="13">
        {(value * 100).toFixed(1)}% Risk
      </text>
    </svg>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("Dashboard");
  const [datasetStats, setDatasetStats] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [featureImportance, setFeatureImportance] = useState({});
  const [causalGraph, setCausalGraph] = useState(null);

  const [patient, setPatient] = useState(initialPatient);
  const [predictResult, setPredictResult] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState("");

  const [reportLoading, setReportLoading] = useState(false);
  const [reportResult, setReportResult] = useState(null);
  const [reportError, setReportError] = useState("");

  const [simulateVariable, setSimulateVariable] = useState("MMSE");
  const [simulateValue, setSimulateValue] = useState(20);
  const [simulateLoading, setSimulateLoading] = useState(false);
  const [simulateResult, setSimulateResult] = useState(null);
  const [simulateError, setSimulateError] = useState("");

  const [globalError, setGlobalError] = useState("");

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        const [statsRes, perfRes, impRes, causalRes] = await Promise.all([
          fetch(`${API_URL}/dataset/stats`),
          fetch(`${API_URL}/model/performance`),
          fetch(`${API_URL}/model/feature_importance`),
          fetch(`${API_URL}/causal/graph`)
        ]);

        if (!statsRes.ok || !perfRes.ok || !impRes.ok || !causalRes.ok) {
          throw new Error("Unable to load dashboard data.");
        }

        setDatasetStats(await statsRes.json());
        setPerformance(await perfRes.json());
        setFeatureImportance(await impRes.json());
        setCausalGraph(await causalRes.json());
      } catch (err) {
        setGlobalError(err.message || "Failed to load initial data.");
      }
    };

    loadDashboard();
  }, []);

  const featureChartData = useMemo(
    () =>
      Object.entries(featureImportance || {}).map(([name, score]) => ({
        name,
        score
      })),
    [featureImportance]
  );

  const performanceChartData = useMemo(() => {
    if (!performance || !performance.classification_report) {
      return [];
    }
    const report = performance.classification_report;
    return [
      {
        className: "Nondemented",
        precision: report["0"]?.precision || 0,
        recall: report["0"]?.recall || 0,
        f1: report["0"]?.["f1-score"] || 0
      },
      {
        className: "Demented",
        precision: report["1"]?.precision || 0,
        recall: report["1"]?.recall || 0,
        f1: report["1"]?.["f1-score"] || 0
      }
    ];
  }, [performance]);

  const trajectoryData = useMemo(() => {
    if (!predictResult?.trajectory) {
      return [];
    }
    const months = [6, 12, 18, 24, 30, 36];
    return months.map((m, i) => ({ month: m, risk: predictResult.trajectory[i] || 0 }));
  }, [predictResult]);

  const causalEdgesForViz = useMemo(() => causalGraph?.edges || [], [causalGraph]);

  const nodePositions = useMemo(() => {
    const names = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"];
    const centerX = 320;
    const centerY = 240;
    const radius = 180;
    return names.map((name, idx) => {
      const angle = (2 * Math.PI * idx) / names.length - Math.PI / 2;
      return {
        name,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      };
    });
  }, []);

  const nodeMap = useMemo(() => {
    const map = {};
    nodePositions.forEach((n) => {
      map[n.name] = n;
    });
    return map;
  }, [nodePositions]);

  const updatePatient = (key, value) => {
    setPatient((prev) => ({ ...prev, [key]: Number(value) }));
  };

  const runPrediction = async () => {
    setPredictLoading(true);
    setPredictError("");
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient)
      });
      if (!res.ok) {
        const errBody = await res.text();
        throw new Error(errBody || "Prediction request failed");
      }
      setPredictResult(await res.json());
    } catch (err) {
      setPredictError(err.message || "Prediction failed");
    } finally {
      setPredictLoading(false);
    }
  };

  const generateReport = async () => {
    setReportLoading(true);
    setReportError("");
    try {
      const res = await fetch(`${API_URL}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient)
      });
      if (!res.ok) {
        const errBody = await res.text();
        throw new Error(errBody || "Report generation failed");
      }
      setReportResult(await res.json());
    } catch (err) {
      setReportError(err.message || "Report generation failed");
    } finally {
      setReportLoading(false);
    }
  };

  const runSimulation = async () => {
    setSimulateLoading(true);
    setSimulateError("");
    try {
      const res = await fetch(`${API_URL}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_data: patient,
          intervention_variable: simulateVariable,
          new_value: Number(simulateValue)
        })
      });
      if (!res.ok) {
        const errBody = await res.text();
        throw new Error(errBody || "Simulation failed");
      }
      setSimulateResult(await res.json());
    } catch (err) {
      setSimulateError(err.message || "Simulation failed");
    } finally {
      setSimulateLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-slate-100">
      <header className="border-b border-cyan-900/40 bg-gradient-to-r from-[#0a0a0f] via-[#0d1326] to-[#140f24]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-5">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-cyan-300">NeuroSynth</h1>
            <p className="text-sm text-slate-300">Neurological Deterioration Prediction Engine</p>
          </div>
          <div className="rounded-full border border-violet-400/30 bg-violet-500/10 px-4 py-2 text-xs text-violet-200">
            Production-grade research platform
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {globalError && (
          <div className="mb-4 flex items-start gap-2 rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">
            <AlertCircle className="mt-0.5 h-4 w-4" />
            <span>{globalError}</span>
          </div>
        )}

        <div className="mb-6 flex flex-wrap gap-2">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`rounded-lg px-4 py-2 text-sm transition ${
                activeTab === tab
                  ? "bg-cyan-500 text-slate-950"
                  : "border border-slate-700 bg-slate-900/60 text-slate-300 hover:border-cyan-600"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {activeTab === "Dashboard" && (
          <section className="space-y-6">
            <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
              <h2 className="text-3xl font-semibold text-cyan-300">NeuroSynth</h2>
              <p className="mt-2 text-slate-300">Neurological Deterioration Prediction Engine</p>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <p className="text-xs uppercase text-slate-400">Dataset Size</p>
                <p className="mt-2 text-2xl font-semibold text-cyan-300">{datasetStats?.n_patients || 0}</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <p className="text-xs uppercase text-slate-400">Model Accuracy</p>
                <p className="mt-2 text-2xl font-semibold text-cyan-300">{performance?.accuracy || 0}</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <p className="text-xs uppercase text-slate-400">Features Analyzed</p>
                <p className="mt-2 text-2xl font-semibold text-cyan-300">8</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <p className="text-xs uppercase text-slate-400">Causal Links Found</p>
                <p className="mt-2 text-2xl font-semibold text-cyan-300">{causalGraph?.edges?.length || 0}</p>
              </div>
            </div>

            <div className="grid gap-6 xl:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <h3 className="mb-3 flex items-center gap-2 text-lg text-cyan-300">
                  <ShieldCheck className="h-4 w-4" /> Model Performance
                </h3>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={performanceChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="className" stroke="#cbd5e1" />
                      <YAxis stroke="#cbd5e1" domain={[0, 1]} />
                      <Tooltip />
                      <Bar dataKey="precision" fill="#00d4ff" />
                      <Bar dataKey="recall" fill="#7c3aed" />
                      <Bar dataKey="f1" fill="#22c55e" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <h3 className="mb-3 flex items-center gap-2 text-lg text-cyan-300">
                  <Gauge className="h-4 w-4" /> Feature Importance
                </h3>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={featureChartData} margin={{ left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#cbd5e1" />
                      <YAxis type="category" dataKey="name" stroke="#cbd5e1" width={80} />
                      <Tooltip />
                      <Bar dataKey="score" fill="#00d4ff" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </section>
        )}

        {activeTab === "Patient Analysis" && (
          <section className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
              <h3 className="mb-4 flex items-center gap-2 text-xl text-cyan-300">
                <Brain className="h-5 w-5" /> Patient Biomarkers
              </h3>
              <div className="space-y-4">
                {biomarkerConfig.map((cfg) => (
                  <div key={cfg.key} className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                    <div className="mb-2 flex items-center justify-between text-sm">
                      <label className="text-slate-300">{cfg.label}</label>
                      <input
                        type="number"
                        value={patient[cfg.key]}
                        min={cfg.min}
                        max={cfg.max}
                        step={cfg.step}
                        onChange={(e) => updatePatient(cfg.key, e.target.value)}
                        className="w-24 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-right"
                      />
                    </div>
                    <input
                      type="range"
                      min={cfg.min}
                      max={cfg.max}
                      step={cfg.step}
                      value={patient[cfg.key]}
                      onChange={(e) => updatePatient(cfg.key, e.target.value)}
                      className="w-full accent-cyan-400"
                    />
                  </div>
                ))}
              </div>

              <button
                onClick={runPrediction}
                disabled={predictLoading}
                className="mt-5 w-full rounded-lg bg-cyan-400 px-4 py-3 font-semibold text-slate-900 disabled:opacity-60"
              >
                {predictLoading ? "Analyzing..." : "Analyze Patient"}
              </button>
              {predictError && <p className="mt-2 text-sm text-red-300">{predictError}</p>}
            </div>

            <div className="space-y-6">
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
                <h3 className="mb-3 text-xl text-cyan-300">Risk Assessment</h3>
                {predictResult ? (
                  <div>
                    <RiskGauge probability={predictResult.probability} />
                    <p className="mt-2 text-lg">
                      Prediction: <span className="font-semibold text-cyan-300">{predictResult.prediction}</span>
                    </p>
                    <p className="text-sm text-slate-300">Confidence: {predictResult.confidence}</p>
                    <p className="text-sm text-slate-300">Risk Level: {predictResult.risk_level}</p>
                    <div className="mt-4 flex flex-wrap gap-2">
                      {(predictResult.top_risk_factors || []).map((item) => (
                        <span
                          key={item.feature}
                          className="rounded-full border border-violet-500/40 bg-violet-500/20 px-3 py-1 text-xs text-violet-200"
                        >
                          {item.feature}: {item.score}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p className="text-slate-400">Run analysis to view risk results.</p>
                )}
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
                <h3 className="mb-3 flex items-center gap-2 text-xl text-cyan-300">
                  <LineChartIcon className="h-5 w-5" /> 36-Month Trajectory
                </h3>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trajectoryData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="month" stroke="#cbd5e1" />
                      <YAxis stroke="#cbd5e1" domain={[0, 1]} />
                      <Tooltip />
                      <Line type="monotone" dataKey="risk" stroke="#00d4ff" strokeWidth={3} dot />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
                <button
                  onClick={generateReport}
                  disabled={reportLoading}
                  className="w-full rounded-lg bg-violet-500 px-4 py-3 font-semibold text-white disabled:opacity-60"
                >
                  {reportLoading ? "Generating Clinical Report..." : "Generate Clinical Report"}
                </button>
                {reportError && <p className="mt-2 text-sm text-red-300">{reportError}</p>}

                {reportResult?.sections && (
                  <div className="mt-4 space-y-3">
                    {Object.entries(reportResult.sections).map(([title, content]) => (
                      <div key={title} className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                        <p className="text-sm font-semibold text-cyan-300">{title}</p>
                        <p className="mt-1 whitespace-pre-wrap text-sm text-slate-300">{content}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {activeTab === "Causal Map" && (
          <section className="space-y-6">
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
              <h3 className="mb-3 flex items-center gap-2 text-xl text-cyan-300">
                <Network className="h-5 w-5" /> Causal Network
              </h3>
              <div className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                <svg viewBox="0 0 640 480" className="w-full min-w-[640px]">
                  {causalEdgesForViz.map((edge, idx) => {
                    const src = nodeMap[edge.from];
                    const dst = nodeMap[edge.to];
                    if (!src || !dst) return null;
                    return (
                      <g key={`${edge.from}-${edge.to}-${idx}`}>
                        <line
                          x1={src.x}
                          y1={src.y}
                          x2={dst.x}
                          y2={dst.y}
                          stroke="#00d4ff"
                          strokeOpacity={0.4 + edge.strength * 0.6}
                          strokeWidth={1 + edge.strength * 4}
                        />
                      </g>
                    );
                  })}
                  {nodePositions.map((node) => (
                    <g key={node.name}>
                      <circle cx={node.x} cy={node.y} r={28} fill="#151a2e" stroke="#7c3aed" strokeWidth="2" />
                      <text x={node.x} y={node.y + 4} textAnchor="middle" fill="#e2e8f0" fontSize="11">
                        {node.name}
                      </text>
                    </g>
                  ))}
                </svg>
              </div>
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                  <p className="text-sm font-semibold text-cyan-300">Top Causes of CDR</p>
                  <ul className="mt-2 space-y-1 text-sm text-slate-300">
                    {(causalGraph?.top_causes_of_CDR || []).map((item) => (
                      <li key={`cdr-${item.variable}`}>
                        {item.variable}: {item.strength}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                  <p className="text-sm font-semibold text-cyan-300">Top Causes of MMSE</p>
                  <ul className="mt-2 space-y-1 text-sm text-slate-300">
                    {(causalGraph?.top_causes_of_MMSE || []).map((item) => (
                      <li key={`mmse-${item.variable}`}>
                        {item.variable}: {item.strength}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
              <h3 className="mb-3 text-xl text-cyan-300">Intervention Simulator</h3>
              <div className="grid gap-4 md:grid-cols-3">
                <div>
                  <label className="mb-2 block text-sm text-slate-300">Variable</label>
                  <select
                    value={simulateVariable}
                    onChange={(e) => setSimulateVariable(e.target.value)}
                    className="w-full rounded border border-slate-700 bg-slate-900 px-3 py-2"
                  >
                    {["MMSE", "SES", "EDUC", "CDR", "Age", "eTIV", "nWBV", "ASF"].map((v) => (
                      <option key={v} value={v}>
                        {v}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="mb-2 block text-sm text-slate-300">New Value</label>
                  <input
                    type="range"
                    min="0"
                    max="30"
                    step="0.5"
                    value={simulateValue}
                    onChange={(e) => setSimulateValue(Number(e.target.value))}
                    className="w-full accent-cyan-400"
                  />
                  <p className="mt-1 text-sm text-slate-300">{simulateValue}</p>
                </div>
                <div className="flex items-end">
                  <button
                    onClick={runSimulation}
                    disabled={simulateLoading}
                    className="w-full rounded-lg bg-cyan-400 px-4 py-2 font-semibold text-slate-900 disabled:opacity-60"
                  >
                    {simulateLoading ? "Simulating..." : "Simulate"}
                  </button>
                </div>
              </div>

              {simulateError && <p className="mt-3 text-sm text-red-300">{simulateError}</p>}

              {simulateResult && (
                <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950/60 p-4 text-sm text-slate-300">
                  <p>Original CDR Risk: {simulateResult.original_CDR_risk}</p>
                  <p>Intervened CDR Risk: {simulateResult.intervened_CDR_risk}</p>
                  <p>Estimated Improvement: {simulateResult.estimated_improvement}</p>
                  <p className="mt-2 text-slate-200">{simulateResult.interpretation}</p>
                </div>
              )}
            </div>
          </section>
        )}

        {activeTab === "Research" && (
          <section className="space-y-5">
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-6">
              <h3 className="mb-3 flex items-center gap-2 text-xl text-cyan-300">
                <FlaskConical className="h-5 w-5" /> Research Context
              </h3>
              <p className="text-slate-300">
                OASIS Longitudinal is an open neuroimaging and cognitive dataset used for studying aging and dementia
                progression. NeuroSynth combines classic ensemble prediction, sequence modeling, and causal discovery to
                generate practical risk analytics.
              </p>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <h4 className="mb-2 font-semibold text-cyan-300">Biomarker Definitions</h4>
                <ul className="space-y-1 text-sm text-slate-300">
                  <li>MMSE: Mini-Mental State Examination cognitive score.</li>
                  <li>CDR: Clinical Dementia Rating progression marker.</li>
                  <li>nWBV: Normalized whole brain volume proxy for atrophy trends.</li>
                  <li>eTIV/ASF: Structural normalization parameters.</li>
                </ul>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                <h4 className="mb-2 font-semibold text-cyan-300">Methodology</h4>
                <ul className="space-y-1 text-sm text-slate-300">
                  <li>Ensemble RF + Gradient Boosting for robust baseline risk.</li>
                  <li>PyTorch LSTM for trajectory forecasting.</li>
                  <li>NOTEARS-style neural causal discovery for directional pathways.</li>
                  <li>LLM-based report generation via Hugging Face Inference API.</li>
                </ul>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 text-sm text-slate-300">
              <p className="mb-2 font-semibold text-cyan-300">Selected References</p>
              <ul className="space-y-1">
                <li>
                  OASIS Dataset: <a className="text-cyan-300" href="https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers">https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers</a>
                </li>
                <li>
                  NOTEARS Paper: <a className="text-cyan-300" href="https://arxiv.org/abs/1803.01422">https://arxiv.org/abs/1803.01422</a>
                </li>
                <li>
                  LSTM for Time-Series: <a className="text-cyan-300" href="https://www.bioinf.jku.at/publications/older/2604.pdf">https://www.bioinf.jku.at/publications/older/2604.pdf</a>
                </li>
              </ul>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
