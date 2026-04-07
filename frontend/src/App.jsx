import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Brain,
  FlaskConical,
  GitBranch,
  HeartPulse,
  Info,
  LineChart as LineChartIcon,
  Microscope,
  ShieldCheck,
  Sparkles,
  Users,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const tabs = ["Dashboard", "Patient Analysis", "Causal Map", "Research", "About"];

const PATIENT_FIELDS = [
  { key: "Age", label: "Age", min: 45, max: 100, step: 1, section: "Demographics" },
  { key: "Gender", label: "Gender", min: 0, max: 2, step: 1, section: "Demographics" },
  { key: "Ethnicity", label: "Ethnicity", min: 0, max: 3, step: 1, section: "Demographics" },
  { key: "EducationLevel", label: "Education Level", min: 0, max: 3, step: 1, section: "Demographics" },

  { key: "BMI", label: "BMI", min: 15, max: 45, step: 0.1, section: "Lifestyle" },
  { key: "Smoking", label: "Smoking", min: 0, max: 1, step: 1, section: "Lifestyle" },
  { key: "AlcoholConsumption", label: "Alcohol Consumption", min: 0, max: 20, step: 0.1, section: "Lifestyle" },
  { key: "PhysicalActivity", label: "Physical Activity", min: 0, max: 10, step: 0.1, section: "Lifestyle" },
  { key: "DietQuality", label: "Diet Quality", min: 0, max: 10, step: 0.1, section: "Lifestyle" },
  { key: "SleepQuality", label: "Sleep Quality", min: 0, max: 10, step: 0.1, section: "Lifestyle" },

  { key: "FamilyHistoryAlzheimers", label: "Family History AD", min: 0, max: 1, step: 1, section: "Medical History" },
  { key: "CardiovascularDisease", label: "Cardiovascular Disease", min: 0, max: 1, step: 1, section: "Medical History" },
  { key: "Diabetes", label: "Diabetes", min: 0, max: 1, step: 1, section: "Medical History" },
  { key: "Depression", label: "Depression", min: 0, max: 1, step: 1, section: "Medical History" },
  { key: "HeadInjury", label: "Head Injury", min: 0, max: 1, step: 1, section: "Medical History" },
  { key: "Hypertension", label: "Hypertension", min: 0, max: 1, step: 1, section: "Medical History" },

  { key: "SystolicBP", label: "Systolic BP", min: 80, max: 220, step: 1, section: "Clinical Measurements" },
  { key: "DiastolicBP", label: "Diastolic BP", min: 40, max: 140, step: 1, section: "Clinical Measurements" },
  { key: "CholesterolTotal", label: "Total Cholesterol", min: 100, max: 400, step: 1, section: "Clinical Measurements" },
  { key: "CholesterolLDL", label: "LDL", min: 40, max: 300, step: 1, section: "Clinical Measurements" },
  { key: "CholesterolHDL", label: "HDL", min: 20, max: 120, step: 1, section: "Clinical Measurements" },
  { key: "CholesterolTriglycerides", label: "Triglycerides", min: 40, max: 500, step: 1, section: "Clinical Measurements" },
  { key: "MMSE", label: "MMSE", min: 0, max: 30, step: 1, section: "Clinical Measurements" },
  { key: "FunctionalAssessment", label: "Functional Assessment", min: 0, max: 10, step: 0.1, section: "Clinical Measurements" },
  { key: "ADL", label: "ADL", min: 0, max: 10, step: 0.1, section: "Clinical Measurements" },

  { key: "MemoryComplaints", label: "Memory Complaints", min: 0, max: 1, step: 1, section: "Symptoms" },
  { key: "BehavioralProblems", label: "Behavioral Problems", min: 0, max: 1, step: 1, section: "Symptoms" },
  { key: "Confusion", label: "Confusion", min: 0, max: 1, step: 1, section: "Symptoms" },
  { key: "Disorientation", label: "Disorientation", min: 0, max: 1, step: 1, section: "Symptoms" },
  { key: "PersonalityChanges", label: "Personality Changes", min: 0, max: 1, step: 1, section: "Symptoms" },
  { key: "DifficultyCompletingTasks", label: "Difficulty Completing Tasks", min: 0, max: 1, step: 1, section: "Symptoms" },
  { key: "Forgetfulness", label: "Forgetfulness", min: 0, max: 1, step: 1, section: "Symptoms" },
];

const sectionOrder = ["Demographics", "Lifestyle", "Medical History", "Clinical Measurements", "Symptoms"];

const defaultPatient = {
  Age: 73,
  Gender: 1,
  Ethnicity: 1,
  EducationLevel: 1,
  BMI: 27.4,
  Smoking: 0,
  AlcoholConsumption: 3.2,
  PhysicalActivity: 4.8,
  DietQuality: 5.2,
  SleepQuality: 5.0,
  FamilyHistoryAlzheimers: 1,
  CardiovascularDisease: 0,
  Diabetes: 0,
  Depression: 0,
  HeadInjury: 0,
  Hypertension: 1,
  SystolicBP: 132,
  DiastolicBP: 82,
  CholesterolTotal: 202,
  CholesterolLDL: 124,
  CholesterolHDL: 49,
  CholesterolTriglycerides: 166,
  MMSE: 24,
  FunctionalAssessment: 6.2,
  MemoryComplaints: 1,
  BehavioralProblems: 0,
  ADL: 6.0,
  Confusion: 0,
  Disorientation: 0,
  PersonalityChanges: 0,
  DifficultyCompletingTasks: 1,
  Forgetfulness: 1,
};

const cardStyle = {
  background: "rgba(255,255,255,0.03)",
  border: "1px solid rgba(255,255,255,0.08)",
  backdropFilter: "blur(10px)",
  WebkitBackdropFilter: "blur(10px)",
};

function RiskGauge({ probability = 0 }) {
  const clamped = Math.max(0, Math.min(1, probability));
  const pct = clamped * 100;
  const angle = (Math.PI * clamped) - Math.PI;
  const r = 74;
  const cx = 92;
  const cy = 92;
  const x2 = cx + r * Math.cos(angle);
  const y2 = cy + r * Math.sin(angle);

  const color = pct >= 80 ? "#ef4444" : pct >= 65 ? "#f59e0b" : pct >= 40 ? "#facc15" : "#10b981";

  return (
    <svg viewBox="0 0 184 120" className="w-full max-w-[250px]">
      <path d="M 18 92 A 74 74 0 0 1 166 92" stroke="#1e293b" strokeWidth="16" fill="none" />
      <path
        d="M 18 92 A 74 74 0 0 1 166 92"
        stroke={color}
        strokeWidth="16"
        fill="none"
        strokeDasharray={`${clamped * 233} 233`}
      />
      <line x1={cx} y1={cy} x2={x2} y2={y2} stroke="#e2e8f0" strokeWidth="4" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="5" fill="#e2e8f0" />
      <text x="92" y="112" fill="#cbd5e1" textAnchor="middle" fontSize="13" fontWeight="600">
        {pct.toFixed(1)}% Risk
      </text>
    </svg>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("Dashboard");
  const [datasetStats, setDatasetStats] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [featureImportance, setFeatureImportance] = useState({});
  const [causalGraph, setCausalGraph] = useState(null);

  const [patient, setPatient] = useState(defaultPatient);
  const [predictResult, setPredictResult] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportResult, setReportResult] = useState(null);
  const [appError, setAppError] = useState("");

  const [simVariable, setSimVariable] = useState("SleepQuality");
  const [simValue, setSimValue] = useState(0.8);
  const [simResult, setSimResult] = useState(null);

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap";
    document.head.appendChild(link);
    return () => document.head.removeChild(link);
  }, []);

  useEffect(() => {
    const load = async () => {
      try {
        const [stats, perf, imp, causal] = await Promise.all([
          fetch(`${API_URL}/dataset/stats`),
          fetch(`${API_URL}/model/performance`),
          fetch(`${API_URL}/model/feature_importance`),
          fetch(`${API_URL}/causal/graph`),
        ]);
        if (!stats.ok || !perf.ok || !imp.ok || !causal.ok) {
          throw new Error("Failed to load dashboard data.");
        }
        setDatasetStats(await stats.json());
        setPerformance(await perf.json());
        setFeatureImportance(await imp.json());
        setCausalGraph(await causal.json());
      } catch (err) {
        setAppError(err.message || "Initialization failed");
      }
    };
    load();
  }, []);

  const topFeatures = useMemo(() => {
    return Object.entries(featureImportance || {})
      .slice(0, 15)
      .map(([name, value]) => ({ name, value }));
  }, [featureImportance]);

  const datasetPie = useMemo(() => {
    if (!datasetStats) return [];
    return [
      { name: "Alzheimer's", value: datasetStats.n_alzheimers || 0 },
      { name: "Healthy", value: datasetStats.n_healthy || 0 },
    ];
  }, [datasetStats]);

  const trajectoryData = useMemo(() => {
    if (!predictResult?.trajectory) return [];
    const months = [6, 12, 18, 24, 30, 36];
    return months.map((m, i) => ({
      month: m,
      risk: predictResult.trajectory[i],
      lower: predictResult.confidence_bands?.lower?.[i] ?? predictResult.trajectory[i],
      upper: predictResult.confidence_bands?.upper?.[i] ?? predictResult.trajectory[i],
    }));
  }, [predictResult]);

  const perfCards = useMemo(() => {
    return [
      { label: "Accuracy", value: performance?.accuracy ?? 0 },
      { label: "F1 Weighted", value: performance?.f1_weighted ?? 0 },
      { label: "AUC-ROC", value: performance?.roc_auc ?? 0 },
      { label: "Precision", value: performance?.precision ?? 0 },
    ];
  }, [performance]);

  const inputBySection = useMemo(() => {
    const grouped = {};
    sectionOrder.forEach((s) => (grouped[s] = []));
    PATIENT_FIELDS.forEach((f) => grouped[f.section].push(f));
    return grouped;
  }, []);

  const updateField = (key, value) => {
    setPatient((prev) => ({ ...prev, [key]: Number(value) }));
  };

  const analyze = async () => {
    setPredictLoading(true);
    setAppError("");
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient),
      });
      if (!res.ok) throw new Error(await res.text());
      setPredictResult(await res.json());
    } catch (err) {
      setAppError(err.message || "Prediction failed");
    } finally {
      setPredictLoading(false);
    }
  };

  const getReport = async () => {
    setReportLoading(true);
    setAppError("");
    try {
      const res = await fetch(`${API_URL}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient),
      });
      if (!res.ok) throw new Error(await res.text());
      setReportResult(await res.json());
    } catch (err) {
      setAppError(err.message || "Report failed");
    } finally {
      setReportLoading(false);
    }
  };

  const simulate = async () => {
    setSimResult(null);
    try {
      const res = await fetch(`${API_URL}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_data: patient, variable: simVariable, new_value: simValue }),
      });
      if (!res.ok) throw new Error(await res.text());
      setSimResult(await res.json());
    } catch (err) {
      setAppError(err.message || "Simulation failed");
    }
  };

  const modifiableVars = causalGraph?.modifiable_interventions?.map((i) => i.variable) || ["SleepQuality", "PhysicalActivity", "MMSE"];

  return (
    <div className="min-h-screen text-slate-100" style={{ background: "#050508", fontFamily: "Inter, sans-serif" }}>
      <header className="sticky top-0 z-40 border-b border-white/10 backdrop-blur-md" style={{ background: "rgba(5,5,8,0.8)" }}>
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <h1 className="text-xl font-bold">🧠 NeuroSynth</h1>
          <nav className="flex flex-wrap gap-2">
            {tabs.map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className="rounded-lg px-3 py-2 text-sm transition-all duration-300"
                style={
                  activeTab === tab
                    ? { background: "linear-gradient(135deg,#00d4ff,#7c3aed)", color: "#050508", fontWeight: 700 }
                    : { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }
                }
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-6">
        {appError && (
          <div className="mb-4 flex items-center gap-2 rounded-xl border border-red-400/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            <AlertTriangle className="h-4 w-4" /> {appError}
          </div>
        )}

        {activeTab === "Dashboard" && (
          <section className="space-y-6">
            <div className="rounded-2xl p-6" style={cardStyle}>
              <p className="text-5xl font-extrabold tracking-tight" style={{ background: "linear-gradient(135deg,#00d4ff,#7c3aed)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                NeuroSynth
              </p>
              <p className="mt-2 text-lg text-slate-300">Neurological Deterioration Prediction Engine</p>
              <p className="mt-1 text-sm text-slate-400">Powered by ensemble ML, causal discovery, and AI clinical reasoning.</p>
            </div>

            <div className="grid gap-4 md:grid-cols-3 xl:grid-cols-6">
              {[
                { icon: Users, label: "Total Patients", value: datasetStats?.n_patients || 0 },
                { icon: ShieldCheck, label: "Model Accuracy", value: performance?.accuracy || 0 },
                { icon: Brain, label: "Features Tracked", value: 32 },
                { icon: GitBranch, label: "Causal Links", value: causalGraph?.edges?.length || 0 },
                { icon: Activity, label: "AD Cases", value: datasetStats?.n_alzheimers || 0 },
                { icon: HeartPulse, label: "Healthy", value: datasetStats?.n_healthy || 0 },
              ].map((item) => (
                <div key={item.label} className="rounded-xl p-4 transition-all duration-300 hover:-translate-y-1" style={cardStyle}>
                  <item.icon className="mb-2 h-4 w-4 text-cyan-300" />
                  <p className="text-xs uppercase text-slate-400">{item.label}</p>
                  <p className="mt-2 text-2xl font-bold text-white">{item.value}</p>
                </div>
              ))}
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {perfCards.map((m) => (
                <div key={m.label} className="rounded-xl p-4" style={cardStyle}>
                  <p className="text-xs uppercase text-slate-400">{m.label}</p>
                  <p className="text-3xl font-bold text-cyan-300">{Number(m.value).toFixed(3)}</p>
                </div>
              ))}
            </div>

            <div className="grid gap-6 xl:grid-cols-2">
              <div className="rounded-xl p-4" style={cardStyle}>
                <h3 className="mb-2 text-lg font-semibold">Top 15 Feature Importance</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={topFeatures} layout="vertical" margin={{ left: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#cbd5e1" />
                      <YAxis type="category" dataKey="name" width={130} stroke="#cbd5e1" />
                      <Tooltip />
                      <Bar dataKey="value">
                        {topFeatures.map((_, i) => (
                          <Cell key={i} fill={i < 5 ? "#ef4444" : i < 10 ? "#f59e0b" : "#10b981"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="rounded-xl p-4" style={cardStyle}>
                <h3 className="mb-2 text-lg font-semibold">Dataset Composition</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={datasetPie} dataKey="value" nameKey="name" outerRadius={110} label>
                        <Cell fill="#ef4444" />
                        <Cell fill="#10b981" />
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-4">
              {[
                ["Data Input", "32-feature patient profile ingestion", Microscope],
                ["ML Analysis", "4-model ensemble computes risk probability", Sparkles],
                ["Causal Discovery", "NOTEARS-MLP infers directional pathways", GitBranch],
                ["Clinical Report", "LLM synthesizes actionable assessment", LineChartIcon],
              ].map(([title, desc, Icon]) => (
                <div key={title} className="rounded-xl p-4" style={cardStyle}>
                  <Icon className="h-5 w-5 text-cyan-300" />
                  <p className="mt-2 font-semibold">{title}</p>
                  <p className="text-sm text-slate-400">{desc}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        {activeTab === "Patient Analysis" && (
          <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="max-h-[78vh] overflow-auto rounded-xl p-4" style={cardStyle}>
              {sectionOrder.map((section) => (
                <div key={section} className="mb-4 rounded-xl border border-white/10 p-3">
                  <p className="mb-2 font-semibold text-cyan-300">{section}</p>
                  <div className="grid gap-3 md:grid-cols-2">
                    {inputBySection[section].map((f) => (
                      <div key={f.key} className="rounded-lg border border-white/10 bg-white/5 p-2">
                        <div className="mb-1 flex items-center justify-between text-xs text-slate-300">
                          <span>{f.label}</span>
                          <span>{patient[f.key]}</span>
                        </div>
                        <input
                          type="range"
                          min={f.min}
                          max={f.max}
                          step={f.step}
                          value={patient[f.key]}
                          onChange={(e) => updateField(f.key, e.target.value)}
                          className="w-full accent-cyan-400"
                        />
                        <input
                          type="number"
                          min={f.min}
                          max={f.max}
                          step={f.step}
                          value={patient[f.key]}
                          onChange={(e) => updateField(f.key, e.target.value)}
                          className="mt-1 w-full rounded border border-white/10 bg-black/30 px-2 py-1 text-sm"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
              <button onClick={analyze} disabled={predictLoading} className="w-full rounded-xl px-4 py-3 font-semibold text-black transition-all duration-300 hover:opacity-90" style={{ background: "linear-gradient(135deg,#00d4ff,#7c3aed)" }}>
                {predictLoading ? "Analyzing..." : "Analyze Patient"}
              </button>
            </div>

            <div className="sticky top-24 space-y-4 self-start">
              <div className="rounded-xl p-4" style={cardStyle}>
                <p className="font-semibold">Risk Gauge</p>
                <RiskGauge probability={predictResult?.probability || 0} />
                <p className="text-sm text-slate-300">Risk Level: <span className="font-semibold">{predictResult?.risk_level || "-"}</span></p>
                <p className="text-sm text-slate-300">Confidence: {predictResult?.confidence || "-"}</p>
              </div>

              <div className="rounded-xl p-4" style={cardStyle}>
                <p className="mb-2 font-semibold">36-Month Trajectory</p>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trajectoryData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="month" stroke="#cbd5e1" />
                      <YAxis domain={[0, 1]} stroke="#cbd5e1" />
                      <Tooltip />
                      <Area type="monotone" dataKey="upper" stroke="none" fill="#7c3aed55" />
                      <Area type="monotone" dataKey="lower" stroke="none" fill="#050508" />
                      <Line type="monotone" dataKey="risk" stroke="#00d4ff" strokeWidth={3} dot />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="rounded-xl p-4" style={cardStyle}>
                <p className="mb-2 font-semibold">Top SHAP Risk Factors</p>
                {(predictResult?.shap_values || []).map((s) => (
                  <div key={s.feature} className="mb-2">
                    <div className="mb-1 flex justify-between text-xs">
                      <span>{s.feature}</span>
                      <span>{s.value.toFixed(4)}</span>
                    </div>
                    <div className="h-2 rounded bg-white/10">
                      <div className="h-2 rounded" style={{ width: `${Math.min(100, Math.abs(s.value) * 100)}%`, background: s.value >= 0 ? "#ef4444" : "#10b981" }} />
                    </div>
                  </div>
                ))}
              </div>

              <div className="rounded-xl p-4" style={cardStyle}>
                <p className="mb-2 font-semibold">Model Agreement</p>
                {Object.entries(predictResult?.individual_model_probs || {}).map(([k, v]) => (
                  <p key={k} className="text-sm text-slate-300">{k}: {Number(v).toFixed(4)}</p>
                ))}
              </div>

              <button onClick={getReport} disabled={reportLoading} className="w-full rounded-xl border border-cyan-400/40 bg-cyan-400/20 px-4 py-2 font-semibold transition-all duration-300 hover:bg-cyan-400/30">
                {reportLoading ? "Generating Report..." : "Generate Clinical Report"}
              </button>

              {reportResult?.sections && (
                <div className="rounded-xl p-4" style={cardStyle}>
                  {Object.entries(reportResult.sections).map(([title, text]) => (
                    <details key={title} className="mb-2 rounded border border-white/10 bg-black/20 p-2" open>
                      <summary className="cursor-pointer font-semibold text-cyan-300">{title}</summary>
                      <p className="mt-1 whitespace-pre-wrap text-sm text-slate-300">{text}</p>
                    </details>
                  ))}
                </div>
              )}
            </div>
          </section>
        )}

        {activeTab === "Causal Map" && (
          <section className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="rounded-xl p-4" style={cardStyle}>
              <h3 className="mb-2 text-lg font-semibold">Causal Network</h3>
              <div className="overflow-auto rounded border border-white/10 bg-black/20 p-3">
                <svg viewBox="0 0 820 520" className="min-w-[820px]">
                  {(causalGraph?.edges || []).slice(0, 80).map((e, idx) => {
                    const vars = causalGraph.variables || [];
                    const from = vars.indexOf(e.from);
                    const to = vars.indexOf(e.to);
                    const n = Math.max(vars.length, 1);
                    const fx = 410 + 180 * Math.cos((2 * Math.PI * from) / n);
                    const fy = 260 + 180 * Math.sin((2 * Math.PI * from) / n);
                    const tx = 410 + 180 * Math.cos((2 * Math.PI * to) / n);
                    const ty = 260 + 180 * Math.sin((2 * Math.PI * to) / n);
                    const cx = (fx + tx) / 2 + (to - from) * 4;
                    const cy = (fy + ty) / 2 - (to - from) * 3;
                    const riskEdge = e.to === "Diagnosis";
                    return (
                      <g key={`${e.from}-${e.to}-${idx}`}>
                        <path d={`M ${fx} ${fy} Q ${cx} ${cy} ${tx} ${ty}`} fill="none" stroke={riskEdge ? "#ef4444" : "#10b981"} strokeWidth={1 + e.strength * 4} strokeOpacity="0.65" />
                      </g>
                    );
                  })}
                  {(causalGraph?.variables || []).map((v, i, arr) => {
                    const x = 410 + 180 * Math.cos((2 * Math.PI * i) / arr.length);
                    const y = 260 + 180 * Math.sin((2 * Math.PI * i) / arr.length);
                    return (
                      <g key={v}>
                        <circle cx={x} cy={y} r="33" fill="#111827" stroke="#00d4ff" strokeWidth="2" />
                        <text x={x} y={y + 4} textAnchor="middle" fill="#e2e8f0" fontSize="11">{v}</text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-xl p-4" style={cardStyle}>
                <p className="font-semibold text-cyan-300">Top Causes of Alzheimer's</p>
                {(causalGraph?.top_causes_of_Diagnosis || []).map((t) => <p key={t.variable} className="text-sm text-slate-300">{t.variable}: {t.strength}</p>)}
                <p className="mt-2 font-semibold text-green-300">Protective Factors</p>
                {(causalGraph?.protective_factors || []).map((t) => <p key={t.variable} className="text-sm text-slate-300">{t.variable}: {t.effect}</p>)}
                <p className="mt-2 font-semibold text-red-300">Risk Amplifiers</p>
                {(causalGraph?.risk_amplifiers || []).map((t) => <p key={t.variable} className="text-sm text-slate-300">{t.variable}: {t.effect}</p>)}
              </div>

              <div className="rounded-xl p-4" style={cardStyle}>
                <p className="mb-2 font-semibold">Intervention Simulator</p>
                <select value={simVariable} onChange={(e) => setSimVariable(e.target.value)} className="mb-2 w-full rounded border border-white/10 bg-black/30 p-2">
                  {modifiableVars.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
                <input type="range" min="0" max="1" step="0.01" value={simValue} onChange={(e) => setSimValue(Number(e.target.value))} className="w-full accent-cyan-400" />
                <p className="text-sm text-slate-400">New normalized value: {simValue.toFixed(2)}</p>
                <button onClick={simulate} className="mt-2 w-full rounded-lg bg-cyan-500 px-3 py-2 font-semibold text-black">Simulate Intervention</button>
                {simResult && (
                  <div className="mt-3 rounded border border-white/10 bg-black/20 p-3 text-sm">
                    <p>Original Risk: {simResult.original_risk}</p>
                    <p>Intervened Risk: {simResult.intervened_risk}</p>
                    <p>Absolute Reduction: {simResult.absolute_risk_reduction}</p>
                    <p>Relative Reduction: {simResult.relative_risk_reduction_pct}%</p>
                    <p className="mt-2 text-slate-300">{simResult.interpretation}</p>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {activeTab === "Research" && (
          <section className="space-y-4">
            <div className="rounded-xl p-4" style={cardStyle}>
              <h3 className="mb-2 text-lg font-semibold">Alzheimer's Disease Context</h3>
              <p className="text-slate-300">Alzheimer's disease is a progressive neurodegenerative condition affecting memory, executive function, and daily living. Early risk identification can support proactive intervention planning.</p>
            </div>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              {["Age", "MMSE", "FunctionalAssessment", "ADL", "MemoryComplaints", "BehavioralProblems", "Depression", "SleepQuality", "PhysicalActivity", "Diagnosis"].map((v) => (
                <div key={v} className="rounded-xl p-3" style={cardStyle}>
                  <p className="font-semibold text-cyan-300">{v}</p>
                  <p className="text-xs text-slate-400">Core causal variable in NeuroSynth modeling.</p>
                </div>
              ))}
            </div>
            <div className="rounded-xl p-4" style={cardStyle}>
              <p className="font-semibold">Methodology</p>
              <p className="text-slate-300">4-model probabilistic ensemble + pseudo-longitudinal LSTM + NOTEARS-style neural causal discovery + Mistral-7B report generation via HuggingFace inference.</p>
            </div>
            <div className="rounded-xl p-4" style={cardStyle}>
              <p className="font-semibold">Dataset</p>
              <p className="text-slate-300">alzheimers_disease_data.csv (2149 patients, 34 total columns including target), based on Rabie El Kharoua 2024 Kaggle dataset framing.</p>
            </div>
            <div className="rounded-xl p-4 text-sm text-slate-300" style={cardStyle}>
              <p className="font-semibold text-cyan-300">Limitations & Ethics</p>
              <p>Predictions are probabilistic and should not be used as standalone medical diagnosis. Clinical interpretation by licensed professionals is required.</p>
            </div>
          </section>
        )}

        {activeTab === "About" && (
          <section className="space-y-4">
            <div className="rounded-xl p-4" style={cardStyle}>
              <h3 className="mb-2 text-lg font-semibold">About NeuroSynth</h3>
              <p className="text-slate-300">NeuroSynth is a production-grade neurological AI platform that integrates predictive modeling, temporal risk simulation, causal inference, and AI-assisted report generation.</p>
            </div>
            <div className="rounded-xl p-4" style={cardStyle}>
              <pre className="overflow-auto rounded bg-black/40 p-3 text-xs text-slate-300">
{`alzheimers_disease_data.csv
  -> DataPipeline
  -> 4-model Ensemble + SHAP
  -> Temporal LSTM Trajectory
  -> NOTEARS Causal Discovery
  -> FastAPI + React + Gradio`}
              </pre>
            </div>
            <div className="rounded-xl p-4" style={cardStyle}>
              <p className="font-semibold">GitHub</p>
              <a href="https://github.com/DisturbedSage5840C/NeuroSynth" className="text-cyan-300 underline" target="_blank" rel="noreferrer">https://github.com/DisturbedSage5840C/NeuroSynth</a>
              <p className="mt-2 text-xs text-slate-400">Research only. Not for direct clinical diagnosis or treatment decisions.</p>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
