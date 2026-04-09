import { useMemo } from 'react';
import { patients } from '../data/mock-data';
import { ForecastChart } from './ForecastChart';
import { ConnectomeGraph } from './ConnectomeGraph';
import { GenomicHeatmap } from './GenomicHeatmap';
import { BiomarkerStrip } from './BiomarkerStrip';
import { RiskBadge } from './UncertaintyBadge';
import { Calendar, MapPin, Stethoscope } from 'lucide-react';
import { PatientInputPanel } from './PatientInputPanel';
import { useAnalysisStore } from '../../../state/analysisStore';
import type { AnalysisResult } from '../types/analysis';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

interface DashboardProps {
  selectedPatientId: string;
}

export function Dashboard({ selectedPatientId }: DashboardProps) {
  const analysisResult = useAnalysisStore((s) => s.result);
  const setResult = useAnalysisStore((s) => s.setResult);
  const patient = patients.find(p => p.id === selectedPatientId) || patients[0];
  const probability = analysisResult?.probability ?? patient.deteriorationProb;
  const riskLevel = useMemo(() => {
    if (!analysisResult) return patient.riskLevel;
    const level = String(analysisResult.risk_level || '').toLowerCase();
    if (level.includes('critical')) return 'critical';
    if (level.includes('high')) return 'high';
    if (level.includes('moderate')) return 'moderate';
    return 'low';
  }, [analysisResult, patient.riskLevel]);

  const disease = analysisResult?.disease_classification;
  const diseaseRows = Object.entries(disease?.disease_probabilities || {})
    .map(([name, prob]) => ({ name, probability: Number(prob) }))
    .sort((a, b) => b.probability - a.probability);

  const diseaseColor = (name: string) => {
    if (name.includes('Alzheimer')) return 'var(--risk-critical)';
    if (name.includes('Parkinson')) return 'var(--risk-high)';
    if (name.includes('Multiple Sclerosis')) return 'var(--risk-moderate)';
    if (name.includes('Epilepsy')) return 'var(--chart-3)';
    if (name.includes('ALS')) return 'var(--risk-critical)';
    if (name.includes('Huntington')) return 'var(--chart-4)';
    return 'var(--primary)';
  };

  return (
    <div className="flex flex-1 overflow-hidden">
      <aside className="w-96 border-r border-border bg-card/40">
        <PatientInputPanel onResult={setResult} />
      </aside>

      <div className="flex-1 overflow-y-auto p-6">
        {/* Patient header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h1 style={{ fontSize: '20px' }} className="text-foreground">{patient.name}</h1>
              <RiskBadge level={riskLevel} />
            </div>
            <div className="mt-1 flex items-center gap-4 text-muted-foreground" style={{ fontSize: '12px' }}>
              <span className="font-mono">{patient.mrn}</span>
              <span>{patient.age}{patient.sex} · {patient.diagnosis}</span>
              <span className="flex items-center gap-1"><MapPin size={11} /> {patient.ward}</span>
              <span className="flex items-center gap-1"><Stethoscope size={11} /> {patient.attendingPhysician}</span>
              <span className="flex items-center gap-1"><Calendar size={11} /> Admitted {patient.admissionDate}</span>
            </div>
          </div>
          <div className="text-right">
            <div className="font-mono text-foreground" style={{ fontSize: '28px', color: `var(--risk-${riskLevel})` }}>
              {Math.round(probability * 100)}%
            </div>
            <div className="text-muted-foreground" style={{ fontSize: '10px' }}>DETERIORATION PROBABILITY</div>
          </div>
        </div>

        {/* Main grid */}
        <div className="space-y-4">
          {/* Row 1: Forecast */}
          <ForecastChart analysisResult={analysisResult as AnalysisResult | null} />

          {/* Row 2: Biomarker strip */}
          <BiomarkerStrip />

          {/* Row 3: Connectome + Genomic */}
          <div className="grid grid-cols-2 gap-4">
            <ConnectomeGraph analysisResult={analysisResult as AnalysisResult | null} />
            <GenomicHeatmap analysisResult={analysisResult as AnalysisResult | null} />
          </div>

          <div className="rounded-lg border border-border bg-card p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-medium text-foreground">Disease Profile</h3>
              <span className="rounded px-2 py-0.5 text-xs" style={{ background: 'var(--secondary)', color: 'var(--muted-foreground)' }}>
                {disease?.confidence || 'N/A'} confidence
              </span>
            </div>

            <div className="mb-3">
              <div
                className="text-lg font-semibold"
                style={{ color: disease?.predicted_disease ? diseaseColor(disease.predicted_disease) : 'var(--muted-foreground)' }}
              >
                {disease?.predicted_disease || 'Run analysis to classify disease type'}
              </div>
            </div>

            {diseaseRows.length ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={diseaseRows} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="name" tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }} interval={0} angle={-15} textAnchor="end" height={70} />
                  <YAxis tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }} domain={[0, 1]} />
                  <Tooltip formatter={(value: number) => `${Math.round(value * 100)}%`} />
                  <Bar dataKey="probability" radius={[4, 4, 0, 0]} fill="var(--primary)" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[220px] items-center justify-center text-xs text-muted-foreground">
                Disease probabilities will appear after analysis.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
