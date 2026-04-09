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
        </div>
      </div>
    </div>
  );
}
