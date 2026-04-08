import { patients } from '../data/mock-data';
import { ForecastChart } from './ForecastChart';
import { ConnectomeGraph } from './ConnectomeGraph';
import { GenomicHeatmap } from './GenomicHeatmap';
import { BiomarkerStrip } from './BiomarkerStrip';
import { RiskBadge } from './UncertaintyBadge';
import { Calendar, MapPin, Stethoscope } from 'lucide-react';

interface DashboardProps {
  selectedPatientId: string;
}

export function Dashboard({ selectedPatientId }: DashboardProps) {
  const patient = patients.find(p => p.id === selectedPatientId) || patients[0];

  return (
    <div className="flex-1 overflow-y-auto p-6">
      {/* Patient header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 style={{ fontSize: '20px' }} className="text-foreground">{patient.name}</h1>
            <RiskBadge level={patient.riskLevel} />
          </div>
          <div className="flex items-center gap-4 mt-1 text-muted-foreground" style={{ fontSize: '12px' }}>
            <span className="font-mono">{patient.mrn}</span>
            <span>{patient.age}{patient.sex} · {patient.diagnosis}</span>
            <span className="flex items-center gap-1"><MapPin size={11} /> {patient.ward}</span>
            <span className="flex items-center gap-1"><Stethoscope size={11} /> {patient.attendingPhysician}</span>
            <span className="flex items-center gap-1"><Calendar size={11} /> Admitted {patient.admissionDate}</span>
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-foreground" style={{ fontSize: '28px', color: `var(--risk-${patient.riskLevel})` }}>
            {Math.round(patient.deteriorationProb * 100)}%
          </div>
          <div className="text-muted-foreground" style={{ fontSize: '10px' }}>DETERIORATION PROBABILITY</div>
        </div>
      </div>

      {/* Main grid */}
      <div className="space-y-4">
        {/* Row 1: Forecast */}
        <ForecastChart />

        {/* Row 2: Biomarker strip */}
        <BiomarkerStrip />

        {/* Row 3: Connectome + Genomic */}
        <div className="grid grid-cols-2 gap-4">
          <ConnectomeGraph />
          <GenomicHeatmap />
        </div>
      </div>
    </div>
  );
}
