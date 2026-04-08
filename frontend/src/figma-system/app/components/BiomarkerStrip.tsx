import { useMemo } from 'react';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';
import { generateBiomarkerData } from '../data/mock-data';
import { Heart, Wind, Thermometer, Droplets } from 'lucide-react';

interface MetricConfig {
  key: string;
  label: string;
  unit: string;
  icon: React.ReactNode;
  color: string;
  domain: [number, number];
  normalRange: [number, number];
}

const metrics: MetricConfig[] = [
  { key: 'heartRate', label: 'HR', unit: 'bpm', icon: <Heart size={12} />, color: '#ef4444', domain: [50, 120], normalRange: [60, 100] },
  { key: 'spo2', label: 'SpO₂', unit: '%', icon: <Droplets size={12} />, color: '#3b82f6', domain: [88, 100], normalRange: [95, 100] },
  { key: 'systolicBP', label: 'SBP', unit: 'mmHg', icon: <Droplets size={12} />, color: '#f97316', domain: [90, 180], normalRange: [100, 140] },
  { key: 'respiratoryRate', label: 'RR', unit: '/min', icon: <Wind size={12} />, color: '#22c55e', domain: [8, 30], normalRange: [12, 20] },
  { key: 'temperature', label: 'Temp', unit: '°C', icon: <Thermometer size={12} />, color: '#a78bfa', domain: [35, 40], normalRange: [36.5, 37.5] },
];

export function BiomarkerStrip() {
  const data = useMemo(() => generateBiomarkerData(), []);

  return (
    <div className="bg-card rounded-lg border border-border p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 style={{ fontSize: '13px' }} className="text-foreground">Real-Time Wearable Biomarkers</h3>
          <p className="text-muted-foreground" style={{ fontSize: '10px' }}>Streaming · 60s window · NeuroWatch v2 Band</p>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[var(--risk-low)] animate-pulse" />
          <span className="text-[var(--risk-low)] font-mono" style={{ fontSize: '10px' }}>LIVE</span>
        </div>
      </div>

      <div className="flex gap-2">
        {metrics.map(m => {
          const lastValue = (data[data.length - 1] as Record<string, number>)[m.key];
          const inRange = lastValue >= m.normalRange[0] && lastValue <= m.normalRange[1];
          return (
            <div key={m.key} className="flex-1 rounded bg-secondary/50 border border-border p-2">
              <div className="flex items-center gap-1 mb-1">
                <span style={{ color: m.color }}>{m.icon}</span>
                <span className="text-muted-foreground" style={{ fontSize: '10px' }}>{m.label}</span>
              </div>
              <div className="flex items-baseline gap-1">
                <span className="font-mono text-foreground" style={{ fontSize: '16px', color: inRange ? m.color : 'var(--risk-critical)' }}>
                  {m.key === 'temperature' ? lastValue.toFixed(1) : Math.round(lastValue)}
                </span>
                <span className="text-muted-foreground font-mono" style={{ fontSize: '9px' }}>{m.unit}</span>
              </div>
              <div style={{ height: '30px', width: '100%', minWidth: '50px' }}>
                <ResponsiveContainer width="100%" height={30} minWidth={50}>
                  <LineChart data={data}>
                    <YAxis domain={m.domain} hide />
                    <Line
                      type="monotone"
                      dataKey={m.key}
                      stroke={m.color}
                      strokeWidth={1}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}