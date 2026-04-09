import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { apiFetch } from '../../../lib/api';

type PerformanceResponse = {
  accuracy?: number;
  f1_weighted?: number;
  roc_auc?: number;
  precision?: number;
  confusion_matrix?: number[][];
};

export function PerformanceDashboard() {
  const perf = useQuery({
    queryKey: ['model-performance'],
    queryFn: () => apiFetch<PerformanceResponse>('/predictions/model/performance'),
  });
  const featureImportance = useQuery({
    queryKey: ['feature-importance'],
    queryFn: () => apiFetch<Record<string, number>>('/predictions/model/feature_importance'),
  });

  const rows = useMemo(
    () =>
      Object.entries(featureImportance.data || {})
        .map(([feature, value]) => ({ feature, value: Number(value) }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 15),
    [featureImportance.data]
  );

  const confusion = perf.data?.confusion_matrix || [[0, 0], [0, 0]];
  const tn = confusion[0]?.[0] || 0;
  const fp = confusion[0]?.[1] || 0;
  const fn = confusion[1]?.[0] || 0;
  const tp = confusion[1]?.[1] || 0;

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <h1 className="mb-4 text-xl text-foreground">Model Performance</h1>

      <div className="mb-4 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard label="Accuracy" value={perf.data?.accuracy} />
        <MetricCard label="F1 Weighted" value={perf.data?.f1_weighted} />
        <MetricCard label="AUC-ROC" value={perf.data?.roc_auc} />
        <MetricCard label="Precision" value={perf.data?.precision} />
      </div>

      <div className="mb-4 rounded-lg border border-border bg-card p-4">
        <h2 className="mb-3 text-sm font-medium text-foreground">Feature Importance (Top 15)</h2>
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={rows} layout="vertical" margin={{ top: 8, right: 12, left: 90, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis type="number" tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }} />
            <YAxis type="category" dataKey="feature" tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }} width={130} />
            <Tooltip formatter={(v: number) => v.toFixed(4)} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {rows.map((_, idx) => {
                const fill = idx < 5 ? 'var(--risk-critical)' : idx < 10 ? 'var(--risk-moderate)' : 'var(--risk-low)';
                return <Cell key={idx} fill={fill} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="rounded-lg border border-border bg-card p-4">
        <h2 className="mb-3 text-sm font-medium text-foreground">Confusion Matrix</h2>
        <div className="grid w-full max-w-md grid-cols-2 gap-2">
          <CellBox label="TN" value={tn} bg="var(--risk-low-bg)" color="var(--risk-low)" />
          <CellBox label="FP" value={fp} bg="var(--risk-high-bg)" color="var(--risk-high)" />
          <CellBox label="FN" value={fn} bg="var(--risk-moderate-bg)" color="var(--risk-moderate)" />
          <CellBox label="TP" value={tp} bg="var(--risk-critical-bg)" color="var(--risk-critical)" />
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-2xl text-foreground">{value != null ? value.toFixed(3) : 'N/A'}</div>
    </div>
  );
}

function CellBox({ label, value, bg, color }: { label: string; value: number; bg: string; color: string }) {
  return (
    <div className="rounded p-3" style={{ background: bg }}>
      <div className="text-xs" style={{ color }}>{label}</div>
      <div className="font-mono text-xl" style={{ color }}>{value}</div>
    </div>
  );
}
