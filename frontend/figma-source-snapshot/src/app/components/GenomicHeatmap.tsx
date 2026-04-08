import { useState } from 'react';
import { genomicRisks } from '../data/mock-data';
import { UncertaintyBadge } from './UncertaintyBadge';

export function GenomicHeatmap() {
  const [hoveredGene, setHoveredGene] = useState<string | null>(null);

  const getRiskColor = (risk: number) => {
    if (risk >= 0.8) return 'var(--risk-critical)';
    if (risk >= 0.6) return 'var(--risk-high)';
    if (risk >= 0.4) return 'var(--risk-moderate)';
    return 'var(--risk-low)';
  };

  const getRiskOpacity = (risk: number) => 0.3 + risk * 0.7;

  return (
    <div className="bg-card rounded-lg border border-border p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 style={{ fontSize: '13px' }} className="text-foreground">Genomic Risk Matrix</h3>
          <UncertaintyBadge confidence={0.79} />
        </div>
        <span className="text-muted-foreground font-mono" style={{ fontSize: '10px' }}>12 variants · Panel v2.4</span>
      </div>

      {/* Heatmap grid */}
      <div className="grid grid-cols-4 gap-1">
        {genomicRisks.map(g => (
          <div
            key={g.gene}
            className={`relative rounded p-2 cursor-pointer transition-all ${hoveredGene === g.gene ? 'ring-1 ring-primary' : ''}`}
            style={{
              backgroundColor: getRiskColor(g.risk),
              opacity: getRiskOpacity(g.risk),
            }}
            onMouseEnter={() => setHoveredGene(g.gene)}
            onMouseLeave={() => setHoveredGene(null)}
          >
            <div className="font-mono text-white" style={{ fontSize: '11px' }}>{g.gene}</div>
            <div className="font-mono text-white/70" style={{ fontSize: '9px' }}>{(g.risk * 100).toFixed(0)}%</div>
          </div>
        ))}
      </div>

      {/* Detail panel */}
      {hoveredGene && (() => {
        const g = genomicRisks.find(x => x.gene === hoveredGene)!;
        return (
          <div className="mt-3 p-2 rounded bg-secondary border border-border" style={{ fontSize: '11px' }}>
            <div className="flex items-center justify-between">
              <span className="text-foreground font-mono">{g.gene} · {g.variant}</span>
              <UncertaintyBadge confidence={g.confidence} />
            </div>
            <div className="text-muted-foreground mt-1">{g.pathway}</div>
            <div className="flex items-center gap-2 mt-1">
              <div className="h-1 flex-1 rounded-full bg-background overflow-hidden">
                <div className="h-full rounded-full" style={{ width: `${g.risk * 100}%`, backgroundColor: getRiskColor(g.risk) }} />
              </div>
              <span className="font-mono text-foreground" style={{ fontSize: '10px' }}>{(g.risk * 100).toFixed(0)}%</span>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
