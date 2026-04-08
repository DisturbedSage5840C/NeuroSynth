import { useMemo, useRef, useEffect, useState, useCallback } from 'react';
import { generateForecastData } from '../data/mock-data';
import { UncertaintyBadge } from './UncertaintyBadge';

const MARGIN = { top: 10, right: 10, bottom: 24, left: 40 };

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function buildMonotonePath(pts: { x: number; y: number }[]): string {
  if (pts.length < 2) return '';
  let d = `M${pts[0].x},${pts[0].y}`;
  for (let i = 1; i < pts.length; i++) {
    const cx = (pts[i - 1].x + pts[i].x) / 2;
    d += ` C${cx},${pts[i - 1].y} ${cx},${pts[i].y} ${pts[i].x},${pts[i].y}`;
  }
  return d;
}

export function ForecastChart() {
  const data = useMemo(() => generateForecastData(), []);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 600, h: 220 });
  const [tooltip, setTooltip] = useState<{ x: number; y: number; idx: number } | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setSize({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const nowIndex = data.findIndex(d => d.actual === undefined);
  const chartW = size.w - MARGIN.left - MARGIN.right;
  const chartH = size.h - MARGIN.top - MARGIN.bottom;

  const xScale = useCallback((i: number) => MARGIN.left + (i / (data.length - 1)) * chartW, [chartW, data.length]);
  const yScale = useCallback((v: number) => MARGIN.top + (1 - v) * chartH, [chartH]);

  // Build paths
  const predictedPts = data.map((d, i) => ({ x: xScale(i), y: yScale(d.predicted) }));
  const actualPts = data.filter(d => d.actual !== undefined).map((d, i) => ({ x: xScale(i), y: yScale(d.actual!) }));
  const upperPts = data.map((d, i) => ({ x: xScale(i), y: yScale(d.upper) }));
  const lowerPts = data.map((d, i) => ({ x: xScale(i), y: yScale(d.lower) }));

  const ciBand = buildMonotonePath(upperPts) + ' L' + lowerPts.map(p => `${p.x},${p.y}`).reverse().join(' L') + ' Z';
  const predictedPath = buildMonotonePath(predictedPts);
  const actualPath = buildMonotonePath(actualPts);

  // Y ticks
  const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  // X ticks every 4 points
  const xTicks = data.map((d, i) => i).filter((_, i) => i % 4 === 0);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left - MARGIN.left;
    const idx = Math.round((mx / chartW) * (data.length - 1));
    if (idx >= 0 && idx < data.length) {
      setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, idx });
    }
  }, [chartW, data.length]);

  return (
    <div className="bg-card rounded-lg border border-border p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 style={{ fontSize: '13px' }} className="text-foreground">Temporal Forecast — Deterioration Probability</h3>
            <UncertaintyBadge confidence={0.84} />
          </div>
          <p className="text-muted-foreground mt-0.5" style={{ fontSize: '11px' }}>TFT Model Output · 72h retrospective + 48h prospective · Updated 2 min ago</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5 bg-primary rounded" />
            <span className="text-muted-foreground" style={{ fontSize: '10px' }}>Predicted</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-0.5 rounded" style={{ backgroundColor: 'var(--risk-low)' }} />
            <span className="text-muted-foreground" style={{ fontSize: '10px' }}>Actual</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded opacity-30 bg-primary" />
            <span className="text-muted-foreground" style={{ fontSize: '10px' }}>95% CI</span>
          </div>
        </div>
      </div>
      <div ref={containerRef} style={{ width: '100%', height: 220, position: 'relative' }}>
        <svg
          width={size.w}
          height={size.h}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip(null)}
          style={{ display: 'block' }}
        >
          <defs>
            <linearGradient id="fci" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.15} />
              <stop offset="95%" stopColor="var(--primary)" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          {/* Grid */}
          {yTicks.map(t => (
            <line key={`yg-${t}`} x1={MARGIN.left} x2={size.w - MARGIN.right} y1={yScale(t)} y2={yScale(t)} stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
          ))}
          {xTicks.map(i => (
            <line key={`xg-${i}`} x1={xScale(i)} x2={xScale(i)} y1={MARGIN.top} y2={MARGIN.top + chartH} stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
          ))}

          {/* CI band */}
          <path d={ciBand} fill="url(#fci)" />

          {/* Critical threshold */}
          <line x1={MARGIN.left} x2={size.w - MARGIN.right} y1={yScale(0.7)} y2={yScale(0.7)} stroke="var(--risk-high)" strokeDasharray="2 4" strokeOpacity={0.5} />

          {/* NOW line */}
          {nowIndex > 0 && (
            <>
              <line x1={xScale(nowIndex - 1)} x2={xScale(nowIndex - 1)} y1={MARGIN.top} y2={MARGIN.top + chartH} stroke="var(--muted-foreground)" strokeDasharray="3 3" />
              <text x={xScale(nowIndex - 1)} y={MARGIN.top - 2} textAnchor="middle" fill="var(--muted-foreground)" style={{ fontSize: 9 }}>NOW</text>
            </>
          )}

          {/* Actual line */}
          <path d={actualPath} fill="none" stroke="var(--risk-low)" strokeWidth={1.5} strokeDasharray="4 2" />

          {/* Predicted line */}
          <path d={predictedPath} fill="none" stroke="var(--primary)" strokeWidth={2} />

          {/* Y axis labels */}
          {yTicks.map(t => (
            <text key={`yl-${t}`} x={MARGIN.left - 4} y={yScale(t) + 3} textAnchor="end" fill="var(--muted-foreground)" style={{ fontSize: 9, fontFamily: 'var(--font-mono)' }}>{`${(t * 100).toFixed(0)}%`}</text>
          ))}

          {/* X axis labels */}
          {xTicks.map(i => (
            <text key={`xl-${i}`} x={xScale(i)} y={size.h - 4} textAnchor="middle" fill="var(--muted-foreground)" style={{ fontSize: 9 }}>{data[i]?.time ?? ''}</text>
          ))}

          {/* Axis lines */}
          <line x1={MARGIN.left} x2={size.w - MARGIN.right} y1={MARGIN.top + chartH} y2={MARGIN.top + chartH} stroke="rgba(255,255,255,0.08)" />
          <line x1={MARGIN.left} x2={MARGIN.left} y1={MARGIN.top} y2={MARGIN.top + chartH} stroke="rgba(255,255,255,0.08)" />

          {/* Tooltip crosshair */}
          {tooltip && tooltip.idx >= 0 && tooltip.idx < data.length && (
            <>
              <line x1={xScale(tooltip.idx)} x2={xScale(tooltip.idx)} y1={MARGIN.top} y2={MARGIN.top + chartH} stroke="var(--muted-foreground)" strokeOpacity={0.3} />
              <circle cx={xScale(tooltip.idx)} cy={yScale(data[tooltip.idx].predicted)} r={3} fill="var(--primary)" />
              {data[tooltip.idx].actual !== undefined && (
                <circle cx={xScale(tooltip.idx)} cy={yScale(data[tooltip.idx].actual!)} r={3} fill="var(--risk-low)" />
              )}
            </>
          )}
        </svg>
        {/* Tooltip popup */}
        {tooltip && tooltip.idx >= 0 && tooltip.idx < data.length && (
          <div
            className="absolute pointer-events-none bg-popover border border-border rounded-md px-2.5 py-1.5 z-10"
            style={{
              left: tooltip.x + 12,
              top: tooltip.y - 10,
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
            }}
          >
            <div className="text-foreground mb-1">{data[tooltip.idx].time}</div>
            <div style={{ color: 'var(--primary)' }}>Predicted: {(data[tooltip.idx].predicted * 100).toFixed(1)}%</div>
            {data[tooltip.idx].actual !== undefined && (
              <div style={{ color: 'var(--risk-low)' }}>Actual: {(data[tooltip.idx].actual! * 100).toFixed(1)}%</div>
            )}
            <div className="text-muted-foreground">CI: {(data[tooltip.idx].lower * 100).toFixed(1)}–{(data[tooltip.idx].upper * 100).toFixed(1)}%</div>
          </div>
        )}
      </div>
    </div>
  );
}