import { useRef, useEffect, useState, useCallback } from 'react';
import { connectomeNodes, connectomeEdges } from '../data/mock-data';
import { UncertaintyBadge } from './UncertaintyBadge';

export function ConnectomeGraph() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; activity: number } | null>(null);
  const animRef = useRef(0);

  const regionColors: Record<string, string> = {
    frontal: '#818cf8', parietal: '#34d399', occipital: '#f59e0b',
    temporal: '#ec4899', limbic: '#06b6d4', subcortical: '#a78bfa', cerebellum: '#fb923c',
  };

  const draw = useCallback((time: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const scaleX = w / 500;
    const scaleY = h / 360;

    ctx.clearRect(0, 0, w, h);

    // Draw edges
    connectomeEdges.forEach(edge => {
      const src = connectomeNodes.find(n => n.id === edge.source)!;
      const tgt = connectomeNodes.find(n => n.id === edge.target)!;
      const isHovered = hoveredNode === edge.source || hoveredNode === edge.target;
      const pulse = Math.sin(time / 1000 + edge.weight * 10) * 0.3 + 0.7;

      ctx.beginPath();
      ctx.moveTo(src.x * scaleX, src.y * scaleY);
      ctx.lineTo(tgt.x * scaleX, tgt.y * scaleY);
      ctx.strokeStyle = isHovered
        ? `rgba(129,140,248,${0.6 * pulse})`
        : `rgba(255,255,255,${0.06 + edge.weight * 0.08 * pulse})`;
      ctx.lineWidth = isHovered ? 2 : edge.weight * 1.5;
      ctx.stroke();
    });

    // Draw nodes
    connectomeNodes.forEach(node => {
      const x = node.x * scaleX;
      const y = node.y * scaleY;
      const isHovered = hoveredNode === node.id;
      const radius = 4 + node.activity * 8 + (isHovered ? 3 : 0);
      const pulse = Math.sin(time / 800 + node.activity * 5) * 2;
      const color = regionColors[node.region] || '#818cf8';

      // Glow
      if (node.activity > 0.7) {
        const glow = ctx.createRadialGradient(x, y, 0, x, y, radius + 8 + pulse);
        glow.addColorStop(0, color + '40');
        glow.addColorStop(1, color + '00');
        ctx.beginPath();
        ctx.arc(x, y, radius + 8 + pulse, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = isHovered ? color : color + 'cc';
      ctx.fill();
      ctx.strokeStyle = isHovered ? '#fff' : color + '60';
      ctx.lineWidth = isHovered ? 2 : 1;
      ctx.stroke();
    });

    animRef.current = requestAnimationFrame(draw);
  }, [hoveredNode]);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);
    const scaleX = canvas.width / 500;
    const scaleY = canvas.height / 360;

    let found: typeof connectomeNodes[0] | null = null;
    for (const node of connectomeNodes) {
      const dx = node.x * scaleX - mx;
      const dy = node.y * scaleY - my;
      if (Math.sqrt(dx * dx + dy * dy) < 16) { found = node; break; }
    }
    setHoveredNode(found?.id || null);
    setTooltip(found ? { x: e.clientX - rect.left, y: e.clientY - rect.top, label: found.label, activity: found.activity } : null);
  };

  return (
    <div className="bg-card rounded-lg border border-border p-4 relative">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 style={{ fontSize: '13px' }} className="text-foreground">Brain Connectome</h3>
          <UncertaintyBadge confidence={0.82} />
        </div>
        <span className="text-muted-foreground font-mono" style={{ fontSize: '10px' }}>fMRI resting-state · 04/07</span>
      </div>
      <canvas
        ref={canvasRef}
        width={500}
        height={340}
        className="w-full rounded cursor-crosshair"
        style={{ height: '220px', background: 'rgba(0,0,0,0.2)' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => { setHoveredNode(null); setTooltip(null); }}
      />
      {tooltip && (
        <div className="absolute bg-popover border border-border rounded px-2 py-1 pointer-events-none z-10"
          style={{ left: tooltip.x + 10, top: tooltip.y - 10, fontSize: '11px' }}>
          <div className="text-foreground">{tooltip.label}</div>
          <div className="text-muted-foreground font-mono" style={{ fontSize: '10px' }}>
            Activity: {(tooltip.activity * 100).toFixed(0)}%
          </div>
        </div>
      )}
      {/* Legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2">
        {Object.entries(regionColors).map(([region, color]) => (
          <div key={region} className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-muted-foreground capitalize" style={{ fontSize: '9px' }}>{region}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
