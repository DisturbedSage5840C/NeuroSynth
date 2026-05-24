/**
 * BrainVisualization3D (Gap 5) — interactive 3D brain colored by SHAP attribution.
 *
 * Procedural anatomical approximation (no external mesh asset required): eight
 * cortical lobes + cerebellum + brainstem, each a noise-displaced sphere positioned
 * to read as a brain. Regions are tinted by their SHAP contribution to the top
 * predicted disease (red = increases risk, green = protective, slate = neutral).
 * Orbit to rotate, hover a region for its name and value.
 *
 * Lazy-load this component (React.lazy) so Three.js stays out of the main bundle.
 */
import { Canvas } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import { useMemo, useRef, useState } from "react";
import * as THREE from "three";

export interface BrainRegion {
  id: string;
  name: string;
  shapValue: number;
  position: [number, number, number];
  scale?: number;
}

interface SHAPValue {
  feature: string;
  value: number;
}

// Anatomical layout (approx, in arbitrary units). x = left/right, y = up, z = front/back.
const REGION_LAYOUT: Array<Omit<BrainRegion, "shapValue">> = [
  { id: "frontal_l", name: "Frontal lobe (L)", position: [-0.55, 0.35, 0.9], scale: 0.62 },
  { id: "frontal_r", name: "Frontal lobe (R)", position: [0.55, 0.35, 0.9], scale: 0.62 },
  { id: "parietal_l", name: "Parietal lobe (L)", position: [-0.55, 0.7, -0.15], scale: 0.55 },
  { id: "parietal_r", name: "Parietal lobe (R)", position: [0.55, 0.7, -0.15], scale: 0.55 },
  { id: "temporal_l", name: "Temporal lobe (L)", position: [-0.95, -0.2, 0.15], scale: 0.5 },
  { id: "temporal_r", name: "Temporal lobe (R)", position: [0.95, -0.2, 0.15], scale: 0.5 },
  { id: "occipital_l", name: "Occipital lobe (L)", position: [-0.45, 0.25, -1.0], scale: 0.5 },
  { id: "occipital_r", name: "Occipital lobe (R)", position: [0.45, 0.25, -1.0], scale: 0.5 },
  { id: "cerebellum", name: "Cerebellum", position: [0, -0.7, -0.9], scale: 0.55 },
  { id: "brainstem", name: "Brainstem", position: [0, -0.85, -0.2], scale: 0.3 },
];

function colorForShap(v: number): THREE.Color {
  if (v > 0.05) {
    const t = Math.min(v / 0.3, 1);
    return new THREE.Color().lerpColors(new THREE.Color("#64748b"), new THREE.Color("#ef4444"), 0.4 + 0.6 * t);
  }
  if (v < -0.05) {
    const t = Math.min(-v / 0.3, 1);
    return new THREE.Color().lerpColors(new THREE.Color("#64748b"), new THREE.Color("#22c55e"), 0.4 + 0.6 * t);
  }
  return new THREE.Color("#4b5563");
}

/** Map a SHAP value list onto anatomical regions (by descending magnitude). */
export function regionsFromShap(shapValues: SHAPValue[]): BrainRegion[] {
  const sorted = [...shapValues].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return REGION_LAYOUT.map((r, i) => ({
    ...r,
    shapValue: sorted[i]?.value ?? 0,
    name: sorted[i] ? `${r.name} · ${sorted[i].feature}` : r.name,
  }));
}

function RegionMesh({ region }: { region: BrainRegion }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const color = useMemo(() => colorForShap(region.shapValue), [region.shapValue]);

  // Noise-displaced sphere for an organic, sulci-like surface.
  const geometry = useMemo(() => {
    const geo = new THREE.IcosahedronGeometry(region.scale ?? 0.5, 4);
    const pos = geo.attributes.position as THREE.BufferAttribute;
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      const n = Math.sin(v.x * 9) * Math.cos(v.y * 9) * Math.sin(v.z * 9);
      v.multiplyScalar(1 + n * 0.06);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geo.computeVertexNormals();
    return geo;
  }, [region.scale]);

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      position={region.position}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
      }}
      onPointerOut={() => setHovered(false)}
    >
      <meshStandardMaterial
        color={color}
        roughness={0.65}
        metalness={0.05}
        transparent
        opacity={hovered ? 1 : 0.9}
        emissive={color}
        emissiveIntensity={hovered ? 0.35 : 0.12}
      />
      {hovered && (
        <Html distanceFactor={6} position={[0, (region.scale ?? 0.5) + 0.15, 0]} center>
          <div
            style={{
              background: "#0b0e14",
              border: "1px solid rgba(255,255,255,0.16)",
              borderRadius: 6,
              padding: "6px 9px",
              color: "#e2e8f0",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 11,
              whiteSpace: "nowrap",
              pointerEvents: "none",
            }}
          >
            {region.name}
            <span style={{ color: region.shapValue >= 0 ? "#ef4444" : "#22c55e", marginLeft: 8 }}>
              {region.shapValue >= 0 ? "+" : ""}
              {region.shapValue.toFixed(3)}
            </span>
          </div>
        </Html>
      )}
    </mesh>
  );
}

interface BrainVisualization3DProps {
  shapValues?: SHAPValue[];
  regions?: BrainRegion[];
  height?: number;
}

export default function BrainVisualization3D({
  shapValues = [],
  regions,
  height = 400,
}: BrainVisualization3DProps) {
  const resolved = useMemo<BrainRegion[]>(
    () => regions ?? regionsFromShap(shapValues),
    [regions, shapValues],
  );

  return (
    <div
      className="rounded-xl border border-border overflow-hidden"
      style={{ width: "100%", height, background: "#0b0e14" }}
    >
      <Canvas camera={{ position: [0, 0.4, 3.6], fov: 45 }} dpr={[1, 2]}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[4, 5, 5]} intensity={0.8} />
        <directionalLight position={[-4, -2, -3]} intensity={0.3} color="#00d4aa" />
        {resolved.map((r) => (
          <RegionMesh key={r.id} region={r} />
        ))}
        <OrbitControls enableZoom enablePan={false} autoRotate autoRotateSpeed={0.6} minDistance={2.5} maxDistance={6} />
      </Canvas>
    </div>
  );
}
