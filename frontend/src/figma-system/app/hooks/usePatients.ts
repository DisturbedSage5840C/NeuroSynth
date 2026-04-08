import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../../../lib/api';
import type { Patient } from '../data/mock-data';

interface ApiPatientSummary {
  patient_id?: string;
  id?: string;
  name?: string;
  updated_at?: string;
}

function toUiPatient(item: ApiPatientSummary, index: number): Patient {
  const id = item.patient_id || item.id || `P-${String(index + 1).padStart(3, '0')}`;
  const nowIso = new Date().toISOString();
  return {
    id,
    name: item.name || `Patient ${id}`,
    age: 60 + (index % 20),
    sex: index % 2 === 0 ? 'M' : 'F',
    mrn: id,
    diagnosis: 'Neurology Monitoring',
    deteriorationProb: 0.4,
    riskLevel: 'moderate',
    lastUpdated: item.updated_at || nowIso,
    admissionDate: nowIso.slice(0, 10),
    ward: 'Neuro',
    attendingPhysician: 'Dr. NeuroSynth',
  };
}

export function usePatients() {
  return useQuery({
    queryKey: ['patients'],
    queryFn: () => apiFetch<{ items: ApiPatientSummary[] }>('/patients'),
    select: (data) => data.items.map(toUiPatient),
  });
}
