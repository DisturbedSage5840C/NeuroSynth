import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../../../lib/api';
import type { Patient } from '../data/mock-data';

export function usePatients() {
  return useQuery({
    queryKey: ['patients'],
    queryFn: () => apiFetch<{ items: Patient[] }>('/patients'),
    select: (data) => data.items,
  });
}
