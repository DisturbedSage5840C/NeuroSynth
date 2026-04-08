import type { ReactNode } from 'react';
import { createBrowserRouter, Navigate } from 'react-router';
import { Layout } from './components/Layout';
import { Dashboard } from './components/Dashboard';
import { ReportViewer } from './components/ReportViewer';
import { DataExplorer } from './components/DataExplorer';
import { LoginPage } from '../../features/auth/LoginPage';
import { useOutletContext } from 'react-router';
import { useAuthStore } from '../../state/authStore';

function RequireAuth({ children }: { children: ReactNode }) {
  const token = useAuthStore((s) => s.accessToken);
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function DashboardPage() {
  const { selectedPatientId } = useOutletContext<{ selectedPatientId: string }>();
  return <Dashboard selectedPatientId={selectedPatientId} />;
}

export const router = createBrowserRouter([
  { path: '/login', Component: LoginPage },
  {
    path: '/',
    element: (
      <RequireAuth>
        <Layout />
      </RequireAuth>
    ),
    children: [
      { index: true, Component: DashboardPage },
      { path: 'report', Component: ReportViewer },
      { path: 'explorer', Component: DataExplorer },
    ],
  },
]);
