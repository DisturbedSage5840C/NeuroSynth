import { createBrowserRouter } from 'react-router';
import { Layout } from './components/Layout';
import { Dashboard } from './components/Dashboard';
import { ReportViewer } from './components/ReportViewer';
import { DataExplorer } from './components/DataExplorer';
import { useOutletContext } from 'react-router';

function DashboardPage() {
  const { selectedPatientId } = useOutletContext<{ selectedPatientId: string }>();
  return <Dashboard selectedPatientId={selectedPatientId} />;
}

function ReportPage() {
  return <ReportViewer />;
}

function ExplorerPage() {
  return <DataExplorer />;
}

export const router = createBrowserRouter([
  {
    path: '/',
    Component: Layout,
    children: [
      { index: true, Component: DashboardPage },
      { path: 'report', Component: ReportPage },
      { path: 'explorer', Component: ExplorerPage },
    ],
  },
]);
