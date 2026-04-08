import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";
import { ConnectomeGraph } from "@/features/figma/components/ConnectomeGraph";
import { ForecastChart } from "@/features/figma/components/ForecastChart";
import { GenomicHeatmap } from "@/features/figma/components/GenomicHeatmap";
import { BiomarkerStrip } from "@/features/figma/components/BiomarkerStrip";
import { ReportViewer } from "@/features/figma/components/ReportViewer";
import { LoginPage } from "@/features/auth/LoginPage";
import { ProtectedRoute } from "@/routes/ProtectedRoute";

function DashboardPage(): JSX.Element {
  return (
    <AppShell>
      <div className="grid gap-4 tablet:grid-cols-2">
        <ErrorBoundary title="Brain Connectome Force Graph">
          <ConnectomeGraph />
        </ErrorBoundary>

        <ErrorBoundary title="TFT Forecast">
          <ForecastChart />
        </ErrorBoundary>

        <ErrorBoundary title="Genomic Risk Heatmap">
          <GenomicHeatmap />
        </ErrorBoundary>

        <ErrorBoundary title="Live Wearables">
          <BiomarkerStrip />
        </ErrorBoundary>
      </div>

      <ErrorBoundary title="Clinical Report Renderer">
        <div className="mt-4 rounded-panel border border-line/70 bg-surface/80">
          <ReportViewer />
        </div>
      </ErrorBoundary>
    </AppShell>
  );
}

export default function App(): JSX.Element {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <ProtectedRoute allowed={["clinician", "researcher", "admin"]}>
            <DashboardPage />
          </ProtectedRoute>
        }
      />
      <Route path="/reports" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
