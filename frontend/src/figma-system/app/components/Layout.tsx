import { useState } from 'react';
import { Outlet, NavLink } from 'react-router';
import { useNavigate } from 'react-router';
import { LayoutDashboard, FileText, Database, Brain, BarChart2, LogOut } from 'lucide-react';
import { PatientSidebar } from './PatientSidebar';
import { patients } from '../data/mock-data';
import { useAuthStore } from '../../../state/authStore';

const navItems = [
  { to: '/', icon: <LayoutDashboard size={16} />, label: 'Dashboard' },
  { to: '/report', icon: <FileText size={16} />, label: 'Report' },
  { to: '/explorer', icon: <Database size={16} />, label: 'Explorer' },
  { to: '/performance', icon: <BarChart2 size={16} />, label: 'Performance' },
];

export function Layout() {
  const [selectedPatientId, setSelectedPatientId] = useState(patients[0].id);
  const navigate = useNavigate();
  const clear = useAuthStore((s) => s.clear);

  const handleLogout = async () => {
    try {
      await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/auth/logout`, {
        method: 'POST',
        credentials: 'include',
      });
    } catch { /* ignore network error */ }
    clear();
    navigate('/login');
  };

  return (
    <div className="h-screen flex bg-background text-foreground overflow-hidden">
      {/* Nav rail */}
      <div className="w-44 flex flex-col py-4 gap-1 px-3 border-r border-border bg-[var(--sidebar)]">
        <div className="px-3 mb-6">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-primary/15 flex items-center justify-center flex-shrink-0">
              <Brain size={14} className="text-primary" />
            </div>
            <div>
              <div className="font-semibold text-foreground" style={{ fontSize: 13 }}>NeuroSynth</div>
              <div className="text-muted-foreground" style={{ fontSize: 9 }}>v3.2 · Clinical AI</div>
            </div>
          </div>
        </div>
        {navItems.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `w-full h-10 rounded-lg flex items-center gap-3 px-3 transition-colors ${
                isActive ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
              }`
            }
            title={item.label}
            end={item.to === '/'}
          >
            <span className="flex items-center gap-3 w-full">
              {item.icon}
              <span style={{ fontSize: 13 }}>{item.label}</span>
            </span>
          </NavLink>
        ))}
        <div className="mt-auto pb-2">
          <button
            onClick={handleLogout}
            className="w-full h-10 rounded-lg flex items-center gap-3 px-3 text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
            title="Sign out"
          >
            <span className="flex items-center gap-3 w-full">
              <LogOut size={16} />
              <span style={{ fontSize: 13 }}>Sign out</span>
            </span>
          </button>
        </div>
      </div>

      {/* Patient sidebar */}
      <PatientSidebar selectedId={selectedPatientId} onSelect={setSelectedPatientId} />

      {/* Main content */}
      <Outlet context={{ selectedPatientId }} />
    </div>
  );
}
