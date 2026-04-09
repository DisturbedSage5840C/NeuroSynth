import { useState } from 'react';
import { Outlet, NavLink } from 'react-router';
import { LayoutDashboard, FileText, Database, Brain, BarChart2 } from 'lucide-react';
import { PatientSidebar } from './PatientSidebar';
import { patients } from '../data/mock-data';

const navItems = [
  { to: '/', icon: <LayoutDashboard size={16} />, label: 'Dashboard' },
  { to: '/report', icon: <FileText size={16} />, label: 'Report' },
  { to: '/explorer', icon: <Database size={16} />, label: 'Explorer' },
  { to: '/performance', icon: <BarChart2 size={16} />, label: 'Performance' },
];

export function Layout() {
  const [selectedPatientId, setSelectedPatientId] = useState(patients[0].id);

  return (
    <div className="h-screen flex bg-background text-foreground overflow-hidden">
      {/* Nav rail */}
      <div className="w-14 flex flex-col items-center py-4 gap-1 border-r border-border bg-[var(--sidebar)]">
        <div className="w-8 h-8 rounded-lg bg-primary/15 flex items-center justify-center mb-4">
          <Brain size={16} className="text-primary" />
        </div>
        {navItems.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `w-10 h-10 rounded-lg flex items-center justify-center transition-colors ${
                isActive ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
              }`
            }
            title={item.label}
            end={item.to === '/'}
          >
            {item.icon}
          </NavLink>
        ))}
      </div>

      {/* Patient sidebar */}
      <PatientSidebar selectedId={selectedPatientId} onSelect={setSelectedPatientId} />

      {/* Main content */}
      <Outlet context={{ selectedPatientId }} />
    </div>
  );
}
