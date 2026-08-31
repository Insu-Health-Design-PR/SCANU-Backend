import {
  Activity,
  Camera,
  FlaskConical,
  Radio,
  Settings,
  Shield,
  Thermometer,
} from 'lucide-react';
import { NavSection } from './sensorStatus';

const NAV: { id: NavSection; label: string; icon: typeof Camera }[] = [
  { id: 'camera', label: 'Cameras', icon: Camera },
  { id: 'thermal', label: 'Thermal', icon: Thermometer },
  { id: 'mmwave', label: 'mmWave', icon: Radio },
  { id: 'metrics', label: 'Metrics', icon: Activity },
  { id: 'playground', label: 'Playground', icon: FlaskConical },
];

interface AppNavProps {
  active: NavSection;
  onChange: (section: NavSection) => void;
}

export function AppNav({ active, onChange }: AppNavProps) {
  return (
    <nav className="w-52 shrink-0 border-r border-white/10 bg-[#0c0f14] flex flex-col">
      <div className="h-14 px-4 flex items-center gap-2.5 border-b border-white/10">
        <div className="w-8 h-8 rounded-lg bg-white/10 border border-white/15 flex items-center justify-center">
          <Shield className="w-4 h-4 text-white" />
        </div>
        <div>
          <div className="text-sm font-semibold text-white leading-tight">Threat Monitor</div>
          <div className="text-[10px] text-white/40 uppercase tracking-wider">Live</div>
        </div>
      </div>

      <div className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto">
        {NAV.map(({ id, label, icon: Icon }) => {
          const selected = active === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => onChange(id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                selected
                  ? 'bg-white/12 text-white font-medium'
                  : 'text-white/55 hover:text-white hover:bg-white/5'
              }`}
            >
              <Icon className={`w-4 h-4 shrink-0 ${selected ? 'text-white' : 'text-white/45'}`} />
              {label}
            </button>
          );
        })}
      </div>

      <div className="p-2 border-t border-white/10">
        <button
          type="button"
          onClick={() => onChange('control')}
          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
            active === 'control'
              ? 'bg-white/12 text-white font-medium'
              : 'text-white/55 hover:text-white hover:bg-white/5'
          }`}
        >
          <Settings className="w-4 h-4 shrink-0" />
          Control
        </button>
      </div>
    </nav>
  );
}
