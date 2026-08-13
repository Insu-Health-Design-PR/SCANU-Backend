import { Activity, AlertTriangle, FlaskConical, LayoutGrid, Settings, Shield, WifiOff, RefreshCw } from 'lucide-react';
import { SystemStatus, OperatorMode } from '../api/client';

type AppPage = 'slabs' | 'slab-detail' | 'playground' | 'control';

interface HeaderProps {
  systemStatus: SystemStatus;
  operatorMode: OperatorMode;
  onOperatorModeChange: (mode: OperatorMode) => void;
  page: AppPage;
  onNavigate: (page: AppPage) => void;
  onBackToSlabs: () => void;
}

const MODES: OperatorMode[] = ['central', 'fallback', 'local'];

export function Header({
  systemStatus,
  operatorMode,
  onOperatorModeChange,
  page,
  onNavigate,
  onBackToSlabs,
}: HeaderProps) {
  const stateStyles = () => {
    switch (systemStatus.state) {
      case 'normal':
        return 'text-green-300 bg-green-500/10 border-green-500/30';
      case 'fallback':
        return 'text-amber-300 bg-amber-500/10 border-amber-500/30';
      case 'recovery':
        return 'text-cyan-300 bg-cyan-500/10 border-cyan-500/30 animate-pulse';
      case 'fault':
        return 'text-red-300 bg-red-500/10 border-red-500/30 animate-pulse';
      default:
        return 'text-white/50 bg-white/5 border-white/10';
    }
  };

  const stateIcon = () => {
    switch (systemStatus.state) {
      case 'normal':
        return <Shield className="w-4 h-4" />;
      case 'fault':
        return <AlertTriangle className="w-4 h-4" />;
      case 'recovery':
        return <RefreshCw className="w-4 h-4 animate-spin" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  const navBtn = (target: AppPage, label: string, icon: typeof LayoutGrid) => {
    const Icon = icon;
    const active = page === target || (target === 'slabs' && page === 'slab-detail');
    return (
      <button
        key={target}
        type="button"
        onClick={() => {
          if (target === 'slabs') onBackToSlabs();
          else onNavigate(target);
        }}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs transition-all ${
          active ? 'bg-white/15 text-white' : 'text-white/50 hover:text-white hover:bg-white/5'
        }`}
      >
        <Icon className="w-3.5 h-3.5" />
        {label}
      </button>
    );
  };

  return (
    <header className="h-14 border-b border-white/10 bg-[#141820] flex items-center justify-between px-6 gap-4 shrink-0">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-white/10 rounded-md flex items-center justify-center border border-white/20">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-lg font-semibold">SCAN-U</h1>
        </div>

        <div className="hidden sm:flex items-center gap-1 p-1 bg-white/[0.04] rounded-lg border border-white/10">
          {navBtn('slabs', 'Slabs', LayoutGrid)}
          {navBtn('playground', 'Playground', FlaskConical)}
          {navBtn('control', 'Control', Settings)}
        </div>

        <div className={`flex items-center gap-2 px-3 py-1 rounded-md border ${stateStyles()}`}>
          {stateIcon()}
          <span className="text-xs font-semibold uppercase tracking-wider">{systemStatus.state}</span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1 p-1 bg-white/5 rounded-md border border-white/10">
          <span className="text-xs text-white/40 px-2">Mode</span>
          {MODES.map((m) => (
            <button
              key={m}
              onClick={() => onOperatorModeChange(m)}
              className={`px-2.5 py-1 rounded text-xs uppercase tracking-wide transition-all ${
                operatorMode === m
                  ? 'bg-white text-black font-semibold'
                  : 'text-white/60 hover:text-white'
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        {!systemStatus.backendOnline && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-500/10 text-amber-300 border border-amber-500/30 animate-pulse">
            <WifiOff className="w-3.5 h-3.5" />
            <span className="text-xs font-medium">Waiting for Backend</span>
          </div>
        )}

        {systemStatus.activeAlerts > 0 && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-red-500/10 text-red-300 border border-red-500/30 animate-pulse">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span className="text-xs font-medium">
              {systemStatus.activeAlerts} alert{systemStatus.activeAlerts > 1 ? 's' : ''}
            </span>
          </div>
        )}
      </div>
    </header>
  );
}
