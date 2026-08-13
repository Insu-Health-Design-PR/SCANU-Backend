import { Play, RotateCcw, Settings, Sliders, Square } from 'lucide-react';
import { useEffect, useState } from 'react';
import { DashboardMetrics, scanuClient } from '../api/client';
import { ScanuSelect } from './ScanuSelect';

interface ControlPanelProps {
  metrics: DashboardMetrics;
}

export function ControlPanel({ metrics }: ControlPanelProps) {
  const [profiles, setProfiles] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedProfile, setSelectedProfile] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void scanuClient.fetchProfiles().then(setProfiles);
  }, []);

  async function act(fn: () => Promise<void>) {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex-1 overflow-y-auto bg-slate-950 p-6">
      <div className="max-w-3xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-white">Control</h1>
          <p className="text-sm text-white/45 mt-1">Layer 8 sensor runners and model profiles</p>
        </div>

        <div className="bg-white/5 rounded-xl border border-white/10 p-6 space-y-4">
          <div className="flex items-center gap-3">
            <Settings className="w-5 h-5 text-white/60" />
            <h3 className="text-lg font-semibold">Sensor runners</h3>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <button
              disabled={busy}
              onClick={() => act(() => scanuClient.runAll())}
              className="flex items-center justify-center gap-2 px-4 py-3 bg-white/10 hover:bg-white/20 rounded-lg text-sm disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              Run all
            </button>
            <button
              disabled={busy}
              onClick={() => act(() => scanuClient.stopAll())}
              className="flex items-center justify-center gap-2 px-4 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg text-sm disabled:opacity-50"
            >
              <Square className="w-4 h-4" />
              Stop all
            </button>
            <button
              disabled={busy}
              onClick={() => act(() => scanuClient.restartAll())}
              className="flex items-center justify-center gap-2 px-4 py-3 bg-white/10 hover:bg-white/20 rounded-lg text-sm disabled:opacity-50"
            >
              <RotateCcw className="w-4 h-4" />
              Restart all
            </button>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {(['webcam', 'thermal', 'mmwave'] as const).map((s) => (
              <div key={s} className="bg-black/20 rounded-lg border border-white/5 p-3 space-y-2">
                <div className="text-xs text-white/50 uppercase">{s}</div>
                <button
                  disabled={busy}
                  onClick={() => act(() => scanuClient.runSensor('layer8-local', s))}
                  className="w-full px-3 py-2 text-sm bg-white/10 hover:bg-white/20 rounded disabled:opacity-50"
                >
                  Run
                </button>
                <button
                  disabled={busy}
                  onClick={() => act(() => scanuClient.stopSensor('layer8-local', s))}
                  className="w-full px-3 py-2 text-sm bg-red-500/15 hover:bg-red-500/25 rounded disabled:opacity-50"
                >
                  Stop
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white/5 rounded-xl border border-white/10 p-6 space-y-4">
          <div className="flex items-center gap-3">
            <Sliders className="w-5 h-5 text-white/60" />
            <h3 className="text-lg font-semibold">Model profile</h3>
          </div>
          <ScanuSelect
            label="Profile"
            value={selectedProfile}
            onValueChange={setSelectedProfile}
            options={profiles.map((p) => p.name)}
            placeholder="Select profile…"
            disabled={busy}
          />
          <button
            disabled={busy || !selectedProfile}
            onClick={() => act(() => scanuClient.applyProfileByName(selectedProfile))}
            className="w-full px-4 py-3 bg-white hover:bg-white/90 text-black font-medium rounded-lg disabled:opacity-40"
          >
            Apply profile
          </button>
        </div>

        <div className="bg-white/5 rounded-xl border border-white/10 p-6">
          <h3 className="text-lg font-semibold mb-4">Host metrics</h3>
          <div className="space-y-4">
            {[
              { label: 'CPU Usage', value: metrics.cpuUsage },
              { label: 'GPU Usage', value: metrics.gpuUsage },
              { label: 'Memory', value: metrics.memoryUsage },
            ].map((item) => (
              <div key={item.label}>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-white/70">{item.label}</span>
                  <span className="font-mono">{item.value.toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full bg-white/80 transition-all" style={{ width: `${item.value}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
