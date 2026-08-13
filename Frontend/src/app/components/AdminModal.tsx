import { X, Settings, Sliders, Play, Square, RotateCcw } from 'lucide-react';
import { useEffect, useState } from 'react';
import { DashboardMetrics, scanuClient } from '../api/client';
import { ModelPlaygroundPanel } from './ModelPlaygroundPanel';

interface AdminModalProps {
  metrics: DashboardMetrics;
  onClose: () => void;
}

export function AdminModal({ metrics, onClose }: AdminModalProps) {
  const [profiles, setProfiles] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedProfile, setSelectedProfile] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void scanuClient.fetchProfiles().then(setProfiles);
  }, []);

  const profileOptions = profiles;

  async function act(fn: () => Promise<void>) {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-6">
      <div className="bg-slate-900 rounded-lg border border-white/20 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-slate-900 border-b border-white/10 p-6 flex items-center justify-between">
          <h2 className="text-2xl font-semibold">Layer 8 control</h2>
          <button onClick={onClose} className="p-2 hover:bg-white/10 rounded transition-all">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          <div className="bg-white/5 rounded-lg border border-white/10 p-6">
            <div className="flex items-center gap-3 mb-6">
              <Settings className="w-5 h-5 text-white/60" />
              <h3 className="text-lg font-semibold">Sensor runners</h3>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <button
                disabled={busy}
                onClick={() => act(() => scanuClient.runAll())}
                className="flex items-center justify-center gap-2 px-4 py-3 bg-white/10 hover:bg-white/20 rounded transition-all disabled:opacity-50"
              >
                <Play className="w-4 h-4" />
                Run all
              </button>
              <button
                disabled={busy}
                onClick={() => act(() => scanuClient.stopAll())}
                className="flex items-center justify-center gap-2 px-4 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded transition-all disabled:opacity-50"
              >
                <Square className="w-4 h-4" />
                Stop all
              </button>
              <button
                disabled={busy}
                onClick={() => act(() => scanuClient.restartAll())}
                className="flex items-center justify-center gap-2 px-4 py-3 bg-white/10 hover:bg-white/20 rounded transition-all disabled:opacity-50"
              >
                <RotateCcw className="w-4 h-4" />
                Restart all
              </button>
            </div>
            <div className="grid grid-cols-3 gap-3 mt-3">
              {(['webcam', 'thermal', 'mmwave'] as const).map((s) => (
                <div key={s} className="flex flex-col gap-2">
                  <div className="text-xs text-white/50 uppercase">{s}</div>
                  <button
                    disabled={busy}
                    onClick={() => act(() => scanuClient.runSensor('layer8-local', s))}
                    className="px-3 py-2 text-sm bg-white/10 hover:bg-white/20 rounded disabled:opacity-50"
                  >
                    Run
                  </button>
                  <button
                    disabled={busy}
                    onClick={() => act(() => scanuClient.stopSensor('layer8-local', s))}
                    className="px-3 py-2 text-sm bg-red-500/15 hover:bg-red-500/25 rounded disabled:opacity-50"
                  >
                    Stop
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white/5 rounded-lg border border-white/10 p-6">
            <div className="flex items-center gap-3 mb-6">
              <Sliders className="w-5 h-5 text-white/60" />
              <h3 className="text-lg font-semibold">Model profile</h3>
            </div>
            <select
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value)}
              className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded text-white focus:outline-none focus:border-white/40"
            >
              <option value="">Select profile…</option>
              {profileOptions.map((p) => (
                <option key={p.id} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
            <button
              disabled={busy || !selectedProfile}
              onClick={() => act(() => scanuClient.applyProfileByName(selectedProfile))}
              className="mt-4 w-full px-4 py-3 bg-white hover:bg-white/90 text-black font-medium rounded transition-all disabled:opacity-40"
            >
              Apply profile
            </button>
            <p className="text-xs text-white/40 mt-2">
              POST /api/ai_camera/profiles/apply_by_name
            </p>
          </div>

          <ModelPlaygroundPanel disabled={busy} />

          <div className="bg-white/5 rounded-lg border border-white/10 p-6">
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
                    <div
                      className="h-full bg-white transition-all"
                      style={{ width: `${item.value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
