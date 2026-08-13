import { Activity, Cpu, HardDrive, Microchip, Shield, Zap } from 'lucide-react';
import { Alert, DashboardMetrics, OperatorMode } from '../api/client';

interface MetricsViewProps {
  metrics: DashboardMetrics;
  alerts: Alert[];
  operatorMode: OperatorMode;
  sensorRunning: { webcam: boolean; thermal: boolean; mmwave: boolean };
  backendOnline: boolean;
}

function Bar({ label, value, icon: Icon }: { label: string; value: number; icon: typeof Cpu }) {
  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-white/70">
          <Icon className="w-4 h-4" />
          <span className="text-sm">{label}</span>
        </div>
        <span className="font-mono text-lg font-semibold">{value.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-cyan-500/80 to-blue-400/80 transition-all duration-500"
          style={{ width: `${Math.min(100, value)}%` }}
        />
      </div>
    </div>
  );
}

export function MetricsView({
  metrics,
  alerts,
  operatorMode,
  sensorRunning,
  backendOnline,
}: MetricsViewProps) {
  const sensors = [
    { key: 'webcam', label: 'Camera', running: sensorRunning.webcam },
    { key: 'thermal', label: 'Thermal', running: sensorRunning.thermal },
    { key: 'mmwave', label: 'mmWave', running: sensorRunning.mmwave },
  ] as const;

  return (
    <div className="flex-1 overflow-y-auto bg-slate-950 p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-white">System metrics</h1>
          <p className="text-sm text-white/45 mt-1">Host resources and inference summary</p>
        </div>

        <div className="grid sm:grid-cols-3 gap-4">
          <Bar label="CPU" value={metrics.cpuUsage} icon={Cpu} />
          <Bar label="GPU" value={metrics.gpuUsage} icon={Microchip} />
          <Bar label="Memory" value={metrics.memoryUsage} icon={HardDrive} />
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { label: 'Infer FPS', value: metrics.fps > 0 ? metrics.fps.toFixed(1) : '—', icon: Zap },
            {
              label: 'Prediction',
              value: metrics.prediction ?? '—',
              icon: Shield,
            },
            {
              label: 'Unsafe %',
              value: metrics.unsafePct != null ? `${metrics.unsafePct}%` : '—',
              icon: Activity,
            },
            { label: 'Operator', value: operatorMode, icon: Activity },
          ].map((item) => (
            <div key={item.label} className="bg-white/5 rounded-xl border border-white/10 p-4">
              <div className="flex items-center gap-2 text-white/50 text-xs mb-2">
                <item.icon className="w-3.5 h-3.5" />
                {item.label}
              </div>
              <div className="text-xl font-semibold capitalize">{item.value}</div>
            </div>
          ))}
        </div>

        <div className="bg-white/5 rounded-xl border border-white/10 p-5">
          <h3 className="text-sm font-medium text-white/70 mb-4">Sensor runners</h3>
          <div className="grid sm:grid-cols-3 gap-3">
            {sensors.map((s) => {
              const state = !backendOnline ? 'disconnected' : s.running ? 'running' : 'idle';
              const cls =
                state === 'running'
                  ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                  : state === 'idle'
                    ? 'border-amber-500/25 bg-amber-500/10 text-amber-200'
                    : 'border-white/10 bg-white/5 text-white/40';
              return (
                <div key={s.key} className={`rounded-lg border p-4 ${cls}`}>
                  <div className="text-sm font-medium">{s.label}</div>
                  <div className="text-xs uppercase tracking-wide mt-1 opacity-80">{state}</div>
                </div>
              );
            })}
          </div>
        </div>

        {alerts.length > 0 && (
          <div className="bg-white/5 rounded-xl border border-white/10 p-5">
            <h3 className="text-sm font-medium text-white/70 mb-3">Recent alerts ({alerts.length})</h3>
            <div className="space-y-2">
              {alerts.slice(0, 6).map((a) => (
                <div key={a.id} className="text-sm text-white/70 border-b border-white/5 pb-2 last:border-0">
                  {a.message}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
