import { AlertTriangle, Users, Activity, Gauge, Zap, Bell, Shield, Download } from 'lucide-react';
import { Alert, DashboardMetrics, OperatorMode } from '../api/client';
import { downloadDataUrl, slabScreenshotName } from '../utils/capturePreview';

interface CompactStatsProps {
  metrics: DashboardMetrics;
  alerts: Alert[];
  operatorMode: OperatorMode;
  slabId?: string;
}

function fmtCount(value: number | null): string {
  return value != null ? String(value) : '—';
}

function predictionClass(prediction: string | null): string {
  if (!prediction) return 'text-white/50 bg-white/5 border-white/10';
  const p = prediction.toLowerCase();
  if (p.includes('unsafe') || p.includes('armed')) {
    return 'text-red-300 bg-red-500/10 border-red-500/30';
  }
  if (p.includes('suspicious')) {
    return 'text-amber-300 bg-amber-500/10 border-amber-500/30';
  }
  return 'text-green-300 bg-green-500/10 border-green-500/30';
}

export function CompactStats({ metrics, alerts, operatorMode, slabId = 'slab' }: CompactStatsProps) {
  const recent = alerts.slice(0, 6);

  return (
    <div className="w-72 border-l border-white/10 bg-black/20 backdrop-blur-sm flex flex-col p-4 gap-3 overflow-y-auto">
      {!metrics.inferActive && (
        <div className="text-xs text-white/40 bg-white/5 border border-white/10 rounded-lg p-2.5">
          {metrics.metricsNote || 'Infer runner stopped — live metrics paused.'}
        </div>
      )}

      <div className="grid grid-cols-2 gap-2">
        <div className="bg-white/5 rounded-lg p-2.5 border border-white/10">
          <div className="flex items-center gap-1.5 mb-1">
            <Gauge className="w-3.5 h-3.5 text-cyan-300" />
            <div className="text-xs text-white/50">Infer FPS</div>
          </div>
          <div className="text-xl font-bold font-mono">
            {metrics.fps > 0 ? metrics.fps.toFixed(1) : '—'}
          </div>
        </div>
        <div className="bg-white/5 rounded-lg p-2.5 border border-white/10">
          <div className="flex items-center gap-1.5 mb-1">
            <Zap className="w-3.5 h-3.5 text-amber-300" />
            <div className="text-xs text-white/50">Frame ms</div>
          </div>
          <div className="text-xl font-bold font-mono">
            {metrics.latency > 0 ? (
              <>
                {metrics.latency.toFixed(0)}
                <span className="text-xs text-white/40">ms</span>
              </>
            ) : (
              '—'
            )}
          </div>
        </div>
      </div>

      <div className={`rounded-lg p-3 border ${predictionClass(metrics.prediction)}`}>
        <div className="flex items-center gap-2 mb-2">
          <Shield className="w-4 h-4" />
          <div className="text-xs uppercase tracking-wide opacity-80">Prediction</div>
        </div>
        <div className="text-lg font-semibold capitalize">{metrics.prediction ?? '—'}</div>
        <div className="text-xs opacity-70 mt-1">
          Gun: {metrics.gunDetected ? 'Yes' : metrics.inferActive ? 'No' : '—'}
        </div>
      </div>

      <div className="bg-white/5 rounded-lg p-3 border border-white/10">
        <div className="flex items-center gap-2 mb-2">
          <Activity className="w-4 h-4 text-white/60" />
          <div className="text-xs text-white/50">Unsafe %</div>
        </div>
        <div className="text-3xl font-bold">
          {metrics.unsafePct != null ? `${metrics.unsafePct}%` : '—'}
        </div>
      </div>

      <div className="bg-white/5 rounded-lg p-3 border border-white/10">
        <div className="flex items-center gap-2 mb-2">
          <Users className="w-4 h-4 text-white/60" />
          <div className="text-xs text-white/50">Persons</div>
        </div>
        <div className="text-2xl font-bold">{fmtCount(metrics.personDetections)}</div>
      </div>

      {metrics.gunDetected ? (
        <div className="bg-red-500/10 rounded-lg p-3 border border-red-500/30 animate-pulse">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <div className="text-xs text-red-400 font-medium">ARMED</div>
          </div>
          <div className="text-3xl font-bold text-red-400">
            {fmtCount(metrics.personsWithGun)}
          </div>
          <div className="text-xs text-red-300/70 mt-1">persons with gun</div>
        </div>
      ) : (
        <div className="bg-white/5 rounded-lg p-3 border border-white/10">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-white/60" />
            <div className="text-xs text-white/50">Armed persons</div>
          </div>
          <div className="text-2xl font-bold">{metrics.inferActive ? '0' : '—'}</div>
          <div className="text-xs text-white/40 mt-1">
            {metrics.inferActive ? 'All clear' : 'No live infer'}
          </div>
        </div>
      )}

      <div className="bg-white/5 rounded-lg p-3 border border-white/10">
        <div className="text-xs text-white/50 mb-1">Operator Mode</div>
        <div className="text-sm font-medium capitalize">{operatorMode}</div>
      </div>

      <div className="bg-white/5 rounded-lg p-3 border border-white/10">
        <div className="flex items-center gap-2 mb-2">
          <Bell className="w-3.5 h-3.5 text-white/60" />
          <div className="text-xs text-white/50 uppercase tracking-wide">Recent Alerts</div>
        </div>
        {recent.length === 0 ? (
          <div className="text-xs text-white/40">No recent alerts</div>
        ) : (
          <div className="space-y-2">
            {recent.map((a) => (
              <div
                key={a.id}
                className={`rounded-lg text-xs border overflow-hidden ${
                  a.severity === 'critical'
                    ? 'bg-red-500/10 border-red-500/30 text-red-100'
                    : a.severity === 'warning'
                      ? 'bg-amber-500/10 border-amber-500/30 text-amber-100'
                      : 'bg-white/5 border-white/10 text-white/70'
                }`}
              >
                {a.screenshotDataUrl && (
                  <img
                    src={a.screenshotDataUrl}
                    alt=""
                    className="w-full h-20 object-cover border-b border-white/10 bg-black"
                  />
                )}
                <div className="p-2 flex items-start gap-2">
                  <AlertTriangle className="w-3 h-3 mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    {a.personId != null ? (
                      <div className="font-medium">
                        Person <span className="font-mono text-red-200">{a.personId}</span>: had gun
                      </div>
                    ) : (
                      <div className="font-medium">{a.message}</div>
                    )}
                    <div className="text-[10px] opacity-60 mt-0.5">
                      {new Date(a.timestamp).toLocaleTimeString()} · {a.sensor}
                      {a.confidence > 0 && ` · ${(a.confidence * 100).toFixed(0)}%`}
                    </div>
                    {a.screenshotDataUrl && (
                      <button
                        type="button"
                        onClick={() =>
                          downloadDataUrl(
                            a.screenshotDataUrl!,
                            slabScreenshotName(slabId, `${a.sensor}-alert-p${a.personId ?? 'x'}`),
                          )
                        }
                        className="mt-1.5 flex items-center gap-1 text-[10px] text-cyan-300 hover:text-cyan-200"
                      >
                        <Download className="w-3 h-3" />
                        Save screenshot
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
