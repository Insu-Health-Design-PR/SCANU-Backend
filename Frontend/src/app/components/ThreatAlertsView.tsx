import { AlertTriangle, Bell, Download } from 'lucide-react';
import { Alert } from '../api/client';
import { downloadDataUrl, slabScreenshotName } from '../utils/capturePreview';

interface ThreatAlertsViewProps {
  alerts: Alert[];
  slabId?: string;
}

function cameraLabel(sensor: Alert['sensor']): string {
  if (sensor === 'multi_camera') return 'Back Camera';
  if (sensor === 'webcam') return 'Front Camera';
  return sensor;
}

export function ThreatAlertsView({ alerts, slabId = 'threat-monitor' }: ThreatAlertsViewProps) {
  return (
    <div className="flex-1 overflow-auto bg-[#0f1117]">
      <div className="max-w-[1200px] mx-auto px-6 py-8">
        <div className="mb-8 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center justify-center">
            <Bell className="w-5 h-5 text-red-300" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold text-white tracking-tight">Alerts</h1>
            <p className="text-sm text-white/45 mt-0.5">
              Unsafe detections from Front and Back cameras — screenshots captured automatically
            </p>
          </div>
          {alerts.length > 0 && (
            <span className="ml-auto px-3 py-1 rounded-full bg-red-500/15 text-red-300 text-sm font-medium border border-red-500/30">
              {alerts.length} active
            </span>
          )}
        </div>

        {alerts.length === 0 ? (
          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-12 text-center">
            <AlertTriangle className="w-10 h-10 text-white/20 mx-auto mb-3" />
            <p className="text-white/60">No alerts yet</p>
            <p className="text-sm text-white/35 mt-1">
              Alerts appear here when a person is flagged unsafe on either camera
            </p>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {alerts.map((a) => (
              <article
                key={a.id}
                className={`rounded-xl border overflow-hidden ${
                  a.severity === 'critical'
                    ? 'bg-red-500/10 border-red-500/30'
                    : a.severity === 'warning'
                      ? 'bg-amber-500/10 border-amber-500/30'
                      : 'bg-white/5 border-white/10'
                }`}
              >
                {a.screenshotDataUrl ? (
                  <img
                    src={a.screenshotDataUrl}
                    alt=""
                    className="w-full h-40 object-cover bg-black border-b border-white/10"
                  />
                ) : (
                  <div className="w-full h-40 bg-black/60 flex items-center justify-center border-b border-white/10">
                    <span className="text-xs text-white/35">Capturing screenshot…</span>
                  </div>
                )}
                <div className="p-4">
                  <div className="flex items-start gap-2">
                    <AlertTriangle
                      className={`w-4 h-4 mt-0.5 shrink-0 ${
                        a.severity === 'critical' ? 'text-red-400' : 'text-amber-400'
                      }`}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-semibold text-white leading-snug">{a.message}</div>
                      <div className="text-xs text-white/45 mt-2 space-y-0.5">
                        <div>{cameraLabel(a.sensor)}</div>
                        <div>{new Date(a.timestamp).toLocaleString()}</div>
                        {a.personId != null && (
                          <div>
                            Person ID: <span className="font-mono text-white/70">{a.personId}</span>
                          </div>
                        )}
                        {a.confidence > 0 && (
                          <div>Confidence: {(a.confidence * 100).toFixed(0)}%</div>
                        )}
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
                          className="mt-3 flex items-center gap-1.5 text-xs text-cyan-300 hover:text-cyan-200"
                        >
                          <Download className="w-3.5 h-3.5" />
                          Download screenshot
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
