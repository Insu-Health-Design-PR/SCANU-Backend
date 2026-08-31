import { useEffect, useState } from 'react';
import { LAYER8 } from '../api/layer8';

interface LiveSummary {
  state: string;
  persons: number;
  points: number;
  alignmentMs: number | null;
  calibrationId: string;
}

const EMPTY: LiveSummary = {
  state: 'STOPPED',
  persons: 0,
  points: 0,
  alignmentMs: null,
  calibrationId: '',
};

export function PointCloudView() {
  const [summary, setSummary] = useState<LiveSummary>(EMPTY);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const [statusResponse, metricsResponse] = await Promise.all([
          fetch(LAYER8.mmwaveLiveStatus(), { cache: 'no-store' }),
          fetch(LAYER8.mmwaveLiveMetrics(), { cache: 'no-store' }),
        ]);
        const status = statusResponse.ok ? await statusResponse.json() : {};
        const metricsEnvelope = metricsResponse.ok ? await metricsResponse.json() : {};
        const metrics = metricsEnvelope?.metrics ?? {};
        if (!cancelled) {
          setSummary({
            state: String(status?.live?.state ?? metrics?.state ?? 'STOPPED'),
            persons: Number(metrics?.fused?.global_person_count ?? 0),
            points: Number(metrics?.fused?.point_count ?? metrics?.fused_points?.length ?? 0),
            alignmentMs:
              metrics?.quality?.alignment_error_ms === null || metrics?.quality?.alignment_error_ms === undefined
                ? null
                : Number(metrics.quality.alignment_error_ms),
            calibrationId: String(metrics?.calibration_id ?? status?.live?.calibration_id ?? ''),
          });
        }
      } catch {
        if (!cancelled) setSummary(EMPTY);
      }
    };
    void poll();
    const timer = window.setInterval(poll, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  return (
    <div className="relative flex h-full w-full flex-col bg-[#030b16]">
      <div className="flex items-center justify-between border-b border-white/10 px-5 py-3">
        <div>
          <h2 className="text-lg font-medium tracking-wide text-white">Calibrated Dual-mmWave Fusion</h2>
          <p className="text-xs text-white/45">Measured A+B post-CFAR returns in one world frame</p>
        </div>
        <span className="rounded bg-cyan-400/10 px-2 py-1 font-mono text-xs text-cyan-300">
          {summary.state}
        </span>
      </div>

      <div className="min-h-0 flex-1 bg-black">
        <img
          src={`${LAYER8.previewMmwave()}?side=fused`}
          alt="Live calibrated dual-mmWave fused dashboard"
          className="h-full w-full object-contain"
        />
      </div>

      <div className="grid grid-cols-4 gap-3 border-t border-white/10 bg-black/50 px-5 py-2 text-xs">
        <Metric label="Global persons" value={String(summary.persons)} />
        <Metric label="Fused points" value={String(summary.points)} />
        <Metric label="Alignment" value={summary.alignmentMs === null ? '—' : `${summary.alignmentMs.toFixed(1)} ms`} />
        <Metric label="Calibration" value={summary.calibrationId || '—'} />
      </div>
      <p className="px-5 py-1 text-[10px] text-amber-300/70">
        Experimental reflectivity evidence does not confirm material or classify a weapon.
      </p>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="truncate">
      <span className="text-white/45">{label} </span>
      <span className="font-mono text-white/90">{value}</span>
    </div>
  );
}
