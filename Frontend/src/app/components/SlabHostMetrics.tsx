import { Cpu, HardDrive, Microchip } from 'lucide-react';
import { DashboardMetrics } from '../api/client';

function HostMetricChip({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: number;
  icon: typeof Cpu;
}) {
  const pct = Math.min(100, Math.max(0, value));
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/10 min-w-[100px]">
      <Icon className="w-3.5 h-3.5 text-white/45 shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2 text-[10px] text-white/45 uppercase tracking-wide">
          <span>{label}</span>
          <span className="font-mono text-white/80 normal-case">{pct.toFixed(0)}%</span>
        </div>
        <div className="h-1 mt-1 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full bg-cyan-400/80 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  );
}

export function SlabHostMetrics({ metrics }: { metrics: DashboardMetrics }) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <HostMetricChip label="CPU" value={metrics.cpuUsage} icon={Cpu} />
      <HostMetricChip label="GPU" value={metrics.gpuUsage} icon={Microchip} />
      <HostMetricChip label="RAM" value={metrics.memoryUsage} icon={HardDrive} />
    </div>
  );
}
