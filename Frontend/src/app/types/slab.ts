import { DashboardMetrics, DashboardSnapshot, Device } from '../api/client';

export type SensorLinkState = 'disconnected' | 'idle' | 'live';

export type ThreatLevel = 'none' | 'sus' | 'medium' | 'high';

export interface SlabRegistryEntry {
  id: string;
  label: string;
  ip: string;
}

/** Known AI Slabs — extend via VITE_SLAB_REGISTRY JSON or backend later. */
export const SLAB_REGISTRY: SlabRegistryEntry[] = parseSlabRegistry();

function parseSlabRegistry(): SlabRegistryEntry[] {
  const raw = import.meta.env.VITE_SLAB_REGISTRY;
  if (raw && String(raw).trim()) {
    try {
      const parsed = JSON.parse(String(raw)) as SlabRegistryEntry[];
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    } catch {
      /* fall through */
    }
  }
  return [{ id: 'layer8-local', label: 'SLAB-001', ip: '127.0.0.1:8088' }];
}

export interface AiSlab {
  id: string;
  slabId: string;
  ip: string;
  threat: ThreatLevel;
  camera: SensorLinkState;
  thermal: SensorLinkState;
  mmwave: SensorLinkState;
  peopleDetected: number | null;
  weaponsDetected: number | null;
  alerts: number;
  online: boolean;
}

export function threatFromMetrics(metrics: DashboardMetrics): ThreatLevel {
  if (metrics.gunDetected) return 'high';
  const p = (metrics.prediction ?? '').toLowerCase();
  if (p.includes('armed') || p.includes('unsafe')) return 'high';
  if ((metrics.unsafePct ?? 0) >= 50) return 'medium';
  if (p.includes('suspicious') || p.includes('sus') || (metrics.unsafePct ?? 0) >= 15) return 'sus';
  return 'none';
}

export function sensorLinkState(
  backendOnline: boolean,
  slabOnline: boolean,
  running: boolean,
): SensorLinkState {
  if (!backendOnline || !slabOnline) return 'disconnected';
  return running ? 'live' : 'idle';
}

export function slabsFromSnapshot(snapshot: DashboardSnapshot): AiSlab[] {
  const { metrics, status, alerts, sensorRunning, devices } = snapshot;
  const deviceById = new Map(devices.map((d) => [d.id, d]));

  return SLAB_REGISTRY.map((reg) => {
    const device = deviceById.get(reg.id);
    const isLocal = reg.id === 'layer8-local';
    const backendOnline = status.backendOnline;
    const slabOnline = isLocal ? backendOnline && Boolean(device) : Boolean(device?.status === 'online');
    const slabAlerts = alerts.filter((a) => a.deviceId === reg.id).length;

    const liveMetrics = isLocal && backendOnline ? metrics : null;
    const liveRunning = isLocal && backendOnline ? sensorRunning : { webcam: false, thermal: false, mmwave: false };

    return {
      id: reg.id,
      slabId: reg.label,
      ip: reg.ip,
      threat: liveMetrics ? threatFromMetrics(liveMetrics) : 'none',
      camera: sensorLinkState(backendOnline && isLocal, slabOnline, liveRunning.webcam),
      thermal: sensorLinkState(backendOnline && isLocal, slabOnline, liveRunning.thermal),
      mmwave: sensorLinkState(backendOnline && isLocal, slabOnline, liveRunning.mmwave),
      peopleDetected: liveMetrics?.personDetections ?? null,
      weaponsDetected: liveMetrics
        ? liveMetrics.weaponDetections > 0
          ? liveMetrics.weaponDetections
          : liveMetrics.gunDetected
            ? liveMetrics.personsWithGun ?? 1
            : 0
        : null,
      alerts: isLocal ? status.activeAlerts || slabAlerts : slabAlerts,
      online: slabOnline,
    };
  });
}

export function slabToDevice(slabId: string, devices: Device[]): Device | undefined {
  return devices.find((d) => d.id === slabId);
}

export const THREAT_LABEL: Record<ThreatLevel, string> = {
  none: 'None',
  sus: 'Sus',
  medium: 'Medium',
  high: 'High',
};

export const THREAT_CLASS: Record<ThreatLevel, string> = {
  none: 'bg-white/5 text-white/50 border-white/10',
  sus: 'bg-amber-500/10 text-amber-200 border-amber-500/30',
  medium: 'bg-orange-500/10 text-orange-200 border-orange-500/35',
  high: 'bg-red-500/15 text-red-300 border-red-500/40',
};

export const LINK_LABEL: Record<SensorLinkState, string> = {
  disconnected: 'Disconnected',
  idle: 'Idle',
  live: 'Live',
};

export const LINK_CLASS: Record<SensorLinkState, string> = {
  disconnected: 'text-white/35',
  idle: 'text-amber-300/90',
  live: 'text-emerald-400',
};
