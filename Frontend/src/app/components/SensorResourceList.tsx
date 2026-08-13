import { ChevronRight } from 'lucide-react';
import { Device } from '../api/client';
import {
  NavSection,
  RUN_STATE_CLASS,
  RUN_STATE_LABEL,
  SensorKey,
  deviceHasSensor,
  navToSensor,
  sensorRunState,
} from './sensorStatus';

interface SensorResourceListProps {
  section: NavSection;
  devices: Device[];
  selectedDeviceId: string | null;
  backendOnline: boolean;
  sensorRunning: Record<SensorKey, boolean>;
  onSelect: (deviceId: string) => void;
  onDeviceContextMenu?: (device: Device) => void;
}

const SECTION_TITLE: Record<string, string> = {
  camera: 'Cameras',
  thermal: 'Thermal sensors',
  mmwave: 'mmWave sensors',
};

export function SensorResourceList({
  section,
  devices,
  selectedDeviceId,
  backendOnline,
  sensorRunning,
  onSelect,
  onDeviceContextMenu,
}: SensorResourceListProps) {
  const sensor = navToSensor(section);
  if (!sensor) return null;

  const filtered = devices.filter((d) => deviceHasSensor(d, sensor));
  const running = filtered.filter(
    (d) => sensorRunState(sensor, backendOnline, sensorRunning, d.status) === 'running',
  ).length;
  const idle = filtered.filter(
    (d) => sensorRunState(sensor, backendOnline, sensorRunning, d.status) === 'idle',
  ).length;
  const disconnected = filtered.filter(
    (d) => sensorRunState(sensor, backendOnline, sensorRunning, d.status) === 'disconnected',
  ).length;

  return (
    <div className="w-72 shrink-0 border-r border-white/10 bg-[#0a0d12] flex flex-col">
      <div className="px-4 py-4 border-b border-white/10">
        <h2 className="text-base font-semibold text-white">{SECTION_TITLE[section] ?? section}</h2>
        <p className="text-xs text-white/40 mt-1">
          {filtered.length} device{filtered.length !== 1 ? 's' : ''}
        </p>
        <div className="flex flex-wrap gap-1.5 mt-3">
          {running > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full border bg-emerald-500/10 text-emerald-300 border-emerald-500/25">
              {running} running
            </span>
          )}
          {idle > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full border bg-amber-500/10 text-amber-200 border-amber-500/25">
              {idle} idle
            </span>
          )}
          {disconnected > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full border bg-white/5 text-white/45 border-white/10">
              {disconnected} disconnected
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="p-4 text-sm text-white/40">No devices with this sensor.</div>
        ) : (
          filtered.map((device) => {
            const state = sensorRunState(sensor, backendOnline, sensorRunning, device.status);
            const selected = device.id === selectedDeviceId;
            return (
              <button
                key={device.id}
                type="button"
                onClick={() => onSelect(device.id)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  onDeviceContextMenu?.(device);
                }}
                className={`w-full text-left px-4 py-3 border-b border-white/5 transition-colors flex items-center gap-3 ${
                  selected ? 'bg-white/10' : 'hover:bg-white/5'
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-white truncate">{device.name}</div>
                  <div className="text-xs text-white/40 truncate mt-0.5">{device.location}</div>
                  <span
                    className={`inline-flex mt-2 text-[10px] font-medium uppercase tracking-wide px-2 py-0.5 rounded border ${RUN_STATE_CLASS[state]}`}
                  >
                    {RUN_STATE_LABEL[state]}
                  </span>
                </div>
                <ChevronRight className={`w-4 h-4 shrink-0 ${selected ? 'text-white' : 'text-white/25'}`} />
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
