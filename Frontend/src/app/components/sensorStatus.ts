import { Device } from '../api/client';

export type NavSection = 'camera' | 'thermal' | 'mmwave' | 'metrics' | 'playground' | 'control';

export type SensorKey = 'webcam' | 'thermal' | 'mmwave';

export type SensorRunState = 'running' | 'idle' | 'disconnected';

export function navToSensor(section: NavSection): SensorKey | null {
  if (section === 'camera') return 'webcam';
  if (section === 'thermal') return 'thermal';
  if (section === 'mmwave') return 'mmwave';
  return null;
}

export function navToCameraView(section: NavSection): 'rgb' | 'thermal' | 'mmwave' {
  if (section === 'thermal') return 'thermal';
  if (section === 'mmwave') return 'mmwave';
  return 'rgb';
}

export function sensorRunState(
  sensor: SensorKey,
  backendOnline: boolean,
  sensorRunning: Record<SensorKey, boolean>,
  deviceStatus?: Device['status'],
): SensorRunState {
  if (!backendOnline || deviceStatus === 'offline') return 'disconnected';
  return sensorRunning[sensor] ? 'running' : 'idle';
}

export function deviceHasSensor(device: Device, sensor: SensorKey): boolean {
  if (sensor === 'webcam') return device.cameras.webcam;
  if (sensor === 'thermal') return device.cameras.thermal;
  return device.mmwave;
}

export const RUN_STATE_LABEL: Record<SensorRunState, string> = {
  running: 'Running',
  idle: 'Idle',
  disconnected: 'Disconnected',
};

export const RUN_STATE_CLASS: Record<SensorRunState, string> = {
  running: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  idle: 'bg-amber-500/10 text-amber-200 border-amber-500/25',
  disconnected: 'bg-white/5 text-white/40 border-white/10',
};
