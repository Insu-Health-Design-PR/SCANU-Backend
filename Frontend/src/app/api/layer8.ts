/**
 * Layer 8 backend routes (SCANU-backend/software/layer8_ui/dashboard_routes.py).
 * Frontend-only — do not add routes on the backend; map UI calls here.
 */

/** Empty string = same-origin (Vite dev proxy → :8088). */
export function layer8Base(): string {
  if (typeof window !== 'undefined' && (window as unknown as { __SCANU_REST_BASE__?: string }).__SCANU_REST_BASE__) {
    return String((window as unknown as { __SCANU_REST_BASE__: string }).__SCANU_REST_BASE__).replace(/\/$/, '');
  }
  const fromEnv = import.meta.env.VITE_LAYER8_API_BASE;
  if (fromEnv !== undefined && fromEnv !== null && String(fromEnv).trim() !== '') {
    return String(fromEnv).replace(/\/$/, '');
  }
  return '';
}

export function layer8Url(path: string): string {
  const base = layer8Base();
  const p = path.startsWith('/') ? path : `/${path}`;
  return base ? `${base}${p}` : p;
}

/** Local host device id used when talking to a single Layer 8 instance. */
export const LAYER8_LOCAL_DEVICE_ID = 'layer8-local';

export const LAYER8 = {
  status: () => layer8Url('/api/status'),
  statusStream: () => layer8Url('/api/status/stream'),
  statusSensor: (sensor: 'thermal' | 'webcam' | 'mmwave') => layer8Url(`/api/status/${sensor}`),

  systemMetrics: () => layer8Url('/api/system/metrics'),
  dashboardMetrics: () => layer8Url('/api/dashboard/metrics'),
  threatMetrics: () => layer8Url('/api/threat/metrics'),

  config: () => layer8Url('/api/config'),
  thermalConfig: () => layer8Url('/api/thermal/config'),
  aiCameraConfig: () => layer8Url('/api/ai_camera/config'),
  mmwaveConfig: () => layer8Url('/api/mmwave/config'),

  runSensor: (sensor: 'thermal' | 'webcam' | 'mmwave') => layer8Url(`/api/run/${sensor}`),
  stopSensor: (sensor: 'thermal' | 'webcam' | 'mmwave') => layer8Url(`/api/stop/${sensor}`),
  restartSensor: (sensor: 'thermal' | 'webcam' | 'mmwave') => layer8Url(`/api/restart/${sensor}`),
  aiCameraRun: () => layer8Url('/api/ai_camera/run'),
  aiCameraStop: () => layer8Url('/api/ai_camera/stop'),
  aiCameraRestart: () => layer8Url('/api/ai_camera/restart'),

  runAll: () => layer8Url('/api/run_all'),
  stopAll: () => layer8Url('/api/stop_all'),
  restartAll: () => layer8Url('/api/restart_all'),

  previewWebcam: () => layer8Url('/api/ai_camera/preview/live'),
  previewThermal: () => layer8Url('/api/thermal/preview/live'),
  previewMmwave: () => layer8Url('/api/mmwave/preview/live'),

  profiles: () => layer8Url('/api/ai_camera/profiles'),
  applyProfileByName: () => layer8Url('/api/ai_camera/profiles/apply_by_name'),
  saveProfileIfNew: () => layer8Url('/api/model/profiles/save_if_new'),
  modelOptions: () => layer8Url('/api/model/options'),
  playgroundInfer: () => layer8Url('/api/playground/infer_image'),
  playgroundSample: () => layer8Url('/api/playground/sample'),

  v4l2Devices: () => layer8Url('/api/devices/v4l2'),
  serialPorts: () => layer8Url('/api/devices/serial'),
  thermalAutoConfigure: () => layer8Url('/api/thermal/auto_configure'),
  mmwaveAutoConfigure: () => layer8Url('/api/mmwave/auto_configure'),
} as const;

export type Layer8SensorStatus = {
  running: boolean;
  pid: number;
  log_tail: string;
  log_file: string;
};

export type Layer8AllStatus = {
  thermal: Layer8SensorStatus;
  webcam: Layer8SensorStatus;
  mmwave: Layer8SensorStatus;
};

/** Runner-aware summary — same payload as Layer 8 HTML dashboard (`GET /api/dashboard/metrics`). */
export type Layer8DashboardMetrics = {
  unsafe_pct?: number | null;
  unsafe_score?: number | null;
  gun_detected?: boolean | null;
  object_gun_peak?: number | null;
  weapon_gun_peak?: number | null;
  persons_with_gun?: number | null;
  persons_total?: number | null;
  prediction?: string | null;
  mmwave_torso_score?: number | null;
  frame?: number | null;
  ts?: number | null;
  note?: string;
};

export type Layer8ThreatMetrics = {
  ts?: number;
  frame?: number;
  unsafe_score?: number;
  unsafe_pct?: number;
  gun_detected?: boolean;
  object_gun_peak?: number;
  weapon_gun_peak?: number;
  persons_with_gun?: number;
  persons_total?: number;
  prediction?: string;
  byte_tracks?: Array<{
    track_id: number;
    display_id: number;
    row: number;
    threat: number;
    object_gun_conf?: number;
    weapon_gun_conf?: number;
    bucket: string;
  }>;
  firearm_tracks?: Array<{
    track_id: number;
    stable_sid: number;
    display_tag: string | null;
    conf: number;
    kind: string;
  }>;
};

export type Layer8SystemMetrics = {
  cpu_percent: number | null;
  load_1m: number | null;
  mem: { used_mb: number; total_mb: number; percent: number } | null;
  gpu: { util_percent: number; mem_used_mb: number; mem_total_mb: number } | null;
};

/** UI sensor name → Layer 8 subprocess id. */
export const UI_SENSOR_TO_LAYER8: Record<'webcam' | 'thermal' | 'mmwave', 'webcam' | 'thermal' | 'mmwave'> = {
  webcam: 'webcam',
  thermal: 'thermal',
  mmwave: 'mmwave',
};
