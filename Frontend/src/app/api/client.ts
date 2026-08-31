// SCAN-U client — Layer 8 REST + SSE only (see layer8.ts).

import {
  LAYER8,
  LAYER8_LOCAL_DEVICE_ID,
  Layer8AllStatus,
  Layer8DashboardMetrics,
  Layer8SystemMetrics,
  Layer8ThreatMetrics,
  UI_SENSOR_TO_LAYER8,
  layer8Url,
} from './layer8';
import { capturePreviewFrame } from '../utils/capturePreview';

export type OperatorMode = 'central' | 'fallback' | 'local';
export type SystemState = 'normal' | 'fallback' | 'recovery' | 'fault';
export type SensorType = 'webcam' | 'thermal' | 'mmwave' | 'multi_camera';
export type CameraSensor = 'webcam' | 'multi_camera';
export type AlertSeverity = 'info' | 'warning' | 'critical';

export interface Device {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'degraded';
  cameras: { webcam: boolean; thermal: boolean };
  mmwave: boolean;
  lastHeartbeat: string;
  fps: number;
  latencyMs: number;
  temperature?: number;
}

export interface SystemStatus {
  mode: OperatorMode;
  state: SystemState;
  backendOnline: boolean;
  activeAlerts: number;
  timestamp: string;
}

export interface DashboardMetrics {
  /** True when thermal or AI Camera infer runner is active. */
  inferActive: boolean;
  gunDetected: boolean;
  prediction: string | null;
  unsafePct: number | null;
  unsafeScore: number | null;
  personsWithGun: number | null;
  personDetections: number | null;
  weaponDetections: number;
  totalDetections: number | null;
  frame: number | null;
  metricsNote: string;
  fps: number;
  latency: number;
  cpuUsage: number;
  gpuUsage: number;
  memoryUsage: number;
}

export interface Alert {
  id: string;
  severity: AlertSeverity;
  type: string;
  message: string;
  deviceId: string;
  sensor: SensorType;
  confidence: number;
  timestamp: string;
  acknowledged: boolean;
  /** ByteTrack display id when alert is person-specific. */
  personId?: number;
  /** JPEG data URL captured at alert time or via Screenshot. */
  screenshotDataUrl?: string;
}

/** Overlay boxes — only populated when Layer 8 sends normalized bbox (not yet). */
export interface Detection {
  id: string;
  type: 'person' | 'weapon';
  confidence: number;
  bbox: [number, number, number, number];
  trackingId: string;
  unsafe: boolean;
  trail: Array<{ x: number; y: number }>;
  timestamp: string;
}

export type PlaygroundParams = {
  conf: number;
  gunConf: number;
  unsafeThreshold: number;
  roiPadFrac: number;
  roiPadPx: number;
  gunImgsz: number;
  gunEmitMin: number;
  personModel: string;
  gunModel: string;
};

export type ModelOptions = {
  gun_checkpoints: string[];
  person_yolo_suggestions: string[];
  person_yolo_options: string[];
};

export type PlaygroundSummary = {
  prediction?: string | null;
  unsafe_pct?: number | null;
  gun_detected?: boolean | null;
  persons_total?: number | null;
  persons_with_gun?: number | null;
  firearms?: Array<{ kind: string; conf: number }>;
};

export interface DashboardSnapshot {
  devices: Device[];
  metrics: DashboardMetrics;
  status: SystemStatus;
  alerts: Alert[];
  detections: Record<string, Detection[]>;
  sensorLogs: Record<'thermal' | 'webcam' | 'mmwave', string>;
  sensorRunning: Record<'thermal' | 'webcam' | 'mmwave' | 'multi_camera', boolean>;
}

type Listener = (snap: DashboardSnapshot) => void;

/** Layer 8 threat JSON has track ids but no normalized bbox — do not fabricate boxes. */
function detectionsFromThreat(_threat: Layer8ThreatMetrics | null): Detection[] {
  return [];
}

function inferRunnerActive(status: Layer8AllStatus | null): boolean {
  return Boolean(
    status?.webcam?.running || status?.thermal?.running || status?.multi_camera?.running,
  );
}

function predictionLooksUnsafe(prediction: string | null | undefined): boolean {
  if (!prediction) return false;
  const p = prediction.toLowerCase();
  return (
    p.includes('unsafe') ||
    p.includes('armed') ||
    p.includes('concealed') ||
    p.includes('suspicious')
  );
}

function isUnsafeThreat(threat: Layer8ThreatMetrics | null): boolean {
  if (!threat) return false;
  if (threat.gun_detected === true) return true;
  if ((threat.persons_with_gun ?? 0) > 0) return true;

  const tracks = threat.byte_tracks ?? [];
  const unsafeTracks = tracks.filter((t) => {
    const weapon = t.weapon_gun_conf ?? 0;
    const object = t.object_gun_conf ?? 0;
    return weapon > 0.05 || object > 0.17 || t.bucket === 'unsafe' || t.threat >= 0.55;
  });
  if (unsafeTracks.length > 0) return true;

  const persons = threat.persons_total ?? 0;
  if (persons <= 0 && tracks.length === 0) return false;

  const unsafeScore = Number(threat.unsafe_score ?? 0);
  const unsafePct = Number(threat.unsafe_pct ?? 0);
  if (unsafeScore >= 0.55 || unsafePct >= 55) return true;
  if (predictionLooksUnsafe(threat.prediction)) return true;

  return false;
}

function metricsFromDashboard(
  dash: Layer8DashboardMetrics | null,
  sys: Layer8SystemMetrics | null,
  inferActive: boolean,
  fps: number,
): DashboardMetrics {
  const host = {
    fps,
    latency: fps > 0 ? 1000 / fps : 0,
    cpuUsage: Number(sys?.cpu_percent ?? 0),
    gpuUsage: Number(sys?.gpu?.util_percent ?? 0),
    memoryUsage: Number(sys?.mem?.percent ?? 0),
  };

  if (!inferActive || !dash) {
    return {
      inferActive: false,
      gunDetected: false,
      prediction: null,
      unsafePct: null,
      unsafeScore: null,
      personsWithGun: null,
      personDetections: null,
      weaponDetections: 0,
      totalDetections: null,
      frame: null,
      metricsNote: dash?.note || 'Start AI Camera or Thermal runner for live metrics.',
      ...host,
    };
  }

  const persons = dash.persons_total ?? null;
  const gunDetected = dash.gun_detected === true;
  const armed = gunDetected ? (dash.persons_with_gun ?? 0) : 0;
  const prediction =
    dash.prediction != null && String(dash.prediction).trim() !== ''
      ? String(dash.prediction)
      : null;

  return {
    inferActive: true,
    gunDetected,
    prediction,
    unsafePct: dash.unsafe_pct ?? null,
    unsafeScore: dash.unsafe_score ?? null,
    personsWithGun: dash.persons_with_gun ?? null,
    personDetections: persons,
    weaponDetections: armed,
    totalDetections: persons != null ? persons : null,
    frame: dash.frame ?? null,
    metricsNote: dash.note || '',
    ...host,
  };
}

function alertsFromCameraThreat(
  threat: Layer8ThreatMetrics | null,
  sensor: CameraSensor,
  running: boolean,
  cameraLabel: string,
): Alert[] {
  if (!running || !threat || !isUnsafeThreat(threat)) {
    return [];
  }

  const frame = threat.frame ?? Date.now();
  const prefix = sensor === 'webcam' ? 'front' : 'back';
  const tracks = threat.byte_tracks ?? [];

  const unsafeTracks = tracks.filter((t) => {
    const weapon = t.weapon_gun_conf ?? 0;
    const object = t.object_gun_conf ?? 0;
    return weapon > 0.05 || object > 0.17 || t.bucket === 'unsafe' || t.threat >= 0.55;
  });

  if (unsafeTracks.length > 0) {
    return unsafeTracks.map((t) => ({
      id: `alert-${prefix}-f${frame}-p${t.display_id}`,
      severity: 'critical' as const,
      type: 'person_unsafe',
      message: `${cameraLabel}: Person ${t.display_id} — unsafe`,
      personId: t.display_id,
      deviceId: LAYER8_LOCAL_DEVICE_ID,
      sensor,
      confidence: Math.max(t.weapon_gun_conf ?? 0, t.object_gun_conf ?? 0, t.threat),
      timestamp: new Date().toISOString(),
      acknowledged: false,
    }));
  }

  const persons = threat.persons_total ?? tracks.length;
  if (persons > 0) {
    const pred = threat.prediction?.trim();
    return [
      {
        id: `alert-${prefix}-f${frame}`,
        severity: 'critical',
        type: 'threat_detected',
        message: `${cameraLabel}: ${pred || 'Unsafe threat detected'} (${persons} person${persons > 1 ? 's' : ''})`,
        deviceId: LAYER8_LOCAL_DEVICE_ID,
        sensor,
        confidence: Number(threat.unsafe_score ?? threat.weapon_gun_peak ?? 0.5),
        timestamp: new Date().toISOString(),
        acknowledged: false,
      },
    ];
  }

  return [];
}

function deviceFromStatus(status: Layer8AllStatus | null): Device {
  const webcamOn = Boolean(status?.webcam?.running);
  const thermalOn = Boolean(status?.thermal?.running);
  const mmwaveOn = Boolean(status?.mmwave?.running);
  const anyRunning = webcamOn || thermalOn || mmwaveOn;

  return {
    id: LAYER8_LOCAL_DEVICE_ID,
    name: 'Threat Monitor',
    location: 'localhost:8088',
    status: anyRunning ? 'online' : status ? 'degraded' : 'offline',
    cameras: { webcam: true, thermal: true },
    mmwave: true,
    lastHeartbeat: new Date().toISOString(),
    fps: 0,
    latencyMs: 0,
  };
}

function sensorRunningFromStatus(
  status: Layer8AllStatus | null,
): Record<'thermal' | 'webcam' | 'mmwave' | 'multi_camera', boolean> {
  return {
    thermal: Boolean(status?.thermal?.running),
    webcam: Boolean(status?.webcam?.running),
    multi_camera: Boolean(status?.multi_camera?.running),
    mmwave: Boolean(status?.mmwave?.running),
  };
}

class ScanUClient {
  private listeners = new Set<Listener>();
  private eventSource: EventSource | null = null;
  private pollTimer: ReturnType<typeof setInterval> | null = null;
  private backendReachable = false;
  private currentOperatorMode: OperatorMode = 'central';
  private lastSensorStatus: Layer8AllStatus | null = null;
  private lastFrameSample: { frame: number; ts: number } | null = null;
  private alertHistory: Alert[] = [];
  private seenAlertIds = new Set<string>();
  private metricsRefreshInFlight = false;
  private screenshotInFlight = false;
  private screenshotQueue: Array<{ alertId: string; sensor: CameraSensor }> = [];

  private snapshot: DashboardSnapshot = {
    devices: [],
    metrics: this.emptyMetrics(),
    status: {
      mode: 'central',
      state: 'fault',
      backendOnline: false,
      activeAlerts: 0,
      timestamp: new Date().toISOString(),
    },
    alerts: [],
    detections: {},
    sensorLogs: { thermal: '', webcam: '', mmwave: '' },
    sensorRunning: { thermal: false, webcam: false, mmwave: false, multi_camera: false },
  };

  constructor() {
    void this.refreshMetrics();
    this.connectStatusStream();
    this.pollTimer = setInterval(() => void this.refreshMetrics(), 2500);
  }

  isConnected() {
    return this.backendReachable;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    fn(this.snapshot);
    return () => this.listeners.delete(fn);
  }

  selectDevice(_deviceId: string) {
    /* single local Layer 8 host */
  }

  setOperatorMode(mode: OperatorMode) {
    this.currentOperatorMode = mode;
    this.snapshot = {
      ...this.snapshot,
      status: { ...this.snapshot.status, mode },
    };
    this.emit();
  }

  async runSensor(_deviceId: string, sensor: SensorType) {
    await fetch(LAYER8.runSensor(UI_SENSOR_TO_LAYER8[sensor]), { method: 'POST' });
    await this.refresh();
  }

  async stopSensor(_deviceId: string, sensor: SensorType) {
    await fetch(LAYER8.stopSensor(UI_SENSOR_TO_LAYER8[sensor]), { method: 'POST' });
    await this.refresh();
  }

  async restartSensor(_deviceId: string, sensor: SensorType) {
    await fetch(LAYER8.restartSensor(UI_SENSOR_TO_LAYER8[sensor]), { method: 'POST' });
    await this.refresh();
  }

  async runAll() {
    await fetch(LAYER8.runAll(), { method: 'POST' });
    await this.refresh();
  }

  async stopAll() {
    await fetch(LAYER8.stopAll(), { method: 'POST' });
    await this.refresh();
  }

  async restartAll() {
    await fetch(LAYER8.restartAll(), { method: 'POST' });
    await this.refresh();
  }

  async applyProfileByName(name: string) {
    await fetch(LAYER8.applyProfileByName(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    await this.refresh();
  }

  async fetchProfiles(): Promise<Array<{ id: string; name: string; description?: string }>> {
    const res = await fetch(LAYER8.profiles());
    if (!res.ok) return [];
    const data = await res.json();
    const list = data?.profiles;
    if (!Array.isArray(list)) return [];
    return list.map((p: { id: string; name: string; description?: string }) => ({
      id: p.id,
      name: p.name,
      description: p.description,
    }));
  }

  async fetchModelOptions(): Promise<ModelOptions> {
    const res = await fetch(LAYER8.modelOptions());
    if (!res.ok) {
      return { gun_checkpoints: [], person_yolo_suggestions: ['yolov8n.pt'], person_yolo_options: ['yolov8n.pt'] };
    }
    return res.json();
  }

  async fetchPlaygroundDefaults(): Promise<Partial<PlaygroundParams>> {
    const res = await fetch(LAYER8.aiCameraConfig());
    if (!res.ok) return {};
    const data = await res.json();
    const w = (data?.webcam ?? {}) as Record<string, unknown>;
    const extra = String(w.weapon_extra_args ?? '');
    const roiFrac = Number(extra.match(/--gun_roi_pad_frac\s+([\d.]+)/)?.[1]);
    const roiPx = Number(extra.match(/--gun_roi_pad_px\s+(\d+)/)?.[1]);
    return {
      conf: Number(w.weapon_conf ?? 0.35),
      gunConf: Number(w.weapon_gun_conf ?? 0.15),
      unsafeThreshold: Number(w.weapon_unsafe_threshold ?? 0.55),
      gunImgsz: Number(w.weapon_gun_imgsz ?? 640),
      gunEmitMin: Number(w.weapon_gun_emit_min_conf ?? 0.08),
      roiPadFrac: Number.isFinite(roiFrac) ? roiFrac : 0.12,
      roiPadPx: Number.isFinite(roiPx) ? roiPx : 48,
      personModel: String(w.person_detection_model ?? w.weapon_yolo_model ?? 'yolov8n.pt'),
      gunModel: String(w.weapon_gun_yolo_model ?? ''),
    };
  }

  async playgroundInfer(body: {
    values?: Record<string, unknown>;
    useSample?: boolean;
    imageB64?: string;
  }): Promise<{ image_b64: string; image_mime: string; summary: PlaygroundSummary }> {
    const res = await fetch(LAYER8.playgroundInfer(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        values: body.values ?? null,
        use_sample: body.useSample ?? true,
        image_b64: body.imageB64 ?? '',
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(String((err as { detail?: string }).detail ?? res.statusText));
    }
    return res.json();
  }

  async saveProfileIfNew(
    name: string,
    values?: Record<string, unknown>,
  ): Promise<{ skipped: boolean; saved_as?: string; existing_id?: string }> {
    const res = await fetch(LAYER8.saveProfileIfNew(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, values: values ?? null }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(String((err as { detail?: string }).detail ?? res.statusText));
    }
    return res.json();
  }

  async fetchWebcamConfig(): Promise<Record<string, unknown>> {
    const res = await fetch(LAYER8.aiCameraConfig());
    if (!res.ok) return {};
    const data = await res.json();
    return (data?.webcam ?? {}) as Record<string, unknown>;
  }

  async saveWebcamConfig(patch: Record<string, unknown>): Promise<void> {
    const res = await fetch(LAYER8.aiCameraConfig(), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ webcam: patch }),
    });
    if (!res.ok) throw new Error('Failed to save webcam config');
  }

  async fetchThermalConfig(): Promise<Record<string, unknown>> {
    const res = await fetch(LAYER8.thermalConfig());
    if (!res.ok) return {};
    const data = await res.json();
    return (data?.thermal ?? {}) as Record<string, unknown>;
  }

  async saveThermalConfig(patch: Record<string, unknown>): Promise<void> {
    const res = await fetch(LAYER8.thermalConfig(), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thermal: patch }),
    });
    if (!res.ok) throw new Error('Failed to save thermal config');
  }

  async fetchMmwaveConfig(): Promise<Record<string, unknown>> {
    const res = await fetch(LAYER8.mmwaveConfig());
    if (!res.ok) return {};
    const data = await res.json();
    return (data?.mmwave ?? {}) as Record<string, unknown>;
  }

  async saveMmwaveConfig(patch: Record<string, unknown>): Promise<void> {
    const res = await fetch(LAYER8.mmwaveConfig(), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mmwave: patch }),
    });
    if (!res.ok) throw new Error('Failed to save mmWave config');
  }

  async fetchV4l2Devices(): Promise<Array<{ index: number; label: string }>> {
    const res = await fetch(LAYER8.v4l2Devices());
    if (!res.ok) return [];
    const data = await res.json();
    const groups = data?.groups ?? data?.devices ?? [];
    if (!Array.isArray(groups)) return [];
    const out: Array<{ index: number; label: string }> = [];
    for (const g of groups) {
      const name = String(g?.name ?? g?.card ?? 'Device');
      const nodes = g?.nodes ?? g?.devices ?? [];
      if (Array.isArray(nodes)) {
        for (const n of nodes) {
          const m = String(n).match(/video(\d+)/);
          if (m) out.push({ index: Number(m[1]), label: `${name} (/dev/video${m[1]})` });
        }
      } else if (g?.index != null) {
        out.push({ index: Number(g.index), label: name });
      }
    }
    return out;
  }

  async fetchSerialPorts(): Promise<string[]> {
    const res = await fetch(LAYER8.serialPorts());
    if (!res.ok) return [];
    const data = await res.json();
    const ports = data?.ports ?? data?.candidates ?? [];
    return Array.isArray(ports) ? ports.map(String) : [];
  }

  async fetchSensorWeaponDefaults(sensor: 'webcam' | 'thermal'): Promise<Partial<PlaygroundParams>> {
    const res = await fetch(sensor === 'thermal' ? LAYER8.thermalConfig() : LAYER8.aiCameraConfig());
    if (!res.ok) return {};
    const data = await res.json();
    const w = ((sensor === 'thermal' ? data?.thermal : data?.webcam) ?? {}) as Record<string, unknown>;
    const extra = String(w.weapon_extra_args ?? '');
    const roiFrac = Number(extra.match(/--gun_roi_pad_frac\s+([\d.]+)/)?.[1]);
    const roiPx = Number(extra.match(/--gun_roi_pad_px\s+(\d+)/)?.[1]);
    return {
      conf: Number(w.weapon_conf ?? 0.35),
      gunConf: Number(w.weapon_gun_conf ?? 0.15),
      unsafeThreshold: Number(w.weapon_unsafe_threshold ?? 0.55),
      gunImgsz: Number(w.weapon_gun_imgsz ?? 640),
      gunEmitMin: Number(w.weapon_gun_emit_min_conf ?? 0.08),
      roiPadFrac: Number.isFinite(roiFrac) ? roiFrac : 0.12,
      roiPadPx: Number.isFinite(roiPx) ? roiPx : 48,
      personModel: String(w.person_detection_model ?? w.weapon_yolo_model ?? 'yolov8n.pt'),
      gunModel: String(w.weapon_gun_yolo_model ?? ''),
    };
  }

  async applyProfileAndRestart(sensor: 'webcam' | 'thermal' | 'mmwave', profileName: string) {
    await this.applyProfileByName(profileName);
    await this.restartSensor('layer8-local', sensor === 'webcam' ? 'webcam' : sensor);
  }

  ackAlert(_alertId: string) {
    /* Layer 8 has no alert-ack endpoint */
  }

  /** Capture current preview stream; optionally attach to a recent alert. */
  async captureScreenshot(sensor: SensorType = 'webcam'): Promise<string | null> {
    const snapSensor =
      sensor === 'multi_camera'
        ? 'multi_camera'
        : sensor === 'thermal'
          ? 'thermal'
          : 'webcam';
    const dataUrl = await capturePreviewFrame('', snapSensor);
    if (!dataUrl) return null;
    if (this.alertHistory.length > 0) {
      const next = [...this.alertHistory];
      next[0] = { ...next[0], screenshotDataUrl: dataUrl };
      this.alertHistory = next;
      this.snapshot = { ...this.snapshot, alerts: next };
      this.emit();
    }
    return dataUrl;
  }

  private mergeAlertHistory(incoming: Alert[]) {
    const newAlerts: Alert[] = [];
    for (const a of incoming) {
      if (this.seenAlertIds.has(a.id)) continue;
      this.seenAlertIds.add(a.id);
      this.alertHistory.unshift(a);
      newAlerts.push(a);
    }
    if (this.alertHistory.length > 32) {
      this.alertHistory = this.alertHistory.slice(0, 32);
    }
    if (this.seenAlertIds.size > 128) {
      this.seenAlertIds = new Set(this.alertHistory.map((a) => a.id));
    }
    for (const a of newAlerts) {
      if (a.severity !== 'critical') continue;
      const sensor: CameraSensor = a.sensor === 'multi_camera' ? 'multi_camera' : 'webcam';
      this.screenshotQueue.push({ alertId: a.id, sensor });
      void this.drainScreenshotQueue();
    }
    return this.alertHistory;
  }

  private async drainScreenshotQueue() {
    if (this.screenshotInFlight || this.screenshotQueue.length === 0) return;
    this.screenshotInFlight = true;
    const job = this.screenshotQueue.shift();
    if (!job) {
      this.screenshotInFlight = false;
      return;
    }
    try {
      const dataUrl = await capturePreviewFrame('', job.sensor);
      if (dataUrl) {
        const idx = this.alertHistory.findIndex((x) => x.id === job.alertId);
        if (idx >= 0) {
          const next = [...this.alertHistory];
          next[idx] = { ...next[idx], screenshotDataUrl: dataUrl };
          this.alertHistory = next;
          this.snapshot = { ...this.snapshot, alerts: next };
          this.emit();
        }
      }
    } finally {
      this.screenshotInFlight = false;
      if (this.screenshotQueue.length > 0) {
        void this.drainScreenshotQueue();
      }
    }
  }

  private connectStatusStream() {
    try {
      this.eventSource?.close();
    } catch {
      /* ignore */
    }
    const es = new EventSource(LAYER8.statusStream());
    this.eventSource = es;

    es.addEventListener('status', (ev: MessageEvent) => {
      try {
        const data = JSON.parse(String(ev.data)) as Layer8AllStatus;
        this.lastSensorStatus = data;
        this.mergeSensorStatus(data);
      } catch {
        /* ignore */
      }
    });

    es.onopen = () => {
      this.backendReachable = true;
      this.snapshot = {
        ...this.snapshot,
        status: { ...this.snapshot.status, backendOnline: true, state: 'normal' },
      };
      this.emit();
      void this.refreshMetrics();
    };

    es.onerror = () => {
      this.backendReachable = false;
      this.snapshot = {
        ...this.snapshot,
        status: { ...this.snapshot.status, backendOnline: false, state: 'fault' },
      };
      this.emit();
    };
  }

  private mergeSensorStatus(sensorStatus: Layer8AllStatus) {
    const device = deviceFromStatus(sensorStatus);
    const inferActive = inferRunnerActive(sensorStatus);

    this.snapshot = {
      ...this.snapshot,
      devices: [device],
      sensorLogs: {
        thermal: sensorStatus.thermal?.log_tail || '',
        webcam: sensorStatus.webcam?.log_tail || '',
        mmwave: sensorStatus.mmwave?.log_tail || '',
      },
      sensorRunning: sensorRunningFromStatus(sensorStatus),
      status: {
        ...this.snapshot.status,
        backendOnline: true,
        state: device.status === 'online' ? 'normal' : 'fallback',
        timestamp: new Date().toISOString(),
      },
    };

    if (!inferActive) {
      this.lastFrameSample = null;
      this.snapshot = {
        ...this.snapshot,
        metrics: metricsFromDashboard(null, null, false, 0),
      };
    }

    this.emit();
  }

  private lastDashboardMetrics: Layer8DashboardMetrics | null = null;

  private computeInferFps(frame: number | null, ts: number | null): number {
    if (frame == null || ts == null) return this.snapshot.metrics.fps;
    const prev = this.lastFrameSample;
    this.lastFrameSample = { frame, ts };
    if (!prev || frame <= prev.frame || ts <= prev.ts) return this.snapshot.metrics.fps;
    const fps = (frame - prev.frame) / (ts - prev.ts);
    return fps > 0 && fps < 240 ? fps : this.snapshot.metrics.fps;
  }

  private async refreshMetrics() {
    if (this.metricsRefreshInFlight) return;
    this.metricsRefreshInFlight = true;
    try {
      const running = sensorRunningFromStatus(this.lastSensorStatus);
      const fetches: Promise<Response>[] = [
        fetch(LAYER8.systemMetrics()),
        fetch(LAYER8.dashboardMetrics()),
      ];
      if (running.webcam) {
        fetches.push(fetch(LAYER8.frontCameraThreatMetrics()));
      }
      if (running.multi_camera) {
        fetches.push(fetch(LAYER8.backCameraThreatMetrics()));
      }

      const results = await Promise.all(fetches);
      const sysRes = results[0];
      const dashRes = results[1];
      let frontThreatRes: Response | null = null;
      let backThreatRes: Response | null = null;
      let ri = 2;
      if (running.webcam) {
        frontThreatRes = results[ri++];
      }
      if (running.multi_camera) {
        backThreatRes = results[ri++];
      }

      if (!dashRes.ok && !this.lastSensorStatus) {
        throw new Error('Layer 8 unreachable');
      }

      const sys: Layer8SystemMetrics | null = sysRes.ok ? await sysRes.json() : null;
      const dash: Layer8DashboardMetrics | null = dashRes.ok ? await dashRes.json() : null;
      const frontThreat: Layer8ThreatMetrics | null = frontThreatRes?.ok
        ? await frontThreatRes.json()
        : null;
      const backThreat: Layer8ThreatMetrics | null = backThreatRes?.ok ? await backThreatRes.json() : null;
      const sensorStatus = this.lastSensorStatus;

      if (dash) this.lastDashboardMetrics = dash;

      const runningNow = sensorRunningFromStatus(sensorStatus);
      const inferActive = inferRunnerActive(sensorStatus);
      const fps = inferActive ? this.computeInferFps(dash?.frame ?? null, dash?.ts ?? null) : 0;
      const metrics = metricsFromDashboard(dash, sys, inferActive, fps);

      const frontAlerts = alertsFromCameraThreat(
        frontThreat,
        'webcam',
        runningNow.webcam,
        'Front Camera',
      );
      const backAlerts = alertsFromCameraThreat(
        backThreat,
        'multi_camera',
        runningNow.multi_camera,
        'Back Camera',
      );
      const freshAlerts = [...frontAlerts, ...backAlerts];
      const alerts =
        freshAlerts.length > 0 ? this.mergeAlertHistory(freshAlerts) : this.alertHistory;
      const device = deviceFromStatus(sensorStatus);
      const threatForDetections = frontThreat?.byte_tracks?.length
        ? frontThreat
        : backThreat ?? frontThreat;

      this.backendReachable = true;
      this.snapshot = {
        ...this.snapshot,
        devices: [device],
        metrics: {
          ...metrics,
          gunDetected:
            metrics.gunDetected ||
            frontThreat?.gun_detected === true ||
            backThreat?.gun_detected === true,
          personDetections: Math.max(
            metrics.personDetections ?? 0,
            frontThreat?.persons_total ?? 0,
            backThreat?.persons_total ?? 0,
          ) || null,
        },
        alerts,
        detections: { [LAYER8_LOCAL_DEVICE_ID]: detectionsFromThreat(threatForDetections) },
        status: {
          ...this.snapshot.status,
          mode: this.currentOperatorMode,
          backendOnline: true,
          state:
            metrics.gunDetected ||
            frontThreat?.gun_detected ||
            backThreat?.gun_detected
              ? 'normal'
              : device.status === 'online'
                ? 'normal'
                : 'fallback',
          activeAlerts: alerts.length,
          timestamp: new Date().toISOString(),
        },
        sensorLogs: sensorStatus
          ? {
              thermal: sensorStatus.thermal?.log_tail || '',
              webcam: sensorStatus.webcam?.log_tail || '',
              mmwave: sensorStatus.mmwave?.log_tail || '',
            }
          : this.snapshot.sensorLogs,
        sensorRunning: runningNow,
      };
      this.emit();
    } catch {
      this.backendReachable = false;
      this.lastDashboardMetrics = null;
      this.lastFrameSample = null;
      this.snapshot = {
        ...this.snapshot,
        metrics: this.emptyMetrics(),
        status: {
          ...this.snapshot.status,
          backendOnline: false,
          state: 'fault',
          activeAlerts: 0,
        },
        devices: [{ ...deviceFromStatus(null), status: 'offline' }],
        detections: {},
        alerts: [],
        sensorRunning: { thermal: false, webcam: false, mmwave: false, multi_camera: false },
      };
      this.alertHistory = [];
      this.seenAlertIds.clear();
      this.emit();
    } finally {
      this.metricsRefreshInFlight = false;
    }
  }

  private async refresh() {
    await this.refreshMetrics();
  }

  private emptyMetrics(): DashboardMetrics {
    return {
      inferActive: false,
      gunDetected: false,
      prediction: null,
      unsafePct: null,
      unsafeScore: null,
      personsWithGun: null,
      personDetections: null,
      weaponDetections: 0,
      totalDetections: null,
      frame: null,
      metricsNote: '',
      fps: 0,
      latency: 0,
      cpuUsage: 0,
      gpuUsage: 0,
      memoryUsage: 0,
    };
  }

  private emit() {
    for (const fn of this.listeners) fn(this.snapshot);
  }
}

export const scanuClient = new ScanUClient();

export const previewUrls = {
  rgb: LAYER8.previewFrontCamera(),
  front: LAYER8.previewFrontCamera(),
  back: LAYER8.previewBackCamera(),
  thermal: LAYER8.previewThermal(),
  mmwave: LAYER8.previewMmwave(),
} as const;

export { layer8Url, LAYER8 };
