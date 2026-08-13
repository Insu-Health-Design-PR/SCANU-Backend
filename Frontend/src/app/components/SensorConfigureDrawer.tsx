import { Loader2, X } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { scanuClient } from '../api/client';
import { ModelPlaygroundPanel } from './ModelPlaygroundPanel';
import { ScanuSelect } from './ScanuSelect';

export type ConfigureSensor = 'webcam' | 'thermal' | 'mmwave';

const SENSOR_LABEL: Record<ConfigureSensor, string> = {
  webcam: 'RGB Camera',
  thermal: 'Thermal Camera',
  mmwave: 'mmWave Radar',
};

const WEBCAM_RES_PRESETS = [
  { label: '720p · 1280×720', w: 1280, h: 720, fps: 30 },
  { label: '1080p · 1920×1080', w: 1920, h: 1080, fps: 30 },
  { label: '1440p · 2560×1440', w: 2560, h: 1440, fps: 30 },
  { label: '4K · 3840×2160', w: 3840, h: 2160, fps: 30 },
];

const THERMAL_RES_PRESETS = [
  { label: '160×120 @ 9fps', w: 160, h: 120, fps: 9 },
  { label: '320×240 @ 9fps', w: 320, h: 240, fps: 9 },
  { label: '640×480 @ 9fps', w: 640, h: 480, fps: 9 },
];

interface SensorConfigureDrawerProps {
  open: boolean;
  sensor: ConfigureSensor;
  onClose: () => void;
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-xl border border-white/10 bg-white/[0.03] overflow-hidden">
      <div className="px-4 py-3 border-b border-white/10 bg-white/[0.02]">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
      </div>
      <div className="p-4 space-y-4">{children}</div>
    </section>
  );
}

export function SensorConfigureDrawer({ open, sensor, onClose }: SensorConfigureDrawerProps) {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [profiles, setProfiles] = useState<Array<{ id: string; name: string; description?: string }>>([]);
  const [v4l2, setV4l2] = useState<Array<{ index: number; label: string }>>([]);
  const [serialPorts, setSerialPorts] = useState<string[]>([]);

  const [webcamDevice, setWebcamDevice] = useState('0');
  const [webcamW, setWebcamW] = useState('1920');
  const [webcamH, setWebcamH] = useState('1080');
  const [webcamFps, setWebcamFps] = useState('30');

  const [thermalDevice, setThermalDevice] = useState('0');
  const [thermalW, setThermalW] = useState('160');
  const [thermalH, setThermalH] = useState('120');
  const [thermalFps, setThermalFps] = useState('9');

  const [cliPort, setCliPort] = useState('');
  const [dataPort, setDataPort] = useState('');
  const [mmwaveConfig, setMmwaveConfig] = useState('');

  const loadConfig = useCallback(async () => {
    setStatus('');
    const [prof, v4, serial] = await Promise.all([
      scanuClient.fetchProfiles(),
      scanuClient.fetchV4l2Devices(),
      scanuClient.fetchSerialPorts(),
    ]);
    setProfiles(prof);
    setV4l2(v4);
    setSerialPorts(serial.length ? serial : ['/dev/ttyUSB0', '/dev/ttyUSB1']);

    if (sensor === 'webcam') {
      const w = await scanuClient.fetchWebcamConfig();
      setWebcamDevice(String(w.webcam_device ?? 0));
      setWebcamW(String(w.webcam_width ?? 1920));
      setWebcamH(String(w.webcam_height ?? 1080));
      setWebcamFps(String(w.fps ?? 30));
    } else if (sensor === 'thermal') {
      const t = await scanuClient.fetchThermalConfig();
      setThermalDevice(String(t.thermal_device ?? 0));
      setThermalW(String(t.thermal_width ?? 160));
      setThermalH(String(t.thermal_height ?? 120));
      setThermalFps(String(t.thermal_fps ?? 9));
    } else {
      const m = await scanuClient.fetchMmwaveConfig();
      setCliPort(String(m.cli_port ?? ''));
      setDataPort(String(m.data_port ?? ''));
      setMmwaveConfig(String(m.config ?? ''));
    }
  }, [sensor]);

  useEffect(() => {
    if (open) void loadConfig();
  }, [open, loadConfig]);

  async function act(fn: () => Promise<void>) {
    setBusy(true);
    setStatus('');
    try {
      await fn();
    } catch (e) {
      setStatus(e instanceof Error ? e.message : 'Action failed');
    } finally {
      setBusy(false);
    }
  }

  async function saveDeviceConfig() {
    if (sensor === 'webcam') {
      await scanuClient.saveWebcamConfig({
        webcam_device: Number(webcamDevice),
        webcam_width: Number(webcamW),
        webcam_height: Number(webcamH),
        fps: Number(webcamFps),
      });
      await scanuClient.restartSensor('layer8-local', 'webcam');
      setStatus('RGB camera settings saved — webcam restarted.');
    } else if (sensor === 'thermal') {
      await scanuClient.saveThermalConfig({
        thermal_device: Number(thermalDevice),
        thermal_width: Number(thermalW),
        thermal_height: Number(thermalH),
        thermal_fps: Number(thermalFps),
      });
      await scanuClient.restartSensor('layer8-local', 'thermal');
      setStatus('Thermal settings saved — thermal runner restarted.');
    } else {
      await scanuClient.saveMmwaveConfig({
        cli_port: cliPort,
        data_port: dataPort,
        config: mmwaveConfig,
      });
      await scanuClient.restartSensor('layer8-local', 'mmwave');
      setStatus('mmWave settings saved — runner restarted.');
    }
  }

  if (!open) return null;

  const deviceOptions =
    v4l2.length > 0
      ? v4l2.map((d) => String(d.index))
      : ['0', '1', '2', '3', '4', '5', '6', '7', '8'];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
      <button
        type="button"
        className="absolute inset-0 bg-black/65 backdrop-blur-[2px]"
        onClick={onClose}
        aria-label="Close configure"
      />
      <div className="relative w-full max-w-5xl max-h-[88vh] rounded-2xl bg-[#12151c] border border-white/15 shadow-2xl flex flex-col overflow-hidden">
        <div className="shrink-0 flex items-center justify-between px-5 py-4 border-b border-white/10 bg-[#141820]">
          <div>
            <div className="text-xs text-white/40 uppercase tracking-wider">Configure</div>
            <div className="text-lg font-semibold text-white">{SENSOR_LABEL[sensor]}</div>
          </div>
          <button type="button" onClick={onClose} className="p-2 rounded-lg hover:bg-white/10 text-white/60">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5 min-h-0">
          <Section title="Section 1 — Camera device & resolution">
            {sensor === 'webcam' && (
              <>
                <ScanuSelect
                  label="Camera device (/dev/videoN)"
                  value={webcamDevice}
                  onValueChange={setWebcamDevice}
                  options={deviceOptions}
                  disabled={busy}
                />
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { label: 'Width', value: webcamW, set: setWebcamW },
                    { label: 'Height', value: webcamH, set: setWebcamH },
                    { label: 'FPS', value: webcamFps, set: setWebcamFps },
                  ].map((f) => (
                    <div key={f.label}>
                      <label className="block text-xs text-white/50 mb-1">{f.label}</label>
                      <input
                        value={f.value}
                        onChange={(e) => f.set(e.target.value)}
                        className="w-full px-3 py-2 bg-white/10 border border-white/15 rounded text-sm"
                      />
                    </div>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  {WEBCAM_RES_PRESETS.map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      disabled={busy}
                      onClick={() => {
                        setWebcamW(String(p.w));
                        setWebcamH(String(p.h));
                        setWebcamFps(String(p.fps));
                      }}
                      className="px-2.5 py-1 text-xs rounded-md bg-white/5 hover:bg-white/10 border border-white/10"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </>
            )}

            {sensor === 'thermal' && (
              <>
                <ScanuSelect
                  label="Thermal device (/dev/videoN)"
                  value={thermalDevice}
                  onValueChange={setThermalDevice}
                  options={deviceOptions}
                  disabled={busy}
                />
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { label: 'Width', value: thermalW, set: setThermalW },
                    { label: 'Height', value: thermalH, set: setThermalH },
                    { label: 'FPS', value: thermalFps, set: setThermalFps },
                  ].map((f) => (
                    <div key={f.label}>
                      <label className="block text-xs text-white/50 mb-1">{f.label}</label>
                      <input
                        value={f.value}
                        onChange={(e) => f.set(e.target.value)}
                        className="w-full px-3 py-2 bg-white/10 border border-white/15 rounded text-sm"
                      />
                    </div>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  {THERMAL_RES_PRESETS.map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      disabled={busy}
                      onClick={() => {
                        setThermalW(String(p.w));
                        setThermalH(String(p.h));
                        setThermalFps(String(p.fps));
                      }}
                      className="px-2.5 py-1 text-xs rounded-md bg-white/5 hover:bg-white/10 border border-white/10"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </>
            )}

            {sensor === 'mmwave' && (
              <>
                <ScanuSelect
                  label="CLI port"
                  value={cliPort || serialPorts[0] || ''}
                  onValueChange={setCliPort}
                  options={serialPorts}
                  disabled={busy}
                />
                <ScanuSelect
                  label="Data port"
                  value={dataPort || serialPorts[1] || serialPorts[0] || ''}
                  onValueChange={setDataPort}
                  options={serialPorts}
                  disabled={busy}
                />
                <div>
                  <label className="block text-xs text-white/50 mb-1">Radar config (.cfg)</label>
                  <input
                    value={mmwaveConfig}
                    onChange={(e) => setMmwaveConfig(e.target.value)}
                    className="w-full px-3 py-2 bg-white/10 border border-white/15 rounded text-sm font-mono text-xs"
                  />
                </div>
              </>
            )}

            <button
              type="button"
              disabled={busy}
              onClick={() => act(saveDeviceConfig)}
              className="w-full py-2.5 rounded-lg bg-white/10 hover:bg-white/15 text-sm font-medium disabled:opacity-50"
            >
              Save device settings & restart
            </button>
          </Section>

          {(sensor === 'webcam' || sensor === 'thermal') && (
            <Section title="Section 2 — Model configurations (quick profiles)">
              {profiles.length === 0 ? (
                <p className="text-sm text-white/40">No profiles on Layer 8.</p>
              ) : (
                <div className="grid grid-cols-1 gap-2 max-h-48 overflow-y-auto">
                  {profiles.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      disabled={busy}
                      onClick={() =>
                        act(async () => {
                          await scanuClient.applyProfileAndRestart(sensor, p.name);
                          setStatus(`Applied profile "${p.name}" and restarted ${SENSOR_LABEL[sensor]}.`);
                        })
                      }
                      className="text-left px-3 py-2.5 rounded-lg bg-white/5 hover:bg-cyan-500/10 border border-white/10 hover:border-cyan-500/30 transition-colors disabled:opacity-50"
                    >
                      <div className="text-sm font-medium text-white">{p.name}</div>
                      {p.description && (
                        <div className="text-xs text-white/40 mt-0.5 truncate">{p.description}</div>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </Section>
          )}

          {sensor === 'mmwave' && (
            <Section title="Section 2 — Model configurations (quick profiles)">
              <p className="text-sm text-white/45">
                mmWave uses radar TLV tracking — no YOLO weapon profile. Tune ports and .cfg in Section 1.
              </p>
            </Section>
          )}

          {(sensor === 'webcam' || sensor === 'thermal') && (
            <Section title="Section 3 — Model configurations (customize)">
              <ModelPlaygroundPanel embedded targetSensor={sensor} disabled={busy} />
            </Section>
          )}

          {sensor === 'mmwave' && (
            <Section title="Section 3 — Model configurations (customize)">
              <p className="text-sm text-white/45">
                Playground tuning applies to RGB and Thermal infer pipelines only.
              </p>
            </Section>
          )}

          {status && (
            <p className={`text-sm ${status.includes('failed') || status.includes('Failed') ? 'text-red-400' : 'text-emerald-400'}`}>
              {status}
            </p>
          )}
        </div>

        {busy && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40 rounded-2xl pointer-events-none">
            <Loader2 className="w-8 h-8 animate-spin text-white/60" />
          </div>
        )}
      </div>
    </div>
  );
}
