import { ImagePlus, Loader2, RotateCcw, Save, Zap } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import sampleImage from '../../assets/playground/sample.jpg';
import { PlaygroundParams, PlaygroundSummary, scanuClient } from '../api/client';
import { ScanuSelect } from './ScanuSelect';

interface ModelPlaygroundPanelProps {
  disabled?: boolean;
  /** Inside Configure modal — tighter chrome, same two-column playground layout. */
  embedded?: boolean;
  targetSensor?: 'webcam' | 'thermal';
}

type ImageSource = 'sample' | 'upload';

function normalizePrediction(raw: string | null | undefined): string {
  if (raw == null || String(raw).trim() === '') return '—';
  const s = String(raw).trim().toLowerCase();
  if (s.startsWith('no_person') || s.startsWith('no person')) return 'No person';
  if (s.startsWith('armed')) return 'Armed';
  if (s.startsWith('unsafe')) return 'Unsafe';
  if (s.startsWith('safe')) return 'Safe';
  if (s.startsWith('suspicious')) return 'Suspicious';
  return String(raw).trim();
}

function predictionTone(raw: string | null | undefined): string {
  const p = String(raw ?? '').toLowerCase();
  if (p.startsWith('armed') || p.startsWith('unsafe')) {
    return 'text-red-300 bg-red-500/15 border-red-500/30';
  }
  if (p.startsWith('suspicious')) {
    return 'text-amber-300 bg-amber-500/15 border-amber-500/30';
  }
  if (p.startsWith('safe') || p.startsWith('no_person')) {
    return 'text-emerald-300 bg-emerald-500/15 border-emerald-500/30';
  }
  return 'text-white/70 bg-white/5 border-white/10';
}

function averageGunConfidence(summary: PlaygroundSummary | null): number | null {
  if (!summary?.firearms || !Array.isArray(summary.firearms) || summary.firearms.length === 0) return null;
  const confs = summary.firearms
    .map((f) => Number((f as unknown as { conf?: unknown }).conf))
    .filter((n) => Number.isFinite(n));
  if (confs.length === 0) return null;
  return confs.reduce((a, b) => a + b, 0) / confs.length;
}

function buildExtraArgs(roiFrac: number, roiPx: number): string {
  return `--yolo_classes 0 --gun_roi_pad_frac ${roiFrac.toFixed(2)} --gun_roi_pad_px ${Math.round(roiPx)} --yolo_device cuda --classifier_device cuda`;
}

function valuesFromParams(p: PlaygroundParams): Record<string, unknown> {
  return {
    person_detection_model: p.personModel,
    weapon_yolo_model: p.personModel,
    weapon_gun_yolo_model: p.gunModel,
    weapon_conf: p.conf,
    weapon_gun_conf: p.gunConf,
    weapon_unsafe_threshold: p.unsafeThreshold,
    weapon_gun_imgsz: p.gunImgsz,
    weapon_gun_emit_min_conf: p.gunEmitMin,
    weapon_extra_args: buildExtraArgs(p.roiPadFrac, p.roiPadPx),
  };
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result ?? '');
      const comma = result.indexOf(',');
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-white/70">{label}</span>
        <span className="font-mono text-white/90">{value.toFixed(step < 0.01 ? 3 : step < 1 ? 2 : 0)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-white"
      />
    </div>
  );
}

export function ModelPlaygroundPanel({ disabled, embedded, targetSensor = 'webcam' }: ModelPlaygroundPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [params, setParams] = useState<PlaygroundParams>({
    conf: 0.35,
    gunConf: 0.15,
    unsafeThreshold: 0.55,
    roiPadFrac: 0.12,
    roiPadPx: 48,
    gunImgsz: 640,
    gunEmitMin: 0.08,
    personModel: 'yolov8n.pt',
    gunModel: '',
  });
  const [personModels, setPersonModels] = useState<string[]>(['yolov8n.pt']);
  const [gunModels, setGunModels] = useState<string[]>([]);
  const [imageSource, setImageSource] = useState<ImageSource>('sample');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState<string | null>(null);
  const [profileName, setProfileName] = useState('playground_custom');
  const [previewUrl, setPreviewUrl] = useState<string>(sampleImage);
  const [summary, setSummary] = useState<PlaygroundSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [statusMsg, setStatusMsg] = useState('');

  const values = useMemo(() => valuesFromParams(params), [params]);
  const inputPreviewUrl = imageSource === 'upload' && uploadPreviewUrl ? uploadPreviewUrl : sampleImage;

  const patch = useCallback((partial: Partial<PlaygroundParams>) => {
    setParams((prev) => ({ ...prev, ...partial }));
  }, []);

  useEffect(() => {
    void (async () => {
      const [opts, defaults] = await Promise.all([
        scanuClient.fetchModelOptions(),
        scanuClient.fetchSensorWeaponDefaults(targetSensor),
      ]);
      const guns = opts.gun_checkpoints ?? [];
      const persons = Array.from(
        new Set([
          ...(opts.person_yolo_suggestions ?? []),
          String(defaults.personModel ?? ''),
        ].filter(Boolean)),
      );
      setGunModels(guns);
      setPersonModels(persons.length ? persons : ['yolov8n.pt']);
      setParams((prev) => ({
        ...prev,
        ...defaults,
        personModel: defaults.personModel ?? prev.personModel,
        gunModel: defaults.gunModel && guns.includes(defaults.gunModel)
          ? defaults.gunModel
          : guns[0] ?? prev.gunModel,
      }));
    })();
  }, [targetSensor]);

  useEffect(() => {
    return () => {
      if (uploadPreviewUrl?.startsWith('blob:')) {
        URL.revokeObjectURL(uploadPreviewUrl);
      }
    };
  }, [uploadPreviewUrl]);

  async function runTry() {
    if (imageSource === 'upload' && !uploadFile) {
      setError('Choose an image to upload first');
      return;
    }
    if (!params.gunModel) {
      setError('Select a gun detection model');
      return;
    }

    setBusy(true);
    setError('');
    setStatusMsg('');
    try {
      let imageB64 = '';
      if (imageSource === 'upload' && uploadFile) {
        imageB64 = await fileToBase64(uploadFile);
      }
      const result = await scanuClient.playgroundInfer({
        values,
        useSample: imageSource === 'sample',
        imageB64: imageSource === 'upload' ? imageB64 : undefined,
      });
      setPreviewUrl(`data:${result.image_mime};base64,${result.image_b64}`);
      setSummary(result.summary);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Inference failed');
    } finally {
      setBusy(false);
    }
  }

  async function saveProfile(goLive: boolean) {
    const name = profileName.trim();
    if (!name) {
      setError('Profile name is required');
      return;
    }
    setBusy(true);
    setError('');
    setStatusMsg('');
    try {
      const saved = await scanuClient.saveProfileIfNew(name, values);
      if (saved.skipped) {
        setStatusMsg(`Profile "${name}" already exists — skipped save.`);
      } else {
        setStatusMsg(`Saved profile "${name}".`);
      }
      if (goLive) {
        await scanuClient.applyProfileByName(name);
        await scanuClient.restartSensor('layer8-local', targetSensor);
        setStatusMsg((m) => `${m} Applied and restarted ${targetSensor}.`.trim());
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setBusy(false);
    }
  }

  function handleUploadSelect(file: File | null) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please choose a JPEG or PNG image');
      return;
    }
    if (uploadPreviewUrl?.startsWith('blob:')) {
      URL.revokeObjectURL(uploadPreviewUrl);
    }
    const url = URL.createObjectURL(file);
    setUploadFile(file);
    setUploadPreviewUrl(url);
    setImageSource('upload');
    setPreviewUrl(url);
    setSummary(null);
    setError('');
  }

  function useSampleImage() {
    if (uploadPreviewUrl?.startsWith('blob:')) {
      URL.revokeObjectURL(uploadPreviewUrl);
    }
    setUploadFile(null);
    setUploadPreviewUrl(null);
    setImageSource('sample');
    setPreviewUrl(sampleImage);
    setSummary(null);
    setError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  useEffect(() => {
    const t = window.setTimeout(() => {
      if (!disabled && !busy && (imageSource === 'sample' || uploadFile)) {
        void runTry();
      }
    }, 600);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    params.conf,
    params.gunConf,
    params.unsafeThreshold,
    params.roiPadFrac,
    params.roiPadPx,
    params.gunImgsz,
    params.gunEmitMin,
    params.personModel,
    params.gunModel,
    imageSource,
    uploadFile,
  ]);

  const previewPane = (
    <div className="space-y-3">
      <div className="w-full aspect-video bg-black/40 rounded border border-white/10 overflow-hidden flex items-center justify-center">
        {busy ? (
          <Loader2 className="w-8 h-8 animate-spin text-white/50" />
        ) : (
          <img
            src={busy ? inputPreviewUrl : previewUrl}
            alt="Playground preview"
            className="w-full h-full object-contain"
          />
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={(e) => handleUploadSelect(e.target.files?.[0] ?? null)}
        />
        <button
          type="button"
          disabled={disabled || busy}
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center gap-2 px-3 py-2 text-sm bg-white/10 hover:bg-white/20 rounded disabled:opacity-50"
        >
          <ImagePlus className="w-4 h-4" />
          Upload image
        </button>
        <button
          type="button"
          disabled={disabled || busy || imageSource === 'sample'}
          onClick={useSampleImage}
          className="flex items-center gap-2 px-3 py-2 text-sm bg-white/10 hover:bg-white/20 rounded disabled:opacity-50"
        >
          <RotateCcw className="w-4 h-4" />
          Use sample
        </button>
      </div>
      <p className="text-xs text-white/40">
        {imageSource === 'sample'
          ? 'Using bundled sample image'
          : uploadFile
            ? `Uploaded: ${uploadFile.name}`
            : 'No upload selected'}
      </p>

      {summary && (
        <div className="rounded-lg border border-white/10 bg-black/30 p-3 space-y-2">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-white/50 uppercase tracking-wide">Prediction</span>
            <span
              className={`text-sm font-semibold px-2.5 py-1 rounded-md border ${predictionTone(summary.prediction)}`}
            >
              {normalizePrediction(summary.prediction)}
            </span>
          </div>
          <div className="text-xs font-mono text-white/55 space-y-0.5">
            <div>
              Gun Confidence:{' '}
              {(() => {
                const avg = averageGunConfidence(summary);
                return avg == null ? '—' : avg.toFixed(3);
              })()}
            </div>
            {summary.persons_total != null && (
              <div>
                persons: {summary.persons_total}
                {summary.persons_with_gun != null ? ` · armed: ${summary.persons_with_gun}` : ''}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  const controlsPane = (
    <div className="space-y-3">
          <ScanuSelect
            label="Person detection model"
            value={params.personModel}
            onValueChange={(v) => patch({ personModel: v })}
            options={personModels}
            disabled={busy}
          />

          <ScanuSelect
            label="Gun detection model"
            value={params.gunModel}
            onValueChange={(v) => patch({ gunModel: v })}
            options={gunModels.length ? gunModels : ['Loading…']}
            disabled={busy || gunModels.length === 0}
          />

          <SliderRow label="Person conf" value={params.conf} min={0.1} max={0.9} step={0.01} onChange={(v) => patch({ conf: v })} />
          <SliderRow label="Gun conf" value={params.gunConf} min={0.05} max={0.8} step={0.01} onChange={(v) => patch({ gunConf: v })} />
          <SliderRow
            label="Unsafe threshold"
            value={params.unsafeThreshold}
            min={0.2}
            max={0.95}
            step={0.01}
            onChange={(v) => patch({ unsafeThreshold: v })}
          />
          <SliderRow
            label="Gun ROI pad frac"
            value={params.roiPadFrac}
            min={0.05}
            max={0.35}
            step={0.01}
            onChange={(v) => patch({ roiPadFrac: v })}
          />
          <SliderRow label="Gun ROI pad px" value={params.roiPadPx} min={16} max={160} step={4} onChange={(v) => patch({ roiPadPx: v })} />
          <SliderRow label="Gun imgsz" value={params.gunImgsz} min={320} max={1280} step={32} onChange={(v) => patch({ gunImgsz: v })} />
          <SliderRow
            label="Gun emit min conf"
            value={params.gunEmitMin}
            min={0.01}
            max={0.5}
            step={0.01}
            onChange={(v) => patch({ gunEmitMin: v })}
          />

          <input
            type="text"
            value={profileName}
            onChange={(e) => setProfileName(e.target.value)}
            placeholder="Profile name"
            className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded text-sm"
          />

          <div className="flex flex-wrap gap-2">
            <button
              disabled={disabled || busy}
              onClick={() => void runTry()}
              className="flex items-center gap-2 px-4 py-2 bg-white/15 hover:bg-white/25 rounded text-sm disabled:opacity-50"
            >
              <Zap className="w-4 h-4" />
              Try
            </button>
            <button
              disabled={disabled || busy}
              onClick={() => void saveProfile(false)}
              className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded text-sm disabled:opacity-50"
            >
              <Save className="w-4 h-4" />
              Save profile
            </button>
            <button
              disabled={disabled || busy}
              onClick={() => void saveProfile(true)}
              className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-white/90 text-black font-medium rounded text-sm disabled:opacity-50"
            >
              Save &amp; go live
            </button>
          </div>

      {error && <p className="text-sm text-red-400">{error}</p>}
      {statusMsg && <p className="text-sm text-emerald-400">{statusMsg}</p>}
    </div>
  );

  return (
    <div className={embedded ? 'space-y-5' : 'bg-white/5 rounded-xl border border-white/10 p-6 space-y-5'}>
      {embedded ? (
        <>
          {previewPane}
          {controlsPane}
        </>
      ) : (
        <div className="grid md:grid-cols-2 gap-6">
          {previewPane}
          {controlsPane}
        </div>
      )}
    </div>
  );
}
