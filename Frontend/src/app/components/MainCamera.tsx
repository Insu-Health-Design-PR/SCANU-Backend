import {
  Camera,
  Thermometer,
  Radio,
  Circle,
  Crosshair,
  Route,
  Tag,
  Server,
  Settings,
  ImageDown,
  SlidersHorizontal,
  LayoutGrid,
  Check,
} from 'lucide-react';
import { Device, Detection, DashboardMetrics, previewUrls } from '../api/client';
import { useState } from 'react';
import { DetectionOverlay } from './DetectionOverlay';
import { SensorConfigureDrawer, ConfigureSensor } from './SensorConfigureDrawer';

export type PreviewLayout = 'single' | 'dual' | 'triple';

const PREVIEW_LAYOUT_OPTIONS: { id: PreviewLayout; label: string; hint: string }[] = [
  { id: 'single', label: 'Single preview', hint: 'One camera — use tabs' },
  { id: 'dual', label: '2 preview', hint: 'Front + Back side by side' },
  { id: 'triple', label: 'Triple view', hint: 'Front + Back + Thermal' },
];

interface MainCameraProps {
  device?: Device;
  detections: Detection[];
  metrics: DashboardMetrics;
  alertsCount: number;
  activeView: 'rgb' | 'back' | 'thermal' | 'mmwave';
  previewLayout: PreviewLayout;
  showBoxes: boolean;
  showTrails: boolean;
  showIds: boolean;
  backendOnline?: boolean;
  sensorRunning: { webcam: boolean; thermal: boolean; mmwave: boolean; multi_camera: boolean };
  onViewChange: (view: 'rgb' | 'back' | 'thermal' | 'mmwave') => void;
  onPreviewLayoutChange: (layout: PreviewLayout) => void;
  onToggleBoxes: () => void;
  onToggleTrails: () => void;
  onToggleIds: () => void;
  onScreenshot?: () => void;
}

function streamUrl(viewType: 'rgb' | 'back' | 'thermal' | 'mmwave'): string {
  if (viewType === 'thermal') return previewUrls.thermal;
  if (viewType === 'mmwave') return previewUrls.mmwave;
  if (viewType === 'back') return previewUrls.back;
  return previewUrls.front;
}

function sensorKey(viewType: 'rgb' | 'back' | 'thermal' | 'mmwave'): 'webcam' | 'multi_camera' | 'thermal' | 'mmwave' {
  if (viewType === 'back') return 'multi_camera';
  if (viewType === 'thermal') return 'thermal';
  if (viewType === 'mmwave') return 'mmwave';
  return 'webcam';
}

function viewLabel(viewType: 'rgb' | 'back' | 'thermal' | 'mmwave'): string {
  if (viewType === 'back') return 'Back Camera';
  if (viewType === 'thermal') return 'Thermal';
  if (viewType === 'mmwave') return 'mmWave';
  return 'Front Camera';
}

function NoStreamPanel({
  title,
  detail,
}: {
  title: string;
  detail: string;
}) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-black">
      <div className="flex flex-col items-center gap-2 px-6 py-4 text-center max-w-md">
        <Server className="w-8 h-8 text-white/30" />
        <p className="text-sm font-medium text-white/70">{title}</p>
        <p className="text-xs text-white/40">{detail}</p>
      </div>
    </div>
  );
}

export function MainCamera({
  device,
  detections,
  metrics,
  alertsCount,
  activeView,
  previewLayout,
  showBoxes,
  showTrails,
  showIds,
  backendOnline = false,
  sensorRunning,
  onViewChange,
  onPreviewLayoutChange,
  onToggleBoxes,
  onToggleTrails,
  onToggleIds,
  onScreenshot,
}: MainCameraProps) {
  const hasAlert = metrics.gunDetected && metrics.inferActive;
  const [showDisplayOptions, setShowDisplayOptions] = useState(false);
  const [showViewMenu, setShowViewMenu] = useState(false);
  const [configureOpen, setConfigureOpen] = useState(false);
  const hasOverlayData = detections.length > 0;
  const singlePreview = previewLayout === 'single';
  const layoutLabel = PREVIEW_LAYOUT_OPTIONS.find((o) => o.id === previewLayout)?.label ?? 'View';

  const configureSensor: ConfigureSensor =
    activeView === 'thermal' ? 'thermal' : activeView === 'mmwave' ? 'mmwave' : 'webcam';

  const renderCameraView = (viewType: 'rgb' | 'back' | 'thermal' | 'mmwave') => {
    const sk = sensorKey(viewType);
    const running = sensorRunning[sk];
    const showStream = backendOnline && running;
    const label = viewLabel(viewType);

    return (
      <div className="relative w-full h-full bg-black rounded-lg border border-white/10 overflow-hidden min-h-[280px]">
        <div className="absolute inset-0">
          {!backendOnline && (
            <NoStreamPanel
              title="Backend offline"
              detail="Start Layer 8 on :8088 — UI polls /api/status/stream"
            />
          )}
          {backendOnline && !running && (
            <NoStreamPanel
              title={`${label} runner stopped`}
              detail={`POST /api/${sk === 'multi_camera' ? 'back_camera' : sk}/run to start live preview`}
            />
          )}
          {showStream && (
            <img
              src={streamUrl(viewType)}
              alt={`${label} live`}
              className="w-full h-full object-contain bg-black"
            />
          )}
        </div>

        {showStream && showBoxes && hasOverlayData && (
          <DetectionOverlay
            detections={detections}
            showTrails={showTrails}
            showIds={showIds}
          />
        )}

        {showStream && hasAlert && viewType !== 'mmwave' && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-red-500/90 px-3 py-1 rounded">
            <Circle className="w-2 h-2 fill-white text-white" />
            <span className="text-xs text-white font-bold uppercase tracking-wider">
              Threat active
            </span>
          </div>
        )}

        {showStream && (
          <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded">
            <Circle className="w-2 h-2 fill-red-500 text-red-500 animate-pulse" />
            <span className="text-xs text-white font-medium">LIVE</span>
          </div>
        )}

        <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm px-2 py-1 rounded">
          <span className="text-xs text-white/80">{label}</span>
        </div>

        {backendOnline && metrics.inferActive && (
          <div className="absolute bottom-3 left-3 flex gap-3 text-xs">
            <div className="bg-black/60 backdrop-blur-sm px-2 py-1 rounded">
              <span className="text-white/50">Persons </span>
              <span className="text-white font-mono">
                {metrics.personDetections ?? '—'}
              </span>
            </div>
            <div className="bg-black/60 backdrop-blur-sm px-2 py-1 rounded">
              <span className="text-white/50">Armed </span>
              <span className={`font-mono ${metrics.gunDetected ? 'text-red-400' : 'text-white'}`}>
                {metrics.gunDetected ? (metrics.personsWithGun ?? '—') : '0'}
              </span>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden min-h-0 bg-slate-950">
      {/* Chrome-style sensor tabs */}
      <div className="shrink-0 flex items-end gap-0 px-4 pt-2 bg-[#1c2028] border-b border-white/10">
        {(
          [
            { id: 'rgb' as const, label: 'Front', icon: Camera },
            { id: 'back' as const, label: 'Back', icon: Camera },
            { id: 'thermal' as const, label: 'Thermal', icon: Thermometer },
            { id: 'mmwave' as const, label: 'mmWave', icon: Radio },
          ] as const
        ).map(({ id, label, icon: Icon }) => {
          const active = singlePreview && activeView === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => {
                if (!singlePreview) onPreviewLayoutChange('single');
                onViewChange(id);
              }}
              className={`relative flex items-center gap-2 px-4 py-2.5 text-sm rounded-t-lg border border-b-0 transition-colors min-w-[120px] ${
                active
                  ? 'bg-slate-950 border-white/15 text-white z-10 -mb-px shadow-[0_-1px_0_0_rgba(255,255,255,0.05)_inset]'
                  : 'bg-white/[0.04] border-transparent text-white/45 hover:bg-white/[0.08] hover:text-white/75 mb-0'
              }`}
            >
              <Icon className="w-3.5 h-3.5 shrink-0" />
              {label}
            </button>
          );
        })}

        <div className="flex-1 min-w-4 border-b border-transparent self-stretch" />

          <div className="flex items-center gap-1.5 pb-2 pr-1">
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowViewMenu((v) => !v)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md border transition-all text-xs ${
                  previewLayout !== 'single'
                    ? 'border-cyan-500/40 bg-cyan-500/10 text-cyan-200'
                    : 'border-white/10 bg-white/[0.04] text-white/80 hover:bg-white/10'
                }`}
              >
                <LayoutGrid className="w-3.5 h-3.5" />
                View
                <span className="text-white/45 hidden sm:inline">· {layoutLabel}</span>
              </button>
              {showViewMenu && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => setShowViewMenu(false)} />
                  <div className="absolute right-0 top-full mt-2 bg-slate-900/98 backdrop-blur-sm border border-white/20 rounded-lg shadow-2xl z-20 p-1.5 min-w-[220px]">
                    {PREVIEW_LAYOUT_OPTIONS.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => {
                          onPreviewLayoutChange(opt.id);
                          setShowViewMenu(false);
                        }}
                        className={`w-full flex items-start gap-2 px-3 py-2.5 rounded-md text-left transition-colors ${
                          previewLayout === opt.id
                            ? 'bg-cyan-500/15 text-cyan-100'
                            : 'text-white/75 hover:bg-white/10'
                        }`}
                      >
                        <span className="mt-0.5 w-4 shrink-0">
                          {previewLayout === opt.id && <Check className="w-4 h-4 text-cyan-400" />}
                        </span>
                        <span>
                          <span className="block text-sm font-medium">{opt.label}</span>
                          <span className="block text-[11px] text-white/40 mt-0.5">{opt.hint}</span>
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>

            <button
              type="button"
              onClick={() => setConfigureOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-white/10 bg-white/[0.04] hover:bg-white/10 transition-all text-xs text-white/80"
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
              Configure
            </button>

            <div className="w-px h-6 bg-white/10 mx-0.5" />

          {onScreenshot && (
            <button
              type="button"
              onClick={onScreenshot}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white/5 hover:bg-white/10 transition-all text-xs text-white/70"
              title="Capture screenshot of active stream"
            >
              <ImageDown className="w-3.5 h-3.5" />
              Screenshot
            </button>
          )}

          {activeView !== 'mmwave' && hasOverlayData && (
            <div className="relative">
              <button
                onClick={() => setShowDisplayOptions(!showDisplayOptions)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-white/5 hover:bg-white/10 transition-all text-xs text-white/70"
              >
                <Settings className="w-3.5 h-3.5" />
                Overlays
              </button>

              {showDisplayOptions && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => setShowDisplayOptions(false)} />
                  <div className="absolute right-0 top-full mt-2 bg-slate-900/95 backdrop-blur-sm border border-white/20 rounded-lg shadow-2xl z-20 p-2 min-w-[180px]">
                    <div className="flex flex-col gap-1">
                      <button
                        onClick={onToggleBoxes}
                        className={`flex items-center gap-2 px-3 py-2 rounded text-sm transition-all ${
                          showBoxes ? 'bg-cyan-500/20 text-cyan-300' : 'bg-white/5 text-white/70 hover:bg-white/10'
                        }`}
                      >
                        <Crosshair className="w-4 h-4" />
                        <span className="flex-1 text-left">Boxes</span>
                        {showBoxes && <Circle className="w-2 h-2 fill-cyan-400 text-cyan-400" />}
                      </button>
                      <button
                        onClick={onToggleIds}
                        className={`flex items-center gap-2 px-3 py-2 rounded text-sm transition-all ${
                          showIds ? 'bg-cyan-500/20 text-cyan-300' : 'bg-white/5 text-white/70 hover:bg-white/10'
                        }`}
                      >
                        <Tag className="w-4 h-4" />
                        <span className="flex-1 text-left">IDs</span>
                        {showIds && <Circle className="w-2 h-2 fill-cyan-400 text-cyan-400" />}
                      </button>
                      <button
                        onClick={onToggleTrails}
                        className={`flex items-center gap-2 px-3 py-2 rounded text-sm transition-all ${
                          showTrails ? 'bg-cyan-500/20 text-cyan-300' : 'bg-white/5 text-white/70 hover:bg-white/10'
                        }`}
                      >
                        <Route className="w-4 h-4" />
                        <span className="flex-1 text-left">Trails</span>
                        {showTrails && <Circle className="w-2 h-2 fill-cyan-400 text-cyan-400" />}
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
          </div>
      </div>

      <SensorConfigureDrawer
        open={configureOpen}
        sensor={configureSensor}
        onClose={() => setConfigureOpen(false)}
      />

      <div className="flex-1 min-h-0 p-4">
        {previewLayout === 'dual' && (
          <div className="grid grid-cols-2 gap-4 h-full">
            {renderCameraView('rgb')}
            {renderCameraView('back')}
          </div>
        )}
        {previewLayout === 'triple' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-full">
            {renderCameraView('rgb')}
            {renderCameraView('back')}
            {renderCameraView('thermal')}
          </div>
        )}
        {previewLayout === 'single' && renderCameraView(activeView)}
      </div>
    </div>
  );
}
