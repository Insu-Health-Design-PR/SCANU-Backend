import { ArrowLeft } from 'lucide-react';
import { Alert, DashboardMetrics, Detection, Device, OperatorMode } from '../api/client';
import { AiSlab, SLAB_REGISTRY } from '../types/slab';
import { CompactStats } from './CompactStats';
import { MainCamera, PreviewLayout } from './MainCamera';
import { SlabHostMetrics } from './SlabHostMetrics';

interface SlabDetailViewProps {
  slab: AiSlab;
  device?: Device;
  detections: Detection[];
  metrics: DashboardMetrics;
  alerts: Alert[];
  operatorMode: OperatorMode;
  backendOnline: boolean;
  sensorRunning: { webcam: boolean; thermal: boolean; mmwave: boolean; multi_camera: boolean };
  activeView: 'rgb' | 'back' | 'thermal' | 'mmwave';
  previewLayout: PreviewLayout;
  showBoxes: boolean;
  showIds: boolean;
  showTrails: boolean;
  onBack: () => void;
  onViewChange: (view: 'rgb' | 'back' | 'thermal' | 'mmwave') => void;
  onPreviewLayoutChange: (layout: PreviewLayout) => void;
  onToggleBoxes: () => void;
  onToggleIds: () => void;
  onToggleTrails: () => void;
  onScreenshot: () => void;
}

export function SlabDetailView({
  slab,
  device,
  detections,
  metrics,
  alerts,
  operatorMode,
  backendOnline,
  sensorRunning,
  activeView,
  previewLayout,
  showBoxes,
  showIds,
  showTrails,
  onBack,
  onViewChange,
  onPreviewLayoutChange,
  onToggleBoxes,
  onToggleIds,
  onToggleTrails,
  onScreenshot,
}: SlabDetailViewProps) {
  const reg = SLAB_REGISTRY.find((r) => r.id === slab.id);

  return (
    <div className="flex-1 flex flex-col overflow-hidden min-w-0">
      <div className="shrink-0 flex items-center gap-4 px-6 py-3 border-b border-white/10 bg-[#141820] flex-wrap">
        <button
          type="button"
          onClick={onBack}
          className="flex items-center gap-2 text-sm text-white/60 hover:text-white transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          All slabs
        </button>
        <div className="h-4 w-px bg-white/10" />
        <div>
          <div className="text-sm font-semibold text-white">{slab.slabId}</div>
          <div className="text-xs text-white/40 font-mono">{reg?.ip ?? slab.ip}</div>
        </div>
        <div className="ml-auto">
          <SlabHostMetrics metrics={metrics} />
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden min-h-0">
        <MainCamera
          device={device}
          detections={detections}
          metrics={metrics}
          alertsCount={alerts.length}
          activeView={activeView}
          previewLayout={previewLayout}
          showBoxes={showBoxes}
          showIds={showIds}
          showTrails={showTrails}
          backendOnline={backendOnline}
          sensorRunning={sensorRunning}
          onViewChange={onViewChange}
          onPreviewLayoutChange={onPreviewLayoutChange}
          onToggleBoxes={onToggleBoxes}
          onToggleIds={onToggleIds}
          onToggleTrails={onToggleTrails}
          onScreenshot={onScreenshot}
        />
        <CompactStats
          metrics={metrics}
          alerts={alerts}
          operatorMode={operatorMode}
          slabId={slab.slabId}
        />
      </div>
    </div>
  );
}
