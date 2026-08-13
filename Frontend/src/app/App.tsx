import { useEffect, useMemo, useState } from 'react';
import { Header } from './components/Header';
import { ConsoleBar } from './components/ConsoleBar';
import { JetsonModal } from './components/JetsonModal';
import { SlabManagementPage } from './components/SlabManagementPage';
import { SlabDetailView } from './components/SlabDetailView';
import { PlaygroundView } from './components/PlaygroundView';
import { ControlPanel } from './components/ControlPanel';
import {
  scanuClient,
  DashboardSnapshot,
  Device,
  OperatorMode,
} from './api/client';
import { slabToDevice, slabsFromSnapshot } from './types/slab';
import { downloadDataUrl, slabScreenshotName } from './utils/capturePreview';

type AppPage = 'slabs' | 'slab-detail' | 'playground' | 'control';

export default function App() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot>(() => ({
    devices: [],
    metrics: {
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
    },
    status: {
      mode: 'central',
      state: 'normal',
      backendOnline: false,
      activeAlerts: 0,
      timestamp: new Date().toISOString(),
    },
    alerts: [],
    detections: {},
    sensorLogs: { thermal: '', webcam: '', mmwave: '' },
    sensorRunning: { thermal: false, webcam: false, mmwave: false },
  }));

  const [page, setPage] = useState<AppPage>('slabs');
  const [selectedSlabId, setSelectedSlabId] = useState<string | null>(null);
  const [operatorMode, setOperatorMode] = useState<OperatorMode>('central');

  const [jetsonModalOpen, setJetsonModalOpen] = useState(false);
  const [selectedJetsonForModal, setSelectedJetsonForModal] = useState<Device | null>(null);

  const [activeView, setActiveView] = useState<'rgb' | 'thermal' | 'mmwave'>('rgb');
  const [previewLayout, setPreviewLayout] = useState<'single' | 'dual' | 'triple'>('single');
  const [showBoxes, setShowBoxes] = useState(true);
  const [showIds, setShowIds] = useState(true);
  const [showTrails, setShowTrails] = useState(true);

  useEffect(() => {
    return scanuClient.subscribe(setSnapshot);
  }, []);

  const { devices, metrics, status, alerts, detections, sensorLogs, sensorRunning } = snapshot;
  const slabs = useMemo(() => slabsFromSnapshot(snapshot), [snapshot]);
  const selectedSlab = slabs.find((s) => s.id === selectedSlabId);
  const selectedDevice = selectedSlabId ? slabToDevice(selectedSlabId, devices) : undefined;
  const deviceDetections = selectedSlabId ? detections[selectedSlabId] ?? [] : [];

  const handleSelectSlab = (slabId: string) => {
    setSelectedSlabId(slabId);
    scanuClient.selectDevice(slabId);
    setPage('slab-detail');
  };

  const handleOperatorModeChange = (mode: OperatorMode) => {
    setOperatorMode(mode);
    scanuClient.setOperatorMode(mode);
  };

  const handleScreenshot = async () => {
    const sensor =
      activeView === 'thermal' ? 'thermal' : activeView === 'mmwave' ? 'mmwave' : 'webcam';
    const dataUrl = await scanuClient.captureScreenshot(sensor);
    if (dataUrl && selectedSlab) {
      downloadDataUrl(dataUrl, slabScreenshotName(selectedSlab.slabId, sensor));
    }
  };

  return (
    <div className="dark size-full flex flex-col bg-slate-950 text-white overflow-hidden">
      <Header
        systemStatus={status}
        operatorMode={operatorMode}
        onOperatorModeChange={handleOperatorModeChange}
        page={page}
        onNavigate={setPage}
        onBackToSlabs={() => {
          setPage('slabs');
          setSelectedSlabId(null);
        }}
      />

      <div className="flex-1 flex overflow-hidden min-h-0">
        {page === 'slabs' && (
          <SlabManagementPage snapshot={snapshot} onSelectSlab={handleSelectSlab} />
        )}

        {page === 'slab-detail' && selectedSlab && (
          <SlabDetailView
            slab={selectedSlab}
            device={selectedDevice}
            detections={deviceDetections}
            metrics={metrics}
            alerts={alerts}
            operatorMode={operatorMode}
            backendOnline={status.backendOnline}
            sensorRunning={sensorRunning}
            activeView={activeView}
            previewLayout={previewLayout}
            showBoxes={showBoxes}
            showIds={showIds}
            showTrails={showTrails}
            onBack={() => {
              setPage('slabs');
              setSelectedSlabId(null);
            }}
            onViewChange={setActiveView}
            onPreviewLayoutChange={setPreviewLayout}
            onToggleBoxes={() => setShowBoxes((v) => !v)}
            onToggleIds={() => setShowIds((v) => !v)}
            onToggleTrails={() => setShowTrails((v) => !v)}
            onScreenshot={() => void handleScreenshot()}
          />
        )}

        {page === 'playground' && <PlaygroundView />}
        {page === 'control' && <ControlPanel metrics={metrics} />}
      </div>

      <ConsoleBar sensorLogs={sensorLogs} backendOnline={status.backendOnline} />

      {jetsonModalOpen && selectedJetsonForModal && (
        <JetsonModal device={selectedJetsonForModal} onClose={() => setJetsonModalOpen(false)} />
      )}
    </div>
  );
}
