import { Camera, Thermometer, Radio, Maximize2, Circle } from 'lucide-react';
import { useState } from 'react';

type CameraType = 'webcam' | 'thermal' | 'mmwave';

interface CameraPanelProps {
  deviceName: string;
  fps: number;
  latencyMs: number;
}

export function CameraPanel({ deviceName, fps, latencyMs }: CameraPanelProps) {
  const [activeCamera, setActiveCamera] = useState<CameraType>('webcam');
  const [showOverlay, setShowOverlay] = useState(true);

  // Mock detection data
  const detections = activeCamera === 'webcam' ? [
    { id: 'p1', type: 'person', x: 120, y: 80, w: 80, h: 180, confidence: 0.94 },
    { id: 'p2', type: 'person', x: 320, y: 100, w: 70, h: 160, confidence: 0.88 },
  ] : [];

  return (
    <div className="flex-1 flex flex-col bg-black/40 rounded-xl border border-white/10 overflow-hidden">
      {/* Camera Selector Tabs */}
      <div className="flex items-center gap-2 p-3 border-b border-white/10 bg-black/20">
        <button
          onClick={() => setActiveCamera('webcam')}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
            activeCamera === 'webcam'
              ? 'bg-white text-black'
              : 'bg-white/5 text-white/60 hover:bg-white/10'
          }`}
        >
          <Camera className="w-4 h-4" />
          RGB Camera
        </button>
        <button
          onClick={() => setActiveCamera('thermal')}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
            activeCamera === 'thermal'
              ? 'bg-white text-black'
              : 'bg-white/5 text-white/60 hover:bg-white/10'
          }`}
        >
          <Thermometer className="w-4 h-4" />
          Thermal
        </button>
        <button
          onClick={() => setActiveCamera('mmwave')}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
            activeCamera === 'mmwave'
              ? 'bg-white text-black'
              : 'bg-white/5 text-white/60 hover:bg-white/10'
          }`}
        >
          <Radio className="w-4 h-4" />
          mmWave
        </button>

        <div className="flex-1" />

        <button
          onClick={() => setShowOverlay(!showOverlay)}
          className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${
            showOverlay ? 'bg-white/20 text-white' : 'bg-white/5 text-white/60'
          }`}
        >
          Overlay
        </button>

        <button className="p-2 rounded-lg bg-white/5 text-white/60 hover:bg-white/10">
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Camera Feed */}
      <div className="flex-1 relative bg-black overflow-hidden">
        {/* Simulated camera feed */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
          {activeCamera === 'webcam' && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-white/20 text-6xl">
                <Camera className="w-32 h-32" />
              </div>
            </div>
          )}

          {activeCamera === 'thermal' && (
            <div className="absolute inset-0 bg-gradient-to-br from-orange-900/40 via-red-900/40 to-yellow-900/40">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-orange-300/30 text-6xl">
                  <Thermometer className="w-32 h-32" />
                </div>
              </div>
              {/* Simulated heat signatures */}
              {showOverlay && (
                <>
                  <div className="absolute left-[20%] top-[30%] w-24 h-32 rounded-full bg-gradient-radial from-yellow-400/60 via-orange-500/40 to-transparent blur-xl" />
                  <div className="absolute right-[25%] top-[35%] w-20 h-28 rounded-full bg-gradient-radial from-yellow-400/50 via-orange-500/30 to-transparent blur-xl" />
                </>
              )}
            </div>
          )}

          {activeCamera === 'mmwave' && (
            <div className="absolute inset-0 bg-black">
              {/* Range rings */}
              <svg className="absolute inset-0 w-full h-full">
                <circle cx="50%" cy="50%" r="30%" fill="none" stroke="rgba(255, 255, 255, 0.1)" strokeWidth="1" />
                <circle cx="50%" cy="50%" r="45%" fill="none" stroke="rgba(255, 255, 255, 0.1)" strokeWidth="1" />
                <circle cx="50%" cy="50%" r="60%" fill="none" stroke="rgba(255, 255, 255, 0.1)" strokeWidth="1" />

                {/* Mock point cloud */}
                {showOverlay && Array.from({ length: 40 }).map((_, i) => {
                  const angle = (i / 40) * Math.PI * 2;
                  const radius = 30 + Math.random() * 30;
                  const cx = 50 + Math.cos(angle) * radius;
                  const cy = 50 + Math.sin(angle) * radius;
                  return (
                    <circle
                      key={i}
                      cx={`${cx}%`}
                      cy={`${cy}%`}
                      r="2"
                      fill="rgb(255, 255, 255)"
                      opacity={0.3 + Math.random() * 0.4}
                    />
                  );
                })}

                {/* Detected objects */}
                <circle cx="35%" cy="40%" r="8" fill="none" stroke="rgb(255, 255, 255)" strokeWidth="2" />
                <circle cx="65%" cy="45%" r="8" fill="none" stroke="rgb(255, 255, 255)" strokeWidth="2" />
              </svg>
            </div>
          )}

          {/* Detection Overlays for RGB */}
          {activeCamera === 'webcam' && showOverlay && detections.map((det) => (
            <div
              key={det.id}
              className="absolute border-2 border-white"
              style={{
                left: det.x,
                top: det.y,
                width: det.w,
                height: det.h,
              }}
            >
              <div className="absolute -top-6 left-0 bg-white text-black text-xs px-2 py-0.5 rounded">
                {det.type} {(det.confidence * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>

        {/* Recording Indicator */}
        <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/60 backdrop-blur-sm px-3 py-2 rounded-lg">
          <Circle className="w-2 h-2 fill-red-500 text-red-500 animate-pulse" />
          <span className="text-xs text-white font-medium">LIVE</span>
        </div>

        {/* Device Name */}
        <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-sm px-3 py-2 rounded-lg">
          <span className="text-xs text-white/80 font-medium">{deviceName}</span>
        </div>
      </div>

      {/* Footer Stats */}
      <div className="h-12 border-t border-white/10 bg-black/20 flex items-center justify-between px-4">
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <span className="text-white/50">FPS</span>
            <span className="font-mono text-white">{fps.toFixed(1)}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-white/50">Latency</span>
            <span className="font-mono text-white">{latencyMs}ms</span>
          </div>
          {activeCamera === 'webcam' && (
            <div className="flex items-center gap-2">
              <span className="text-white/50">Detections</span>
              <span className="font-mono text-white">{detections.length}</span>
            </div>
          )}
        </div>

        <div className="text-xs text-white/40">
          {new Date().toLocaleTimeString('en-US')}
        </div>
      </div>
    </div>
  );
}
