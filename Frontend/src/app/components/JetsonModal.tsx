import { X, Camera, Thermometer, Radio } from 'lucide-react';
import { Device } from '../api/client';

interface JetsonModalProps {
  device: Device;
  onClose: () => void;
}

export function JetsonModal({ device, onClose }: JetsonModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-slate-900 rounded-lg border border-white/20 w-full max-w-md p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold">{device.name}</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-white/10 rounded transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="space-y-4">
          {/* Status */}
          <div>
            <div className="text-xs text-white/50 mb-1">Status</div>
            <div className="flex items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  device.status === 'online' ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="text-sm font-medium capitalize">{device.status}</span>
            </div>
          </div>

          {/* Location */}
          <div>
            <div className="text-xs text-white/50 mb-1">Location</div>
            <div className="text-sm">{device.location}</div>
          </div>

          {/* Sensors */}
          <div>
            <div className="text-xs text-white/50 mb-2">Available Sensors</div>
            <div className="flex gap-2">
              {device.cameras.webcam && (
                <div className="flex items-center gap-1 px-3 py-1.5 bg-white/10 rounded text-sm">
                  <Camera className="w-4 h-4" />
                  RGB Camera
                </div>
              )}
              {device.cameras.thermal && (
                <div className="flex items-center gap-1 px-3 py-1.5 bg-white/10 rounded text-sm">
                  <Thermometer className="w-4 h-4" />
                  Thermal
                </div>
              )}
              {device.mmwave && (
                <div className="flex items-center gap-1 px-3 py-1.5 bg-white/10 rounded text-sm">
                  <Radio className="w-4 h-4" />
                  mmWave
                </div>
              )}
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-white/5 rounded p-3">
              <div className="text-xs text-white/50 mb-1">FPS</div>
              <div className="text-lg font-mono">{device.fps.toFixed(1)}</div>
            </div>
            <div className="bg-white/5 rounded p-3">
              <div className="text-xs text-white/50 mb-1">Latency</div>
              <div className="text-lg font-mono">{device.latencyMs}ms</div>
            </div>
            {device.temperature && (
              <div className="bg-white/5 rounded p-3">
                <div className="text-xs text-white/50 mb-1">Temperature</div>
                <div className="text-lg font-mono">{device.temperature}°C</div>
              </div>
            )}
            <div className="bg-white/5 rounded p-3">
              <div className="text-xs text-white/50 mb-1">Last Heartbeat</div>
              <div className="text-xs">
                {new Date(device.lastHeartbeat).toLocaleTimeString('en-US')}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
