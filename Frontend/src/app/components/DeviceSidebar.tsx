import { Camera, Radio, Thermometer, ChevronRight, Circle } from 'lucide-react';
import { Device } from '../api/client';
import { useState } from 'react';

interface DeviceSidebarProps {
  devices: Device[];
  selectedDeviceId: string | null;
  onSelectDevice: (deviceId: string) => void;
}

export function DeviceSidebar({ devices, selectedDeviceId, onSelectDevice }: DeviceSidebarProps) {
  const [expandedDevice, setExpandedDevice] = useState<string | null>(null);

  const getStatusColor = (status: Device['status']) => {
    switch (status) {
      case 'online':
        return 'bg-white';
      case 'degraded':
        return 'bg-white/50';
      case 'offline':
        return 'bg-red-400';
    }
  };

  return (
    <div className="w-80 border-r border-white/10 bg-black/20 backdrop-blur-sm flex flex-col">
      <div className="p-4 border-b border-white/10">
        <h2 className="font-semibold text-sm text-white/70 uppercase tracking-wider">
          Jetson Devices
        </h2>
        <p className="text-xs text-white/40 mt-1">{devices.length} units detected</p>
      </div>

      <div className="flex-1 overflow-y-auto">
        {devices.map((device) => {
          const isSelected = selectedDeviceId === device.id;
          const isExpanded = expandedDevice === device.id;

          return (
            <div key={device.id} className="border-b border-white/5">
              <button
                onClick={() => {
                  onSelectDevice(device.id);
                  setExpandedDevice(isExpanded ? null : device.id);
                }}
                className={`w-full p-4 text-left transition-all hover:bg-white/5 ${
                  isSelected ? 'bg-white/10 border-l-2 border-white' : ''
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${getStatusColor(device.status)}`} />
                      <h3 className="font-medium text-sm">{device.name}</h3>
                    </div>
                    <p className="text-xs text-white/50 mt-1">{device.location}</p>

                    {/* Sensor Icons */}
                    <div className="flex items-center gap-2 mt-2">
                      {device.cameras.webcam && (
                        <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-white/10 text-white/80">
                          <Camera className="w-3 h-3" />
                          <span className="text-xs">RGB</span>
                        </div>
                      )}
                      {device.cameras.thermal && (
                        <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-white/10 text-white/80">
                          <Thermometer className="w-3 h-3" />
                          <span className="text-xs">IR</span>
                        </div>
                      )}
                      {device.mmwave && (
                        <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-white/10 text-white/80">
                          <Radio className="w-3 h-3" />
                          <span className="text-xs">mmW</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <ChevronRight
                    className={`w-4 h-4 text-white/40 transition-transform ${
                      isExpanded ? 'rotate-90' : ''
                    }`}
                  />
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="mt-3 pt-3 border-t border-white/10 space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-white/50">FPS</span>
                      <span className="text-white font-mono">{device.fps.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-white/50">Latency</span>
                      <span className="text-white font-mono">{device.latencyMs}ms</span>
                    </div>
                    {device.temperature && (
                      <div className="flex justify-between text-xs">
                        <span className="text-white/50">Temperature</span>
                        <span className="text-white font-mono">{device.temperature}°C</span>
                      </div>
                    )}
                    <div className="flex justify-between text-xs">
                      <span className="text-white/50">Last Heartbeat</span>
                      <span className="text-white/60">
                        {new Date(device.lastHeartbeat).toLocaleTimeString('en-US')}
                      </span>
                    </div>
                  </div>
                )}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
