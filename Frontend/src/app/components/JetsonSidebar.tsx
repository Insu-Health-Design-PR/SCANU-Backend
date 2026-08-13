import { Circle } from 'lucide-react';
import { Device } from '../api/client';

interface JetsonSidebarProps {
  devices: Device[];
  selectedDeviceId: string | null;
  onJetsonClick: (device: Device) => void;
  onJetsonRightClick: (device: Device) => void;
}

export function JetsonSidebar({ devices, selectedDeviceId, onJetsonClick, onJetsonRightClick }: JetsonSidebarProps) {
  return (
    <div className="w-28 border-r border-white/10 bg-black/20 backdrop-blur-sm flex flex-col">
      <div className="p-3 border-b border-white/10">
        <div className="text-xs text-white/50 font-medium">Devices</div>
      </div>

      <div className="flex-1 overflow-y-auto py-2">
        {devices.map((device) => (
          <button
            key={device.id}
            onClick={() => onJetsonClick(device)}
            onContextMenu={(e) => {
              e.preventDefault();
              onJetsonRightClick(device);
            }}
            className={`w-full p-3 flex flex-col items-center gap-2 transition-all hover:bg-white/5 ${
              selectedDeviceId === device.id ? 'bg-white/10' : ''
            }`}
          >
            <Circle
              className={`w-3 h-3 ${
                device.status === 'online'
                  ? 'fill-green-500 text-green-500'
                  : device.status === 'degraded'
                  ? 'fill-amber-400 text-amber-400 animate-pulse'
                  : 'fill-red-500 text-red-500'
              }`}
            />
            <div className="text-xs text-center font-medium break-words w-full">
              {device.name}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
