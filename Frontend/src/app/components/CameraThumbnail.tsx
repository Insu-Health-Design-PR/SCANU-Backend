import { Camera, Thermometer } from 'lucide-react';

interface CameraThumbnailProps {
  type: 'rgb' | 'thermal';
  title: string;
  deviceName: string;
  isActive: boolean;
  onClick: () => void;
}

export function CameraThumbnail({ type, title, deviceName, isActive, onClick }: CameraThumbnailProps) {
  return (
    <button
      onClick={onClick}
      className={`relative w-full aspect-video rounded-lg overflow-hidden border-2 transition-all ${
        isActive ? 'border-white' : 'border-white/20 hover:border-white/40'
      }`}
    >
      {/* Camera Feed Placeholder */}
      <div className={`absolute inset-0 ${
        type === 'thermal'
          ? 'bg-gradient-to-br from-orange-900/40 via-red-900/40 to-yellow-900/40'
          : 'bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900'
      }`}>
        <div className="absolute inset-0 flex items-center justify-center">
          {type === 'thermal' ? (
            <Thermometer className="w-8 h-8 text-orange-300/30" />
          ) : (
            <Camera className="w-8 h-8 text-white/20" />
          )}
        </div>
      </div>

      {/* Overlay Info */}
      <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded">
        <span className="text-xs text-white font-medium">{title}</span>
      </div>

      <div className="absolute bottom-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded">
        <span className="text-xs text-white/70">{deviceName}</span>
      </div>
    </button>
  );
}
