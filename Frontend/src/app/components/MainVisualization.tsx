import { Radio } from 'lucide-react';

export function MainVisualization() {
  return (
    <div className="w-full h-full bg-black rounded-lg border border-white/10 overflow-hidden">
      {/* Header */}
      <div className="h-12 border-b border-white/10 bg-black/40 backdrop-blur-sm flex items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Radio className="w-4 h-4 text-white/60" />
          <span className="text-sm font-medium">Point Cloud</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="text-xs text-white/50">FPS</div>
          <div className="text-xs text-white font-mono">28.4</div>
        </div>
      </div>

      {/* Point Cloud Visualization */}
      <div className="relative h-[calc(100%-3rem)] bg-black">
        <svg className="absolute inset-0 w-full h-full">
          {/* Grid lines */}
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />

          {/* Range circles */}
          <circle cx="50%" cy="50%" r="20%" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
          <circle cx="50%" cy="50%" r="35%" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
          <circle cx="50%" cy="50%" r="50%" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />

          {/* Point cloud dots */}
          {Array.from({ length: 80 }).map((_, i) => {
            const angle = (i / 80) * Math.PI * 2;
            const radius = 15 + Math.random() * 35;
            const cx = 50 + Math.cos(angle) * radius;
            const cy = 50 + Math.sin(angle) * radius;
            return (
              <circle
                key={i}
                cx={`${cx}%`}
                cy={`${cy}%`}
                r="1.5"
                fill="rgb(255, 255, 255)"
                opacity={0.2 + Math.random() * 0.5}
              />
            );
          })}

          {/* Detected objects with labels */}
          <g>
            <circle cx="35%" cy="40%" r="12" fill="none" stroke="rgb(255, 255, 255)" strokeWidth="2" />
            <text x="35%" y="55%" fill="white" fontSize="10" textAnchor="middle">Target 1</text>
          </g>
          <g>
            <circle cx="65%" cy="45%" r="12" fill="none" stroke="rgb(255, 255, 255)" strokeWidth="2" />
            <text x="65%" y="60%" fill="white" fontSize="10" textAnchor="middle">Target 2</text>
          </g>
        </svg>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur-sm p-3 rounded border border-white/10">
          <div className="text-xs text-white/70 space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-white" />
              <span>Detection Points</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full border border-white" />
              <span>Tracked Objects</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
