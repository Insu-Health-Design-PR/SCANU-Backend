import { Detection } from '../api/client';

interface DetectionOverlayProps {
  detections: Detection[];
  showTrails?: boolean;
  showIds?: boolean;
}

export function DetectionOverlay({
  detections,
  showTrails = true,
  showIds = true,
}: DetectionOverlayProps) {
  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      {showTrails &&
        detections
          .filter((d) => d.type === 'person' && d.trail.length > 1)
          .map((d) => {
            const points = d.trail.map((p) => `${p.x * 100},${p.y * 100}`).join(' ');
            const stroke = d.unsafe ? '#f87171' : '#38bdf8';
            return (
              <polyline
                key={`trail-${d.trackingId}`}
                points={points}
                fill="none"
                stroke={stroke}
                strokeWidth="0.25"
                strokeOpacity="0.7"
                strokeLinecap="round"
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
                strokeDasharray="0.5 0.5"
              />
            );
          })}

      {detections.map((d) => {
        const [x, y, w, h] = d.bbox;
        const isWeapon = d.type === 'weapon';
        const color = isWeapon || d.unsafe ? '#ef4444' : '#22d3ee';
        return (
          <g key={d.id}>
            <rect
              x={x * 100}
              y={y * 100}
              width={w * 100}
              height={h * 100}
              fill="none"
              stroke={color}
              strokeWidth={isWeapon ? '0.5' : '0.3'}
              vectorEffect="non-scaling-stroke"
              opacity={isWeapon ? 1 : 0.9}
            >
              {isWeapon && (
                <animate
                  attributeName="opacity"
                  values="0.4;1;0.4"
                  dur="0.8s"
                  repeatCount="indefinite"
                />
              )}
            </rect>
            {showIds && (
              <g>
                <rect
                  x={x * 100}
                  y={Math.max(0, y * 100 - 3)}
                  width={Math.max(8, d.trackingId.length * 1.5)}
                  height="2.6"
                  fill={color}
                  opacity="0.85"
                />
                <text
                  x={x * 100 + 0.5}
                  y={Math.max(2, y * 100 - 1)}
                  fontSize="2"
                  fill="#000"
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  {d.trackingId} {(d.confidence * 100).toFixed(0)}%
                </text>
              </g>
            )}
            {d.unsafe && d.type === 'person' && (
              <rect
                x={x * 100 - 0.5}
                y={y * 100 - 0.5}
                width={w * 100 + 1}
                height={h * 100 + 1}
                fill="none"
                stroke="#ef4444"
                strokeWidth="0.15"
                strokeDasharray="1 1"
                vectorEffect="non-scaling-stroke"
                opacity="0.6"
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
