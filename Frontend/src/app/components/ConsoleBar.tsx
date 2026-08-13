import { useMemo, useState } from 'react';
import { ChevronUp, ChevronDown, Terminal } from 'lucide-react';

interface ConsoleBarProps {
  sensorLogs: Record<'thermal' | 'webcam' | 'mmwave', string>;
  backendOnline: boolean;
}

function parseLogTail(tail: string, sensor: string): { time: string; level: string; message: string }[] {
  if (!tail.trim()) return [];
  const lines = tail.trim().split('\n').slice(-40);
  return lines.map((line) => {
    const msg = line.trim();
    const level = /error|fail|exception/i.test(msg)
      ? 'ERROR'
      : /warn/i.test(msg)
        ? 'WARN'
        : 'INFO';
    return {
      time: '',
      level,
      message: `[${sensor}] ${msg}`,
    };
  });
}

export function ConsoleBar({ sensorLogs, backendOnline }: ConsoleBarProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const logs = useMemo(() => {
    if (!backendOnline) {
      return [
        {
          time: '',
          level: 'WARN',
          message: 'Layer 8 backend not reachable — check :8088 and Vite proxy',
        },
      ];
    }
    const merged = [
      ...parseLogTail(sensorLogs.webcam, 'webcam'),
      ...parseLogTail(sensorLogs.thermal, 'thermal'),
      ...parseLogTail(sensorLogs.mmwave, 'mmwave'),
    ];
    if (merged.length === 0) {
      return [
        {
          time: '',
          level: 'INFO',
          message: 'Connected — waiting for sensor runner log output',
        },
      ];
    }
    return merged.slice(-60);
  }, [sensorLogs, backendOnline]);

  return (
    <div className="border-t border-white/10 bg-black/40 backdrop-blur-sm">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full h-10 px-4 flex items-center justify-between hover:bg-white/5 transition-all"
      >
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-white/60" />
          <span className="text-sm text-white/70">Sensor logs (Layer 8)</span>
          <span className="text-xs text-white/40">({logs.length} lines)</span>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-white/40" />
        ) : (
          <ChevronUp className="w-4 h-4 text-white/40" />
        )}
      </button>

      {isExpanded && (
        <div className="h-48 overflow-y-auto border-t border-white/10 bg-black/60 p-4">
          <div className="space-y-1 font-mono text-xs">
            {logs.map((log, i) => (
              <div key={i} className="flex items-start gap-3 hover:bg-white/5 p-1 rounded">
                {log.time ? <span className="text-white/40">{log.time}</span> : null}
                <span
                  className={`font-semibold shrink-0 ${
                    log.level === 'ERROR'
                      ? 'text-red-400'
                      : log.level === 'WARN'
                        ? 'text-amber-400'
                        : 'text-cyan-400'
                  }`}
                >
                  [{log.level}]
                </span>
                <span className="text-white/80 break-all">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
