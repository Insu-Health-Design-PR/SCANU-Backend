import { ChevronUp, ChevronDown, Terminal, Info, AlertTriangle, XCircle } from 'lucide-react';
import { useState, useEffect } from 'react';

type LogLevel = 'info' | 'warning' | 'error';

interface LogEntry {
  id: string;
  level: LogLevel;
  message: string;
  timestamp: Date;
}

export function ConsolePanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: '1',
      level: 'info',
      message: 'System initialized successfully',
      timestamp: new Date(Date.now() - 10000),
    },
    {
      id: '2',
      level: 'info',
      message: 'Connected to Jetson Alpha (jetson-01)',
      timestamp: new Date(Date.now() - 8000),
    },
    {
      id: '3',
      level: 'info',
      message: 'RGB camera stream active - 28 FPS',
      timestamp: new Date(Date.now() - 5000),
    },
    {
      id: '4',
      level: 'warning',
      message: 'Jetson Gamma (jetson-03) - elevated temperature detected (61°C)',
      timestamp: new Date(Date.now() - 3000),
    },
  ]);

  const [filterLevel, setFilterLevel] = useState<LogLevel | 'all'>('all');

  const filteredLogs = filterLevel === 'all'
    ? logs
    : logs.filter(log => log.level === filterLevel);

  const getLogIcon = (level: LogLevel) => {
    switch (level) {
      case 'info':
        return <Info className="w-4 h-4 text-white/60" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-white/60" />;
      case 'error':
        return <XCircle className="w-4 h-4 text-red-400" />;
    }
  };

  const getLogColor = (level: LogLevel) => {
    switch (level) {
      case 'info':
        return 'text-white/60';
      case 'warning':
        return 'text-white/60';
      case 'error':
        return 'text-red-400/70';
    }
  };

  return (
    <div className="border-t border-white/10 bg-black/40 backdrop-blur-sm">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full h-12 px-4 flex items-center justify-between hover:bg-white/5 transition-all"
      >
        <div className="flex items-center gap-3">
          <Terminal className="w-4 h-4 text-white/60" />
          <h3 className="font-semibold text-sm text-white/70 uppercase tracking-wider">
            System Console
          </h3>
          <span className="px-2 py-0.5 rounded bg-white/10 text-white/70 text-xs font-medium">
            {logs.length} events
          </span>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setFilterLevel('all');
              }}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                filterLevel === 'all' ? 'bg-white/20 text-white' : 'bg-white/5 text-white/50'
              }`}
            >
              All
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setFilterLevel('info');
              }}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                filterLevel === 'info' ? 'bg-white/20 text-white' : 'bg-white/5 text-white/50'
              }`}
            >
              Info
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setFilterLevel('warning');
              }}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                filterLevel === 'warning' ? 'bg-white/20 text-white' : 'bg-white/5 text-white/50'
              }`}
            >
              Warn
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setFilterLevel('error');
              }}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                filterLevel === 'error' ? 'bg-red-500/30 text-red-400' : 'bg-white/5 text-white/50'
              }`}
            >
              Error
            </button>
          </div>

          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-white/40" />
          ) : (
            <ChevronUp className="w-4 h-4 text-white/40" />
          )}
        </div>
      </button>

      {/* Expanded Console */}
      {isExpanded && (
        <div className="h-64 overflow-y-auto border-t border-white/10 bg-black/60">
          <div className="p-4 space-y-2 font-mono text-xs">
            {filteredLogs.map((log) => (
              <div key={log.id} className="flex items-start gap-3 hover:bg-white/5 p-2 rounded">
                {getLogIcon(log.level)}
                <div className="flex-1">
                  <div className="flex items-baseline gap-3">
                    <span className="text-white/40">
                      {log.timestamp.toLocaleTimeString('en-US')}
                    </span>
                    <span className={`uppercase font-semibold ${getLogColor(log.level)}`}>
                      [{log.level}]
                    </span>
                    <span className="text-white/80">{log.message}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
