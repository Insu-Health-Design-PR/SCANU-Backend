import { Settings, Activity, AlertTriangle, Users } from 'lucide-react';
import { DashboardMetrics } from '../api/client';

interface RightPanelProps {
  metrics: DashboardMetrics;
}

export function RightPanel({ metrics }: RightPanelProps) {
  return (
    <div className="w-80 border-l border-white/10 bg-black/20 backdrop-blur-sm overflow-y-auto">
      {/* Detection Summary */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-4 h-4 text-white/60" />
          <h3 className="text-sm font-semibold">Detection Summary</h3>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="text-xs text-white/50 mb-1">Total</div>
            <div className="text-2xl font-bold">{metrics.totalDetections}</div>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="text-xs text-white/50 mb-1">Persons</div>
            <div className="text-2xl font-bold">{metrics.personDetections}</div>
          </div>
          {metrics.weaponDetections > 0 && (
            <div className="col-span-2 bg-red-500/10 rounded-lg p-3 border border-red-500/30 animate-pulse">
              <div className="flex items-center gap-2 mb-1">
                <AlertTriangle className="w-4 h-4 text-red-400" />
                <div className="text-xs text-red-400">WEAPON DETECTED</div>
              </div>
              <div className="text-2xl font-bold text-red-400">{metrics.weaponDetections}</div>
            </div>
          )}
        </div>
      </div>

      {/* Confidence Meter */}
      <div className="p-4 border-b border-white/10">
        <div className="text-xs text-white/50 mb-2">Avg Confidence</div>
        <div className="flex items-center gap-3">
          <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-white transition-all"
              style={{ width: `${metrics.avgConfidence * 100}%` }}
            />
          </div>
          <div className="text-sm font-mono text-white">
            {(metrics.avgConfidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* System Controls */}
      <div className="p-4 border-b border-white/10">
        <div className="flex items-center gap-2 mb-3">
          <Settings className="w-4 h-4 text-white/60" />
          <h3 className="text-sm font-semibold">Controls</h3>
        </div>

        <div className="space-y-2">
          <button className="w-full px-3 py-2 rounded bg-white/10 hover:bg-white/20 text-white text-xs transition-all text-left">
            <div className="font-medium">Mode: Central</div>
            <div className="text-white/50">Server inference active</div>
          </button>

          <div className="grid grid-cols-2 gap-2">
            <button className="px-3 py-2 rounded bg-white/10 hover:bg-white/20 text-white text-xs transition-all">
              Start All
            </button>
            <button className="px-3 py-2 rounded bg-red-500/20 hover:bg-red-500/30 text-red-400 text-xs transition-all">
              Stop All
            </button>
          </div>
        </div>
      </div>

      {/* AI Configuration */}
      <div className="p-4 border-b border-white/10">
        <h3 className="text-sm font-semibold mb-3">AI Config</h3>

        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-white/50">Confidence Threshold</span>
              <span className="text-white font-mono">75%</span>
            </div>
            <div className="h-1.5 bg-white/10 rounded-full">
              <div className="h-full w-3/4 bg-white rounded-full" />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-white/50">Target FPS</span>
              <span className="text-white font-mono">30</span>
            </div>
            <div className="h-1.5 bg-white/10 rounded-full">
              <div className="h-full w-1/2 bg-white rounded-full" />
            </div>
          </div>

          <div>
            <label className="text-xs text-white/50 block mb-1">Model Profile</label>
            <select className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded text-white text-xs">
              <option>Balanced</option>
              <option>High Accuracy</option>
              <option>High Speed</option>
            </select>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="p-4">
        <h3 className="text-sm font-semibold mb-3">System Health</h3>

        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs">
            <span className="text-white/50">CPU Usage</span>
            <span className="font-mono">{metrics.cpuUsage.toFixed(0)}%</span>
          </div>
          <div className="flex justify-between items-center text-xs">
            <span className="text-white/50">GPU Usage</span>
            <span className="font-mono">{metrics.gpuUsage.toFixed(0)}%</span>
          </div>
          <div className="flex justify-between items-center text-xs">
            <span className="text-white/50">Memory</span>
            <span className="font-mono">{metrics.memoryUsage.toFixed(0)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
