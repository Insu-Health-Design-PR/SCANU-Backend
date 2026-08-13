import { AlertTriangle, User, Shield, TrendingUp, Activity } from 'lucide-react';
import { DashboardMetrics, Alert } from '../api/client';

interface DetectionSidebarProps {
  metrics: DashboardMetrics;
  alerts: Alert[];
}

export function DetectionSidebar({ metrics, alerts }: DetectionSidebarProps) {
  return (
    <div className="w-80 border-l border-white/10 bg-black/20 backdrop-blur-sm flex flex-col">
      {/* Detection Stats */}
      <div className="p-4 border-b border-white/10">
        <h2 className="font-semibold text-sm text-white/70 uppercase tracking-wider mb-4">
          Detection Stats
        </h2>

        <div className="space-y-3">
          {/* Total Detections */}
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-white/50">Total Detections</span>
              <Activity className="w-4 h-4 text-white/60" />
            </div>
            <div className="text-2xl font-bold text-white">{metrics.totalDetections}</div>
            <div className="text-xs text-white/40 mt-1">Last 24 hours</div>
          </div>

          {/* Person Detections */}
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-white/50">Person Detections</span>
              <User className="w-4 h-4 text-white/60" />
            </div>
            <div className="text-2xl font-bold text-white">{metrics.personDetections}</div>
            <div className="text-xs text-white/40 mt-1">Active tracks</div>
          </div>

          {/* Weapon Detections */}
          {metrics.weaponDetections > 0 ? (
            <div className="bg-red-500/10 rounded-lg p-3 border border-red-500/30 animate-pulse">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-red-400/70">Weapon Detections</span>
                <AlertTriangle className="w-4 h-4 text-red-400" />
              </div>
              <div className="text-2xl font-bold text-red-400">{metrics.weaponDetections}</div>
              <div className="text-xs text-red-400/50 mt-1">Immediate attention required</div>
            </div>
          ) : (
            <div className="bg-white/5 rounded-lg p-3 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-white/50">Weapon Detections</span>
                <Shield className="w-4 h-4 text-white/60" />
              </div>
              <div className="text-2xl font-bold text-white">0</div>
              <div className="text-xs text-white/40 mt-1">All clear</div>
            </div>
          )}

          {/* Average Confidence */}
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-white/50">Avg Confidence</span>
              <TrendingUp className="w-4 h-4 text-white/60" />
            </div>
            <div className="text-2xl font-bold text-white">
              {(metrics.avgConfidence * 100).toFixed(1)}%
            </div>
            <div className="w-full bg-white/10 rounded-full h-1.5 mt-2">
              <div
                className="bg-white h-1.5 rounded-full transition-all"
                style={{ width: `${metrics.avgConfidence * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Active Alerts */}
      <div className="flex-1 overflow-y-auto p-4">
        <h3 className="font-semibold text-sm text-white/70 uppercase tracking-wider mb-3">
          Active Alerts
        </h3>

        {alerts.length === 0 ? (
          <div className="text-center py-8">
            <Shield className="w-12 h-12 text-white/20 mx-auto mb-2" />
            <p className="text-sm text-white/40">No active alerts</p>
            <p className="text-xs text-white/30 mt-1">System is secure</p>
          </div>
        ) : (
          <div className="space-y-2">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-3 rounded-lg border ${
                  alert.severity === 'critical'
                    ? 'bg-red-500/10 border-red-500/30'
                    : 'bg-white/5 border-white/10'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <AlertTriangle
                      className={`w-4 h-4 ${
                        alert.severity === 'critical'
                          ? 'text-red-400'
                          : 'text-white/60'
                      }`}
                    />
                    <span
                      className={`text-xs font-medium uppercase tracking-wide ${
                        alert.severity === 'critical'
                          ? 'text-red-400'
                          : 'text-white/60'
                      }`}
                    >
                      {alert.severity}
                    </span>
                  </div>
                  <span className="text-xs text-white/40">
                    {new Date(alert.timestamp).toLocaleTimeString('en-US')}
                  </span>
                </div>

                <p className="text-sm text-white mb-2">{alert.message}</p>

                <div className="flex items-center gap-2 text-xs text-white/50">
                  <span>{alert.deviceId}</span>
                  <span>•</span>
                  <span>{alert.sensor}</span>
                  <span>•</span>
                  <span className="text-white/70">{(alert.confidence * 100).toFixed(0)}%</span>
                </div>

                <div className="flex gap-2 mt-3">
                  <button className="flex-1 px-3 py-1.5 rounded bg-white/10 hover:bg-white/20 text-xs font-medium transition-all">
                    Acknowledge
                  </button>
                  <button className="flex-1 px-3 py-1.5 rounded bg-red-500/20 hover:bg-red-500/30 text-red-400 text-xs font-medium transition-all">
                    Escalate
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* System Performance */}
      <div className="p-4 border-t border-white/10 bg-black/40">
        <h3 className="font-semibold text-xs text-white/50 uppercase tracking-wider mb-3">
          System Performance
        </h3>
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-xs text-white/50">CPU</span>
            <div className="flex items-center gap-2">
              <div className="w-24 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white transition-all"
                  style={{ width: `${metrics.cpuUsage}%` }}
                />
              </div>
              <span className="text-xs text-white font-mono w-10 text-right">
                {metrics.cpuUsage.toFixed(0)}%
              </span>
            </div>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-xs text-white/50">GPU</span>
            <div className="flex items-center gap-2">
              <div className="w-24 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white transition-all"
                  style={{ width: `${metrics.gpuUsage}%` }}
                />
              </div>
              <span className="text-xs text-white font-mono w-10 text-right">
                {metrics.gpuUsage.toFixed(0)}%
              </span>
            </div>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-xs text-white/50">Memory</span>
            <div className="flex items-center gap-2">
              <div className="w-24 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white transition-all"
                  style={{ width: `${metrics.memoryUsage}%` }}
                />
              </div>
              <span className="text-xs text-white font-mono w-10 text-right">
                {metrics.memoryUsage.toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
