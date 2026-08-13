import { Settings, Play, Square, RotateCcw, Sliders, Save } from 'lucide-react';
import { useState } from 'react';
import { scanuClient, OperatorMode } from '../api/client';

export function AdminPanel() {
  const [operatorMode, setOperatorMode] = useState<OperatorMode>('central');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.75);
  const [fpsTarget, setFpsTarget] = useState(30);

  const handleModeChange = async (mode: OperatorMode) => {
    setOperatorMode(mode);
    scanuClient.setOperatorMode(mode);
  };

  return (
    <div className="w-80 border-l border-white/10 bg-black/20 backdrop-blur-sm overflow-y-auto p-4 space-y-4">
      {/* System Controls */}
      <div className="bg-white/5 rounded-lg border border-white/10 p-4">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-4 h-4 text-white/60" />
          <h2 className="font-semibold text-sm">System Controls</h2>
        </div>

        {/* Operator Mode Selection */}
        <div className="mb-4">
          <label className="text-xs font-medium text-white/50 mb-2 block">Operator Mode</label>
          <div className="space-y-2">
            <button
              onClick={() => handleModeChange('central')}
              className={`w-full p-2 rounded text-left transition-all text-xs ${
                operatorMode === 'central'
                  ? 'bg-white/20 border border-white/50'
                  : 'bg-white/5 hover:bg-white/10'
              }`}
            >
              <div className="font-medium">Central</div>
            </button>
            <button
              onClick={() => handleModeChange('fallback')}
              className={`w-full p-2 rounded text-left transition-all text-xs ${
                operatorMode === 'fallback'
                  ? 'bg-white/20 border border-white/50'
                  : 'bg-white/5 hover:bg-white/10'
              }`}
            >
              <div className="font-medium">Fallback</div>
            </button>
            <button
              onClick={() => handleModeChange('local')}
              className={`w-full p-2 rounded text-left transition-all text-xs ${
                operatorMode === 'local'
                  ? 'bg-white/20 border border-white/50'
                  : 'bg-white/5 hover:bg-white/10'
              }`}
            >
              <div className="font-medium">Local</div>
            </button>
          </div>
        </div>

        {/* Sensor Controls */}
        <div>
          <label className="text-xs font-medium text-white/50 mb-2 block">Sensor Controls</label>
          <div className="space-y-2">
            <button className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded text-xs bg-white/10 hover:bg-white/20 text-white transition-all">
              <Play className="w-3 h-3" />
              <span>Start All</span>
            </button>
            <button className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded text-xs bg-red-500/20 hover:bg-red-500/30 text-red-400 transition-all">
              <Square className="w-3 h-3" />
              <span>Stop All</span>
            </button>
            <button className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded text-xs bg-white/10 hover:bg-white/20 text-white transition-all">
              <RotateCcw className="w-3 h-3" />
              <span>Restart</span>
            </button>
          </div>
        </div>
      </div>

      {/* Configuration */}
      <div className="bg-white/5 rounded-lg border border-white/10 p-4">
        <div className="flex items-center gap-2 mb-4">
          <Sliders className="w-4 h-4 text-white/60" />
          <h2 className="font-semibold text-sm">AI Configuration</h2>
        </div>

        <div className="space-y-4">
          {/* Confidence Threshold */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-medium text-white/50">Confidence</label>
              <span className="text-xs font-mono text-white">{(confidenceThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, rgb(255, 255, 255) 0%, rgb(255, 255, 255) ${confidenceThreshold * 100}%, rgba(255,255,255,0.1) ${confidenceThreshold * 100}%, rgba(255,255,255,0.1) 100%)`,
              }}
            />
          </div>

          {/* FPS Target */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="text-xs font-medium text-white/50">Target FPS</label>
              <span className="text-xs font-mono text-white">{fpsTarget}</span>
            </div>
            <input
              type="range"
              min="15"
              max="60"
              step="5"
              value={fpsTarget}
              onChange={(e) => setFpsTarget(parseInt(e.target.value))}
              className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, rgb(255, 255, 255) 0%, rgb(255, 255, 255) ${((fpsTarget - 15) / 45) * 100}%, rgba(255,255,255,0.1) ${((fpsTarget - 15) / 45) * 100}%, rgba(255,255,255,0.1) 100%)`,
              }}
            />
          </div>

          {/* Model Profile */}
          <div>
            <label className="text-xs font-medium text-white/50 mb-2 block">Model Profile</label>
            <select className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded text-white text-xs focus:outline-none focus:border-white/40">
              <option value="balanced">Balanced</option>
              <option value="accuracy">High Accuracy</option>
              <option value="speed">High Speed</option>
              <option value="lowpower">Low Power</option>
            </select>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-white/10 space-y-2">
          <button className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded bg-white hover:bg-white/90 text-black text-xs font-medium transition-all">
            <Save className="w-3 h-3" />
            Save Config
          </button>
          <button className="w-full px-3 py-2 rounded bg-white/10 hover:bg-white/20 text-white text-xs transition-all">
            Reset
          </button>
        </div>
      </div>

      {/* Device Management */}
      <div className="bg-white/5 rounded-lg border border-white/10 p-4">
        <h3 className="font-semibold text-sm mb-3">Devices</h3>

        <div className="space-y-2">
          {['jetson-01', 'jetson-02', 'jetson-03'].map((deviceId) => (
            <div key={deviceId} className="flex items-center justify-between p-2 bg-white/5 rounded">
              <div className="text-xs">{deviceId}</div>
              <div className="flex gap-1">
                <button className="p-1 rounded bg-white/10 hover:bg-white/20 transition-all">
                  <Play className="w-3 h-3 text-white" />
                </button>
                <button className="p-1 rounded bg-white/10 hover:bg-white/20 transition-all">
                  <Square className="w-3 h-3 text-red-400" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* System Metrics */}
      <div className="bg-white/5 rounded-lg border border-white/10 p-4">
        <h3 className="font-semibold text-sm mb-3">System Health</h3>
        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs">
            <span className="text-white/50">CPU</span>
            <span className="font-mono">45%</span>
          </div>
          <div className="flex justify-between items-center text-xs">
            <span className="text-white/50">GPU</span>
            <span className="font-mono">62%</span>
          </div>
          <div className="flex justify-between items-center text-xs">
            <span className="text-white/50">Memory</span>
            <span className="font-mono">58%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
