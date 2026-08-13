import { ModelPlaygroundPanel } from './ModelPlaygroundPanel';

export function PlaygroundView() {
  return (
    <div className="flex-1 overflow-y-auto bg-slate-950 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-white">Model playground</h1>
          <p className="text-sm text-white/45 mt-1">
            Tune detection params on a sample or uploaded image before saving a profile
          </p>
        </div>
        <ModelPlaygroundPanel />
      </div>
    </div>
  );
}
