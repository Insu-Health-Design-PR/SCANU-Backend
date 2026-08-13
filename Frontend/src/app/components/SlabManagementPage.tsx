import { Search } from 'lucide-react';
import { useMemo, useState } from 'react';
import { DashboardSnapshot } from '../api/client';
import {
  AiSlab,
  LINK_CLASS,
  LINK_LABEL,
  THREAT_CLASS,
  THREAT_LABEL,
  slabsFromSnapshot,
} from '../types/slab';

interface SlabManagementPageProps {
  snapshot: DashboardSnapshot;
  onSelectSlab: (slabId: string) => void;
}

function StatusCell({ state }: { state: AiSlab['camera'] }) {
  return (
    <span className={`text-sm font-medium ${LINK_CLASS[state]}`}>
      {LINK_LABEL[state]}
    </span>
  );
}

function CountCell({ value }: { value: number | null }) {
  if (value == null) return <span className="text-white/30">—</span>;
  return <span className="font-mono text-sm text-white/90">{value}</span>;
}

export function SlabManagementPage({ snapshot, onSelectSlab }: SlabManagementPageProps) {
  const [query, setQuery] = useState('');
  const slabs = useMemo(() => slabsFromSnapshot(snapshot), [snapshot]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return slabs;
    return slabs.filter(
      (s) =>
        s.slabId.toLowerCase().includes(q) ||
        s.ip.toLowerCase().includes(q) ||
        s.id.toLowerCase().includes(q),
    );
  }, [slabs, query]);

  return (
    <div className="flex-1 overflow-auto bg-[#0f1117]">
      <div className="max-w-[1400px] mx-auto px-6 py-8">
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-white tracking-tight">SCANU — Slab Management</h1>
          <p className="text-sm text-white/45 mt-1">
            AI Slabs (Jetson / server nodes) — each slab runs 1 camera, 1 thermal, and 1 mmWave sensor
          </p>
        </div>

        <div className="mb-4 flex items-center gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/35" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by Slab ID or IP…"
              className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder:text-white/35 focus:outline-none focus:border-white/25"
            />
          </div>
          <div className="text-xs text-white/40">{filtered.length} slab{filtered.length !== 1 ? 's' : ''}</div>
        </div>

        <div className="rounded-xl border border-white/10 overflow-hidden bg-[#141820] shadow-xl shadow-black/20">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-white/10 bg-white/[0.03]">
                {[
                  'Slab ID',
                  'Threats',
                  'Camera',
                  'Thermal Camera',
                  'mmWave',
                  'People',
                  'Weapons',
                  'Alerts',
                  'IP',
                ].map((col) => (
                  <th
                    key={col}
                    className="px-4 py-3 text-[11px] font-semibold uppercase tracking-wider text-white/45 whitespace-nowrap"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={9} className="px-4 py-12 text-center text-sm text-white/40">
                    No slabs match your filter
                  </td>
                </tr>
              ) : (
                filtered.map((slab, idx) => (
                  <tr
                    key={slab.id}
                    onClick={() => onSelectSlab(slab.id)}
                    className={`border-b border-white/5 cursor-pointer transition-colors hover:bg-cyan-500/[0.06] ${
                      idx % 2 === 0 ? 'bg-transparent' : 'bg-white/[0.015]'
                    }`}
                  >
                    <td className="px-4 py-3.5">
                      <div className="font-medium text-cyan-300 hover:underline">{slab.slabId}</div>
                      {!slab.online && (
                        <div className="text-[10px] text-white/35 mt-0.5 uppercase tracking-wide">offline</div>
                      )}
                    </td>
                    <td className="px-4 py-3.5">
                      <span
                        className={`inline-flex text-[11px] font-semibold uppercase tracking-wide px-2 py-0.5 rounded border ${THREAT_CLASS[slab.threat]}`}
                      >
                        {THREAT_LABEL[slab.threat]}
                      </span>
                    </td>
                    <td className="px-4 py-3.5">
                      <StatusCell state={slab.camera} />
                    </td>
                    <td className="px-4 py-3.5">
                      <StatusCell state={slab.thermal} />
                    </td>
                    <td className="px-4 py-3.5">
                      <StatusCell state={slab.mmwave} />
                    </td>
                    <td className="px-4 py-3.5">
                      <CountCell value={slab.peopleDetected} />
                    </td>
                    <td className="px-4 py-3.5">
                      <CountCell value={slab.weaponsDetected} />
                    </td>
                    <td className="px-4 py-3.5">
                      <CountCell value={slab.alerts > 0 ? slab.alerts : slab.alerts === 0 ? 0 : null} />
                    </td>
                    <td className="px-4 py-3.5 font-mono text-xs text-white/55">{slab.ip}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
