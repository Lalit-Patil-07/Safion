import React, { useState, useRef, useCallback, useEffect } from 'react';
import { RefreshCw, Trash2, Loader } from 'lucide-react';

const API = '';

const fmt = (iso) => iso ? new Date(iso).toLocaleString() : '—';

const VIOLATION_COLORS = {
  'NO-Hardhat':     '#EF4444',
  'NO-Mask':        '#F59E0B',
  'NO-Safety Vest': '#EC4899',
};
const vColor = (type) => VIOLATION_COLORS[type] || '#6B7280';

const ViolationChip = ({ type }) => (
  <span className="text-xs px-2 py-0.5 rounded-full text-white font-medium"
        style={{ backgroundColor: vColor(type) }}>{type}</span>
);

const identityLabel = (v) => {
  if (!v.identity_id) return "Unidentified";
  const conf = v.confidence != null ? ` (${v.confidence.toFixed(2)})` : "";
  return `${v.name}${conf}`;
};

const ViolationLogPage = ({ onModalImage }) => {
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(true);

  const firstLoad = useRef(true);

  const load = useCallback(() => {
    if (firstLoad.current) setLoading(true);
    fetch(`${API}/violations`)
      .then(r => r.json())
      .then(setViolations)
      .catch(() => [])
      .finally(() => {
        setLoading(false);
        firstLoad.current = false;
      });
  }, []);

  useEffect(() => {
    load();
    const intervalId = setInterval(load, 5000);
    return () => clearInterval(intervalId);
  }, [load]);

  const clear = async () => {
    if (!window.confirm('Clear all violations?')) return;
    await fetch(`${API}/violations/clear`, { method: 'POST' });
    setViolations([]);
  };

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 className="text-2xl font-bold text-text">Violations</h2>
          <p className="text-sm text-text-secondary">{violations.length} logged</p>
        </div>
        <div className="flex gap-2">
          <button onClick={load} className="p-2 border border-border rounded-lg hover:bg-border text-text-secondary"><RefreshCw size={14} /></button>
          <button onClick={clear} className="flex items-center gap-2 px-3 py-2 bg-red-500 text-white text-sm rounded-lg hover:bg-red-600">
            <Trash2 size={13} /> Clear all
          </button>
        </div>
      </div>

      <div className="bg-card border border-border rounded-xl overflow-hidden">
        {loading ? (
          <div className="flex justify-center py-16"><Loader size={28} className="animate-spin text-text-secondary" /></div>
        ) : violations.length === 0 ? (
          <p className="text-sm text-text-secondary text-center py-14">No violations recorded.</p>
        ) : (
          <div className="divide-y divide-border">
            {violations.map(v => {
              const faceDetected = v.match_score != null;
              return (
                <div key={v.id} className="flex items-start gap-4 p-4 hover:bg-card-secondary transition-all">
                  {v.image_path ? (
                    <img src={`${API}${v.image_path}`} alt="violation"
                         className="w-14 h-14 object-cover rounded-lg cursor-pointer hover:opacity-80 flex-shrink-0"
                         onClick={() => onModalImage(`${API}${v.image_path}`)} />
                  ) : <div className="w-14 h-14 bg-card-secondary rounded-lg flex-shrink-0" />}

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-sm font-semibold ${v.identity_id ? "text-text" : "text-text-secondary italic"}`}>
                        {identityLabel(v)}
                      </span>
                      <ViolationChip type={v.violation_type} />
                    </div>

                    <div className="text-xs text-text-secondary">{fmt(v.timestamp)}</div>

                    <div className="flex items-center gap-1 mt-0.5">
                      <span className={`inline-block w-1.5 h-1.5 rounded-full ${faceDetected ? "bg-green-500" : "bg-gray-400"}`} />
                      <span className="text-xs text-text-secondary">
                        Face: {faceDetected ? "detected" : "not detected"}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default ViolationLogPage;