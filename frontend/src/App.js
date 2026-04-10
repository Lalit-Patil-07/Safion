import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Camera, Upload, Video, AlertCircle, CheckCircle, XCircle,
  Settings, Play, StopCircle, Loader, Shield, ShieldOff, ShieldCheck,
  X, Tv, Plus, Trash2, Maximize, Minimize, Users, ChevronLeft,
  ChevronRight, Search, Filter, GitMerge, Edit3, Clock, Activity,
  BarChart2, Eye, Check, AlertTriangle, RefreshCw
} from 'lucide-react';

const API = '';

// ── Violation type colors ─────────────────────────────────────────────────────
const VIOLATION_COLORS = {
  'NO-Hardhat':     '#EF4444',
  'NO-Mask':        '#F59E0B',
  'NO-Safety Vest': '#EC4899',
};
const violationColor = (type) => VIOLATION_COLORS[type] || '#6B7280';

// ── Utility ───────────────────────────────────────────────────────────────────
const ago = (iso) => {
  if (!iso) return 'Never';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1)  return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)  return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
};
const fmt = (iso) => iso ? new Date(iso).toLocaleString() : '—';

// ── Shared UI atoms ───────────────────────────────────────────────────────────
const Badge = ({ confirmed, archived }) => {
  if (archived)   return <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500">Archived</span>;
  if (confirmed)  return <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700">Confirmed</span>;
  return <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-700">Unconfirmed</span>;
};

const ViolationBadge = ({ type }) => (
  <span className="text-xs px-2 py-0.5 rounded-full text-white font-medium"
        style={{ backgroundColor: violationColor(type) }}>
    {type}
  </span>
);

const FaceAvatar = ({ src, label, size = 'md' }) => {
  const dim = size === 'lg' ? 'w-24 h-24' : size === 'sm' ? 'w-10 h-10' : 'w-14 h-14';
  return src
    ? <img src={`${API}${src}`} alt={label}
           className={`${dim} rounded-full object-cover border-2 border-border flex-shrink-0`} />
    : <div className={`${dim} rounded-full bg-card-secondary border-2 border-border flex items-center justify-center flex-shrink-0`}>
        <Users size={size === 'lg' ? 36 : size === 'sm' ? 16 : 24} className="text-text-secondary" />
      </div>;
};

const ImageModal = ({ imageUrl, onClose }) => {
  if (!imageUrl) return null;
  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50" onClick={onClose}>
      <div className="relative max-w-4xl max-h-[90vh] p-4" onClick={e => e.stopPropagation()}>
        <img src={imageUrl} alt="Evidence" className="w-full h-full object-contain rounded-lg" />
        <button onClick={onClose}
                className="absolute -top-2 -right-2 bg-card rounded-full p-2 text-text hover:bg-border">
          <X size={22} />
        </button>
      </div>
    </div>
  );
};

// ── Sidebar ───────────────────────────────────────────────────────────────────
const NAV = [
  { id: 'dashboard',   icon: BarChart2, label: 'Dashboard'    },
  { id: 'identities',  icon: Users,     label: 'Identities'   },
  { id: 'violations',  icon: AlertTriangle, label: 'Violations' },
  { id: 'monitor',     icon: Tv,        label: 'Live Monitor' },
  { id: 'settings',    icon: Settings,  label: 'Settings'     },
];

const Sidebar = ({ view, setView, open, setOpen, serverStatus }) => (
  <>
    {open && <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={() => setOpen(false)} />}
    <aside className={`fixed top-0 left-0 h-full bg-card-secondary border-r border-border z-50 transition-all duration-300
                       ${open ? 'translate-x-0 w-56' : '-translate-x-full lg:translate-x-0 w-0 lg:w-16'}`}>
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="p-3 border-b border-border flex items-center justify-between min-h-[56px]">
          {open ? (
            <>
              <button onClick={() => setOpen(false)} className="p-2 text-text hover:bg-border rounded-md">
                <ChevronLeft size={18} />
              </button>
              <span className="text-lg font-bold text-text">Safion</span>
              <div className="w-9" />
            </>
          ) : (
            <div className="flex justify-center w-full">
              <button onClick={() => setOpen(true)} className="p-2 text-text hover:bg-border rounded-md">
                <Shield size={18} className="text-primary" />
              </button>
            </div>
          )}
        </div>

        {/* Nav */}
        <nav className="flex flex-col gap-1 p-2 flex-1">
          {NAV.map(({ id, icon: Icon, label }) => (
            <button key={id}
              onClick={() => { setView(id); if (window.innerWidth < 1024) setOpen(false); }}
              title={!open ? label : ''}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all
                          ${view === id ? 'bg-primary text-white' : 'text-text-secondary hover:bg-border'}
                          ${open ? 'justify-start' : 'justify-center'}`}>
              <Icon size={17} className="flex-shrink-0" />
              {open && <span className="whitespace-nowrap">{label}</span>}
            </button>
          ))}
        </nav>

        {/* Status */}
        <div className="p-3 border-t border-border">
          <div className={`flex items-center gap-3 px-3 py-2 ${open ? '' : 'justify-center'}`}>
            <div className={`w-2 h-2 rounded-full flex-shrink-0
              ${serverStatus === 'connected' ? 'bg-green-500' : serverStatus === 'degraded' ? 'bg-amber-500' : 'bg-red-500'}`} />
            {open && (
              <div>
                <div className="text-xs text-text-secondary">Server</div>
                <div className={`text-xs font-semibold capitalize
                  ${serverStatus === 'connected' ? 'text-green-500' : serverStatus === 'degraded' ? 'text-amber-500' : 'text-red-500'}`}>
                  {serverStatus}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </aside>
  </>
);

// ── Dashboard ─────────────────────────────────────────────────────────────────
const StatCard = ({ label, value, sub, icon: Icon, color = 'text-text' }) => (
  <div className="bg-card border border-border rounded-xl p-5 flex items-start gap-4">
    <div className={`p-3 rounded-lg bg-card-secondary ${color}`}>
      <Icon size={20} />
    </div>
    <div>
      <div className="text-2xl font-bold text-text">{value ?? '—'}</div>
      <div className="text-sm font-medium text-text">{label}</div>
      {sub && <div className="text-xs text-text-secondary mt-0.5">{sub}</div>}
    </div>
  </div>
);

const DashboardPage = ({ setView }) => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/face/stats`)
      .then(r => r.json())
      .then(setStats)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="p-6 max-w-5xl">
      <h2 className="text-2xl font-bold text-text mb-6">Dashboard</h2>
      {loading ? (
        <div className="flex justify-center py-20"><Loader size={32} className="animate-spin text-text-secondary" /></div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          <StatCard label="Total identities"  value={stats?.total_identities}  icon={Users}         color="text-blue-500" />
          <StatCard label="Confirmed"         value={stats?.confirmed_count}   icon={CheckCircle}    color="text-green-500" />
          <StatCard label="Unconfirmed"       value={stats?.unconfirmed_count} icon={AlertCircle}    color="text-amber-500" />
          <StatCard label="Total violations"  value={stats?.total_violations}  icon={AlertTriangle}  color="text-red-500" />
          <StatCard label="Violations today"  value={stats?.violations_today}  icon={Clock}          color="text-purple-500" />
          <StatCard label="Repeat offenders"  value={stats?.repeat_offenders}  icon={Activity}       color="text-orange-500"
                    sub="≥3 violations" />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <button onClick={() => setView('identities')}
                className="bg-card border border-border rounded-xl p-5 text-left hover:border-primary transition-all">
          <div className="text-lg font-semibold text-text mb-1">Manage identities</div>
          <div className="text-sm text-text-secondary">Review, name, and merge detected individuals</div>
        </button>
        <button onClick={() => setView('monitor')}
                className="bg-card border border-border rounded-xl p-5 text-left hover:border-primary transition-all">
          <div className="text-lg font-semibold text-text mb-1">Start monitoring</div>
          <div className="text-sm text-text-secondary">Open webcam, RTSP stream, or upload video</div>
        </button>
      </div>
    </div>
  );
};

// ── Identity card ─────────────────────────────────────────────────────────────
const IdentityCard = ({ identity, onSelect, onMergeStart, merging, onMergeTarget }) => {
  const [editing, setEditing]   = useState(false);
  const [newLabel, setNewLabel] = useState(identity.label);
  const inputRef = useRef(null);

  useEffect(() => { if (editing) inputRef.current?.focus(); }, [editing]);

  const save = async () => {
    if (!newLabel.trim() || newLabel === identity.label) { setEditing(false); return; }
    await fetch(`${API}/face/identity/${identity.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: newLabel.trim() }),
    });
    setEditing(false);
    window.location.reload(); // simple refresh; replace with state update in prod
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') save();
    if (e.key === 'Escape') { setEditing(false); setNewLabel(identity.label); }
  };

  return (
    <div
      onClick={() => {
        if (merging) { onMergeTarget(identity); return; }
        if (!editing) onSelect(identity);
      }}
      className={`bg-card border rounded-xl p-4 cursor-pointer transition-all group
        ${merging ? 'border-amber-400 hover:border-amber-500 ring-2 ring-amber-200' : 'border-border hover:border-primary'}`}>

      {/* Thumbnail */}
      <div className="flex items-start gap-3 mb-3">
        <FaceAvatar src={identity.thumbnail} label={identity.label} />
        <div className="flex-1 min-w-0">
          {editing ? (
            <input ref={inputRef} value={newLabel}
                   onChange={e => setNewLabel(e.target.value)}
                   onKeyDown={handleKeyDown}
                   onClick={e => e.stopPropagation()}
                   className="w-full text-sm font-semibold text-text bg-background border border-primary rounded px-2 py-1 outline-none"
            />
          ) : (
            <div className="flex items-center gap-1.5">
              <span className="text-sm font-semibold text-text truncate">{identity.label}</span>
              <button onClick={e => { e.stopPropagation(); setEditing(true); }}
                      className="opacity-0 group-hover:opacity-100 p-0.5 text-text-secondary hover:text-primary transition-all">
                <Edit3 size={12} />
              </button>
            </div>
          )}
          <Badge confirmed={identity.is_confirmed} archived={identity.is_archived} />
        </div>
      </div>

      {/* Stats */}
      <div className="flex justify-between text-xs text-text-secondary">
        <span>{identity.violation_count ?? 0} violations</span>
        <span>{ago(identity.last_seen)}</span>
      </div>

      {/* Actions */}
      {!merging && (
        <div className="flex gap-2 mt-3 opacity-0 group-hover:opacity-100 transition-all">
          <button onClick={e => { e.stopPropagation(); onSelect(identity); }}
                  className="flex-1 text-xs px-2 py-1.5 bg-card-secondary hover:bg-border rounded text-text-secondary transition-all flex items-center justify-center gap-1">
            <Eye size={11} /> View
          </button>
          <button onClick={e => { e.stopPropagation(); onMergeStart(identity); }}
                  className="flex-1 text-xs px-2 py-1.5 bg-card-secondary hover:bg-border rounded text-text-secondary transition-all flex items-center justify-center gap-1">
            <GitMerge size={11} /> Merge
          </button>
        </div>
      )}
    </div>
  );
};

// ── Identity detail panel ─────────────────────────────────────────────────────
const IdentityDetail = ({ identity, onClose, onModalImage }) => {
  const [data, setData]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [newLabel, setNewLabel] = useState('');

  useEffect(() => {
    setLoading(true);
    fetch(`${API}/face/identity/${identity.id}/violations`)
      .then(r => r.json())
      .then(d => { setData(d); setNewLabel(d.identity.label); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [identity.id]);

  const save = async () => {
    if (!newLabel.trim()) return;
    await fetch(`${API}/face/identity/${identity.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: newLabel.trim() }),
    });
    setEditing(false);
    // Reload data
    fetch(`${API}/face/identity/${identity.id}/violations`)
      .then(r => r.json()).then(setData).catch(() => {});
  };

  const id = data?.identity;

  return (
    <div className="fixed inset-0 bg-black/40 z-40 flex justify-end" onClick={onClose}>
      <div className="w-full max-w-lg bg-card h-full shadow-2xl overflow-y-auto"
           onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border sticky top-0 bg-card z-10">
          <h3 className="text-lg font-semibold text-text">Identity Detail</h3>
          <button onClick={onClose} className="p-1.5 hover:bg-border rounded text-text-secondary">
            <X size={18} />
          </button>
        </div>

        {loading ? (
          <div className="flex justify-center py-20"><Loader size={32} className="animate-spin text-text-secondary" /></div>
        ) : (
          <div className="p-5 space-y-6">
            {/* Profile */}
            <div className="flex items-start gap-4">
              <FaceAvatar src={id?.thumbnail} label={id?.label} size="lg" />
              <div className="flex-1">
                {editing ? (
                  <div className="flex items-center gap-2">
                    <input autoFocus value={newLabel} onChange={e => setNewLabel(e.target.value)}
                           onKeyDown={e => { if (e.key === 'Enter') save(); if (e.key === 'Escape') setEditing(false); }}
                           className="flex-1 text-sm font-semibold text-text bg-background border border-primary rounded px-2 py-1 outline-none" />
                    <button onClick={save} className="p-1.5 bg-primary text-white rounded"><Check size={14} /></button>
                    <button onClick={() => setEditing(false)} className="p-1.5 hover:bg-border rounded text-text-secondary"><X size={14} /></button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-bold text-text">{id?.label}</span>
                    <button onClick={() => setEditing(true)} className="p-1 text-text-secondary hover:text-primary">
                      <Edit3 size={14} />
                    </button>
                  </div>
                )}
                <div className="mt-1"><Badge confirmed={id?.is_confirmed} /></div>
              </div>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: 'Violations', value: id?.violation_count },
                { label: 'Embeddings', value: id?.embedding_count },
                { label: 'Last seen', value: ago(id?.last_seen) },
              ].map(({ label, value }) => (
                <div key={label} className="bg-card-secondary rounded-lg p-3 text-center">
                  <div className="text-lg font-bold text-text">{value ?? '—'}</div>
                  <div className="text-xs text-text-secondary">{label}</div>
                </div>
              ))}
            </div>

            {/* Violation type breakdown */}
            {data?.type_counts && Object.keys(data.type_counts).length > 0 && (
              <div>
                <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-2">By type</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(data.type_counts).map(([type, count]) => (
                    <div key={type} className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs text-white"
                         style={{ backgroundColor: violationColor(type) }}>
                      {type} <span className="font-bold">×{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Timeline */}
            <div>
              <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-3">Violation timeline</div>
              {data?.violations?.length === 0 ? (
                <p className="text-sm text-text-secondary text-center py-8">No violations recorded.</p>
              ) : (
                <div className="space-y-3">
                  {data?.violations?.map(v => (
                    <div key={v.id} className="flex items-start gap-3 p-3 bg-card-secondary rounded-lg">
                      {v.image_path && (
                        <img src={`${API}${v.image_path}`} alt="violation"
                             className="w-14 h-14 object-cover rounded cursor-pointer hover:opacity-80 flex-shrink-0"
                             onClick={() => onModalImage(`${API}${v.image_path}`)} />
                      )}
                      <div className="flex-1 min-w-0">
                        <ViolationBadge type={v.violation_type} />
                        <div className="text-xs text-text-secondary mt-1">{fmt(v.timestamp)}</div>
                        {v.stream_id && (
                          <div className="text-xs text-text-secondary truncate">Stream: {v.stream_id.slice(0, 8)}</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* First seen */}
            <div className="text-xs text-text-secondary text-center pt-2 border-t border-border">
              First seen: {fmt(id?.created_at)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ── Identity Management Page ──────────────────────────────────────────────────
const IdentityManagementPage = ({ onModalImage }) => {
  const [identities, setIdentities]   = useState([]);
  const [total, setTotal]             = useState(0);
  const [pages, setPages]             = useState(1);
  const [page, setPage]               = useState(1);
  const [search, setSearch]           = useState('');
  const [confirmed, setConfirmed]     = useState('all');
  const [selected, setSelected]       = useState(null);
  const [loading, setLoading]         = useState(true);
  const [mergeSource, setMergeSource] = useState(null);   // identity being merged FROM
  const [clustering, setClustering]   = useState(false);
  const searchRef = useRef(null);

  const load = useCallback(async (p = page, s = search, c = confirmed) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ page: p, limit: 24, search: s, confirmed: c, sort: 'last_seen' });
      const r = await fetch(`${API}/face/identities?${params}`);
      const d = await r.json();
      setIdentities(d.identities || []);
      setTotal(d.total || 0);
      setPages(d.pages || 1);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [page, search, confirmed]);

  useEffect(() => { load(page, search, confirmed); }, [page, confirmed]);

  // Debounced search
  useEffect(() => {
    const t = setTimeout(() => { setPage(1); load(1, search, confirmed); }, 400);
    return () => clearTimeout(t);
  }, [search]);

  const handleMerge = async (target) => {
    if (!mergeSource || target.id === mergeSource.id) { setMergeSource(null); return; }
    if (!window.confirm(`Merge "${mergeSource.label}" into "${target.label}"?\nThis cannot be undone.`)) {
      setMergeSource(null); return;
    }
    await fetch(`${API}/face/identity/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_id: mergeSource.id, target_id: target.id }),
    });
    setMergeSource(null);
    load(page, search, confirmed);
  };

  const runClustering = async () => {
    setClustering(true);
    try {
      const r = await fetch(`${API}/face/cluster`, { method: 'POST' });
      const d = await r.json();
      alert(`Clustering complete:\n• ${d.clusters_found} clusters found\n• ${d.identities_merged} identities merged\n• ${d.embeddings_reassigned} embeddings reassigned`);
      load(page, search, confirmed);
    } finally {
      setClustering(false);
    }
  };

  return (
    <div className="p-6 h-full">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div>
          <h2 className="text-2xl font-bold text-text">Identities</h2>
          <p className="text-sm text-text-secondary">{total} person{total !== 1 ? 's' : ''} detected</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={runClustering} disabled={clustering}
                  className="flex items-center gap-2 px-3 py-2 bg-card border border-border rounded-lg text-sm text-text-secondary hover:bg-border disabled:opacity-50 transition-all">
            <RefreshCw size={14} className={clustering ? 'animate-spin' : ''} />
            {clustering ? 'Clustering…' : 'Run clustering'}
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-5">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px]">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary" />
          <input ref={searchRef} value={search} onChange={e => { setSearch(e.target.value); }}
                 placeholder="Search by name…"
                 className="w-full pl-9 pr-4 py-2 text-sm bg-card border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary" />
        </div>

        {/* Confirmed filter */}
        <div className="flex rounded-lg overflow-hidden border border-border text-sm">
          {[['all', 'All'], ['false', 'Unconfirmed'], ['true', 'Confirmed']].map(([val, lbl]) => (
            <button key={val} onClick={() => { setConfirmed(val); setPage(1); }}
                    className={`px-3 py-2 transition-all ${confirmed === val ? 'bg-primary text-white' : 'bg-card text-text-secondary hover:bg-border'}`}>
              {lbl}
            </button>
          ))}
        </div>
      </div>

      {/* Merge mode banner */}
      {mergeSource && (
        <div className="mb-4 px-4 py-3 bg-amber-50 border border-amber-300 rounded-lg flex items-center justify-between text-sm">
          <span className="text-amber-800">
            <strong>Merge mode:</strong> Click another identity to merge <em>"{mergeSource.label}"</em> into it.
          </span>
          <button onClick={() => setMergeSource(null)}
                  className="text-amber-600 hover:text-amber-800 font-semibold">Cancel</button>
        </div>
      )}

      {/* Grid */}
      {loading ? (
        <div className="flex justify-center py-24">
          <Loader size={36} className="animate-spin text-text-secondary" />
        </div>
      ) : identities.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <Users size={48} className="text-text-secondary mb-3" />
          <p className="text-lg font-semibold text-text">No identities found</p>
          <p className="text-sm text-text-secondary">Start a stream to begin detecting people.</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-3">
          {identities.map(identity => (
            <IdentityCard key={identity.id}
              identity={identity}
              onSelect={setSelected}
              onMergeStart={setMergeSource}
              merging={!!mergeSource && mergeSource.id !== identity.id}
              onMergeTarget={handleMerge}
            />
          ))}
        </div>
      )}

      {/* Pagination */}
      {pages > 1 && (
        <div className="flex items-center justify-center gap-3 mt-6">
          <button disabled={page === 1} onClick={() => setPage(p => p - 1)}
                  className="p-2 rounded-lg border border-border hover:bg-border disabled:opacity-40">
            <ChevronLeft size={16} />
          </button>
          <span className="text-sm text-text-secondary">Page {page} of {pages}</span>
          <button disabled={page === pages} onClick={() => setPage(p => p + 1)}
                  className="p-2 rounded-lg border border-border hover:bg-border disabled:opacity-40">
            <ChevronRight size={16} />
          </button>
        </div>
      )}

      {/* Detail panel */}
      {selected && (
        <IdentityDetail identity={selected}
          onClose={() => { setSelected(null); load(page, search, confirmed); }}
          onModalImage={onModalImage} />
      )}
    </div>
  );
};

// ── Violation Log ─────────────────────────────────────────────────────────────
const ViolationLogPage = ({ onModalImage }) => {
  const [violations, setViolations] = useState([]);
  const [loading, setLoading]       = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    fetch(`${API}/violations`)
      .then(r => r.json())
      .then(setViolations)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, []);

  const clear = async () => {
    if (!window.confirm('Clear all violations? This cannot be undone.')) return;
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
          <button onClick={load} className="px-3 py-2 bg-card border border-border rounded-lg text-sm text-text-secondary hover:bg-border">
            <RefreshCw size={14} />
          </button>
          <button onClick={clear}
                  className="flex items-center gap-2 px-3 py-2 bg-red-500 text-white text-sm rounded-lg hover:bg-red-600">
            <Trash2 size={14} /> Clear all
          </button>
        </div>
      </div>

      <div className="bg-card border border-border rounded-xl overflow-hidden">
        {loading ? (
          <div className="flex justify-center py-20"><Loader size={32} className="animate-spin text-text-secondary" /></div>
        ) : violations.length === 0 ? (
          <p className="text-sm text-text-secondary text-center py-16">No violations recorded.</p>
        ) : (
          <div className="divide-y divide-border">
            {violations.map(v => (
              <div key={v.id} className="flex items-start gap-4 p-4 hover:bg-card-secondary transition-all">
                {v.image_path ? (
                  <img src={`${API}${v.image_path}`} alt="violation"
                       className="w-16 h-16 object-cover rounded-lg cursor-pointer hover:opacity-80 flex-shrink-0"
                       onClick={() => onModalImage(`${API}${v.image_path}`)} />
                ) : (
                  <div className="w-16 h-16 bg-card-secondary rounded-lg flex-shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-semibold text-text truncate">{v.name}</span>
                    <ViolationBadge type={v.violation_type} />
                  </div>
                  <div className="text-xs text-text-secondary">{fmt(v.timestamp)}</div>
                  {v.match_score != null && (
                    <div className="text-xs text-text-secondary">Match confidence: {(v.match_score * 100).toFixed(1)}%</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// ── Live Monitor (stream controls) ────────────────────────────────────────────
const LiveMonitorPage = ({ activeStreams, stopStream, startStream, serverStatus, handleVideoUpload, fileInputRef, isLoading, zoomedId, setZoomedId }) => {
  const zoomed = zoomedId ? activeStreams[zoomedId] : null;

  if (zoomed) {
    const others = Object.values(activeStreams).filter(s => s.streamId !== zoomedId);
    const hasViolation = zoomed.detections?.some(d => !d.safe);
    return (
      <div className="p-6 h-full flex gap-6">
        <div className="flex-1 flex flex-col min-h-0">
          <h2 className="text-2xl font-bold text-text mb-4 flex-shrink-0">Live Monitor</h2>
          <div className="flex-1 bg-card border border-border rounded-xl p-4 flex flex-col min-h-0">
            <div className="relative bg-black rounded-lg overflow-hidden flex-1">
              <img src={zoomed.videoSrc} className="w-full h-full object-contain" alt={zoomed.name} />
            </div>
            <div className="flex items-center justify-between mt-3 flex-shrink-0">
              <span className="font-semibold text-text">{zoomed.name}</span>
              <div className="flex items-center gap-3">
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm
                  ${hasViolation ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                  {hasViolation ? <ShieldOff size={16} /> : <ShieldCheck size={16} />}
                  {hasViolation ? 'Violation detected' : 'All clear'}
                </div>
                <button onClick={() => setZoomedId(null)} className="p-2 hover:bg-border rounded-lg text-text-secondary">
                  <Minimize size={18} />
                </button>
              </div>
            </div>
          </div>
        </div>
        {others.length > 0 && (
          <aside className="w-64 flex-shrink-0 pt-16 overflow-y-auto space-y-3">
            {others.map(s => (
              <div key={s.streamId} onClick={() => setZoomedId(s.streamId)}
                   className="bg-card border border-border rounded-xl p-2 cursor-pointer hover:border-primary transition-all">
                <div className="bg-black rounded-lg aspect-video overflow-hidden mb-2">
                  <img src={s.videoSrc} className="w-full h-full object-cover" alt={s.name} />
                </div>
                <p className="text-xs font-semibold text-text truncate">{s.name}</p>
              </div>
            ))}
          </aside>
        )}
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-2xl font-bold text-text">Live Monitor</h2>
        <div className="flex gap-2">
          <button onClick={() => startStream('webcam', '0', 'Webcam')}
                  disabled={serverStatus !== 'connected' || isLoading}
                  className="flex items-center gap-2 px-4 py-2 bg-primary text-white text-sm rounded-lg hover:opacity-90 disabled:opacity-50">
            <Camera size={15} /> Webcam
          </button>
          <button onClick={() => fileInputRef.current?.click()}
                  disabled={serverStatus !== 'connected' || isLoading}
                  className="flex items-center gap-2 px-4 py-2 bg-primary text-white text-sm rounded-lg hover:opacity-90 disabled:opacity-50">
            <Upload size={15} /> Upload video
          </button>
        </div>
      </div>

      {Object.keys(activeStreams).length === 0 ? (
        <div className="flex flex-col items-center justify-center text-center bg-card border border-border rounded-xl p-16 h-80">
          {isLoading ? <Loader size={40} className="animate-spin text-text-secondary mb-3" /> : <Tv size={48} className="text-text-secondary mb-3" />}
          <p className="text-base font-semibold text-text">{isLoading ? 'Starting stream…' : 'No active streams'}</p>
          <p className="text-sm text-text-secondary">Use the buttons above to begin monitoring.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {Object.values(activeStreams).map(stream => {
            const hasViolation = stream.detections?.some(d => !d.safe);
            return (
              <div key={stream.streamId} className="bg-card border border-border rounded-xl p-4">
                <div className="relative bg-black rounded-lg aspect-video overflow-hidden mb-3 group">
                  <img src={stream.videoSrc} className="w-full h-full object-contain" alt={stream.name} />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-all flex items-center justify-center">
                    <button onClick={() => setZoomedId(stream.streamId)}
                            className="p-3 bg-white/20 text-white rounded-full hover:bg-white/40 backdrop-blur-sm">
                      <Maximize size={22} />
                    </button>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-text text-sm">{stream.name}</span>
                  <button onClick={() => stopStream(stream.streamId)} className="p-1.5 text-red-500 hover:bg-border rounded">
                    <StopCircle size={18} />
                  </button>
                </div>
                <div className={`flex items-center gap-2 mt-2 px-3 py-2 rounded-lg text-sm
                  ${hasViolation ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                  {hasViolation ? <ShieldOff size={15} /> : <ShieldCheck size={15} />}
                  {hasViolation ? 'Safety violation' : 'All clear'}
                </div>
                <div className="flex justify-between text-xs text-text-secondary mt-2">
                  <span>FPS: {stream.stats?.fps?.toFixed(1) ?? '0.0'}</span>
                  <span>Violations: {stream.stats?.violation_count ?? 0}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ── Settings ──────────────────────────────────────────────────────────────────
const SettingsPage = ({ rtspStreams, setRtspStreams, startStream, serverStatus }) => {
  const add    = () => setRtspStreams(p => [...p, { id: Date.now().toString(), name: `Camera ${p.length + 1}`, url: '' }]);
  const remove = id => setRtspStreams(p => p.filter(s => s.id !== id));
  const update = (id, f, v) => setRtspStreams(p => p.map(s => s.id === id ? { ...s, [f]: v } : s));

  return (
    <div className="p-6 max-w-3xl">
      <h2 className="text-2xl font-bold text-text mb-5">Settings</h2>
      <div className="bg-card border border-border rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-text">RTSP Streams</h3>
          <button onClick={add} className="flex items-center gap-2 px-3 py-1.5 bg-primary text-white text-sm rounded-lg hover:opacity-90">
            <Plus size={14} /> Add
          </button>
        </div>
        {rtspStreams.length === 0 ? (
          <p className="text-sm text-text-secondary text-center py-8">No RTSP streams configured.</p>
        ) : (
          <div className="space-y-3">
            {rtspStreams.map(s => (
              <div key={s.id} className="flex items-center gap-3">
                <input value={s.name} onChange={e => update(s.id, 'name', e.target.value)}
                       placeholder="Camera name"
                       className="w-1/3 px-3 py-2 text-sm bg-background border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary" />
                <input value={s.url} onChange={e => update(s.id, 'url', e.target.value)}
                       placeholder="rtsp://..."
                       className="flex-1 px-3 py-2 text-sm bg-background border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary" />
                <button onClick={() => startStream('rtsp', s.url, s.name)}
                        disabled={!s.url || serverStatus !== 'connected'}
                        className="px-3 py-2 bg-green-600 text-white rounded-lg hover:opacity-90 disabled:opacity-40">
                  <Play size={14} />
                </button>
                <button onClick={() => remove(s.id)} className="p-2 text-red-500 hover:bg-border rounded-lg">
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [view, setView]               = useState('dashboard');
  const [serverStatus, setServerStatus] = useState('checking');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rtspStreams, setRtspStreams]  = useState([]);
  const [isLoading, setIsLoading]     = useState(false);
  const [activeStreams, setActiveStreams] = useState({});
  const [modalImage, setModalImage]   = useState(null);
  const [zoomedId, setZoomedId]       = useState(null);
  const fileInputRef = useRef(null);

  // ── Health check ────────────────────────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API}/health`);
      const d = await r.json();
      setServerStatus(r.ok && d.model_loaded ? 'connected' : 'degraded');
    } catch {
      setServerStatus('disconnected');
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const i = setInterval(checkHealth, 10000);
    return () => clearInterval(i);
  }, [checkHealth]);

  // ── Stream management ────────────────────────────────────────────────────────
  const startStream = useCallback(async (source_type, source_path, name) => {
    setIsLoading(true);
    try {
      const r = await fetch(`${API}/stream/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_type, source_path, name }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || 'Failed');
      setActiveStreams(prev => ({
        ...prev,
        [d.stream_id]: {
          streamId: d.stream_id, name: d.name,
          videoSrc: `${API}/stream/video_feed/${d.stream_id}?t=${Date.now()}`,
          stats: { fps: 0, violation_count: 0 }, detections: [],
        }
      }));
      setView('monitor');
    } catch (e) {
      alert(`Failed to start: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stopStream = useCallback(async (streamId) => {
    if (zoomedId === streamId) setZoomedId(null);
    await fetch(`${API}/stream/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stream_id: streamId }),
    }).catch(() => {});
    setActiveStreams(prev => { const n = { ...prev }; delete n[streamId]; return n; });
  }, [zoomedId]);

  // Poll detections
  useEffect(() => {
    if (!Object.keys(activeStreams).length) return;
    const i = setInterval(async () => {
      for (const sid of Object.keys(activeStreams)) {
        try {
          const r = await fetch(`${API}/stream/detections/${sid}`);
          if (!r.ok) { if (r.status === 404) stopStream(sid); continue; }
          const d = await r.json();
          setActiveStreams(prev => prev[sid]
            ? { ...prev, [sid]: { ...prev[sid], stats: d, detections: d.last_detections || [] } }
            : prev);
        } catch {}
      }
    }, 1000);
    return () => clearInterval(i);
  }, [activeStreams, stopStream]);

  const handleVideoUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsLoading(true);
    const fd = new FormData();
    fd.append('video', file);
    try {
      const r = await fetch(`${API}/upload/video`, { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || 'Upload failed');
      startStream('video', d.path, file.name);
    } catch (e) {
      alert(`Upload failed: ${e.message}`);
    } finally {
      e.target.value = ''; setIsLoading(false);
    }
  };

  const renderPage = () => {
    switch (view) {
      case 'dashboard':   return <DashboardPage setView={setView} />;
      case 'identities':  return <IdentityManagementPage onModalImage={setModalImage} />;
      case 'violations':  return <ViolationLogPage onModalImage={setModalImage} />;
      case 'monitor':     return (
        <LiveMonitorPage activeStreams={activeStreams} stopStream={stopStream} startStream={startStream}
          serverStatus={serverStatus} handleVideoUpload={handleVideoUpload}
          fileInputRef={fileInputRef} isLoading={isLoading}
          zoomedId={zoomedId} setZoomedId={setZoomedId} />
      );
      case 'settings':    return (
        <SettingsPage rtspStreams={rtspStreams} setRtspStreams={setRtspStreams}
          startStream={startStream} serverStatus={serverStatus} />
      );
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-background text-text font-sans flex">
      <ImageModal imageUrl={modalImage} onClose={() => setModalImage(null)} />
      <Sidebar view={view} setView={setView} open={sidebarOpen} setOpen={setSidebarOpen} serverStatus={serverStatus} />
      <main className={`flex-1 h-screen overflow-y-auto transition-all duration-300 ${sidebarOpen ? 'lg:ml-56' : 'lg:ml-16'}`}>
        {renderPage()}
      </main>
      <input ref={fileInputRef} type="file" accept="video/*" onChange={handleVideoUpload} className="hidden" />
    </div>
  );
}