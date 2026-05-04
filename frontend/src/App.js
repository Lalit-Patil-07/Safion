import React, { useState, useRef, useEffect, useCallback } from 'react';
import ViolationLogPage from "./ViolationLogPage";
import { useAuth } from './auth/AuthContext';
import LoginPage from './auth/LoginPage';
import {
  Camera, Upload, Video, CheckCircle, XCircle, Settings,
  Play, StopCircle, Loader, Shield, ShieldOff, ShieldCheck,
  X, Tv, Plus, Trash2, Maximize, Minimize, Users, ChevronLeft,
  ChevronRight, Search, GitMerge, Edit3, Clock, Activity,
  BarChart2, Eye, Check, AlertTriangle, RefreshCw, Inbox,
  Star, TrendingUp, Filter, Zap, Link2, LogOut
} from 'lucide-react';
import API from './config';


const VIOLATION_COLORS = {
  'NO-Hardhat':     '#EF4444',
  'NO-Mask':        '#F59E0B',
  'NO-Safety Vest': '#EC4899',
};
const vColor = (type) => VIOLATION_COLORS[type] || '#6B7280';

const ago = (iso) => {
  if (!iso) return 'Never';
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return 'Just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
};
const fmt = (iso) => iso ? new Date(iso).toLocaleString() : '—';

// ── Confidence indicator ──────────────────────────────────────────────────────
const ConfidenceBar = ({ score, showLabel = false }) => {
  const pct = Math.round((score || 0) * 100);
  const color = pct >= 70 ? '#22c55e' : pct >= 40 ? '#f59e0b' : '#ef4444';
  const label = pct >= 70 ? 'High' : pct >= 40 ? 'Medium' : 'Low';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-card-secondary rounded-full overflow-hidden">
        <div style={{ width: `${pct}%`, backgroundColor: color, height: '100%' }} className="rounded-full transition-all" />
      </div>
      {showLabel && <span className="text-xs text-text-secondary">{label}</span>}
    </div>
  );
};

// ── Shared atoms ──────────────────────────────────────────────────────────────
const StatusBadge = ({ is_confirmed, is_archived }) => {
  if (is_archived)  return <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500">Archived</span>;
  if (is_confirmed) return <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700">Confirmed</span>;
  return <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-700">Unconfirmed</span>;
};

const ViolationChip = ({ type }) => (
  <span className="text-xs px-2 py-0.5 rounded-full text-white font-medium"
        style={{ backgroundColor: vColor(type) }}>{type}</span>
);

const FaceAvatar = ({ src, label, size = 'md' }) => {
  const s = size === 'lg' ? 'w-20 h-20' : size === 'sm' ? 'w-9 h-9' : 'w-14 h-14';
  return src
    ? <img src={`${API}${src}`} alt={label} className={`${s} rounded-full object-cover border-2 border-border flex-shrink-0 bg-card-secondary`} />
    : <div className={`${s} rounded-full bg-card-secondary border-2 border-border flex items-center justify-center flex-shrink-0`}>
        <Users size={size === 'lg' ? 28 : size === 'sm' ? 14 : 20} className="text-text-secondary" />
      </div>;
};

const ImageModal = ({ imageUrl, onClose }) => {
  if (!imageUrl) return null;
  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50" onClick={onClose}>
      <div className="relative max-w-3xl max-h-[90vh] p-4" onClick={e => e.stopPropagation()}>
        <img src={imageUrl} alt="Evidence" className="w-full h-full object-contain rounded-xl" />
        <button onClick={onClose} className="absolute -top-2 -right-2 bg-card rounded-full p-2 hover:bg-border"><X size={20} /></button>
      </div>
    </div>
  );
};

// ── Sidebar ───────────────────────────────────────────────────────────────────

const SafionLogo = ({ className, width, height }) => (
  <svg width={width} height={height} viewBox="0 0 123 152" className={className}>
    <rect x="8"  y="8"   width="20" height="20" rx="8" fill="#181826"/>
    <rect x="37" y="8"   width="20" height="20" rx="8" fill="#F54F00"/>
    <rect x="66" y="8"   width="20" height="20" rx="8" fill="#F54F00"/>
    <rect x="95" y="8"   width="20" height="20" rx="8" fill="#F54F00"/>
    <rect x="8"  y="37"  width="20" height="20" rx="8" fill="#F54F00"/>
    <rect x="37" y="37"  width="20" height="20" rx="8" fill="#181826"/>
    <rect x="66" y="37"  width="20" height="20" rx="8" fill="#181826"/>
    <rect x="95" y="37"  width="20" height="20" rx="8" fill="#181826"/>
    <rect x="8"  y="66"  width="20" height="20" rx="8" fill="#181826"/>
    <rect x="37" y="66"  width="20" height="20" rx="8" fill="#F54F00"/>
    <rect x="66" y="66"  width="20" height="20" rx="8" fill="#F54F00"/>
    <rect x="95" y="66"  width="20" height="20" rx="8" fill="#181826"/>
    <rect x="8"  y="95"  width="20" height="20" rx="8" fill="#181826"/>
    <rect x="37" y="95"  width="20" height="20" rx="8" fill="#181826"/>
    <rect x="66" y="95"  width="20" height="20" rx="8" fill="#181826"/>
    <rect x="95" y="95"  width="20" height="20" rx="8" fill="#F54F00"/>
    <rect x="8"  y="124" width="20" height="20" rx="8" fill="#F54F00"/>
    <rect x="37" y="124" width="20" height="20" rx="8" fill="#F54F00"/>
    <rect x="66" y="124" width="20" height="20" rx="8" fill="#F54F00"/>
    <rect x="95" y="124" width="20" height="20" rx="8" fill="#181826"/>
  </svg>
);

const NAV = [
  { id: 'dashboard',   icon: BarChart2,    label: 'Dashboard' },
  { id: 'identities',  icon: Users,        label: 'Identities' },
  { id: 'import',      icon: Upload,       label: 'Import Identities' },
  { id: 'review',      icon: Inbox,        label: 'Review Queue' },
  { id: 'suggestions', icon: Zap,          label: 'Merge Suggestions' },
  { id: 'violations',  icon: AlertTriangle,label: 'Violations' },
  { id: 'monitor',     icon: Tv,           label: 'Live Monitor' },
  { id: 'settings',    icon: Settings,     label: 'Settings' },
];

const Sidebar = ({ view, setView, open, setOpen, serverStatus, reviewCount, suggestionsCount, user, onLogout }) => (
  <>
    {open && <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={() => setOpen(false)} />}
    <aside className={`fixed top-0 left-0 h-full bg-card-secondary border-r border-border z-50 transition-all duration-300
                       ${open ? 'translate-x-0 w-56' : 'translate-x-0 w-16'}`}>
      <div className="flex flex-col h-full">
        <div className="p-3 border-b border-border relative min-h-[56px] overflow-hidden">
          {open ? (
            <div className="flex flex-col items-center justify-center w-full py-2 gap-4">
              <button onClick={() => setOpen(false)} className="absolute top-3 left-3 p-2 text-text hover:bg-border rounded-md z-10"><ChevronLeft size={17} /></button>
              <SafionLogo width="64" height="82" className="flex-shrink-0" />
              <span className="font-sans font-medium uppercase text-text" style={{ fontSize: '18px', letterSpacing: '0.42em' }}>SAFION</span>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center w-full h-full py-2">
              <button onClick={() => setOpen(true)} className="p-2 text-text hover:bg-border rounded-md">
                <SafionLogo width="24" height="30" className="flex-shrink-0" />
              </button>
            </div>
          )}
        </div>

        <nav className="flex flex-col gap-1 p-2 flex-1">
          {NAV.map(({ id, icon: Icon, label }) => (
            <button key={id}
              onClick={() => { setView(id); if (window.innerWidth < 1024) setOpen(false); }}
              title={!open ? label : ''}
              className={`relative flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all
                          ${view === id ? 'bg-primary text-white' : 'text-text-secondary hover:bg-border'}
                          ${open ? 'justify-start' : 'justify-center'}`}>
              <Icon size={16} className="flex-shrink-0" />
              {open && <span className="whitespace-nowrap">{label}</span>}
              {id === 'review' && reviewCount > 0 && (
                <span className={`${open ? 'ml-auto' : 'absolute top-1 right-1'} text-xs px-1.5 py-0.5 rounded-full font-bold
                                  ${view === id ? 'bg-white text-primary' : 'bg-amber-500 text-white'}`}>
                  {reviewCount > 99 ? '99+' : reviewCount}
                </span>
              )}
              {id === 'suggestions' && suggestionsCount > 0 && (
                <span className={`${open ? 'ml-auto' : 'absolute top-1 right-1'} text-xs px-1.5 py-0.5 rounded-full font-bold
                                  ${view === id ? 'bg-white text-primary' : 'bg-blue-500 text-white'}`}>
                  {suggestionsCount > 99 ? '99+' : suggestionsCount}
                </span>
              )}
            </button>
          ))}
        </nav>

        <div className="p-3 border-t border-border space-y-1">
          <div className={`flex items-center gap-3 px-3 py-2 ${open ? '' : 'justify-center'}`}>
            <div className={`w-2 h-2 rounded-full flex-shrink-0
              ${serverStatus === 'connected' ? 'bg-green-500' : serverStatus === 'degraded' ? 'bg-amber-500' : 'bg-red-500'}`} />
            {open && (
              <div className="flex-1 min-w-0">
                <div className="text-xs text-text-secondary">Server</div>
                <div className={`text-xs font-semibold capitalize
                  ${serverStatus === 'connected' ? 'text-green-500' : serverStatus === 'degraded' ? 'text-amber-500' : 'text-red-500'}`}>
                  {serverStatus}
                </div>
              </div>
            )}
          </div>
          <button
            onClick={onLogout}
            title={!open ? 'Sign out' : ''}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-xs font-medium
                        text-text-secondary hover:bg-border hover:text-red-400 transition-all
                        ${open ? 'justify-start' : 'justify-center'}`}>
            <LogOut size={14} className="flex-shrink-0" />
            {open && <span>{user?.username || 'Sign out'}</span>}
          </button>
        </div>
      </div>
    </aside>
  </>
);

// ── Dashboard ─────────────────────────────────────────────────────────────────
const StatCard = ({ label, value, icon: Icon, color, sub }) => (
  <div className="bg-card border border-border rounded-xl p-5 flex items-start gap-4">
    <div className={`p-3 rounded-xl ${color} bg-opacity-10`} style={{ backgroundColor: `${color}18` }}>
      <Icon size={19} style={{ color }} />
    </div>
    <div>
      <div className="text-2xl font-bold text-text">{value ?? '—'}</div>
      <div className="text-sm text-text">{label}</div>
      {sub && <div className="text-xs text-text-secondary mt-0.5">{sub}</div>}
    </div>
  </div>
);

const DashboardPage = ({ setView }) => {
  const [stats, setStats] = useState(null);
  useEffect(() => {
    fetch(`${API}/face/stats`).then(r => r.json()).then(setStats).catch(() => {});
  }, []);

  return (
    <div className="p-6 pt-12 lg:pt-6 w-full max-w-7xl mx-auto">
      <h2 className="text-2xl font-bold text-text mb-6">Dashboard</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <StatCard label="Total identities"  value={stats?.total_identities}  icon={Users}         color="#3b82f6" />
        <StatCard label="Confirmed"         value={stats?.confirmed_count}   icon={CheckCircle}    color="#22c55e" />
        <StatCard label="Needs review"      value={stats?.unconfirmed_count} icon={Inbox}          color="#f59e0b" />
        <StatCard label="Total violations"  value={stats?.total_violations}  icon={AlertTriangle}  color="#ef4444" />
        <StatCard label="Today"             value={stats?.violations_today}  icon={Clock}          color="#8b5cf6" />
        <StatCard label="Repeat offenders"  value={stats?.repeat_offenders}  icon={TrendingUp}     color="#f97316" sub="≥3 violations" />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <button onClick={() => setView('review')}
                className="bg-amber-50 border border-amber-200 rounded-xl p-5 text-left hover:border-amber-400 transition-all">
          <div className="text-base font-semibold text-amber-800 mb-1">Review unconfirmed identities</div>
          <div className="text-sm text-amber-600">Confirm, merge, or archive auto-detected people</div>
        </button>
        <button onClick={() => setView('monitor')}
                className="bg-card border border-border rounded-xl p-5 text-left hover:border-primary transition-all">
          <div className="text-base font-semibold text-text mb-1">Start monitoring</div>
          <div className="text-sm text-text-secondary">Open webcam, RTSP stream, or upload video</div>
        </button>
      </div>
    </div>
  );
};

// ── Review Queue ──────────────────────────────────────────────────────────────
const ReviewQueuePage = ({ onCountChange, onModalImage }) => {
  const [items, setItems]       = useState([]);
  const [loading, setLoading]   = useState(true);
  const [renaming, setRenaming] = useState(null);
  const [newLabel, setNewLabel] = useState('');
  const renameRef = useRef(null);

  const load = useCallback(async () => {
    setLoading(true);
    const r = await fetch(`${API}/face/review-queue`);
    const d = await r.json().catch(() => []);
    setItems(d);
    onCountChange(d.length);
    setLoading(false);
  }, [onCountChange]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (renaming) renameRef.current?.focus(); }, [renaming]);

  const confirm = async (id) => {
    const identity = items.find(i => i.id === id);
    if (!identity) return;
    await fetch(`${API}/face/identity/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: identity.label }),
    });
    setItems(prev => prev.filter(i => i.id !== id));
    onCountChange(items.length - 1);
  };

  const archive = async (id) => {
    await fetch(`${API}/face/identity/${id}`, { method: 'DELETE' });
    setItems(prev => prev.filter(i => i.id !== id));
    onCountChange(items.length - 1);
  };

  const saveRename = async (id) => {
    if (!newLabel.trim()) { setRenaming(null); return; }
    const r = await fetch(`${API}/face/identity/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: newLabel.trim() }),
    });
    if (r.ok) {
      setItems(prev => prev.filter(i => i.id !== id));
      onCountChange(items.length - 1);
    }
    setRenaming(null);
  };

  if (loading) return <div className="flex justify-center py-24"><Loader size={32} className="animate-spin text-text-secondary" /></div>;

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 className="text-2xl font-bold text-text">Review Queue</h2>
          <p className="text-sm text-text-secondary">
            {items.length} unconfirmed {items.length === 1 ? 'identity' : 'identities'} — high-risk first
          </p>
        </div>
        <button onClick={load} className="p-2 border border-border rounded-lg hover:bg-border text-text-secondary">
          <RefreshCw size={15} />
        </button>
      </div>

      {items.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center bg-card border border-border rounded-xl">
          <CheckCircle size={48} className="text-green-500 mb-3" />
          <p className="text-lg font-semibold text-text">Queue is clear</p>
          <p className="text-sm text-text-secondary">All detected identities have been reviewed.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {items.map(identity => (
            <div key={identity.id}
                 className="bg-card border border-border rounded-xl p-4 flex items-start gap-4 hover:border-primary/30 transition-all">
              <FaceAvatar src={identity.thumbnail} label={identity.label} />
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2 mb-1">
                  <div className="flex-1 min-w-0">
                    {renaming === identity.id ? (
                      <div className="flex items-center gap-2">
                        <input ref={renameRef} value={newLabel} onChange={e => setNewLabel(e.target.value)}
                               onKeyDown={e => { if (e.key === 'Enter') saveRename(identity.id); if (e.key === 'Escape') setRenaming(null); }}
                               className="flex-1 text-sm font-semibold text-text bg-background border border-primary rounded px-2 py-1 outline-none"
                               placeholder="Enter name…" />
                        <button onClick={() => saveRename(identity.id)} className="p-1.5 bg-primary text-white rounded"><Check size={12} /></button>
                        <button onClick={() => setRenaming(null)} className="p-1.5 hover:bg-border rounded text-text-secondary"><X size={12} /></button>
                      </div>
                    ) : (
                      <span className="text-sm font-semibold text-text truncate">{identity.label}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="text-xs text-text-secondary">{ago(identity.last_seen)}</span>
                  </div>
                </div>
                <div className="flex items-center gap-4 mb-2 text-xs text-text-secondary">
                  <span>{identity.violation_count ?? 0} violations</span>
                  <span>{identity.embedding_count ?? 0} embeddings</span>
                  <div className="flex items-center gap-1.5 flex-1 max-w-[120px]">
                    <span>Confidence</span>
                    <ConfidenceBar score={identity.identity_confidence} />
                  </div>
                </div>
                {identity.preview_images?.length > 0 && (
                  <div className="flex gap-2 mb-3">
                    {identity.preview_images.map((img, i) => (
                      <img key={i} src={`${API}${img}`} alt="preview"
                           className="w-14 h-14 object-cover rounded-lg cursor-pointer hover:opacity-80 border border-border"
                           onClick={() => onModalImage(`${API}${img}`)} />
                    ))}
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <button onClick={() => confirm(identity.id)}
                          className="flex items-center gap-1.5 px-3 py-1.5 bg-green-500 text-white text-xs font-semibold rounded-lg hover:bg-green-600 transition-all">
                    <Check size={12} /> Confirm
                  </button>
                  <button onClick={() => { setRenaming(identity.id); setNewLabel(identity.label); }}
                          className="flex items-center gap-1.5 px-3 py-1.5 bg-card-secondary border border-border text-xs font-semibold text-text-secondary rounded-lg hover:bg-border transition-all">
                    <Edit3 size={12} /> Rename & confirm
                  </button>
                  <button onClick={() => archive(identity.id)}
                          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-text-secondary rounded-lg hover:bg-border transition-all">
                    <X size={12} /> Archive
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ── Identity Card ─────────────────────────────────────────────────────────────
const IdentityCard = ({ identity, onSelect, onMergeStart, merging, onMergeTarget, onRefresh }) => {
  const [editing,  setEditing]  = useState(false);
  const [newLabel, setNewLabel] = useState(identity.label);
  const inputRef = useRef(null);

  useEffect(() => { if (editing) inputRef.current?.focus(); }, [editing]);

  const save = async () => {
    const label = newLabel.trim();
    if (!label || label === identity.label) { setEditing(false); return; }
    const r = await fetch(`${API}/face/identity/${identity.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    });
    setEditing(false);
    if (r.ok) onRefresh();
  };

  return (
    <div onClick={() => { if (merging) { onMergeTarget(identity); return; } if (!editing) onSelect(identity); }}
         className={`bg-card border rounded-xl p-3 cursor-pointer transition-all group
           ${merging ? 'border-amber-400 ring-2 ring-amber-100' : 'border-border hover:border-primary'}`}>

      <div className="flex items-start gap-2.5 mb-2">
        <FaceAvatar src={identity.thumbnail} label={identity.label} size="sm" />
        <div className="flex-1 min-w-0">
          {editing ? (
            <input ref={inputRef} value={newLabel} onChange={e => setNewLabel(e.target.value)}
                   onKeyDown={e => { if (e.key === 'Enter') save(); if (e.key === 'Escape') { setEditing(false); setNewLabel(identity.label); } }}
                   onClick={e => e.stopPropagation()}
                   className="w-full text-xs font-semibold text-text bg-background border border-primary rounded px-1.5 py-0.5 outline-none" />
          ) : (
            <div className="flex items-center gap-1">
              <span className="text-xs font-semibold text-text truncate">{identity.label}</span>
              <button onClick={e => { e.stopPropagation(); setEditing(true); }}
                      className="opacity-0 group-hover:opacity-100 p-0.5 text-text-secondary hover:text-primary">
                <Edit3 size={10} />
              </button>
            </div>
          )}
          <StatusBadge is_confirmed={identity.is_confirmed} is_archived={identity.is_archived} />
        </div>
      </div>

      <div className="mb-2">
        <ConfidenceBar score={identity.identity_confidence} />
      </div>

      <div className="flex justify-between text-xs text-text-secondary">
        <span>{identity.violation_count ?? 0} violations</span>
        <span>{ago(identity.last_seen)}</span>
      </div>

      {!merging && (
        <div className="flex gap-1.5 mt-2 opacity-0 group-hover:opacity-100 transition-all">
          <button onClick={e => { e.stopPropagation(); onSelect(identity); }}
                  className="flex-1 text-xs py-1 bg-card-secondary hover:bg-border rounded text-text-secondary flex items-center justify-center gap-1">
            <Eye size={10} /> View
          </button>
          <button onClick={e => { e.stopPropagation(); onMergeStart(identity); }}
                  className="flex-1 text-xs py-1 bg-card-secondary hover:bg-border rounded text-text-secondary flex items-center justify-center gap-1">
            <GitMerge size={10} /> Merge
          </button>
        </div>
      )}
    </div>
  );
};

// ── Identity Detail Panel ─────────────────────────────────────────────────────
const IdentityDetail = ({ identity, onClose, onModalImage, onRefresh }) => {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [newLabel,setNewLabel]= useState('');
  const [samples, setSamples] = useState([]);
  const [similar, setSimilar] = useState([]);

  const load = useCallback(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/face/identity/${identity.id}/violations`).then(r => r.json()),
      fetch(`${API}/face/identity/${identity.id}/samples`).then(r => r.json()).catch(() => []),
      fetch(`${API}/face/identity/${identity.id}/similarity?limit=3`).then(r => r.json()).catch(() => ({ similar: [] })),
    ]).then(([vdata, sampleData, simData]) => {
      setData(vdata);
      setNewLabel(vdata.identity?.label || '');
      setSamples(Array.isArray(sampleData) ? sampleData : []);
      setSimilar(simData.similar || []);
    }).catch(() => {}).finally(() => setLoading(false));
  }, [identity.id]);

  useEffect(() => { load(); }, [load]);

  const save = async () => {
    if (!newLabel.trim()) return;
    const r = await fetch(`${API}/face/identity/${identity.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: newLabel.trim() }),
    });
    if (r.ok) { setEditing(false); load(); onRefresh(); }
  };

  const id = data?.identity;

  return (
    <div className="fixed inset-0 bg-black/40 z-40 flex justify-end" onClick={onClose}>
      <div className="w-full max-w-lg bg-card h-full shadow-2xl overflow-y-auto"
           onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between p-4 border-b border-border sticky top-0 bg-card z-10">
          <h3 className="text-base font-semibold text-text">Identity Detail</h3>
          <button onClick={onClose} className="p-1.5 hover:bg-border rounded text-text-secondary"><X size={16} /></button>
        </div>

        {loading ? (
          <div className="flex justify-center py-20"><Loader size={28} className="animate-spin text-text-secondary" /></div>
        ) : (
          <div className="p-5 space-y-5">
            <div className="flex items-start gap-4">
              <FaceAvatar src={id?.thumbnail} label={id?.label} size="lg" />
              <div className="flex-1">
                {editing ? (
                  <div className="flex items-center gap-2">
                    <input autoFocus value={newLabel} onChange={e => setNewLabel(e.target.value)}
                           onKeyDown={e => { if (e.key === 'Enter') save(); if (e.key === 'Escape') setEditing(false); }}
                           className="flex-1 text-sm font-semibold text-text bg-background border border-primary rounded px-2 py-1 outline-none" />
                    <button onClick={save} className="p-1.5 bg-primary text-white rounded"><Check size={13} /></button>
                    <button onClick={() => setEditing(false)} className="p-1.5 hover:bg-border rounded text-text-secondary"><X size={13} /></button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-bold text-text">{id?.label}</span>
                    <button onClick={() => setEditing(true)} className="p-1 text-text-secondary hover:text-primary"><Edit3 size={13} /></button>
                  </div>
                )}
                <div className="mt-1"><StatusBadge is_confirmed={id?.is_confirmed} /></div>
              </div>
            </div>

            {samples.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-2">
                  Face samples (highest confidence)
                </div>
                <div className="flex gap-2 flex-wrap">
                  {samples.map((s, i) => (
                    <div key={i} className="relative group">
                      <img src={`${API}${s.image_path}`} alt={`sample ${i + 1}`}
                           className="w-16 h-16 object-cover rounded-xl cursor-pointer hover:opacity-80 border border-border"
                           onClick={() => onModalImage(`${API}${s.image_path}`)} />
                      {s.match_score != null && (
                        <span className="absolute bottom-0.5 right-0.5 text-xs bg-black/70 text-white px-1 rounded text-[10px]">
                          {Math.round(s.match_score * 100)}%
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="bg-card-secondary rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">Identity Confidence</span>
                <span className="text-sm font-bold text-text">{Math.round((id?.identity_confidence || 0) * 100)}%</span>
              </div>
              <ConfidenceBar score={id?.identity_confidence} showLabel />
              <p className="text-xs text-text-secondary mt-2">
                Based on match consistency across detections. Higher = more stable identity.
              </p>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {[
                { label: 'Violations', value: id?.violation_count },
                { label: 'Embeddings', value: id?.embedding_count },
                { label: 'Last seen',  value: ago(id?.last_seen) },
              ].map(({ label, value }) => (
                <div key={label} className="bg-card-secondary rounded-xl p-3 text-center">
                  <div className="text-base font-bold text-text">{value ?? '—'}</div>
                  <div className="text-xs text-text-secondary">{label}</div>
                </div>
              ))}
            </div>

            {similar.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-2 flex items-center gap-1.5">
                  <Link2 size={12} /> Potentially same person
                </div>
                <div className="space-y-2">
                  {similar.map(s => (
                    <div key={s.identity_id}
                         className="flex items-center gap-3 p-2.5 bg-card-secondary rounded-xl border border-border">
                      <FaceAvatar src={s.thumbnail} label={s.label} size="sm" />
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-semibold text-text truncate">{s.label}</div>
                        <StatusBadge is_confirmed={s.is_confirmed} />
                      </div>
                      <div className="text-xs font-bold text-amber-600 flex-shrink-0">
                        {Math.round(s.similarity * 100)}% similar
                      </div>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-text-secondary mt-2">
                  Consider merging if these are the same person. Use the Merge Suggestions page for bulk review.
                </p>
              </div>
            )}

            {data?.type_counts && Object.keys(data.type_counts).length > 0 && (
              <div>
                <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-2">By type</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(data.type_counts).map(([type, count]) => (
                    <span key={type} className="text-xs px-2.5 py-1 rounded-full text-white font-medium"
                          style={{ backgroundColor: vColor(type) }}>
                      {type} ×{count}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div>
              <div className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-3">Violation timeline</div>
              {!data?.violations?.length ? (
                <p className="text-sm text-text-secondary text-center py-6">No violations recorded.</p>
              ) : (
                <div className="space-y-2">
                  {data.violations.map(v => (
                    <div key={v.id} className="flex items-start gap-3 p-3 bg-card-secondary rounded-xl">
                      {v.image_path && (
                        <img src={`${API}${v.image_path}`} alt="violation"
                             className="w-12 h-12 object-cover rounded-lg cursor-pointer hover:opacity-80 flex-shrink-0"
                             onClick={() => onModalImage(`${API}${v.image_path}`)} />
                      )}
                      <div className="flex-1 min-w-0">
                        <ViolationChip type={v.violation_type} />
                        <div className="text-xs text-text-secondary mt-1">{fmt(v.timestamp)}</div>
                        {v.match_score != null && (
                          <div className="text-xs text-text-secondary">Match: {(v.match_score * 100).toFixed(0)}%</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="text-xs text-text-secondary text-center pt-2 border-t border-border">
              First seen: {fmt(id?.created_at)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ── Merge Suggestions Panel ───────────────────────────────────────────────────
const MergeSuggestionsPage = ({ onCountChange }) => {
  const [data,     setData]     = useState(null);
  const [loading,  setLoading]  = useState(true);
  const [dismissed,setDismissed]= useState(new Set());
  const [merging,  setMerging]  = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    const r = await fetch(`${API}/face/merge-suggestions`);
    const d = await r.json().catch(() => ({ suggestions: [] }));
    setData(d);
    const visible = (d.suggestions || []).filter(s => !dismissed.has(`${s.identity_a_id}:${s.identity_b_id}`));
    onCountChange(visible.length);
    setLoading(false);
  }, [dismissed, onCountChange]);

  useEffect(() => { load(); }, []);

  const handleMerge = async (s) => {
    const key = `${s.identity_a_id}:${s.identity_b_id}`;
    setMerging(key);
    try {
      const r = await fetch(`${API}/face/identity/merge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_id: s.identity_a_id, target_id: s.identity_b_id }),
      });
      if (r.ok) {
        setDismissed(prev => new Set([...prev, key]));
        onCountChange(prev => Math.max(0, prev - 1));
      }
    } finally { setMerging(null); }
  };

  const handleDismiss = (s) => {
    const key = `${s.identity_a_id}:${s.identity_b_id}`;
    setDismissed(prev => new Set([...prev, key]));
    onCountChange(prev => Math.max(0, prev - 1));
  };

  const visible = (data?.suggestions || []).filter(
    s => !dismissed.has(`${s.identity_a_id}:${s.identity_b_id}`)
  );

  const simColor = (sim) => sim >= 0.85 ? '#22c55e' : sim >= 0.75 ? '#f59e0b' : '#6b7280';

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 className="text-2xl font-bold text-text">Merge Suggestions</h2>
          <p className="text-sm text-text-secondary">
            {loading ? 'Analysing…' : `${visible.length} pair${visible.length !== 1 ? 's' : ''} may be the same person`}
          </p>
        </div>
        <button onClick={load} className="p-2 border border-border rounded-lg hover:bg-border text-text-secondary">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="mb-5 p-4 bg-blue-50 border border-blue-200 rounded-xl text-sm text-blue-800">
        <strong>How this works:</strong> The system compares identities using multiple face prototypes.
        Pairs shown here have high embedding similarity and may represent the same physical person.
        Review and merge — or dismiss if incorrect.
      </div>

      {loading ? (
        <div className="flex justify-center py-20"><Loader size={32} className="animate-spin text-text-secondary" /></div>
      ) : visible.length === 0 ? (
        <div className="flex flex-col items-center py-20 text-center bg-card border border-border rounded-xl">
          <Zap size={44} className="text-text-secondary mb-3" />
          <p className="text-base font-semibold text-text">No suggestions right now</p>
          <p className="text-sm text-text-secondary">
            All identities appear distinct, or not enough data has been collected yet.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {visible.map(s => {
            const key = `${s.identity_a_id}:${s.identity_b_id}`;
            const isMerging = merging === key;
            return (
              <div key={key}
                   className="bg-card border border-border rounded-xl p-4 hover:border-primary/30 transition-all">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold" style={{ color: simColor(s.similarity) }}>
                      {Math.round(s.similarity * 100)}% similar
                    </span>
                    <div className="w-24 h-1.5 bg-card-secondary rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{
                        width: `${s.similarity * 100}%`,
                        backgroundColor: simColor(s.similarity),
                      }} />
                    </div>
                  </div>
                  <span className="text-xs text-text-secondary">
                    {s.similarity >= 0.85 ? 'Very likely same person' : s.similarity >= 0.75 ? 'Possibly same person' : 'May be same person'}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                  {[
                    { id: s.identity_a_id, label: s.identity_a_label, thumb: s.thumbnail_a },
                    { id: s.identity_b_id, label: s.identity_b_label, thumb: s.thumbnail_b },
                  ].map((person, i) => (
                    <div key={i} className="flex items-center gap-3 p-3 bg-card-secondary rounded-xl">
                      <FaceAvatar src={person.thumb} label={person.label} />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-semibold text-text truncate">{person.label}</div>
                        <div className="text-xs text-text-secondary">ID: {person.id.slice(0, 8)}</div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleMerge(s)}
                    disabled={isMerging}
                    className="flex items-center gap-1.5 px-4 py-2 bg-primary text-white text-sm font-semibold rounded-lg hover:opacity-90 disabled:opacity-50 transition-all">
                    {isMerging ? <Loader size={13} className="animate-spin" /> : <GitMerge size={13} />}
                    Merge A → B
                  </button>
                  <button
                    onClick={() => handleDismiss(s)}
                    className="flex items-center gap-1.5 px-4 py-2 bg-card-secondary border border-border text-sm text-text-secondary rounded-lg hover:bg-border transition-all">
                    <X size={13} /> Not the same
                  </button>
                  <p className="text-xs text-text-secondary ml-auto">
                    Merge archives "{s.identity_a_label}" into "{s.identity_b_label}"
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ── Identity Management ───────────────────────────────────────────────────────
const IdentityManagementPage = ({ onModalImage }) => {
  const [identities, setIdentities] = useState([]);
  const [total, setTotal]     = useState(0);
  const [pages, setPages]     = useState(1);
  const [page,  setPage]      = useState(1);
  const [search,    setSearch]    = useState('');
  const [confirmed, setConfirmed] = useState('all');
  const [selected,  setSelected]  = useState(null);
  const [loading,   setLoading]   = useState(true);
  const [mergeSource, setMergeSource] = useState(null);
  const [clustering, setClustering]   = useState(false);

  const load = useCallback(async (p = page, s = search, c = confirmed) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ page: p, limit: 24, search: s, confirmed: c, sort: 'last_seen' });
      const r = await fetch(`${API}/face/identities?${params}`);
      const d = await r.json();
      setIdentities(d.identities || []);
      setTotal(d.total || 0);
      setPages(d.pages || 1);
    } catch {}
    finally { setLoading(false); }
  }, [page, search, confirmed]);

  useEffect(() => { load(page, search, confirmed); }, [page, confirmed]);

  useEffect(() => {
    const t = setTimeout(() => { setPage(1); load(1, search, confirmed); }, 400);
    return () => clearTimeout(t);
  }, [search]);

  const handleMerge = async (target) => {
    if (!mergeSource || target.id === mergeSource.id) { setMergeSource(null); return; }
    if (!window.confirm(`Merge "${mergeSource.label}" into "${target.label}"?`)) { setMergeSource(null); return; }
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
      alert(`Consolidation complete:\n• ${d.clusters_found} clusters\n• ${d.identities_merged} merged\n• ${d.embeddings_reassigned} reassigned`);
      load(page, search, confirmed);
    } finally { setClustering(false); }
  };

  return (
    <div className="p-6">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <h2 className="text-2xl font-bold text-text">Identities</h2>
          <p className="text-sm text-text-secondary">{total} detected</p>
        </div>
        <button onClick={runClustering} disabled={clustering}
                className="flex items-center gap-2 px-3 py-2 bg-card border border-border rounded-lg text-sm text-text-secondary hover:bg-border disabled:opacity-50">
          <RefreshCw size={13} className={clustering ? 'animate-spin' : ''} />
          {clustering ? 'Running…' : 'Consolidate'}
        </button>
      </div>

      <div className="flex flex-wrap gap-3 mb-4">
        <div className="relative flex-1 min-w-[180px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary" />
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search…"
                 className="w-full pl-8 pr-3 py-2 text-sm bg-card border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary" />
        </div>
        <div className="flex rounded-lg overflow-hidden border border-border text-sm">
          {[['all', 'All'], ['false', 'Unconfirmed'], ['true', 'Confirmed']].map(([val, lbl]) => (
            <button key={val} onClick={() => { setConfirmed(val); setPage(1); }}
                    className={`px-3 py-2 transition-all ${confirmed === val ? 'bg-primary text-white' : 'bg-card text-text-secondary hover:bg-border'}`}>
              {lbl}
            </button>
          ))}
        </div>
      </div>

      {mergeSource && (
        <div className="mb-4 px-4 py-3 bg-amber-50 border border-amber-300 rounded-xl flex items-center justify-between text-sm">
          <span className="text-amber-800">Click a target identity to merge <strong>"{mergeSource.label}"</strong> into it.</span>
          <button onClick={() => setMergeSource(null)} className="text-amber-600 font-semibold hover:text-amber-800">Cancel</button>
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-20"><Loader size={32} className="animate-spin text-text-secondary" /></div>
      ) : identities.length === 0 ? (
        <div className="flex flex-col items-center py-20 text-center">
          <Users size={44} className="text-text-secondary mb-3" />
          <p className="text-base font-semibold text-text">No identities found</p>
          <p className="text-sm text-text-secondary">Start a stream to begin detection.</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-3">
          {identities.map(identity => (
            <IdentityCard key={identity.id} identity={identity}
              onSelect={setSelected} onMergeStart={setMergeSource}
              merging={!!mergeSource && mergeSource.id !== identity.id}
              onMergeTarget={handleMerge}
              onRefresh={() => load(page, search, confirmed)} />
          ))}
        </div>
      )}

      {pages > 1 && (
        <div className="flex items-center justify-center gap-3 mt-6">
          <button disabled={page === 1} onClick={() => setPage(p => p - 1)}
                  className="p-2 rounded-lg border border-border hover:bg-border disabled:opacity-40"><ChevronLeft size={15} /></button>
          <span className="text-sm text-text-secondary">Page {page} of {pages}</span>
          <button disabled={page === pages} onClick={() => setPage(p => p + 1)}
                  className="p-2 rounded-lg border border-border hover:bg-border disabled:opacity-40"><ChevronRight size={15} /></button>
        </div>
      )}

      {selected && (
        <IdentityDetail identity={selected}
          onClose={() => { setSelected(null); load(page, search, confirmed); }}
          onModalImage={onModalImage}
          onRefresh={() => load(page, search, confirmed)} />
      )}
    </div>
  );
};

// ── Identity Importer ─────────────────────────────────────────────────────────
const ImportPage = () => {
  const [file,      setFile]      = useState(null);
  const [dragging,  setDragging]  = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result,    setResult]    = useState(null);
  const [error,     setError]     = useState('');
  const importFileRef = useRef(null);

  const selectFile = (f) => {
    setResult(null);
    setError('');
    if (!f) return;
    if (!f.name.toLowerCase().endsWith('.zip')) {
      setError('Only .zip files are accepted. Please choose a valid archive.');
      return;
    }
    setFile(f);
  };

  const clearFile = (e) => {
    e?.stopPropagation();
    setFile(null);
    setResult(null);
    setError('');
    if (importFileRef.current) importFileRef.current.value = '';
  };

  const onDragOver  = (e) => { e.preventDefault(); setDragging(true);  };
  const onDragLeave = ()  => setDragging(false);
  const onDrop      = (e) => {
    e.preventDefault();
    setDragging(false);
    selectFile(e.dataTransfer.files?.[0]);
  };

  const handleUpload = async () => {
    if (!file || uploading) return;
    setUploading(true);
    setResult(null);
    setError('');

    const fd = new FormData();
    fd.append('file', file);

    try {
      const r    = await fetch(`${API}/face/import`, { method: 'POST', body: fd });
      const data = await r.json().catch(() => ({}));

      if (r.status === 400 || r.status === 503) {
        setError(data.error || `Server returned ${r.status}.`);
      } else if (r.status === 200 || r.status === 207) {
        setResult({ ...data, partial: r.status === 207 });
      } else {
        setError(data.error || `Unexpected status ${r.status}.`);
      }
    } catch {
      setError('Network error — could not reach the server.');
    } finally {
      setUploading(false);
    }
  };

  const fmtBytes = (b) => {
    if (b < 1024)    return `${b} B`;
    if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
    return `${(b / 1048576).toFixed(1)} MB`;
  };

  return (
    <div className="p-6 max-w-2xl">
      <h2 className="text-2xl font-bold text-text mb-1">Import Identities</h2>
      <p className="text-sm text-text-secondary mb-6">
        Bulk-enroll known individuals by uploading a ZIP archive containing their
        photos and an optional CSV manifest.
      </p>

      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !file && importFileRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && !file && importFileRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-10 text-center transition-all select-none
          ${dragging
            ? 'border-primary bg-primary/5 scale-[1.01]'
            : file
              ? 'border-green-400 bg-green-50/60 cursor-default'
              : 'border-border hover:border-primary/60 hover:bg-card-secondary cursor-pointer'
          }`}
      >
        <input
          ref={importFileRef}
          type="file"
          accept=".zip"
          className="hidden"
          onChange={(e) => selectFile(e.target.files?.[0])}
        />

        {file ? (
          <div className="flex flex-col items-center gap-2">
            <div className="w-12 h-12 rounded-full bg-green-100 flex items-center justify-center">
              <CheckCircle size={24} className="text-green-500" />
            </div>
            <p className="text-sm font-semibold text-text">{file.name}</p>
            <p className="text-xs text-text-secondary">{fmtBytes(file.size)}</p>
            <button
              onClick={clearFile}
              className="mt-1 flex items-center gap-1 text-xs text-red-400 hover:text-red-600 transition-colors"
            >
              <X size={11} /> Remove
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 pointer-events-none">
            <div className={`w-14 h-14 rounded-full flex items-center justify-center transition-colors
                             ${dragging ? 'bg-primary/10' : 'bg-card-secondary'}`}>
              <Upload size={26} className={dragging ? 'text-primary' : 'text-text-secondary'} />
            </div>
            <div>
              <p className="text-sm font-semibold text-text">
                {dragging ? 'Release to select' : 'Drop a ZIP file here, or click to browse'}
              </p>
              <p className="text-xs text-text-secondary mt-1">.zip only · max 100 MB</p>
            </div>
          </div>
        )}
      </div>

      {/* Format guide */}
      <div className="mt-4 bg-card border border-border rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-border bg-card-secondary">
          <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
            Expected ZIP structure
          </span>
        </div>
        <div className="px-4 py-3 font-mono text-xs text-text-secondary space-y-0.5 leading-relaxed">
          <p><span className="text-amber-500">📁</span> images/</p>
          <p className="pl-4">
            <span className="text-blue-400">📁</span> Alice Smith/
            <span className="text-text-secondary/50 font-sans ml-2">← folder name becomes identity label</span>
          </p>
          <p className="pl-8"><span className="text-green-400">🖼</span> photo1.jpg</p>
          <p className="pl-8"><span className="text-green-400">🖼</span> photo2.jpg</p>
          <p className="pl-4"><span className="text-blue-400">📁</span> Bob Jones/</p>
          <p className="pl-8"><span className="text-green-400">🖼</span> headshot.png</p>
          <p className="mt-1">
            <span className="text-purple-400">📄</span> identities.csv
            <span className="text-text-secondary/50 font-sans ml-2">← optional · columns: name, external_id, metadata</span>
          </p>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mt-4 flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
          <XCircle size={16} className="text-red-500 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Result panel */}
      {result && (
        <div className={`mt-4 rounded-xl border overflow-hidden
          ${result.partial ? 'border-amber-200' : 'border-green-200'}`}>
          <div className={`flex items-center gap-2.5 px-4 py-3 border-b
            ${result.partial
              ? 'bg-amber-50 border-amber-200 text-amber-700'
              : 'bg-green-50 border-green-200 text-green-700'
            }`}>
            {result.partial
              ? <AlertTriangle size={16} className="flex-shrink-0" />
              : <CheckCircle   size={16} className="flex-shrink-0" />
            }
            <span className="text-sm font-semibold">
              {result.partial ? 'Partial success — some identities had issues' : 'Import complete'}
            </span>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-3 divide-x divide-border bg-card">
            {[
              { label: 'Planned',  value: result.total_identities, cls: 'text-text' },
              { label: 'Created',  value: result.created,          cls: 'text-green-600' },
              { label: 'Failed',   value: result.failed,
                cls: result.failed > 0 ? 'text-red-500' : 'text-text-secondary' },
            ].map(({ label, value, cls }) => (
              <div key={label} className="flex flex-col items-center py-4 gap-0.5">
                <span className={`text-2xl font-bold ${cls}`}>{value}</span>
                <span className="text-xs text-text-secondary">{label}</span>
              </div>
            ))}
          </div>

          {/* Error list */}
          {result.errors?.length > 0 && (
            <div className="border-t border-border bg-card">
              <div className="px-4 py-2.5 border-b border-border bg-card-secondary flex items-center justify-between">
                <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                  Issues ({result.errors.length})
                </span>
                <span className="text-xs text-text-secondary">
                  These identities were skipped or partially processed
                </span>
              </div>
              <div className="divide-y divide-border max-h-52 overflow-y-auto">
                {result.errors.map((e, i) => (
                  <div key={i} className="flex items-start gap-3 px-4 py-2.5">
                    <XCircle size={13} className="text-red-400 flex-shrink-0 mt-0.5" />
                    <div className="text-xs min-w-0">
                      <span className="font-semibold text-text">{e.identity}</span>
                      <span className="text-text-secondary"> — {e.error}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Upload button */}
      <button
        onClick={handleUpload}
        disabled={!file || uploading}
        className="mt-5 w-full flex items-center justify-center gap-2.5 py-3 bg-primary text-white
                   text-sm font-semibold rounded-xl hover:opacity-90 disabled:opacity-40
                   disabled:cursor-not-allowed transition-all"
      >
        {uploading
          ? <><Loader size={15} className="animate-spin" /> Uploading…</>
          : <><Upload size={15} /> Start Import</>
        }
      </button>

      {uploading && (
        <p className="text-center text-xs text-text-secondary mt-3">
          Generating embeddings — this may take a moment for large archives…
        </p>
      )}
    </div>
  );
};

// ── Live Monitor ──────────────────────────────────────────────────────────────
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
            <div className="relative bg-black rounded-xl overflow-hidden flex-1">
              <img src={zoomed.videoSrc} className="w-full h-full object-contain" alt={zoomed.name} />
            </div>
            <div className="flex items-center justify-between mt-3">
              <span className="font-semibold text-text">{zoomed.name}</span>
              <div className="flex items-center gap-3">
                <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm
                  ${hasViolation ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                  {hasViolation ? <ShieldOff size={15} /> : <ShieldCheck size={15} />}
                  {hasViolation ? 'Violation' : 'All clear'}
                </div>
                <button onClick={() => setZoomedId(null)} className="p-2 hover:bg-border rounded-lg text-text-secondary">
                  <Minimize size={17} />
                </button>
              </div>
            </div>
          </div>
        </div>
        {others.length > 0 && (
          <aside className="w-60 flex-shrink-0 pt-16 overflow-y-auto space-y-3">
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
                  className="flex items-center gap-2 px-3 py-2 bg-primary text-white text-sm rounded-lg hover:opacity-90 disabled:opacity-50">
            <Camera size={14} /> Webcam
          </button>
          <button onClick={() => fileInputRef.current?.click()}
                  disabled={serverStatus !== 'connected' || isLoading}
                  className="flex items-center gap-2 px-3 py-2 bg-primary text-white text-sm rounded-lg hover:opacity-90 disabled:opacity-50">
            <Upload size={14} /> Upload
          </button>
        </div>
      </div>

      {Object.keys(activeStreams).length === 0 ? (
        <div className="flex flex-col items-center justify-center text-center bg-card border border-border rounded-xl p-16 h-72">
          {isLoading ? <Loader size={36} className="animate-spin text-text-secondary mb-3" /> : <Tv size={44} className="text-text-secondary mb-3" />}
          <p className="text-base font-semibold text-text">{isLoading ? 'Starting…' : 'No active streams'}</p>
          <p className="text-sm text-text-secondary">Use the buttons above to begin.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {Object.values(activeStreams).map(stream => {
            const hasViolation = stream.detections?.some(d => !d.safe);
            return (
              <div key={stream.streamId} className="bg-card border border-border rounded-xl p-4">
                <div className="relative bg-black rounded-xl aspect-video overflow-hidden mb-3 group">
                  <img src={stream.videoSrc} className="w-full h-full object-contain" alt={stream.name} />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-all flex items-center justify-center">
                    <button onClick={() => setZoomedId(stream.streamId)}
                            className="p-3 bg-white/20 text-white rounded-full hover:bg-white/40 backdrop-blur-sm">
                      <Maximize size={20} />
                    </button>
                  </div>
                </div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-text">{stream.name}</span>
                  <button onClick={() => stopStream(stream.streamId)} className="p-1.5 text-red-500 hover:bg-border rounded">
                    <StopCircle size={17} />
                  </button>
                </div>
                <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm mb-2
                  ${hasViolation ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                  {hasViolation ? <ShieldOff size={14} /> : <ShieldCheck size={14} />}
                  {hasViolation ? 'Safety violation' : 'All clear'}
                </div>
                <div className="flex justify-between text-xs text-text-secondary">
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
    <div className="p-6 pt-12 lg:pt-6 max-w-3xl">
      <h2 className="text-2xl font-bold text-text mb-5">Settings</h2>
      <div className="bg-card border border-border rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-text">RTSP Streams</h3>
          <button onClick={add} className="flex items-center gap-2 px-3 py-1.5 bg-primary text-white text-sm rounded-lg hover:opacity-90">
            <Plus size={13} /> Add
          </button>
        </div>
        {rtspStreams.length === 0 ? (
          <p className="text-sm text-text-secondary text-center py-8">No RTSP streams configured.</p>
        ) : (
          <div className="space-y-3">
            {rtspStreams.map(s => (
              <div key={s.id} className="flex items-center gap-3">
                <input value={s.name} onChange={e => update(s.id, 'name', e.target.value)} placeholder="Name"
                       className="w-1/3 px-3 py-2 text-sm bg-background border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary" />
                <input value={s.url} onChange={e => update(s.id, 'url', e.target.value)} placeholder="rtsp://…"
                       className="flex-1 px-3 py-2 text-sm bg-background border border-border rounded-lg text-text focus:outline-none focus:ring-2 focus:ring-primary" />
                <button onClick={() => startStream('rtsp', s.url, s.name)} disabled={!s.url || serverStatus !== 'connected'}
                        className="px-3 py-2 bg-green-600 text-white rounded-lg hover:opacity-90 disabled:opacity-40">
                  <Play size={13} />
                </button>
                <button onClick={() => remove(s.id)} className="p-2 text-red-500 hover:bg-border rounded-lg"><Trash2 size={14} /></button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// ── App root ──────────────────────────────────────────────────────────────────
export default function App() {
  const { isAuthenticated, user, logout } = useAuth();
  const [view, setView]           = useState('dashboard');
  const [serverStatus, setServerStatus] = useState('checking');
  const [sidebarOpen, setSidebarOpen]   = useState(true);
  const [rtspStreams, setRtspStreams]    = useState([]);
  const [isLoading, setIsLoading]       = useState(false);
  const [activeStreams, setActiveStreams]= useState({});
  const [modalImage, setModalImage]     = useState(null);
  const [zoomedId, setZoomedId]         = useState(null);
  const [reviewCount, setReviewCount]       = useState(0);
  const [suggestionsCount, setSuggestionsCount] = useState(0);
  const fileInputRef = useRef(null);

  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API}/health`);
      const d = await r.json();
      setServerStatus(r.ok && d.model_loaded ? 'connected' : 'degraded');
    } catch { setServerStatus('disconnected'); }
  }, []);

  useEffect(() => { checkHealth(); const i = setInterval(checkHealth, 10000); return () => clearInterval(i); }, [checkHealth]);

  useEffect(() => {
    fetch(`${API}/face/review-queue`)
      .then(r => r.json())
      .then(d => setReviewCount(Array.isArray(d) ? d.length : 0))
      .catch(() => {});
    fetch(`${API}/face/merge-suggestions`)
      .then(r => r.json())
      .then(d => setSuggestionsCount((d.suggestions || []).length))
      .catch(() => {});
  }, []);

  const startStream = useCallback(async (source_type, source_path, name) => {
    setIsLoading(true);
    try {
      const r = await fetch(`${API}/stream/start`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
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
    } catch (e) { alert(`Failed: ${e.message}`); }
    finally { setIsLoading(false); }
  }, []);

  const stopStream = useCallback(async (streamId) => {
    if (zoomedId === streamId) setZoomedId(null);
    await fetch(`${API}/stream/stop`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stream_id: streamId }),
    }).catch(() => {});
    setActiveStreams(prev => { const n = { ...prev }; delete n[streamId]; return n; });
  }, [zoomedId]);

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
    const file = e.target.files?.[0]; if (!file) return;
    setIsLoading(true);
    const fd = new FormData(); fd.append('video', file);
    try {
      const r = await fetch(`${API}/upload/video`, { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || 'Upload failed');
      startStream('video', d.path, file.name);
    } catch (e) { alert(`Upload failed: ${e.message}`); }
    finally { e.target.value = ''; setIsLoading(false); }
  };

  const renderPage = () => {
    switch (view) {
      case 'dashboard':   return <DashboardPage setView={setView} />;
      case 'identities':  return <IdentityManagementPage onModalImage={setModalImage} />;
      case 'import':      return <ImportPage />;
      case 'review':      return <ReviewQueuePage onCountChange={setReviewCount} onModalImage={setModalImage} />;
      case 'suggestions': return <MergeSuggestionsPage onCountChange={setSuggestionsCount} />;
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

  if (!isAuthenticated) return <LoginPage />;

  return (
    <div className="min-h-screen bg-background text-text font-sans flex">
      <ImageModal imageUrl={modalImage} onClose={() => setModalImage(null)} />
      <Sidebar view={view} setView={setView} open={sidebarOpen} setOpen={setSidebarOpen}
               serverStatus={serverStatus} reviewCount={reviewCount} suggestionsCount={suggestionsCount}
               user={user} onLogout={logout} />
      <main className={`flex-1 h-screen overflow-y-auto transition-all duration-300 pt-16 lg:pt-0 ${sidebarOpen ? 'ml-56 lg:ml-56' : 'ml-16 lg:ml-16'}`}>
        {renderPage()}
      </main>
      <input ref={fileInputRef} type="file" accept="video/*" onChange={handleVideoUpload} className="hidden" />
    </div>
  );
}