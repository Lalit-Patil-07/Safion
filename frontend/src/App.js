import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Upload, Video, AlertCircle, CheckCircle, XCircle, Settings, Play, StopCircle, Loader, Shield, ShieldOff, ShieldCheck, BookCopy, X, Menu, Tv, Plus, Trash2, Maximize, Minimize, UserPlus, Users } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';

// --- Constants ---
const API_BASE_URL = 'http://localhost:5000';

const PPE_CLASSES = {
    0: { name: "Hardhat", color: "#3B82F6", safe: true },
    1: { name: "Mask", color: "#10B981", safe: true },
    2: { name: "NO-Hardhat", color: "#F56565", safe: false },
    3: { name: "NO-Mask", color: "#F59E0B", safe: false },
    4: { name: "NO-Safety Vest", color: "#EC4899", safe: false },
    5: { name: "Person", color: "#FBBF24", safe: true },
    6: { name: "Safety Cone", color: "#8B5CF6", safe: true },
    7: { name: "Safety Vest", color: "#059669", safe: true },
    8: { name: "Machinery", color: "#6366F1", safe: true },
    9: { name: "Vehicle", color: "#14B8A6", safe: true }
};

// --- Helper & UI Components ---

const ImageModal = ({ imageUrl, onClose }) => {
    if (!imageUrl) return null;
    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50" onClick={onClose}>
            <div className="relative max-w-4xl max-h-[90vh] p-4" onClick={e => e.stopPropagation()}>
                <img src={imageUrl} alt="Violation Evidence" className="w-full h-full object-contain rounded-lg" />
                <button onClick={onClose} className="absolute -top-2 -right-2 bg-card rounded-full p-2 text-text hover:bg-border transition-colors">
                    <X size={24} />
                </button>
            </div>
        </div>
    );
};

// --- Navigation ---
const Sidebar = ({ view, setView }) => {
    const navItems = [
        { id: 'live', icon: Tv, label: 'Live Detection' },
        { id: 'violations', icon: BookCopy, label: 'Violations Log' },
        { id: 'identity', icon: UserPlus, label: 'Identity Recognition' },
        { id: 'settings', icon: Settings, label: 'Settings' },
    ];

    return (
        <aside className="w-64 bg-card-secondary p-4 border-r border-border flex flex-col">
            <h1 className="text-2xl font-bold text-text mb-8">Safion</h1>
            <nav className="flex flex-col gap-2">
                {navItems.map(item => (
                    <button
                        key={item.id}
                        onClick={() => setView(item.id)}
                        className={`flex items-center gap-3 px-4 py-3 rounded-md font-semibold text-sm transition-colors ${view === item.id ? 'bg-primary text-white' : 'text-text-secondary hover:bg-border'}`}
                    >
                        <item.icon size={20} />
                        <span>{item.label}</span>
                    </button>
                ))}
            </nav>
        </aside>
    );
}

// --- Page Components ---

const ViolationLog = ({ violations, onImageClick, clearViolations }) => {
    const formatTimestamp = (isoString) => {
        const date = new Date(isoString);
        return date.toLocaleString();
    };

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-3xl font-bold text-text">Violation Log</h2>
                <button
                    onClick={clearViolations}
                    className="flex items-center gap-2 px-4 py-2 bg-accent-red text-white hover:opacity-90 rounded-md font-semibold text-sm transition-all"
                >
                    <Trash2 size={16} /> Clear Violations
                </button>
            </div>
            <div className="bg-card rounded-lg p-4 shadow-lg border border-border">
                <div className="space-y-4 max-h-[80vh] overflow-y-auto">
                    {violations.length === 0 ? (
                        <p className="text-text-secondary text-center py-12">No violations recorded yet.</p>
                    ) : (
                        violations.map(v => (
                            <div key={v.id} className="flex items-start gap-4 p-3 bg-card-secondary border border-border rounded-lg">
                                <img
                                    src={`${API_BASE_URL}${v.image_path}?t=${new Date(v.timestamp).getTime()}`}
                                    alt={`Violation by ${v.name}`}
                                    className="w-24 h-24 object-cover rounded-md cursor-pointer hover:opacity-80 transition-opacity"
                                    onClick={() => onImageClick(`${API_BASE_URL}${v.image_path}`)}
                                />
                                <div className="flex-1">
                                    <p className="font-bold text-text">{v.name}</p>
                                    <p className="text-sm text-accent-red font-semibold">{v.violation_type}</p>
                                    <p className="text-xs text-text-secondary mt-1">{formatTimestamp(v.timestamp)}</p>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

const IdentityRecognitionPage = () => {
    const [unknowns, setUnknowns] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [selectedIds, setSelectedIds] = useState(new Set());

    const fetchUnknowns = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/violators/unknown`);
            const data = await response.json();
            setUnknowns(data);
        } catch (error) {
            console.error("Failed to fetch unknown violators:", error);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchUnknowns();
    }, [fetchUnknowns]);

    const handleMerge = async () => {
        if (selectedIds.size < 2) {
            alert("Please select at least two images to merge.");
            return;
        }
        const name = prompt("Enter a name for the merged identity:");
        if (name && name.trim()) {
            try {
                await fetch(`${API_BASE_URL}/face/merge`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, violation_ids: Array.from(selectedIds) }),
                });
                setSelectedIds(new Set());
                fetchUnknowns(); // Refresh
            } catch (error) {
                console.error("Failed to merge faces:", error);
            }
        }
    };

    const toggleSelection = (id) => {
        setSelectedIds(prev => {
            const newSelection = new Set(prev);
            if (newSelection.has(id)) {
                newSelection.delete(id);
            } else {
                newSelection.add(id);
            }
            return newSelection;
        });
    };

    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-3xl font-bold text-text">Identity Recognition</h2>
                <button
                    onClick={handleMerge}
                    disabled={selectedIds.size < 2}
                    className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md font-semibold text-sm disabled:bg-border disabled:cursor-not-allowed"
                >
                    <Users size={16} /> Merge {selectedIds.size} Selected
                </button>
            </div>
            <p className="text-text-secondary mb-6 max-w-4xl">
                Select multiple images of the same unidentified person below. Once selected, click 'Merge Selected' to assign a single name to them.
            </p>
            {isLoading ? (
                <div className="flex justify-center items-center h-96"><Loader size={48} className="text-text-secondary animate-spin" /></div>
            ) : unknowns.length > 0 ? (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                    {unknowns.map(v => (
                         <div key={v.id} className={`relative rounded-lg overflow-hidden cursor-pointer border-4 ${selectedIds.has(v.id) ? 'border-primary' : 'border-transparent'}`} onClick={() => toggleSelection(v.id)}>
                             <img src={`${API_BASE_URL}${v.image_path}`} alt="violator" className="w-full h-40 object-cover" />
                             {selectedIds.has(v.id) && (
                                 <div className="absolute inset-0 bg-primary/70 flex items-center justify-center">
                                     <CheckCircle size={40} className="text-white" />
                                 </div>
                             )}
                         </div>
                    ))}
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center text-center bg-card rounded-lg p-12 shadow-lg border border-border h-96">
                    <UserPlus size={64} className="mx-auto mb-4 text-text-secondary" />
                    <p className="text-lg font-semibold text-text">No Unknown Violators Found</p>
                    <p className="text-text-secondary">The system has not detected any unidentified individuals.</p>
                </div>
            )}
        </div>
    );
};


const SettingsPage = ({ rtspStreams, setRtspStreams, startStream, serverStatus }) => {

    const addStream = () => {
        setRtspStreams(prev => [...prev, { id: uuidv4(), name: `Stream ${prev.length + 1}`, url: '' }]);
    };

    const removeStream = (id) => {
        setRtspStreams(prev => prev.filter(s => s.id !== id));
    };

    const updateStream = (id, field, value) => {
        setRtspStreams(prev => prev.map(s => s.id === id ? { ...s, [field]: value } : s));
    };

    return (
        <div className="p-6">
            <h2 className="text-3xl font-bold text-text mb-6">Settings</h2>
            <div className="max-w-4xl">
                <div className="bg-card rounded-lg p-6 shadow-lg border border-border">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-xl font-semibold text-text">RTSP Configuration</h3>
                        <button onClick={addStream} className="flex items-center gap-2 px-4 py-2 bg-primary text-white hover:opacity-90 rounded-md font-semibold text-sm transition-all">
                            <Plus size={16} /> Add Stream
                        </button>
                    </div>
                    <div className="space-y-4">
                        {rtspStreams.map((stream) => (
                            <div key={stream.id} className="flex items-center gap-4 p-3 bg-card-secondary border border-border rounded-lg">
                                <input
                                    type="text"
                                    value={stream.name}
                                    onChange={(e) => updateStream(stream.id, 'name', e.target.value)}
                                    placeholder="Stream Name (e.g., Entrance Cam)"
                                    className="w-1/3 px-3 py-2 bg-background rounded-md border border-border focus:outline-none focus:ring-2 focus:ring-primary text-text"
                                />
                                <input
                                    type="text"
                                    value={stream.url}
                                    onChange={(e) => updateStream(stream.id, 'url', e.target.value)}
                                    placeholder="rtsp://..."
                                    className="flex-1 px-3 py-2 bg-background rounded-md border border-border focus:outline-none focus:ring-2 focus:ring-primary text-text"
                                />
                                <button
                                    onClick={() => startStream('rtsp', stream.url, stream.name)}
                                    disabled={!stream.url || serverStatus !== 'connected'}
                                    className="px-4 py-2 bg-green-600 text-white hover:opacity-90 rounded-md font-semibold transition-all disabled:bg-border disabled:cursor-not-allowed flex items-center justify-center gap-2"
                                >
                                    <Play size={16} />
                                </button>
                                <button onClick={() => removeStream(stream.id)} className="p-2 text-accent-red hover:bg-border rounded-md">
                                    <Trash2 size={18} />
                                </button>
                            </div>
                        ))}
                        {rtspStreams.length === 0 && (
                            <p className="text-text-secondary text-center py-8">No RTSP streams configured. Click 'Add Stream' to start.</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};


const LiveDetectionPage = ({
    activeStreams,
    stopStream,
    startStream,
    serverStatus,
    handleVideoUpload,
    fileInputRef,
    isLoading,
    zoomedStreamId,
    setZoomedStreamId
}) => {
    const zoomedStream = zoomedStreamId ? activeStreams[zoomedStreamId] : null;

    // Render the "theater mode" if a stream is zoomed
    if (zoomedStream) {
        const otherStreams = Object.values(activeStreams).filter(s => s.streamId !== zoomedStreamId);
        const hasViolation = zoomedStream.detections && zoomedStream.detections.some(d => !d.safe);
        return (
            <div className="p-6 h-full flex flex-col md:flex-row gap-6">
                <div className="flex-1 flex flex-col min-h-0">
                    <h2 className="text-3xl font-bold text-text mb-6 flex-shrink-0">Live Detection</h2>
                    <div className="flex-1 bg-card rounded-lg p-4 shadow-lg border border-border flex flex-col min-h-0">
                        <div className="relative bg-black rounded-md overflow-hidden flex-1">
                            <img src={zoomedStream.videoSrc} className="w-full h-full object-contain" alt={`Stream for ${zoomedStream.name}`} />
                        </div>
                        <div className="flex justify-between items-center mt-4 flex-shrink-0">
                            <h3 className="text-xl font-semibold text-text">{zoomedStream.name}</h3>
                            <div className="flex items-center gap-4">
                               <div className={`flex items-center gap-2 p-2 rounded-md ${hasViolation ? 'bg-accent-red/20' : 'bg-accent-green/20'}`}>
                                    {hasViolation ? (<ShieldOff className="text-accent-red" size={20} />) : (<ShieldCheck className="text-accent-green" size={20} />)}
                                </div>
                                <button onClick={() => setZoomedStreamId(null)} className="p-2 text-text-secondary hover:bg-border rounded-md">
                                    <Minimize size={22} />
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <aside className="w-full md:w-72 flex flex-col flex-shrink-0">
                    <h3 className="text-xl font-semibold text-text mb-4 mt-0 md:mt-16">Other Streams</h3>
                    <div className="space-y-4 overflow-y-auto">
                        {otherStreams.map(stream => (
                            <div key={stream.streamId} className="bg-card rounded-lg p-2 shadow-md border border-border cursor-pointer hover:border-primary transition-all" onClick={() => setZoomedStreamId(stream.streamId)}>
                                <div className="relative bg-black rounded-md overflow-hidden aspect-video mb-2">
                                    <img src={stream.videoSrc} className="w-full h-full object-cover" alt={`Thumbnail for ${stream.name}`} />
                                </div>
                                <p className="text-sm font-semibold text-text truncate">{stream.name}</p>
                            </div>
                        ))}
                         {otherStreams.length === 0 && <p className="text-sm text-text-secondary text-center py-4">No other active streams.</p>}
                    </div>
                </aside>
            </div>
        );
    }

    // Render the default grid view
    return (
        <div className="p-6">
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-3xl font-bold text-text">Live Detection</h2>
                <div className="flex gap-4">
                    <button onClick={() => startStream('webcam', '0', 'Webcam')} disabled={serverStatus !== 'connected' || isLoading} className="flex items-center gap-2 px-4 py-2 bg-primary text-white hover:opacity-90 rounded-md font-semibold text-sm transition-all disabled:bg-border disabled:cursor-not-allowed">
                        <Camera size={18} /> Start Webcam
                    </button>
                    <button onClick={() => fileInputRef.current?.click()} disabled={serverStatus !== 'connected' || isLoading} className="flex items-center gap-2 px-4 py-2 bg-primary text-white hover:opacity-90 rounded-md font-semibold text-sm transition-all disabled:bg-border disabled:cursor-not-allowed">
                        <Upload size={18} /> Upload Video
                    </button>
                </div>
            </div>

            {Object.keys(activeStreams).length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                    {Object.values(activeStreams).map(stream => {
                        const hasViolation = stream.detections && stream.detections.some(d => !d.safe);
                        return (
                            <div key={stream.streamId} className="bg-card rounded-lg p-4 shadow-lg border border-border">
                                <div className="relative bg-black rounded-md overflow-hidden aspect-video mb-4 group">
                                    <img src={stream.videoSrc} className="w-full h-full object-contain" alt={`Stream for ${stream.name}`} />
                                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                        <button onClick={() => setZoomedStreamId(stream.streamId)} className="p-3 bg-white/20 text-white rounded-full hover:bg-white/40 transition-colors backdrop-blur-sm">
                                            <Maximize size={24} />
                                        </button>
                                    </div>
                                </div>
                                <div className="flex justify-between items-center mb-3">
                                    <h3 className="text-lg font-semibold text-text">{stream.name}</h3>
                                    <button onClick={() => stopStream(stream.streamId)} className="p-2 text-accent-red hover:bg-border rounded-md">
                                        <StopCircle size={22} />
                                    </button>
                                </div>
                                <div className={`flex items-center gap-3 p-3 rounded-md mb-3 ${hasViolation ? 'bg-accent-red/20 border border-accent-red/50' : 'bg-accent-green/20 border border-accent-green/50'}`}>
                                    {hasViolation ? (<><ShieldOff className="text-accent-red" size={20} /> <span className="font-semibold text-accent-red">Safety Violation</span></>) :
                                        (<><ShieldCheck className="text-accent-green" size={20} /> <span className="font-semibold text-accent-green">All Clear</span></>)
                                    }
                                </div>
                                <div className="space-y-2 text-sm text-text-secondary">
                                    <div className="flex justify-between"><span>Detection FPS:</span><span className="font-semibold text-text">{(stream.stats && stream.stats.fps) ? stream.stats.fps.toFixed(1) : '0.0'}</span></div>
                                    <div className="flex justify-between"><span>Violations:</span><span className="font-semibold text-accent-red">{stream.stats ? stream.stats.violation_count : 0}</span></div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center text-center bg-card rounded-lg p-12 shadow-lg border border-border h-96">
                    {isLoading ? <Loader size={48} className="mx-auto mb-4 text-text-secondary animate-spin" /> : <Video size={64} className="mx-auto mb-4 text-text-secondary" />}
                    <p className="text-lg font-semibold text-text">{isLoading ? 'Starting stream...' : 'No active streams'}</p>
                    <p className="text-text-secondary">Start a stream from the controls above or the Settings page.</p>
                </div>
            )}
        </div>
    );
};


// --- Main App Component ---
function App() {
    const [view, setView] = useState('live'); // 'live', 'violations', 'settings'
    const [serverStatus, setServerStatus] = useState('checking');
    const [rtspStreams, setRtspStreams] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [activeStreams, setActiveStreams] = useState({});
    const [violations, setViolations] = useState([]);
    const [modalImageUrl, setModalImageUrl] = useState(null);
    const [zoomedStreamId, setZoomedStreamId] = useState(null);

    const fileInputRef = useRef(null);

    const checkServerHealth = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Server error');
            setServerStatus(data.model_loaded ? 'connected' : 'degraded');
        } catch (error) {
            setServerStatus('disconnected');
        }
    }, []);

    const fetchViolations = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/violations`);
            if (!response.ok) throw new Error('Failed to fetch violations');
            const data = await response.json();
            setViolations(data);
        } catch (error) {
            console.error('❌ Violation fetch failed:', error);
        }
    }, []);

    const clearViolations = async () => {
        if (window.confirm('Are you sure you want to clear all violations? This action cannot be undone.')) {
            try {
                await fetch(`${API_BASE_URL}/violations/clear`, { method: 'POST' });
                fetchViolations();
            } catch (error) {
                console.error('Failed to clear violations:', error);
            }
        }
    };

    useEffect(() => {
        checkServerHealth();
        const interval = setInterval(checkServerHealth, 10000);
        return () => clearInterval(interval);
    }, [checkServerHealth]);

    const stopStream = useCallback(async (streamId) => {
        if (zoomedStreamId === streamId) {
            setZoomedStreamId(null);
        }
        if (!streamId || !activeStreams[streamId]) return;
        try {
            await fetch(`${API_BASE_URL}/stream/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stream_id: streamId }),
            });
        } catch (error) {
            console.error('❌ Error stopping stream:', error);
        } finally {
            setActiveStreams(prev => {
                const newStreams = { ...prev };
                delete newStreams[streamId];
                return newStreams;
            });
        }
    }, [activeStreams, zoomedStreamId]);

    const startStream = useCallback(async (source_type, source_path, name) => {
        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/stream/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source_type, source_path, name }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to start stream');

            setActiveStreams(prev => ({
                ...prev,
                [data.stream_id]: {
                    streamId: data.stream_id,
                    name: data.name,
                    videoSrc: `${API_BASE_URL}/stream/video_feed/${data.stream_id}?t=${Date.now()}`,
                    stats: { fps: 0, violation_count: 0 },
                    detections: [],
                }
            }));
            setView('live');
        } catch (error) {
            console.error('❌ Error starting stream:', error);
            alert(`Failed to start stream: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        let violationInterval;
        if (view === 'violations') {
            fetchViolations();
            violationInterval = setInterval(fetchViolations, 5000);
        }
        return () => {
            if (violationInterval) clearInterval(violationInterval);
        };
    }, [view, fetchViolations]);

    useEffect(() => {
        if (Object.keys(activeStreams).length === 0) return;

        const interval = setInterval(async () => {
            for (const streamId of Object.keys(activeStreams)) {
                try {
                    const response = await fetch(`${API_BASE_URL}/stream/detections/${streamId}`);
                    if (!response.ok) {
                        if (response.status === 404) {
                            stopStream(streamId);
                        }
                        continue;
                    }
                    const data = await response.json();
                    setActiveStreams(prev => {
                       if (!prev[streamId]) return prev;
                       return {
                         ...prev,
                         [streamId]: {
                            ...prev[streamId],
                            stats: data,
                            detections: data.last_detections,
                         }
                       }
                    });
                } catch (error) {
                    // console.error(`Stats poll for ${streamId} failed:`, error.message);
                }
            }
        }, 1000);
        return () => clearInterval(interval);
    }, [activeStreams, stopStream]);

    const handleVideoUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setIsLoading(true);
        const formData = new FormData();
        formData.append('video', file);
        try {
            const uploadResponse = await fetch(`${API_BASE_URL}/upload/video`, {
                method: 'POST',
                body: formData,
            });
            const uploadData = await uploadResponse.json();
            if (!uploadResponse.ok) throw new Error(uploadData.error || 'Upload failed');
            startStream('video', uploadData.path, file.name);
        } catch (error) {
            alert(`Video upload failed: ${error.message}`);
        } finally {
            e.target.value = '';
            setIsLoading(false);
        }
    };

    const renderView = () => {
        switch (view) {
            case 'live':
                return <LiveDetectionPage
                    activeStreams={activeStreams}
                    isLoading={isLoading}
                    stopStream={stopStream}
                    startStream={startStream}
                    serverStatus={serverStatus}
                    handleVideoUpload={handleVideoUpload}
                    fileInputRef={fileInputRef}
                    zoomedStreamId={zoomedStreamId}
                    setZoomedStreamId={setZoomedStreamId}
                />;
            case 'violations':
                return <ViolationLog violations={violations} onImageClick={setModalImageUrl} clearViolations={clearViolations} />;
            case 'identity':
                return <IdentityRecognitionPage />;
            case 'settings':
                return <SettingsPage
                    rtspStreams={rtspStreams}
                    setRtspStreams={setRtspStreams}
                    startStream={startStream}
                    serverStatus={serverStatus}
                />;
            default:
                return null;
        }
    }


    return (
        <div className="min-h-screen bg-background text-text font-sans flex">
            <ImageModal imageUrl={modalImageUrl} onClose={() => setModalImageUrl(null)} />
            <Sidebar view={view} setView={setView} />

            <main className="flex-1 h-screen overflow-y-auto">
                <header className="bg-card p-4 border-b border-border shadow-sm sticky top-0 z-10 flex justify-end items-center">
                    <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${serverStatus === 'connected' ? 'bg-accent-green' : serverStatus === 'degraded' ? 'bg-accent-yellow' : 'bg-accent-red'}`}></div>
                        <span className="text-sm text-text-secondary capitalize">{serverStatus}</span>
                    </div>
                </header>
                {renderView()}
            </main>
            <input ref={fileInputRef} type="file" accept="video/*" onChange={handleVideoUpload} className="hidden" />
        </div>
    );
}

export default App;