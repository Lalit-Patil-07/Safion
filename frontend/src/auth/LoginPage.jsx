import React, { useState } from 'react';
import { Shield, Loader } from 'lucide-react';
import { useAuth } from './AuthContext';
import API from '../config';


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

export default function LoginPage() {
  const { login } = useAuth();
  const [identifier, setIdentifier] = useState('');
  const [password, setPassword]     = useState('');
  const [error, setError]           = useState('');
  const [loading, setLoading]       = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/v1/auth/login`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ username: identifier, email: identifier, password }),
      });
      const data = await r.json();
      if (!r.ok) { setError(data.error || 'Login failed.'); return; }
      login(data);
    } catch {
      setError('Unable to reach server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="bg-card border border-border rounded-xl p-8 w-full max-w-sm shadow-2xl">
        <div className="flex flex-col items-center mb-8">
          <SafionLogo width="64" height="82" className="mb-4" />
          <span className="font-sans font-medium uppercase text-text" style={{ fontSize: '18px', letterSpacing: '0.42em' }}>SAFION</span>
          <p className="text-xs text-text-secondary mt-2">Safety monitoring system</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1.5">
              Username or email
            </label>
            <input
              type="text"
              value={identifier}
              onChange={e => setIdentifier(e.target.value)}
              required
              autoFocus
              autoComplete="username"
              className="w-full px-3 py-2.5 text-sm bg-background border border-border rounded-lg text-text
                         focus:outline-none focus:ring-2 focus:ring-primary placeholder:text-text-secondary/50"
              placeholder="admin"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1.5">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
              autoComplete="current-password"
              className="w-full px-3 py-2.5 text-sm bg-background border border-border rounded-lg text-text
                         focus:outline-none focus:ring-2 focus:ring-primary placeholder:text-text-secondary/50"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 bg-primary text-white text-sm font-semibold rounded-lg
                       hover:opacity-90 disabled:opacity-50 flex items-center justify-center gap-2 transition-all mt-2"
          >
            {loading && <Loader size={14} className="animate-spin" />}
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  );
}