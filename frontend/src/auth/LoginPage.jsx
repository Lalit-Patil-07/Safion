import React, { useState } from 'react';
import { Shield, Loader } from 'lucide-react';
import { useAuth } from './AuthContext';

const API = '';

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
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 rounded-lg" style={{ backgroundColor: '#5a51a218' }}>
            <Shield size={22} className="text-primary" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-text leading-none">Safion</h1>
            <p className="text-xs text-text-secondary mt-0.5">Safety monitoring system</p>
          </div>
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