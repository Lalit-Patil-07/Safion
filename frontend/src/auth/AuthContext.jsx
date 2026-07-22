import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import API from '../config';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch current user on mount (JWT cookie is auto-sent via the api.js
  // fetch patch).  No localStorage — eliminates XSS-based token side-
  // channel leakage of the full user object.
  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/api/v1/auth/me`, { credentials: 'include' })
      .then(r => (r.ok ? r.json() : null))
      .then(data => { if (!cancelled) setUser(data); })
      .catch(() => { if (!cancelled) setUser(null); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const isAuthenticated = !!user && !loading;

  const login = useCallback((data) => {
    setUser(data.user);
  }, []);

  const logout = useCallback(() => {
    fetch(`${API}/api/v1/auth/logout`, { method: 'POST', credentials: 'include' })
      .catch(() => {});
    setUser(null);
  }, []);

  useEffect(() => {
    window.addEventListener('auth:logout', logout);
    return () => window.removeEventListener('auth:logout', logout);
  }, [logout]);

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}