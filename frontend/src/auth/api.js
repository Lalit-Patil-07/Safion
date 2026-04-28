// Patches global fetch to attach Authorization header on every non-auth request.
// Import once (in index.js) as a side-effect — no other files need changing.

const _originalFetch = window.fetch.bind(window);

window.fetch = async (url, options = {}) => {
  const isAuthRoute = typeof url === 'string' && url.includes('/api/v1/auth/');

  if (!isAuthRoute) {
    const token = localStorage.getItem('access_token');
    if (token) {
      options = {
        ...options,
        headers: { Authorization: `Bearer ${token}`, ...options.headers },
      };
    }
  }

  const response = await _originalFetch(url, options);

  if (response.status === 401 && !isAuthRoute) {
    window.dispatchEvent(new Event('auth:logout'));
  }

  return response;
};

export const apiFetch = window.fetch;