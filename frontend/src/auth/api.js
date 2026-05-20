// Patches global fetch to send cookies and CSRF headers on every request.
// Import once (in index.js) as a side-effect — no other files need changing.

const _originalFetch = window.fetch.bind(window);

function getCookie(name) {
  const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
  return match ? match[2] : null;
}

window.fetch = async (url, options = {}) => {
  // Always include credentials so httpOnly cookies are sent
  options.credentials = options.credentials || 'include';

  const isAuthRoute = typeof url === 'string' && url.includes('/api/v1/auth/');
  const isSafeMethod = !options.method || options.method === 'GET' || options.method === 'HEAD';

  // Add CSRF header for state-changing requests to non-auth routes
  if (!isAuthRoute && !isSafeMethod) {
    const csrfToken = getCookie('csrf_access_token');
    if (csrfToken) {
      options.headers = {
        'X-CSRF-Token': csrfToken,
        ...options.headers,
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
