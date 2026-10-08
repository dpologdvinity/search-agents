// Client for the search-agents backend.
//
// The API defaults to the page's own origin. A page hosted elsewhere sets
// window.SEARCH_AGENTS_API = "https://<backend>" before loading this module.

export const API_BASE = (window.SEARCH_AGENTS_API || '').replace(/\/$/, '');

export async function getJSON(path, params = {}) {
  const url = new URL(API_BASE + path, location.href);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

export async function postJSON(path, body) {
  const res = await fetch(new URL(API_BASE + path, location.href), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join('; ') : data.detail;
    throw new Error(detail || `${res.status}`);
  }
  return data;
}

function wsURL(path) {
  const url = new URL(API_BASE + path, location.href);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

// One WebSocket, reused across solves. solve() sends a request and resolves
// with the final "result" message; onMessage sees every message on the way.
export class SearchSocket {
  constructor(path) {
    this.path = path;
    this.ws = null;
    this.pending = null; // {resolve, reject, onMessage}
  }

  async _open() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    await new Promise((resolve, reject) => {
      const ws = new WebSocket(wsURL(this.path));
      ws.onopen = () => { this.ws = ws; resolve(); };
      ws.onerror = () => reject(new Error('could not connect to the search server'));
      ws.onclose = () => {
        if (this.ws === ws) this.ws = null;
        if (this.pending) {
          this.pending.reject(new Error('connection to the search server closed'));
          this.pending = null;
        }
      };
      ws.onmessage = (event) => this._dispatch(JSON.parse(event.data));
    });
  }

  _dispatch(msg) {
    const p = this.pending;
    if (!p) return;
    p.onMessage?.(msg);
    if (msg.type === 'result') {
      this.pending = null;
      p.resolve(msg);
    } else if (msg.type === 'error') {
      this.pending = null;
      p.reject(new Error(msg.message));
    }
  }

  get busy() { return this.pending !== null; }

  async solve(request, onMessage) {
    if (this.pending) throw new Error('a search is already running');
    await this._open();
    return new Promise((resolve, reject) => {
      this.pending = { resolve, reject, onMessage };
      this.ws.send(JSON.stringify(request));
    });
  }

  cancel() {
    if (this.pending && this.ws) this.ws.send(JSON.stringify({ type: 'cancel' }));
  }
}

export function formatNumber(x) {
  return x === undefined || x === null ? '—' : Math.round(x).toLocaleString('en-US');
}

export function formatSeconds(s) {
  if (s === undefined || s === null) return '—';
  return s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`;
}
