const API_ROOT = import.meta.env.VITE_API_ROOT || '';

async function request(path, options = {}) {
  const { timeoutMs = 0, ...fetchOptions } = options;
  const controller = timeoutMs ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const response = await fetch(`${API_ROOT}/api${path}`, {
      headers: { 'Content-Type': 'application/json', ...fetchOptions.headers },
      ...fetchOptions,
      ...(controller ? { signal: controller.signal } : {}),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.detail || payload.error || `Request failed (${response.status})`;
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return payload;
  } catch (error) {
    if (controller?.signal.aborted) {
      throw new Error('Agent request timed out after 180 seconds. The local model may still be working; reload state before retrying.');
    }
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

export const api = {
  state: (tenant) => request(`/state?tenant_id=${encodeURIComponent(tenant)}`),
  graph: (tenant) => request(`/graph?tenant_id=${encodeURIComponent(tenant)}`),
  concurrencyEvidence: () => request('/evaluation/concurrency'),
  tenants: () => request('/tenants'),
  runAgent: (agent, tenant, incidentId, prompt) => request(`/agents/${agent}/run`, {
    method: 'POST', body: JSON.stringify({ tenant_id: tenant, incident_id: incidentId, ...(prompt ? { prompt } : {}) }),
    timeoutMs: 180000,
  }),
  correctFact: (tenant, factId, correctedText) => request('/facts/correct', {
    method: 'POST', body: JSON.stringify({ tenant_id: tenant, fact_id: factId, corrected_text: correctedText }),
  }),
  taskAction: (tenant, taskId, action) => request(`/tasks/${encodeURIComponent(taskId)}/${action}`, {
    method: 'POST', body: JSON.stringify({ tenant_id: tenant }),
  }),
  approveRunbook: (tenant, runbookId) => request(`/runbooks/${encodeURIComponent(runbookId)}/approve`, {
    method: 'POST', body: JSON.stringify({ tenant_id: tenant }),
  }),
  resetDemo: (confirmation) => request('/demo/reset', {
    method: 'POST', body: JSON.stringify({ tenant_id: 'acme-retail', confirmation }),
  }),
  search: (tenant, query, mode = 'hierarchical') => request('/memories/search', {
    method: 'POST', body: JSON.stringify({ tenant_id: tenant, query, mode, k: 8 }),
  }),
};

export function normalizeState(payload) {
  if (!payload || typeof payload !== 'object' || !payload.tenant_id) {
    throw new Error('The API returned an incomplete state payload.');
  }
  return {
    ...payload,
    categories: Array.isArray(payload.categories) ? payload.categories : [],
    timeline: Array.isArray(payload.timeline) ? payload.timeline : [],
    facts: Array.isArray(payload.facts) ? payload.facts : [],
    agents: Array.isArray(payload.agents) ? payload.agents : [],
    incident: payload.incident || {},
    task: payload.task || null,
    handoff: payload.handoff || null,
    backend: payload.backend || { db_mode: 'demo', status: 'unknown' },
  };
}
