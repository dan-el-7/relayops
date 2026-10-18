import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, normalizeState } from './api.js';

const AGENT_META = {
  triage: { label: 'Triage', icon: '◈', purpose: 'Frames the incident and captures the signal', color: 'triage' },
  investigator: { label: 'Investigator', icon: '⌕', purpose: 'Follows dependencies and gathers evidence', color: 'investigator' },
  resolver: { label: 'Resolver', icon: '✓', purpose: 'Acts on a current, claimed handoff', color: 'resolver' },
};

const AGENT_REQUEST = {
  investigator: 'Find the current database pool and customer impact. Also use one additional search of the runbooks_procedures category to verify mitigation guidance before summarizing.',
};
const APPEARANCE_THEMES = [
  { id: 'plum', label: 'Plum dusk' },
  { id: 'aurora', label: 'Northern lights' },
  { id: 'blue-hour', label: 'Blue hour' },
];
function readAppearanceTheme() {
  try {
    const saved = window.localStorage.getItem('relayops-appearance');
    return APPEARANCE_THEMES.some((theme) => theme.id === saved) ? saved : 'plum';
  } catch { return 'plum'; }
}
function shouldShowAppearanceOnboarding() {
  try { return window.localStorage.getItem('relayops-appearance-onboarding') !== 'complete'; }
  catch { return false; }
}

const idOf = (value) => value?.id ?? value?.task_id ?? value?.incident_id ?? '';
const titleOf = (value) => value?.title || value?.name || value?.summary || 'Untitled';
const textOf = (value) => value?.text || value?.content || value?.summary || value?.value || value?.excerpt || '';
const timeOf = (value) => {
  const raw = value?.created_at || value?.timestamp || value?.updated_at || value?.observed_at || value?.at;
  if (!raw) return '—';
  const date = new Date(raw);
  return Number.isNaN(date.getTime()) ? String(raw) : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};
const isCurrent = (fact) => fact?.status === 'current' || fact?.status === 'active' || (!fact?.status && !fact?.valid_to);
const isStale = (task) => task?.status === 'stale' || task?.stale === true || task?.is_stale === true;
const isTerminalTask = (status) => ['done', 'completed', 'resolved', 'cancelled', 'canceled'].includes(String(status || '').trim().toLowerCase());
const humanize = (value = '') => String(value).replaceAll('_', ' ').replace(/\b\w/g, (x) => x.toUpperCase());
function retrievalOrigin(item) {
  const sources = (Array.isArray(item?.match_sources) ? item.match_sources : []).map((source) => String(source).toLowerCase());
  const retrieval = String(item?.retrieval_source || '').toLowerCase();
  const hasBm25 = sources.some((source) => source.includes('bm25')) || retrieval.includes('bm25');
  const hasVector = sources.some((source) => ['hnsw', 'vector', 'dense'].some((kind) => source.includes(kind))) || retrieval.includes('hybrid');
  if (retrieval.includes('hybrid') || (hasBm25 && hasVector)) return 'Hybrid · HNSW + BM25';
  if (hasBm25) return 'BM25';
  return '';
}
function llmStatus(backend) {
  backend = backend || {};
  const mode = String(backend.llm_mode || backend.agent_mode || backend.model_mode || '').trim().toLowerCase().replaceAll('-', '_').replaceAll(' ', '_');
  const model = String(backend.llm_model || '').trim();
  const provider = String(backend.llm_provider || '').trim();
  const endpoint = String(backend.llm_endpoint_host || '').trim();
  if (['chat_completions', 'openai_compatible', 'openai_compatible_chat'].includes(mode)) {
    const providerLabel = provider || 'OpenAI-compatible';
    const modelLabel = model || 'model not reported';
    const endpointText = endpoint ? ` · ${endpoint}` : '';
    return {
      topbar: `Live remote · ${providerLabel} · ${modelLabel}`,
      sidebar: `Live remote · ${providerLabel} · ${model.split('/').at(-1) || 'model not reported'}`,
      title: `Live remote inference · Provider ${providerLabel} · Model ${modelLabel}${endpointText}`,
    };
  }
  if (mode === 'local') {
    const modelLabel = model || backend.local_model || 'model not reported';
    return { topbar: `Local · ${modelLabel}`, sidebar: `Local · ${modelLabel}`, title: `Live local inference · ${modelLabel}` };
  }
  if (mode === 'openai') {
    const modelLabel = model || 'model not reported';
    return { topbar: `OpenAI · ${modelLabel}`, sidebar: `OpenAI · ${modelLabel}`, title: `Live OpenAI Responses inference · Model ${modelLabel}` };
  }
  if (mode === 'scripted') return { topbar: 'Scripted agents', sidebar: 'Scripted agents', title: 'Deterministic scripted agent mode' };
  const unknown = mode ? `Agent mode · ${humanize(mode)}` : 'Agent mode unavailable';
  return { topbar: unknown, sidebar: unknown, title: 'The API did not report an agent mode.' };
}

function useDialogFocus(open, onClose, dialogRef) {
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!open || !dialog) return undefined;
    const previousFocus = document.activeElement;
    const focusableSelector = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
    const focusableItems = () => Array.from(dialog.querySelectorAll(focusableSelector)).filter((element) => element.getClientRects().length > 0 && element.getAttribute('aria-hidden') !== 'true');
    if (!dialog.hasAttribute('tabindex')) dialog.setAttribute('tabindex', '-1');
    const initialFocus = dialog.querySelector('[data-dialog-initial-focus]') || focusableItems()[0] || dialog;
    initialFocus.focus();

    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        closeRef.current?.();
        return;
      }
      if (event.key !== 'Tab') return;
      const items = focusableItems();
      if (!items.length) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      if (!dialog.contains(document.activeElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      if (previousFocus?.isConnected && typeof previousFocus.focus === 'function') previousFocus.focus();
    };
  }, [open, dialogRef]);
}

export default function App() {
  const [tenant, setTenant] = useState('acme-retail');
  const [tenants, setTenants] = useState([]);
  const [state, setState] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState(null);
  const [view, setView] = useState('incident');
  const [theme, setTheme] = useState(readAppearanceTheme);
  const [appearanceOnboardingOpen, setAppearanceOnboardingOpen] = useState(shouldShowAppearanceOnboarding);
  const [agentLaneExpanded, setAgentLaneExpanded] = useState(false);
  const [tab, setTab] = useState('facts');
  const [query, setQuery] = useState('checkout cache dependency');
  const [searchResult, setSearchResult] = useState(null);
  const [agentEvidence, setAgentEvidence] = useState([]);
  const [selectedFact, setSelectedFact] = useState(null);
  const [correction, setCorrection] = useState('');
  const [resetOpen, setResetOpen] = useState(false);
  const [graphOpen, setGraphOpen] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [graphError, setGraphError] = useState('');
  const [graphLoading, setGraphLoading] = useState(false);
  const [concurrencyEvidence, setConcurrencyEvidence] = useState(null);
  const [concurrencyError, setConcurrencyError] = useState('');
  const tenantEpoch = useRef(0);
  const sessionMenuRef = useRef(null);
  const factDialogRef = useRef(null);
  const resetDialogRef = useRef(null);
  const appearanceDialogRef = useRef(null);
  useDialogFocus(Boolean(selectedFact), () => setSelectedFact(null), factDialogRef);
  useDialogFocus(resetOpen, () => setResetOpen(false), resetDialogRef);
  useDialogFocus(appearanceOnboardingOpen, () => setAppearanceOnboardingOpen(false), appearanceDialogRef);

  const switchTenant = (tenantId) => {
    if (tenantId === tenant) return;
    tenantEpoch.current += 1;
    setState(null);
    setSearchResult(null);
    setAgentEvidence([]);
    setBusy('');
    setSelectedFact(null);
    setNotice(null);
    setGraphData(null);
    setConcurrencyEvidence(null);
    setGraphOpen(false);
    setTab('facts');
    setTenant(tenantId);
  };
  const finishAppearanceOnboarding = () => {
    try { window.localStorage.setItem('relayops-appearance-onboarding', 'complete'); } catch { /* Continue for this session if storage is unavailable. */ }
    setAppearanceOnboardingOpen(false);
  };

  const openGraph = async () => {
    const requestEpoch = tenantEpoch.current;
    setGraphOpen(true);
    setGraphLoading(true);
    setGraphError('');
    setConcurrencyError('');
    const [graphResult, evidenceResult] = await Promise.allSettled([api.graph(tenant), api.concurrencyEvidence()]);
    if (tenantEpoch.current !== requestEpoch) return;
    if (graphResult.status === 'fulfilled') setGraphData(graphResult.value);
    else setGraphError(graphResult.reason?.message || 'Could not load this tenant graph.');
    if (evidenceResult.status === 'fulfilled') setConcurrencyEvidence(evidenceResult.value);
    else setConcurrencyError(evidenceResult.reason?.message || 'No concurrency report is available.');
    setGraphLoading(false);
  };

  const load = useCallback(async (tenantId = tenant) => {
    const requestEpoch = tenantEpoch.current;
    setError('');
    try {
      const payload = await api.state(tenantId);
      if (requestEpoch !== tenantEpoch.current) return;
      setState(normalizeState(payload));
    } catch (e) {
      if (requestEpoch !== tenantEpoch.current) return;
      setError(e.message || 'Could not load incident state.');
      setState(null);
    }
  }, [tenant]);

  useEffect(() => { load(tenant); }, [load, tenant]);
  useEffect(() => {
    try { window.localStorage.setItem('relayops-appearance', theme); } catch { /* Appearance remains session-local if storage is unavailable. */ }
  }, [theme]);
  useEffect(() => {
    if (sessionMenuRef.current) sessionMenuRef.current.open = false;
  }, [view]);
  useEffect(() => {
    let active = true;
    setGraphData(null);
    api.graph(tenant).then((payload) => { if (active) setGraphData(payload); }).catch(() => { if (active) setGraphData(null); });
    return () => { active = false; };
  }, [tenant]);
  useEffect(() => {
    api.tenants().then((payload) => setTenants(payload.items || [])).catch(() => setTenants([]));
  }, []);
  useEffect(() => {
    if (!notice) return undefined;
    const timer = setTimeout(() => setNotice(null), 3600);
    return () => clearTimeout(timer);
  }, [notice]);
  const mutate = async (key, action, success) => {
    const requestEpoch = tenantEpoch.current;
    const tenantId = tenant;
    setBusy(key);
    setError('');
    try {
      await action();
      if (requestEpoch !== tenantEpoch.current) return false;
      await load(tenantId);
      if (requestEpoch !== tenantEpoch.current) return false;
      try {
        const freshGraph = await api.graph(tenantId);
        if (requestEpoch === tenantEpoch.current) setGraphData(freshGraph);
      } catch {
        if (requestEpoch === tenantEpoch.current) setGraphData(null);
      }
      if (requestEpoch !== tenantEpoch.current) return false;
      setNotice({ text: success, kind: 'success' });
      return true;
    } catch (e) {
      if (requestEpoch !== tenantEpoch.current) return false;
      setNotice({ text: e.message || 'Action failed', kind: 'error' });
      return false;
    } finally { if (requestEpoch === tenantEpoch.current) setBusy(''); }
  };

  const runAgent = async (agent) => {
    setSearchResult(null);
    let retrievedEvidence = [];
    const succeeded = await mutate(`agent-${agent}`, async () => {
      const result = await api.runAgent(agent, tenant, idOf(state?.incident), AGENT_REQUEST[agent]);
      retrievedEvidence = Array.isArray(result.evidence) ? result.evidence : [];
    }, `${AGENT_META[agent].label} completed and saved to incident memory.`);
    if (succeeded) {
      setAgentEvidence(retrievedEvidence);
      if (agent === 'investigator' || agent === 'resolver') { setTab('evidence'); setView('memory'); }
    }
  };

  const correct = async (event) => {
    event.preventDefault();
    if (!selectedFact || !correction.trim()) return;
    const succeeded = await mutate('correct', () => api.correctFact(tenant, idOf(selectedFact), correction.trim()), 'Correction saved. Handoff marked stale; refresh it before acting.');
    if (succeeded) { setSelectedFact(null); setCorrection(''); }
  };

  const task = state?.task;
  const incident = state?.incident || {};
  const allFacts = state?.facts || [];
  const currentFacts = useMemo(() => allFacts.filter(isCurrent), [allFacts]);
  const oldFacts = useMemo(() => allFacts.filter((f) => !isCurrent(f)), [allFacts]);
  const backend = state?.backend || {};
  const dbMode = backend.mode || backend.db_mode;
  const evidence = agentEvidence.length ? agentEvidence : (state?.handoff?.evidence?.length ? state.handoff.evidence : task?.evidence || []);
  const categories = state?.categories || [];

  const search = async (event) => {
    event.preventDefault();
    if (!query.trim()) return;
    const requestEpoch = tenantEpoch.current;
    const tenantId = tenant;
    setBusy('search');
    try {
      const result = await api.search(tenantId, query.trim(), 'hierarchical');
      if (requestEpoch !== tenantEpoch.current) return;
      setSearchResult(result);
      setTab('evidence');
      setView('memory');
    } catch (e) { if (requestEpoch === tenantEpoch.current) setNotice({ text: e.message || 'Memory search failed', kind: 'error' }); }
    finally { if (requestEpoch === tenantEpoch.current) setBusy(''); }
  };

  if (error && !state) return <div className="app-shell"><Sidebar tenant={tenant} tenants={tenants} onTenant={switchTenant} backend={null} view={view} onView={setView} /><main className="main"><Topbar view={view} /><section className="error-card card"><div className="error-icon">!</div><h1>RelayOps could not connect</h1><p>{error}</p><p>Start the local API, then retry. The interface will show Demo / degraded whenever FalkorDB is unavailable.</p><button className="btn btn-primary" onClick={() => load(tenant)}>↻ &nbsp; Retry connection</button></section></main></div>;
  if (!state) return <div className="boot"><span className="spinner" /><span>Loading incident memory…</span></div>;

  const taskId = idOf(task);
  const actionState = String(task?.status || '').toLowerCase();
  const taskComplete = isTerminalTask(actionState);
  const isClaimed = !taskComplete && (['claimed', 'in_progress', 'in progress', 'active'].includes(actionState) || Boolean(task?.owner));
  const taskStale = !taskComplete && (isStale(task) || String(state.handoff?.status || '').toLowerCase() === 'stale');
  const canRunAction = isClaimed && !taskStale && !taskComplete;
  const incidentTitle = titleOf(incident);
  const severity = incident.severity || incident.priority || 'SEV-2';
  const incidentStatus = incident.status || 'Investigating';
  const tenantName = tenants.find((t) => t.id === tenant)?.name || state.tenant_name || tenant;
  const appAgentStatus = (agentId) => state.agents.find((a) => a.id === agentId || a.name?.toLowerCase() === agentId)?.status || 'ready';
  const agentLaneStatus = Object.entries(AGENT_META).map(([id, meta]) => `${meta.label} ${humanize(appAgentStatus(id))}`).join(' · ');
  const pageTitles = { incident: 'Incident room', timeline: 'Timeline', memory: 'Memory', runbooks: 'Runbooks' };
  const pageDescriptions = {
    incident: 'The current incident, its lineage, and the next safe action.',
    timeline: 'Events recorded for this incident, with source and time.',
    memory: 'Search shared memory, inspect current facts, and trace retrieval.',
    runbooks: 'Review the candidate recovery guidance and its approval state.',
  };
  const selectedAppearance = APPEARANCE_THEMES.find((option) => option.id === theme) || APPEARANCE_THEMES[0];

  return <div className="app-shell" data-theme={theme}>
    <AmbientShader theme={theme} />
    <Sidebar tenant={tenant} tenants={tenants} onTenant={switchTenant} backend={backend} view={view} onView={setView} />
    <main className="main">
      <Topbar tenantName={tenantName} backend={backend} view={view} />
      <div className="page">
        <div className="page-heading">
          <div><div className="eyebrow">{tenantName} / {view === 'incident' ? 'Live handoff' : pageTitles[view]}</div><h1>{pageTitles[view]}</h1><p className="subhead">{pageDescriptions[view]}</p></div>
          <div className="heading-actions">
            <details className="heading-menu" ref={sessionMenuRef}>
              <summary className="btn heading-menu-trigger">Session actions <span aria-hidden="true">⌄</span></summary>
              <div className="heading-menu-items">
                <label className="theme-picker"><span>Ambient theme</span><select aria-label="Ambient theme" value={theme} onChange={(event) => setTheme(event.target.value)}>{APPEARANCE_THEMES.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}</select></label>
                <button className="btn" aria-label="Restart session" onClick={() => window.location.reload()} title="Reload the app and restore saved incident state">↻ &nbsp; Restart session</button>
                {tenant === 'acme-retail' && <button className="btn btn-danger-ghost" aria-label="Reset seeded demo graph" disabled={!backend.demo_reset_enabled} title={backend.demo_reset_enabled ? 'Reset the isolated local demo graph' : 'Demo reset is disabled in production'} onClick={() => { if (sessionMenuRef.current) sessionMenuRef.current.open = false; setResetOpen(true); }}>Reset seeded demo</button>}
              </div>
            </details>
            <button className="btn btn-primary" disabled={busy === 'agent-triage'} onClick={() => runAgent('triage')}>{busy === 'agent-triage' ? 'Running…' : <>＋ <span>New triage</span></>}</button>
          </div>
        </div>

        {view === 'incident' && <div className="incident-layout">
          <IncidentBanner incident={incident} title={incidentTitle} severity={severity} status={incidentStatus} facts={currentFacts.length} task={task} taskStale={taskStale} taskComplete={taskComplete} />
          <GraphOverview payload={graphData} tenantName={tenantName} loading={!graphData} onExplore={openGraph} />
          <div className="content-grid incident-content-grid">
            <section className="left-column">
            <details className="agent-lane-disclosure" open={agentLaneExpanded} onToggle={(event) => setAgentLaneExpanded(event.currentTarget.open)}>
            <summary className="agent-lane-summary"><span><strong>Agent lane · 3 roles</strong><small>{agentLaneStatus}</small></span><b>{agentLaneExpanded ? 'Collapse' : 'Expand'}</b></summary>
            <Card title="Agent handoff lane" icon="⇄" meta="Shared state · persisted in graph">
              <div className="agent-stack">{Object.entries(AGENT_META).map(([id, meta]) => {
                const agent = state.agents.find((a) => a.id === id || a.role === id || a.name?.toLowerCase() === id) || {};
                const status = String(agent.status || appAgentStatus(id)).toLowerCase();
                const resolverBlocked = id === 'resolver' && (!canRunAction || taskStale);
                return <div className="agent-card" key={id}><div className={`agent-avatar ${meta.color}`}>{meta.icon}</div><div className="agent-info"><strong>{meta.label}</strong><small>{agent.capability || agent.summary || agent.last_action || meta.purpose}</small></div><div className="agent-state"><span className={`tag ${status.includes('complete') ? 'tag-green' : status.includes('work') || status.includes('run') ? 'tag-blue' : 'tag-amber'}`}>{humanize(status)}</span><button className="btn btn-small" disabled={busy === `agent-${id}` || resolverBlocked} title={resolverBlocked ? 'Claim the task and refresh supporting facts before running Resolver.' : ''} onClick={() => runAgent(id)}>{busy === `agent-${id}` ? 'Running…' : 'Run agent'}</button></div></div>;
              })}</div>
              {task ? <div className={`handoff ${taskStale ? 'handoff-stale' : ''}`}>
                <div className="handoff-top"><strong>Persisted task · {titleOf(task)}</strong><span className={`tag ${taskStale ? 'tag-red' : isClaimed ? 'tag-blue' : 'tag-amber'}`}>{taskStale ? 'Refresh required' : humanize(task.status || 'open')}</span></div>
                <p>{task.description || task.summary || state.handoff?.summary || (Array.isArray(state.handoff?.next_actions) ? state.handoff.next_actions.join(' · ') : '') || 'Investigate the affected cache dependency and verify the current service path.'}</p>
                {taskStale && <div className="stale-note">A supporting fact changed. Refresh this handoff before the next simulated action.</div>}
                <div className="handoff-footer"><span>{state.handoff?.from_agent ? `${humanize(state.handoff.from_agent)} → ${humanize(state.handoff.to_agent || 'Resolver')}` : `Owner: ${task.owner || 'Unclaimed'}`}</span><span>Updated {timeOf(task)}</span></div>
              </div> : <Empty title="No persisted handoff" detail="Run Investigator to create a task for the next agent." />}
            </Card>
            </details>
            <button className="timeline-view-link" onClick={() => setView('timeline')}><span className="timeline-view-icon">◷</span><span><strong>Timeline</strong><small>{state.timeline.length} recorded events</small></span><b>Open →</b></button>
          </section>

          <section className="right-column">
            <Card title={taskComplete ? 'Task complete' : 'Next safe action'} icon={taskComplete ? '✓' : '⤷'} meta={taskComplete ? 'Handoff closed' : taskStale ? 'Blocked by stale evidence' : 'Task-gated'}>
              <div className="action-card">
                <div className="action-kicker"><span>↳</span> {taskComplete ? 'Completed handoff' : taskStale ? 'Revalidation needed' : !isClaimed ? 'Claim required' : 'Resolver recommendation'}</div>
                <h3>{taskComplete ? 'No further action is pending.' : task?.next_action || state.handoff?.next_actions?.[0] || (taskStale ? 'Refresh the handoff before acting' : 'Verify the active cache dependency')}</h3>
                <p>{taskComplete ? `This task was completed${task?.updated_at ? ` at ${timeOf(task)}` : ''}. The previous recommendation remains in the incident history.` : taskStale ? 'RelayOps prevents this simulated action until evidence is refreshed.' : !isClaimed ? 'Claim the persisted handoff before running the resolver action.' : 'The resolver acts only after checking the current handoff evidence. Infrastructure actions are simulated.'}</p>
                {!taskComplete && task && <div className="action-buttons">{taskStale ? <button className="btn btn-primary btn-small" disabled={busy === 'refresh'} onClick={() => mutate('refresh', () => api.taskAction(tenant, taskId, 'refresh'), 'Handoff refreshed against current facts.')}>{busy === 'refresh' ? 'Refreshing…' : 'Refresh evidence'}</button> : !isClaimed ? <button className="btn btn-primary btn-small" disabled={busy === 'claim'} onClick={() => mutate('claim', () => api.taskAction(tenant, taskId, 'claim'), 'Task claimed for this shift.')}>{busy === 'claim' ? 'Claiming…' : 'Claim handoff'}</button> : <button className="btn btn-primary btn-small" disabled={!canRunAction || busy === 'simulate'} onClick={() => mutate('simulate', () => api.taskAction(tenant, taskId, 'simulate-action'), 'Simulated action recorded.')}>{busy === 'simulate' ? 'Recording…' : 'Run simulated action'}</button>}</div>}
              </div>
            </Card>
          </section>
        </div>
        </div>}

        {['timeline', 'memory'].includes(view) && <IncidentContextStrip incident={incident} task={task} taskStale={taskStale} tenantName={tenantName} status={incidentStatus} severity={severity} onOpen={() => setView('incident')} />}

        {view === 'timeline' && <Card title="Incident timeline" icon="◷" meta={`${state.timeline.length} recorded events`} action={<button className="icon-btn" aria-label="Reload incident" onClick={() => load(tenant)}>↻</button>}>
          {state.timeline.length ? <div className="timeline">{state.timeline.map((event, index) => <TimelineEvent key={idOf(event) || index} event={event} />)}</div> : <Empty title="No incident events yet" detail="Run triage to record the first event in shared memory." />}
        </Card>}

        {view === 'memory' && <div className="content-grid memory-content-grid">
          <section className="left-column">
            <Card title="Memory control plane" icon="⌘" meta={state.backend?.db_mode === 'falkordb' ? 'FalkorDB graph' : 'Demo memory'}>
              <form className="memory-search" onSubmit={search}><span className="search-symbol">⌕</span><input aria-label="Search incident memory" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search incident memory…" /><button className="btn btn-small" type="submit" disabled={busy === 'search'}>{busy === 'search' ? 'Searching…' : 'Search'}</button></form>
              <div className="routing-summary"><span className="route-status"><span className="status-dot" /> {searchResult ? 'Latest retrieval route' : 'Current incident context'}</span><span className="route-mode">{searchResult?.fallback_used ? 'Global fallback' : 'Hierarchical'}</span></div>
              <div className="category-pills">{(searchResult?.routed_categories?.length ? searchResult.routed_categories : categories.slice(0, 3)).map((cat) => <span className="category-pill" key={cat.id}>{cat.name || cat.id}{cat.score != null && <small>score {Number(cat.score).toFixed(2)}</small>}</span>)}</div>
              <div className="memory-tabs" role="tablist" aria-label="Memory views">{[['facts','Facts'],['history','History'],['evidence','Evidence']].map(([id,label]) => <button className={`memory-tab ${tab === id ? 'active' : ''}`} role="tab" aria-selected={tab === id} key={id} onClick={() => setTab(id)}>{label}{id === 'facts' ? ` ${currentFacts.length}` : id === 'history' ? ` ${oldFacts.length}` : ''}</button>)}</div>
              <MemoryPanel tab={tab} current={currentFacts} old={oldFacts} evidence={searchResult?.results || evidence} categories={categories} onCorrect={(fact) => { setSelectedFact(fact); setCorrection(textOf(fact)); }} />
              {tab === 'facts' && currentFacts.length > 0 && <div className="correction-box"><div className="correction-label"><span>Correct a dependency</span><span>human approved</span></div><button className="btn btn-correction" onClick={() => { setSelectedFact(currentFacts[0]); setCorrection(textOf(currentFacts[0])); }}>＋ &nbsp; Correct selected fact</button></div>}
            </Card>
          </section>
          <section className="right-column">
            <Card title="Retrieval trace" icon="⌁" meta="Evidence path">
              <div className="trace-steps">{(searchResult?.routed_categories?.length ? searchResult.routed_categories : categories.slice(0, 2)).map((cat, i) => <span className="trace-step" key={cat.id}><span className="trace-number">0{i + 1}</span><span>{cat.name || cat.id}</span>{i === 0 && <span className="trace-arrow">→</span>}</span>)}</div>
              <div className="trace-foot">{searchResult ? `${searchResult.results?.length || 0} memories · ${searchResult.db_mode === 'falkordb' ? 'FalkorDB retrieval' : 'demo/degraded retrieval'} · ${searchResult.fallback_used ? 'fallback used' : 'no fallback'}${searchResult.timings_ms?.total != null ? ` · ${searchResult.timings_ms.total} ms` : ''}` : `${humanize(backend.router_mode || 'hierarchical')} route · current memories · bounded incident evidence`}</div>
            </Card>
          </section>
        </div>}

        {view === 'runbooks' && <RunbooksView runbook={state.runbook} incident={incident} task={task} busy={busy} onApprove={() => mutate('approve', () => api.approveRunbook(tenant, state.runbook.id), 'Runbook approved for this incident.')} />}
      </div>
    </main>
    <MobileBottomNav view={view} onView={setView} />
    {notice && <div className={`toast ${notice.kind === 'error' ? 'toast-error' : ''}`} role="status">{notice.text}</div>}
    {appearanceOnboardingOpen && <div className="appearance-onboarding-backdrop"><section className="appearance-onboarding-card" ref={appearanceDialogRef} role="dialog" aria-modal="true" aria-labelledby="appearance-onboarding-title" aria-describedby="appearance-onboarding-copy">
      <div className="appearance-onboarding-kicker">A QUICK WORKSPACE SETUP</div>
      <h2 id="appearance-onboarding-title">Choose your atmosphere</h2>
      <p id="appearance-onboarding-copy">Pick an ambient color theme. The soft shader updates as you choose and stays behind the incident work.</p>
      <div className="appearance-theme-options" aria-label="Ambient themes">{APPEARANCE_THEMES.map((option) => <button type="button" key={option.id} className={`appearance-theme-option ${theme === option.id ? 'selected' : ''}`} aria-pressed={theme === option.id} data-dialog-initial-focus={theme === option.id ? 'true' : undefined} onClick={() => setTheme(option.id)}>
        <span className={`appearance-theme-preview preview-${option.id}`} aria-hidden="true"><i /><i /><i /></span>
        <span className="appearance-theme-copy"><strong>{option.label}</strong><small>{option.id === 'plum' ? 'Warm violet glow' : option.id === 'aurora' ? 'Teal and aurora green' : 'Cool blue dusk'}</small></span>
      </button>)}</div>
      <button type="button" className="btn btn-primary appearance-onboarding-continue" onClick={finishAppearanceOnboarding}>Enter RelayOps · {selectedAppearance.label}</button>
      <small className="appearance-onboarding-note">You can change the theme any time in Session actions.</small>
    </section></div>}
    {graphOpen && <GraphDrawer tenantName={tenantName} payload={graphData} loading={graphLoading} error={graphError} concurrencyEvidence={concurrencyEvidence} concurrencyError={concurrencyError} onClose={() => setGraphOpen(false)} />}
    {selectedFact && <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && setSelectedFact(null)}><form className="modal" ref={factDialogRef} role="dialog" aria-modal="true" aria-labelledby="correct-fact-title" onSubmit={correct}><div className="modal-kicker">Append-only fact revision</div><h2 id="correct-fact-title">Correct the current fact</h2><p>The previous version stays in history. RelayOps will mark dependent handoffs stale so the next agent refreshes evidence.</p><label htmlFor="corrected-fact">Updated fact</label><textarea id="corrected-fact" rows="3" value={correction} onChange={(e) => setCorrection(e.target.value)} data-dialog-initial-focus /><div className="modal-actions"><button type="button" className="btn" onClick={() => setSelectedFact(null)}>Cancel</button><button className="btn btn-primary" disabled={busy === 'correct' || !correction.trim()}>{busy === 'correct' ? 'Saving…' : 'Save correction'}</button></div></form></div>}
    {resetOpen && <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && setResetOpen(false)}><section className="modal reset-modal" ref={resetDialogRef} role="dialog" aria-modal="true" aria-labelledby="reset-demo-title" aria-describedby="reset-demo-description"><div className="reset-mark">!</div><div className="modal-kicker">Destructive demo action</div><h2 id="reset-demo-title">Reset the seeded incident?</h2><p id="reset-demo-description">This permanently deletes and reseeds the isolated <strong>Acme Retail demo graph</strong>, including its incident history, task and memory revisions. This cannot be undone. Other tenant graphs are not affected.</p><div className="modal-actions"><button className="btn" data-dialog-initial-focus onClick={() => setResetOpen(false)}>Cancel</button><button className="btn btn-danger" disabled={busy === 'reset'} onClick={() => mutate('reset', async () => { await api.resetDemo('RESET ISOLATED DEMO TENANT'); setResetOpen(false); }, 'Acme Retail demo graph was reset.')}>{busy === 'reset' ? 'Resetting…' : 'Delete and reseed demo'}</button></div></section></div>}
  </div>;
}

function RunbooksView({ runbook, incident, task, busy, onApprove }) {
  const status = runbook?.approval_status || 'pending confirmation';
  const steps = Array.isArray(runbook?.steps) ? runbook.steps : [];
  return <div className="content-grid runbooks-content-grid">
    <section className="left-column">
      <Card title={runbook?.title || 'Runbook candidate'} icon="≡" meta="Incident recovery guidance">
        {runbook?.id ? <div className="runbook-view-body">
          <span className={`tag ${status === 'approved' ? 'tag-green' : 'tag-amber'}`}>{humanize(status)}</span>
          <p>{runbook.description || runbook.summary || 'This runbook was selected as the recovery candidate for the current handoff.'}</p>
          <p className="runbook-approval-note">Approval records this guidance for this incident. It does not execute an operational change.</p>
          {steps.length > 0 && <ol className="runbook-steps">{steps.map((step, index) => <li key={index}>{typeof step === 'string' ? step : step.title || step.description || JSON.stringify(step)}</li>)}</ol>}
          {status !== 'approved' && <button className="btn btn-primary" disabled={busy === 'approve'} onClick={onApprove}>{busy === 'approve' ? 'Saving…' : 'Approve runbook for this incident'}</button>}
        </div> : <Empty title="No runbook candidate" detail="Run Investigator to attach recovery guidance to the incident handoff." />}
      </Card>
    </section>
    <section className="right-column">
      <Card title="Incident context" icon="↳" meta="Current handoff">
        <dl className="runbook-context">
          <div><dt>Incident</dt><dd>{titleOf(incident)}</dd></div>
          <div><dt>Owner</dt><dd>{task?.owner || 'Unclaimed'}</dd></div>
          <div><dt>Task state</dt><dd>{humanize(task?.status || 'open')}</dd></div>
          <div><dt>Next action</dt><dd>{task?.next_action || task?.description || 'No next action recorded.'}</dd></div>
        </dl>
      </Card>
    </section>
  </div>;
}

function IncidentContextStrip({ incident, task, taskStale, tenantName, status, severity, onOpen }) {
  const title = titleOf(incident);
  const incidentId = incident.id || incident.incident_id || 'Active incident';
  const service = incident.service_name || incident.service || incident.service_id || 'Checkout';
  const taskStatus = !task ? 'No handoff' : isTerminalTask(task.status) ? 'Complete' : taskStale ? 'Refresh required' : task.owner ? 'Claimed' : humanize(task.status || 'open');
  const taskTone = isTerminalTask(task?.status) ? 'tag-green' : taskStale ? 'tag-red' : 'tag-blue';
  return <section className="incident-context-strip" aria-label="Current incident context">
    <div className="incident-context-copy"><span className="context-kicker">{tenantName} · ACTIVE INCIDENT</span><strong>{title}</strong><small>{incidentId} · {service}</small></div>
    <div className="incident-context-state"><span className="tag tag-amber">{severity}</span><span className="tag tag-blue">{humanize(status)}</span><span className={`tag ${taskTone}`}>Handoff · {taskStatus}</span></div>
    <button className="btn incident-context-open" onClick={onOpen}>Incident room <span aria-hidden="true">↗</span></button>
  </section>;
}

function GraphOverview({ payload, tenantName, loading, onExplore }) {
  const [selected, setSelected] = useState(null);
  const nodes = payload?.graph_nodes || [];
  const labelsFor = (node) => node.labels || [];
  const roleFor = (node) => {
    const labels = labelsFor(node);
    const p = node.properties || {};
    if (labels.includes('Incident')) return 'Incident';
    if (labels.includes('Service')) return 'Service';
    if (labels.includes('Handoff')) return 'Handoff';
    if (labels.includes('Task')) return 'Task';
    if (labels.includes('Runbook')) return 'Runbook';
    if (labels.includes('ActionEvent')) return 'Action';
    if (labels.includes('Category')) return 'Category';
    if (labels.includes('Memory')) return p.status === 'superseded' || p.valid_to ? 'Previous fact' : (labels.includes('CurrentMemory') ? 'Current memory' : humanize(p.kind || 'Memory'));
    return labels[0] || 'Record';
  };
  const labelOf = (node) => {
    const p = node?.properties || {};
    const slot = p.slot_key ? String(p.slot_key).split('.').at(-1).replaceAll('_', ' ') : '';
    const slotValue = slot ? String(p.text || '').match(/\b(?:is|was)\s+(\d+)\b/i)?.[1] : '';
    if (slot && slotValue) return `${slot}: ${slotValue}`;
    return String(p.name || p.title || slot || p.summary || p.text || p.id || p.kind || roleFor(node));
  };
  const priority = (node) => ({ Incident: 0, Service: 1, 'Current memory': 2, 'Previous fact': 3, Handoff: 4, Task: 5, Runbook: 6, Category: 7, Action: 8 })[roleFor(node)] ?? 10;
  const chosen = useMemo(() => {
    if (!nodes.length) return [];
    const wanted = [];
    const addFirst = (predicate, count = 1) => nodes.filter(predicate).slice(0, count).forEach((n) => { if (!wanted.some((x) => x.graph_id === n.graph_id)) wanted.push(n); });
    addFirst((n) => roleFor(n) === 'Incident');
    addFirst((n) => roleFor(n) === 'Service');
    addFirst((n) => roleFor(n) === 'Current memory' && String(n.properties?.slot_key || '').includes('db_pool'));
    if (!wanted.some((n) => roleFor(n) === 'Current memory')) addFirst((n) => roleFor(n) === 'Current memory' && n.properties?.kind === 'fact');
    addFirst((n) => roleFor(n) === 'Previous fact');
    addFirst((n) => roleFor(n) === 'Handoff');
    addFirst((n) => roleFor(n) === 'Task');
    addFirst((n) => roleFor(n) === 'Runbook');
    addFirst((n) => roleFor(n) === 'Category');
    return wanted.slice(0, 11);
  }, [payload]);
  const chosenIds = new Set(chosen.map((node) => node.graph_id));
  const edgePriority = (edge) => ({ SUPERSEDES: 0, AFFECTS: 1, HAS_HANDOFF: 2, HAS_TASK: 3, USES_RUNBOOK: 4, HANDOFF_FOR: 5, ABOUT: 6, IN_CATEGORY: 7, RECORDED_ACTION: 8 })[edge.type] ?? 10;
  const links = (payload?.graph_edges || []).filter((edge) => chosenIds.has(edge.source) && chosenIds.has(edge.target) && ['AFFECTS', 'HAS_HANDOFF', 'HANDOFF_FOR', 'USES_RUNBOOK', 'SUPERSEDES'].includes(edge.type) || (edge.type === 'IN_CATEGORY' && chosenIds.has(edge.source) && chosenIds.has(edge.target) && roleFor(chosen.find((node) => node.graph_id === edge.source)) === 'Current memory')).sort((a, b) => edgePriority(a) - edgePriority(b)).slice(0, 8);
  const visibleLinks = links.map((edge) => edge.type === 'SUPERSEDES'
    ? { ...edge, displaySource: edge.target, displayTarget: edge.source, displayType: 'SUPERSEDED BY' }
    : { ...edge, displaySource: edge.source, displayTarget: edge.target, displayType: edge.type });
  const positions = useMemo(() => {
    const placed = {};
    const occupied = new Set();
    const slots = { Incident: [0, 0], Service: [1, 0], 'Previous fact': [2, 0], 'Current memory': [2, 1], Handoff: [3, 0], Task: [3, 1], Runbook: [3, 2], Category: [1, 1] };
    for (const node of chosen) {
      const role = roleFor(node);
      const [col, baseRow] = slots[role] || [1, 0];
      let row = baseRow;
      while (occupied.has(`${col}:${row}`)) row += 1;
      occupied.add(`${col}:${row}`);
      placed[node.graph_id] = { x: 96 + col * 208, y: 58 + row * 66 };
    }
    return placed;
  }, [chosen]);
  const mapHeight = Math.max(238, ...Object.values(positions).map((point) => point.y + 38));
  const selectedNode = chosen.find((node) => node.graph_id === selected) || null;
  const tone = (node) => roleFor(node) === 'Previous fact' ? 'old' : (node.properties?.is_stale || node.properties?.stale || node.properties?.status === 'stale' ? 'stale' : roleFor(node) === 'Current memory' ? 'current' : 'record');
  const nodeLabel = (node) => labelOf(node).replaceAll('_', ' ');
  const mobileLabel = (id) => nodeLabel(chosen.find((node) => node.graph_id === id));
  return <section className="graph-overview" aria-label="Incident graph topology">
    <div className="graph-overview-head"><div><div className="graph-kicker">Memory lineage <span>FALKORDB</span></div><h2>How this incident connects</h2><p>{loading ? 'Loading the tenant graph…' : `Curated path · ${chosen.length} records · ${links.length} stored relationships`}</p></div><button className="btn graph-explore" onClick={onExplore}>Explore full graph <span aria-hidden="true">↗</span></button></div>
    {loading ? <div className="graph-overview-loading"><span className="spinner" /> Reading tenant-scoped topology</div> : chosen.length ? <>
      <div className="graph-mini-map">
        <svg viewBox={`0 0 800 ${mapHeight}`} role="group" aria-label="Selected incident, memory, and handoff lineage">
          <title>Selected incident, memory, and handoff lineage</title>
          <defs><marker id="relay-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" /></marker></defs>
          {visibleLinks.map((edge, index) => {
            const a = positions[edge.displaySource], b = positions[edge.displayTarget]; if (!a || !b) return null;
            const sameColumn = a.x === b.x;
            const startX = sameColumn ? a.x : a.x + (a.x < b.x ? 72 : -72);
            const endX = sameColumn ? b.x : b.x + (b.x < a.x ? 72 : -72);
            const startY = sameColumn ? a.y + (a.y < b.y ? 26 : -26) : a.y;
            const endY = sameColumn ? b.y + (b.y < a.y ? 26 : -26) : b.y;
            const midX = (startX + endX) / 2, midY = (startY + endY) / 2;
            return <g className="mini-edge" key={`${edge.source}-${edge.target}-${edge.type}-${index}`}><path d={`M ${startX} ${startY} L ${endX} ${endY}`} markerEnd="url(#relay-arrow)" /><text x={midX} y={midY - 5}>{edge.displayType}</text></g>;
          })}
          {chosen.map((node) => { const point = positions[node.graph_id]; const active = selected === node.graph_id; const label = nodeLabel(node); return <g key={node.graph_id} className={`mini-node mini-${tone(node)} ${active ? 'is-selected' : ''}`} transform={`translate(${point.x} ${point.y})`} onClick={() => setSelected(node.graph_id)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelected(node.graph_id); } }} tabIndex="0" role="button" aria-label={`${label}, ${roleFor(node)}`}><rect x="-70" y="-24" width="140" height="48" rx="9" /><text className="mini-kind" y="-5">{roleFor(node).toUpperCase()}</text><text className="mini-label" y="10">{label.length > 22 ? `${label.slice(0, 21)}…` : label}</text></g>; })}
        </svg>
      </div>
      <div className="graph-mobile-map">{visibleLinks.length ? visibleLinks.map((edge, index) => <button key={`${edge.source}-${edge.target}-${edge.type}-${index}`} className="mobile-edge" onClick={() => setSelected(edge.displayTarget)}><span>{mobileLabel(edge.displaySource)}</span><small>{edge.displayType}</small><span>{mobileLabel(edge.displayTarget)}</span><b>›</b></button>) : chosen.map((node) => <button key={node.graph_id} className="mobile-edge" onClick={() => setSelected(node.graph_id)}><span>{roleFor(node)}</span><small>{nodeLabel(node)}</small><b>›</b></button>)}</div>
      <div className="graph-overview-foot"><div className="graph-legend-inline"><span><i className="current" />Current</span><span><i className="old" />Superseded</span><span><i className="record" />Operational record</span></div><div className="graph-selection">{selectedNode ? <><strong>{labelOf(selectedNode)}</strong><span>{roleFor(selectedNode)} · {Object.entries(selectedNode.properties || {}).filter(([, value]) => value != null).slice(0, 3).map(([key, value]) => `${humanize(key)} ${String(value)}`).join(' · ')}</span></> : <span>Select a node or relationship to inspect the lineage.</span>}</div></div>
      <p className="graph-layout-note">Lines show stored FalkorDB relationships; node spacing groups record types for readability.</p>
    </> : <div className="graph-overview-empty"><span>Graph unavailable</span><small>Incident state is still available. Use Explore full graph to retry.</small></div>}
  </section>;
}

function GraphDrawer({ tenantName, payload, loading, error, concurrencyEvidence, concurrencyError, onClose }) {
  const [selected, setSelected] = useState(null);
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef(null);
  const dialogRef = useRef(null);
  useDialogFocus(true, onClose, dialogRef);
  const nodes = payload?.graph_nodes || [];
  const edges = payload?.graph_edges || [];
  const positions = useMemo(() => Object.fromEntries(nodes.map((node, index) => [node.graph_id, {
    x: 90 + (index % 6) * 160, y: 72 + Math.floor(index / 6) * 112,
  }])), [nodes]);
  const canvasHeight = Math.max(600, Math.ceil(nodes.length / 6) * 112 + 80);
  const selectedNode = nodes.find((node) => node.graph_id === selected) || null;
  const labelOfNode = (node) => {
    const props = node.properties || {};
    return String(props.name || props.title || props.id || props.kind || node.labels?.[0] || 'Node');
  };
  const edgeSummary = (edge) => `${labelOfNode(nodes.find((node) => node.graph_id === edge.source) || {})} ${edge.type} ${labelOfNode(nodes.find((node) => node.graph_id === edge.target) || {})}`;
  const nodeTone = (node) => {
    const p = node.properties || {};
    if (p.is_stale || p.stale || p.status === 'stale') return 'stale';
    if (node.labels?.includes('CurrentMemory')) return 'current';
    if (p.status === 'superseded' || (p.valid_to && p.kind === 'fact')) return 'old';
    return String(node.labels?.[0] || 'other').toLowerCase();
  };
  const beginPan = (event) => { drag.current = { x: event.clientX, y: event.clientY, pan }; event.currentTarget.setPointerCapture(event.pointerId); };
  const movePan = (event) => { if (drag.current) setPan({ x: drag.current.pan.x + event.clientX - drag.current.x, y: drag.current.pan.y + event.clientY - drag.current.y }); };
  const stopPan = () => { drag.current = null; };
  return <div className="graph-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
    <section className="graph-drawer" ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="graph-drawer-title">
      <header className="graph-head"><div><div className="modal-kicker">FalkorDB · {tenantName}</div><h2 id="graph-drawer-title">Incident graph</h2><p>Lines show stored relationships. Node spacing is arranged for readability, not graph distance. Select a node to inspect its fields.</p></div><button className="icon-btn" data-dialog-initial-focus aria-label="Close graph explorer" onClick={onClose}>×</button></header>
      {loading && <div className="graph-message"><span className="spinner" /> Loading tenant graph…</div>}
      {error && <div className="graph-message graph-error">{error}</div>}
      {!loading && !error && payload && <>
        <div className="graph-summary"><span>{nodes.length} returned nodes</span><span>{edges.length} returned relationships</span><small>Tenant-scoped bounded snapshot · query limit {payload.limits?.nodes || 80} nodes / {payload.limits?.edges || 160} relationships</small></div>
        <div className="graph-legend"><span><i className="legend-current" />Current memory</span><span><i className="legend-old" />Superseded fact</span><span><i className="legend-stale" />Stale work</span><span><i className="legend-entity" />Operational record</span></div>
        <div className="graph-workspace">
          <div className="graph-canvas" onWheel={(e) => { e.preventDefault(); setScale((s) => Math.max(.5, Math.min(2, s * (e.deltaY < 0 ? 1.08 : .92)))); }} onPointerDown={beginPan} onPointerMove={movePan} onPointerUp={stopPan} onPointerCancel={stopPan}>
            <svg viewBox={`0 0 1000 ${canvasHeight}`} style={{ height: `${canvasHeight}px` }} role="group" aria-label="Interactive FalkorDB graph">
              <title>Interactive FalkorDB graph. Lines represent stored relationships; spacing is for readability.</title>
              <g transform={`translate(${pan.x} ${pan.y}) translate(500 300) scale(${scale}) translate(-500 -300)`}>
                {edges.map((edge, index) => { const a = positions[edge.source], b = positions[edge.target]; if (!a || !b) return null; const related = selected === edge.source || selected === edge.target; return <g key={`${edge.source}-${edge.target}-${edge.type}-${index}`} className={`graph-edge ${related ? 'related' : ''}`}><title>{edgeSummary(edge)}</title><line x1={a.x} y1={a.y} x2={b.x} y2={b.y} />{related && <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 4}>{edge.type}</text>}</g>; })}
                {nodes.map((node) => { const point = positions[node.graph_id]; const selectedNode = selected === node.graph_id; const tone = nodeTone(node); const stateLabel = tone === 'current' || tone === 'old' || tone === 'stale' ? humanize(tone) : (node.labels || []).filter((x) => x !== 'CurrentMemory').join(' · '); return <g key={node.graph_id} className={`graph-node tone-${tone} ${selectedNode ? 'selected' : ''}`} transform={`translate(${point.x} ${point.y})`} onPointerDown={(e) => e.stopPropagation()} onClick={() => setSelected(node.graph_id)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelected(node.graph_id); } }} role="button" tabIndex="0" aria-label={`${labelOfNode(node)} node`}><rect x="-74" y="-30" width="148" height="60" rx="12" /><text className="graph-node-label" y="-4">{labelOfNode(node).slice(0, 19)}{labelOfNode(node).length > 19 ? '…' : ''}</text><text className="graph-node-type" y="17">{stateLabel}</text></g>; })}
              </g>
            </svg>
          </div>
          <aside className="graph-details"><strong>{selectedNode ? labelOfNode(selectedNode) : 'Node details'}</strong>{selectedNode ? <><small>{(selectedNode.labels || []).join(' · ')}{['current','old','stale'].includes(nodeTone(selectedNode)) ? ` · ${humanize(nodeTone(selectedNode))}` : ''}</small><dl>{Object.entries(selectedNode.properties || {}).filter(([, value]) => value != null).map(([key, value]) => <div key={key}><dt>{humanize(key)}</dt><dd>{Array.isArray(value) ? value.join(' · ') : String(value)}</dd></div>)}</dl></> : <p>Select any node in the graph to inspect its stored properties.</p>}</aside>
        </div>
        <details className="graph-node-index"><summary>Browse nodes ({nodes.length})</summary><ul>{nodes.map((node) => <li key={node.graph_id}><button aria-pressed={selected === node.graph_id} onClick={() => setSelected(node.graph_id)}>{labelOfNode(node)} <small>{(node.labels || []).join(' · ')}</small></button></li>)}</ul></details>
        <details className="graph-relationship-index"><summary>Browse stored relationships ({edges.length})</summary><ul>{edges.map((edge, index) => <li key={`${edge.source}-${edge.target}-${edge.type}-${index}`}>{edgeSummary(edge)}</li>)}</ul></details>
        <ConcurrencyEvidence report={concurrencyEvidence} error={concurrencyError} />
      </>}
    </section>
  </div>;
}

function ConcurrencyEvidence({ report, error }) {
  const reads = report?.results?.reads;
  const writes = report?.results?.episode_writes;
  const claims = report?.results?.atomic_claims;
  const checksPassed = Boolean(report?.checks?.exactly_one_claim_winner && report?.checks?.cross_tenant_isolation && report?.checks?.all_read_write_requests_succeeded && report?.checks?.cleanup === 'passed');
  return <details className="concurrency-evidence">
    <summary><span><strong>Concurrency evidence</strong><small>{report ? report.classification : error || 'Loading report…'}</small></span><b>{report ? `${reads?.successes ?? 0}/${reads?.requests ?? 0} reads · ${writes?.successes ?? 0}/${writes?.requests ?? 0} writes` : 'Report unavailable'}</b></summary>
    {report ? <div className="concurrency-body">
      <div className="concurrency-metrics"><div><small>Parallel reads</small><strong>{reads?.successes}/{reads?.requests}</strong><span>p50 {reads?.p50_ms} ms · p95 {reads?.p95_ms} ms</span></div><div><small>Episode writes</small><strong>{writes?.successes}/{writes?.requests}</strong><span>p50 {writes?.p50_ms} ms · p95 {writes?.p95_ms} ms</span></div><div><small>Atomic claim race</small><strong>{claims?.winner_count} winner · {claims?.loss_count} conflicts</strong><span>{claims?.requests} concurrent claim attempts</span></div></div>
      <div className="concurrency-foot"><span className={checksPassed ? 'checks-pass' : 'checks-fail'}>{checksPassed ? 'Passed: reads/writes · one winner · isolation · cleanup' : 'One or more report checks failed'}</span><span>Run {report.run_id} · {report.generated_at_utc ? new Date(report.generated_at_utc).toLocaleString() : 'time not reported'}</span><span>{report.environment?.embedding?.model || 'Embedding model not reported'} · {report.environment?.gpu?.name || 'GPU not reported'}</span></div>
      <p className="concurrency-disclaimer">{report.classification}. {report.background_load_conditions || 'Background load conditions were not reported.'}</p>
      {report.aggregation_note && <p className="concurrency-disclaimer">Report note: {report.aggregation_note}</p>}
    </div> : <p className="concurrency-error">{error || 'The concurrency report is being loaded.'}</p>}
  </details>;
}

function Sidebar({ tenant, tenants, onTenant, backend, view, onView }) {
  const embeddingProvider = String(backend?.embedding_provider || backend?.embedding_mode || 'unknown').toLowerCase();
  const embeddingLabel = embeddingProvider.includes('fastembed') ? 'FastEmbed' : embeddingProvider.includes('lexical') ? 'Lexical fallback' : embeddingProvider.includes('demo-hash') ? 'Demo hash fallback' : humanize(embeddingProvider);
  const agentStatus = llmStatus(backend);
  return <aside className="sidebar">
    <div className="brand"><span className="brand-mark">R</span><span className="brand-name">relayops</span></div>
    <label className="workspace-label" htmlFor="tenant-select">WORKSPACE</label>
    <div className="workspace-switch"><span className="tenant-dot" /><select id="tenant-select" value={tenant} onChange={(e) => onTenant(e.target.value)}>{tenants.length ? tenants.map((t) => <option key={t.id} value={t.id}>{t.name}</option>) : <option value={tenant}>{tenant}</option>}</select><span className="chevron">⌄</span></div>
    <div><p className="side-label">OPERATIONS</p><nav className="nav-list" aria-label="Workspace views"><button className={`nav-item ${view === 'incident' ? 'active' : ''}`} aria-label="Incident room" aria-current={view === 'incident' ? 'page' : undefined} onClick={() => onView('incident')}><span className="nav-icon">◫</span><span><span className="nav-label-long">Incident room</span><span className="nav-label-short">Incident</span></span></button><button className={`nav-item ${view === 'timeline' ? 'active' : ''}`} aria-current={view === 'timeline' ? 'page' : undefined} onClick={() => onView('timeline')}><span className="nav-icon">◷</span><span>Timeline</span></button><button className={`nav-item ${view === 'memory' ? 'active' : ''}`} aria-current={view === 'memory' ? 'page' : undefined} onClick={() => onView('memory')}><span className="nav-icon">⌘</span><span>Memory</span></button><button className={`nav-item ${view === 'runbooks' ? 'active' : ''}`} aria-current={view === 'runbooks' ? 'page' : undefined} onClick={() => onView('runbooks')}><span className="nav-icon">≡</span><span>Runbooks</span></button></nav></div>
    <div className="sidebar-spacer" />
    <div className="db-card"><div className="db-card-head"><span>MEMORY ENGINE</span><span className={`engine-dot ${backend?.mode === 'falkordb' && backend?.db_connected ? 'connected' : ''}`} /></div><strong>{backend?.mode === 'falkordb' && backend?.db_connected ? 'FalkorDB' : 'Demo / degraded'}</strong><div className="db-meta"><span>Graph</span><span>{backend?.db_connected ? 'Connected' : 'Unavailable'}</span></div><div className="db-meta"><span>Encoder</span><span title={embeddingProvider}>{embeddingLabel}</span></div><div className="db-meta"><span>Agents</span><span title={agentStatus.title}>{agentStatus.sidebar}</span></div></div>
    <div className="sidebar-foot"><span className="avatar">ON</span><span><strong>On-call engineer</strong><small>Current shift</small></span><span className="more">···</span></div>
  </aside>;
}

function AmbientShader({ theme }) {
  const canvasRef = useRef(null);
  const cursorFieldRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas?.getContext('webgl', { alpha: false, antialias: false, powerPreference: 'low-power' });
    if (!gl) return undefined;
    const vertexSource = 'attribute vec2 position; void main(){ gl_Position=vec4(position,0.0,1.0); }';
    const fragmentSource = `
      precision mediump float;
      uniform vec2 resolution;
      uniform float time;
      uniform vec3 colorA;
      uniform vec3 colorB;
      uniform vec3 colorC;
      uniform vec3 colorBase;
      uniform vec2 cursorPosition;
      uniform float cursorStrength;
      uniform vec2 cursorMotion;
      uniform float cursorRadius;
      float hash(vec2 q) {
        return fract(sin(dot(q, vec2(127.1, 311.7))) * 43758.5453);
      }
      float noise(vec2 q) {
        vec2 i = floor(q);
        vec2 f = fract(q);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
          mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
      }
      float fbm(vec2 q) {
        float value = 0.0;
        float amplitude = 0.55;
        mat2 turn = mat2(0.8, 0.6, -0.6, 0.8);
        for (int octave = 0; octave < 3; octave++) {
          value += amplitude * noise(q);
          q = turn * q * 2.03;
          amplitude *= 0.5;
        }
        return value;
      }
      void main() {
        float scale = min(resolution.x, resolution.y);
        vec2 p = (gl_FragCoord.xy - 0.5 * resolution) / scale;
        vec2 cursor = (cursorPosition - vec2(0.5)) * resolution / scale;
        vec2 delta = p - cursor;
        vec2 travel = cursorMotion * resolution / scale;
        float contact = exp(-dot(delta, delta) / (2.0 * cursorRadius * cursorRadius)) * cursorStrength;
        // A soft local displacement lets the existing color field respond to
        // contact without drawing a second cursor or a visible circular rim.
        vec2 drift = travel / (1.0 + length(travel));
        vec2 fieldPoint = p + delta * contact * 0.10 + drift * contact * cursorRadius * 0.10;
        float t = time * 0.11;
        vec2 flow = vec2(t * 0.68, -t * 0.48);
        float h = fbm(fieldPoint * 1.55 + flow + 0.20 * fbm(fieldPoint * 2.7 - flow * 0.55));
        float h2 = fbm(fieldPoint * 2.0 + vec2(-t * 0.44, t * 0.58) + 12.7);
        vec3 field = mix(colorA, colorB, smoothstep(0.34, 0.78, h2));
        field = mix(field, colorC, smoothstep(0.44, 0.86, (h + h2) * 0.5) * 0.62);
        vec3 color = mix(colorBase, field, 0.20 + 0.62 * smoothstep(0.24, 0.86, h));
        float contour = 1.0 - smoothstep(0.008, 0.045, abs(fract(h * 9.0 - t * 0.025) - 0.5));
        float secondContour = 1.0 - smoothstep(0.008, 0.048, abs(fract(h2 * 7.0 + t * 0.018) - 0.5));
        color += field * (contour * 0.043 + secondContour * 0.025);
        float bloom = exp(-dot(p - vec2(0.0, 0.12), p - vec2(0.0, 0.12)) * 3.2);
        color += field * bloom * 0.045;
        color = mix(color, mix(colorA, colorBase, 0.55), contact * 0.055);
        float vignette = smoothstep(1.1, 0.3, length(p));
        color *= mix(0.96, 1.0, vignette);
        gl_FragColor = vec4(color, 1.0);
      }
    `;
    let program;
    let buffer;
    let frame = 0;
    let running = false;
    let lastDraw = 0;
    const cursorTarget = { x: 0.5, y: 0.5, active: 0 };
    const cursor = { x: 0.5, y: 0.5, vx: 0, vy: 0, active: 0 };
    let lastPhysicsAt = performance.now();
    let lastPointerAt = 0;
    let pointerInside = false;
    let uniforms;
    const compile = (kind, source) => {
      const shader = gl.createShader(kind);
      if (!shader) throw new Error('Unable to allocate ambient shader.');
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const message = gl.getShaderInfoLog(shader) || 'Ambient shader compilation failed.';
        gl.deleteShader(shader);
        throw new Error(message);
      }
      return shader;
    };
    try {
      const vertex = compile(gl.VERTEX_SHADER, vertexSource);
      const fragment = compile(gl.FRAGMENT_SHADER, fragmentSource);
      program = gl.createProgram();
      if (!program) throw new Error('Unable to create ambient shader program.');
      gl.attachShader(program, vertex);
      gl.attachShader(program, fragment);
      gl.linkProgram(program);
      gl.deleteShader(vertex);
      gl.deleteShader(fragment);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || 'Ambient shader linking failed.');
      gl.useProgram(program);
      uniforms = Object.fromEntries(['resolution','time','colorA','colorB','colorC','colorBase','cursorPosition','cursorStrength','cursorMotion','cursorRadius'].map((name) => [name, gl.getUniformLocation(program, name)]));
      buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
      const position = gl.getAttribLocation(program, 'position');
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      const palettes = {
        plum: [[.67,.52,.77],[.53,.70,.84],[.86,.55,.72],[.96,.945,.965]],
        aurora: [[.37,.72,.63],[.51,.63,.85],[.74,.66,.89],[.925,.96,.95]],
        'blue-hour': [[.47,.60,.84],[.39,.73,.82],[.68,.60,.85],[.925,.95,.98]],
      };
      const colors = palettes[theme] || palettes.plum;
      ['colorA','colorB','colorC','colorBase'].forEach((name, index) => gl.uniform3fv(uniforms[name], new Float32Array(colors[index])));
    } catch {
      if (program) gl.deleteProgram(program);
      if (buffer) gl.deleteBuffer(buffer);
      return undefined;
    }

    const resize = () => {
      const ratio = Math.min(window.devicePixelRatio || 1, 1.25);
      const width = Math.max(1, Math.round(window.innerWidth * ratio));
      const height = Math.max(1, Math.round(window.innerHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, width, height);
      }
    };
    const updateCursorField = () => {
      const field = cursorFieldRef.current;
      if (!field) return;
      const speed = Math.hypot(cursor.vx * window.innerWidth, cursor.vy * window.innerHeight);
      const stretch = motionQuery.matches ? 0 : Math.min(speed / 6000, 0.10);
      const angle = Math.atan2(-cursor.vy * window.innerHeight, cursor.vx * window.innerWidth);
      field.style.transform = `translate3d(${cursor.x * window.innerWidth}px, ${(1 - cursor.y) * window.innerHeight}px, 0) translate(-50%, -50%) rotate(${angle}rad) scale(${1 + stretch}, ${1 - stretch * 0.45})`;
      field.style.opacity = String(cursor.active);
    };
    const advanceCursor = (timestamp) => {
      const delta = Math.min(0.05, Math.max(0.001, (timestamp - lastPhysicsAt) / 1000));
      lastPhysicsAt = timestamp;
      if (motionQuery.matches) {
        cursor.x = cursorTarget.x;
        cursor.y = cursorTarget.y;
        cursor.vx = 0;
        cursor.vy = 0;
        cursor.active = 0;
      } else {
        // Small fixed substeps keep the near-critical spring equally calm
        // on high refresh displays and after a delayed animation frame.
        const steps = Math.ceil(delta / (1 / 120));
        const step = delta / steps;
        for (let index = 0; index < steps; index++) {
          cursor.vx += ((cursorTarget.x - cursor.x) * 520 - cursor.vx * 42) * step;
          cursor.vy += ((cursorTarget.y - cursor.y) * 520 - cursor.vy * 42) * step;
          cursor.x += cursor.vx * step;
          cursor.y += cursor.vy * step;
        }
        // Contact brightens with movement, then settles to a quiet presence.
        const resting = timestamp - lastPointerAt > 160;
        const strength = cursorTarget.active * (resting ? 0.30 : 0.85);
        cursor.active += (strength - cursor.active) * (1 - Math.exp(-10 * delta));
        if (Math.abs(cursorTarget.x - cursor.x) < 0.0001 && Math.abs(cursor.vx) < 0.001) { cursor.x = cursorTarget.x; cursor.vx = 0; }
        if (Math.abs(cursorTarget.y - cursor.y) < 0.0001 && Math.abs(cursor.vy) < 0.001) { cursor.y = cursorTarget.y; cursor.vy = 0; }
      }
      updateCursorField();
    };
    const draw = (time) => {
      resize();
      gl.useProgram(program);
      gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
      gl.uniform1f(uniforms.time, time);
      gl.uniform2f(uniforms.cursorPosition, cursor.x, cursor.y);
      gl.uniform1f(uniforms.cursorStrength, cursor.active);
      gl.uniform2f(uniforms.cursorMotion, cursor.vx, cursor.vy);
      gl.uniform1f(uniforms.cursorRadius, 28 / Math.max(1, Math.min(window.innerWidth, window.innerHeight)));
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const stop = () => { running = false; cancelAnimationFrame(frame); frame = 0; };
    const animate = (timestamp) => {
      if (!running || document.hidden) return;
      advanceCursor(timestamp);
      if (timestamp - lastDraw >= 33) {
        draw(timestamp * 0.001);
        lastDraw = timestamp;
      }
      frame = requestAnimationFrame(animate);
    };
    const start = () => {
      if (running || document.hidden || motionQuery.matches) return;
      running = true;
      lastPhysicsAt = performance.now();
      frame = requestAnimationFrame(animate);
    };
    const onMotionChange = () => {
      stop();
      if (motionQuery.matches) { advanceCursor(performance.now()); draw(0); }
      else start();
    };
    const onPointerMove = (event) => {
      if (event.pointerType === 'touch') return;
      cursorTarget.x = Math.max(0, Math.min(1, event.clientX / window.innerWidth));
      cursorTarget.y = Math.max(0, Math.min(1, 1 - event.clientY / window.innerHeight));
      // Enter at the pointer, rather than flying in from the old location.
      if (!pointerInside) {
        cursor.x = cursorTarget.x;
        cursor.y = cursorTarget.y;
        cursor.vx = 0;
        cursor.vy = 0;
      }
      pointerInside = true;
      lastPointerAt = performance.now();
      cursorTarget.active = 1;
      if (motionQuery.matches) { advanceCursor(performance.now()); draw(0); }
    };
    const onPointerLeave = (event) => {
      if (event.relatedTarget == null) {
        pointerInside = false;
        cursorTarget.active = 0;
        if (motionQuery.matches) { advanceCursor(performance.now()); draw(0); }
      }
    };
    const onWindowBlur = () => {
      pointerInside = false;
      cursorTarget.active = 0;
      if (motionQuery.matches) { advanceCursor(performance.now()); draw(0); }
    };
    const onVisibilityChange = () => {
      if (document.hidden) stop();
      else if (motionQuery.matches) { advanceCursor(performance.now()); draw(0); }
      else start();
    };
    window.addEventListener('resize', resize);
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('pointerout', onPointerLeave, { passive: true });
    window.addEventListener('blur', onWindowBlur);
    document.addEventListener('visibilitychange', onVisibilityChange);
    motionQuery.addEventListener?.('change', onMotionChange);
    resize();
    advanceCursor(performance.now());
    draw(0);
    if (!motionQuery.matches) start();
    return () => {
      stop();
      window.removeEventListener('resize', resize);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerout', onPointerLeave);
      window.removeEventListener('blur', onWindowBlur);
      document.removeEventListener('visibilitychange', onVisibilityChange);
      motionQuery.removeEventListener?.('change', onMotionChange);
      if (program) gl.deleteProgram(program);
      if (buffer) gl.deleteBuffer(buffer);
    };
  }, [theme]);
  return <>
    <canvas ref={canvasRef} className="ambient-shader" aria-hidden="true" />
    <div ref={cursorFieldRef} className="cursor-force-field" aria-hidden="true" />
  </>;
}

function MobileBottomNav({ view, onView }) {
  const items = [
    ['incident', '◫', 'Incident', 'Incident room'],
    ['timeline', '◷', 'Timeline', 'Timeline'],
    ['memory', '⌘', 'Memory', 'Memory'],
    ['runbooks', '≡', 'Runbooks', 'Runbooks'],
  ];
  return <nav className="mobile-bottom-nav" aria-label="Workspace views">{items.map(([id, icon, label, accessibleLabel]) => <button key={id} className={view === id ? 'active' : ''} aria-label={accessibleLabel} aria-current={view === id ? 'page' : undefined} onClick={() => onView(id)}><span className="mobile-bottom-icon" aria-hidden="true">{icon}</span><span>{label}</span></button>)}</nav>;
}

function Topbar({ tenantName, backend, view }) {
  const connected = backend?.mode === 'falkordb' && backend?.db_connected;
  const embeddingProvider = String(backend?.embedding_provider || backend?.embedding_mode || 'unknown').toLowerCase();
  const agentStatus = llmStatus(backend);
  const encoder = embeddingProvider.includes('fastembed') ? 'FastEmbed' : embeddingProvider.includes('lexical') ? 'Lexical fallback' : embeddingProvider.includes('demo-hash') ? 'Demo hash fallback' : humanize(embeddingProvider);
  const encoderShort = embeddingProvider.includes('fastembed') ? 'FastEmbed' : embeddingProvider.includes('lexical') ? 'Lexical' : embeddingProvider.includes('demo-hash') ? 'Hash fallback' : humanize(embeddingProvider);
  const viewTitle = { incident: 'Incident room', timeline: 'Timeline', memory: 'Memory', runbooks: 'Runbooks' }[view] || 'Incident room';
  return <header className="topbar"><div className="breadcrumbs"><span>Operations</span><span>／</span><strong>{viewTitle}</strong></div><div className="top-actions"><span className={`connection ${connected ? 'good' : 'degraded'}`} role="status" aria-label={connected ? 'FalkorDB connected' : 'Demo / degraded memory engine'} title={connected ? 'FalkorDB connected' : 'Demo / degraded memory engine'}><i aria-hidden="true" /><span className="connection-label">{connected ? 'FalkorDB connected' : 'Demo / degraded'}</span></span><span className="top-divider" /><span className="encoder-mode" role="img" aria-label={`Embedding provider: ${encoder}`} title={`Embedding provider: ${encoder}`}><span className="encoder-full" aria-hidden="true">Embed · {encoder}</span><span className="encoder-short" aria-hidden="true">Embed · {encoderShort}</span></span><span className="llm-mode" title={agentStatus.title}>{agentStatus.topbar}</span><span className="avatar">ON</span></div></header>;
}

function IncidentBanner({ incident, title, severity, status, facts, task, taskStale, taskComplete }) {
  const openHandoffs = task && !isTerminalTask(task.status) ? 1 : 0;
  const handoffLabel = !task ? 'No handoff' : taskComplete ? 'Handoff complete' : taskStale ? 'Refresh required' : task.owner ? 'Handoff claimed' : `Handoff ${humanize(task.status || 'open')}`;
  const handoffTone = taskComplete ? 'tag-green' : taskStale ? 'tag-red' : 'tag-blue';
  return <div className="incident-banner"><div className="incident-severity">⚠</div><div className="incident-info"><div className="incident-title-row"><span className="incident-title">{title}</span><span className="tag tag-amber">{severity}</span><span className={`tag ${String(status).toLowerCase().includes('resolved') ? 'tag-green' : 'tag-blue'}`}>{humanize(status)}</span><span className={`tag ${handoffTone}`}>{handoffLabel}</span></div><div className="incident-meta"><span>{incident.id || incident.incident_id || 'Active incident'}</span><span>·</span><span>{incident.service_name || incident.service || incident.service_id || 'Checkout'}</span><span>·</span><span>{incident.started_at ? `Started ${timeOf({ created_at: incident.started_at })}` : 'Active shift'}</span></div>{incident.summary && <div className="incident-summary">{incident.summary}</div>}</div><div className="incident-stats"><div className="stat"><strong>{facts}</strong><small>current facts</small></div><div className="stat"><strong>{openHandoffs}</strong><small>open handoffs</small></div></div></div>;
}

function Card({ title, icon, meta, action, children }) {
  return <article className="card"><div className="card-head"><div className="card-title"><span className="section-icon">{icon}</span><div><h2>{title}</h2>{meta && <small className="subtle">{meta}</small>}</div></div>{action}</div><div className="card-inner">{children}</div></article>;
}

function TimelineEvent({ event, index }) {
  const type = String(event.kind || event.type || event.actor || '').toLowerCase();
  const user = type.includes('user') || type.includes('correction');
  const system = type.includes('system') || type.includes('handoff') || type.includes('task');
  const detail = String(event.detail || event.text || event.description || event.summary || event.message || JSON.stringify(event));
  return <div className="timeline-item"><div className={`timeline-marker ${user ? 'user' : system ? 'system' : ''}`}>{user ? '↺' : system ? '⇄' : '•'}</div><div className="timeline-copy"><strong>{event.label || event.title || event.agent || event.actor || humanize(type || 'Incident event')}</strong>{detail.length > 160 ? <details className="event-detail"><summary>{detail.slice(0, 140).trimEnd()}… <span>Read full event</span></summary><p>{detail}</p></details> : <p>{detail}</p>}{event.source_ref && <small className="source-label">Source · {event.source_ref}</small>}</div><span className="timeline-time">{timeOf(event)}</span></div>;
}

function MemoryPanel({ tab, current, old, evidence, categories, onCorrect }) {
  if (tab === 'evidence') {
    const rows = Array.isArray(evidence) ? evidence : [];
    return <div className="memory-list">{rows.length ? rows.map((item, i) => <MemoryItem key={idOf(item) || i} item={item} categories={categories} showRetrieval />) : <Empty title="No retrieved evidence yet" detail="Search memory or run Investigator to expose the sources behind the handoff." />}</div>;
  }
  const rows = tab === 'history' ? old : current;
  return <div className="memory-list">{rows.length ? rows.map((item, i) => <MemoryItem key={idOf(item) || i} item={item} categories={categories} old={tab === 'history'} onCorrect={onCorrect} />) : <Empty title={tab === 'history' ? 'No superseded facts' : 'No current facts'} detail={tab === 'history' ? 'Older revisions will appear here after a correction.' : 'Run Triage or Investigator to populate incident memory.'} />}</div>;
}

function MemoryItem({ item, categories, old, onCorrect, showRetrieval = false }) {
  const kind = item.kind || 'fact';
  const related = (item.category_ids || []).map((id) => categories.find((c) => c.id === id)?.name || id).filter(Boolean);
  const relations = Array.isArray(item.relations) ? item.relations.map((rel) => typeof rel === 'string' ? rel : [rel.type || rel.label || rel.name, rel.target || rel.name || rel.node].filter(Boolean).join(' → ')).filter(Boolean) : [];
  const visibleText = item.key ? `${humanize(item.key)}: ${textOf(item)}` : textOf(item) || titleOf(item);
  const origin = showRetrieval ? retrievalOrigin(item) : '';
  const sources = (Array.isArray(item.match_sources) ? item.match_sources : []).map((source) => String(source).toLowerCase());
  const ranks = item.retrieval_ranks || {};
  const hasHnsw = sources.includes('hnsw');
  const hasBm25 = sources.includes('bm25');
  const rankingDetails = showRetrieval ? [
    hasHnsw && item.score != null ? `HNSW cosine similarity ${Number(item.score).toFixed(3)}${ranks.hnsw ? ` · rank ${ranks.hnsw}` : ''}` : null,
    hasBm25 && item.bm25_score != null ? `BM25 raw score ${Number(item.bm25_score).toFixed(3)}${ranks.bm25 ? ` · rank ${ranks.bm25}` : ''}` : null,
    item.rrf_score != null ? `RRF fusion score ${Number(item.rrf_score).toFixed(5)}` : null,
  ].filter(Boolean) : [];
  return <div className={`memory-item ${old ? 'superseded' : ''}`}><div className="memory-meta"><span className="memory-kind"><i className={`kind-dot kind-${kind}`} />{humanize(kind)}</span><span className={`tag ${old ? 'tag-muted' : 'tag-green'}`}>{old ? 'Superseded' : 'Current'}</span></div><p>{visibleText}</p>{origin && <div className="retrieval-origin" title={`Retrieved by ${origin}`}><span aria-hidden="true">⌁</span>{origin}</div>}{rankingDetails.length > 0 && <details className="retrieval-details"><summary>Ranking details</summary><ul>{rankingDetails.map((detail) => <li key={detail}>{detail}</li>)}</ul></details>}<div className="memory-source"><span>{item.source_ref ? `Source · ${item.source_ref}` : 'Incident memory'}</span><span>{timeOf(item)}</span></div>{related.length > 0 && <div className="memory-categories">{related.map((name) => <span key={name}>{name}</span>)}</div>}{relations.length > 0 && <div className="relation-list"><span>Graph links</span>{relations.map((relation, i) => <span key={`${relation}-${i}`}>{relation}</span>)}</div>}{!old && onCorrect && kind === 'fact' && <button className="text-action" onClick={() => onCorrect(item)}>Correct fact ↗</button>}</div>;
}

function Empty({ title, detail }) { return <div className="empty-state"><span>◇</span><strong>{title}</strong><p>{detail}</p></div>; }
