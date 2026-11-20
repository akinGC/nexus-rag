'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const state = { health: null, token: '', role: 'employee', sessionId: null, histories: new Map(), documents: [], view: 'ask', generation: 0, chatBusy: false, controller: null, epochController: new AbortController() };
const modes = { vector: 'Vector', hybrid: 'Hybrid', hybrid_rerank: 'Hybrid + rerank', all: 'All modes' };
const metricLabels = { recall_at_k: 'Recall @ K', precision_at_k: 'Precision @ K', mrr: 'MRR', ndcg: 'NDCG', citation_accuracy: 'Citation accuracy', answer_correctness: 'Answer correctness', avg_latency_ms: 'Average latency', failure_rate: 'Failure rate', faithfulness: 'Faithfulness' };

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}
function display(value) { return typeof value === 'object' && value !== null ? JSON.stringify(value, null, 2) : String(value ?? '—'); }
function list(value) { return Array.isArray(value) ? value : []; }
function icon(name) { const element = node('i'); element.dataset.lucide = name; element.setAttribute('aria-hidden', 'true'); return element; }
function refreshIcons() { if (window.lucide) { window.lucide.createIcons(); document.body.classList.add('icons-ready'); } }
function action(label, iconName, callback, className = 'button secondary compact') {
  const button = node('button', className); button.type = 'button'; button.append(icon(iconName), node('span', '', label)); button.addEventListener('click', callback); return button;
}
function empty(container, message, error = false, retry) {
  const block = node('div', `empty-state${error ? ' error' : ''}`); block.append(node('p', '', message));
  if (retry) block.append(action('Retry', 'refresh-cw', retry));
  container.replaceChildren(block); refreshIcons();
}
function feedback(selector, text, error = false) { const element = $(selector); element.textContent = text; element.classList.toggle('error', error); }
function date(value) { if (!value) return 'Date unavailable'; const parsed = new Date(value); return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }); }
function safeUrl(value) { try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; } catch { return null; } }
function sourceLocation(source) { return [source.section, source.page != null ? `Page ${source.page}` : null, source.paragraph != null ? `Paragraph ${source.paragraph}` : null].filter(Boolean).join(' · '); }
function historyKey() { return `${state.role}:${state.sessionId ?? 'new'}`; }
function history() { const key = historyKey(); if (!state.histories.has(key)) state.histories.set(key, []); return state.histories.get(key); }
function canRequest() { return !!state.health && (state.health.mode === 'local' || !!state.token); }

async function api(path, options = {}) {
  if (!state.health) throw new Error('Workspace is not connected.');
  if (state.health.mode === 'production' && !state.token) throw new Error('A production token is required.');
  const headers = new Headers(options.headers);
  if (state.health.mode === 'local') headers.set('X-Demo-User', state.role);
  else headers.set('Authorization', `Bearer ${state.token}`);
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  const response = await fetch(path, { ...options, headers, signal: options.signal ?? state.epochController.signal });
  let data;
  try { data = await response.json(); } catch { if (!response.ok) throw new Error(`Request failed (${response.status}).`); return {}; }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : typeof data.error === 'string' ? data.error : `Request failed (${response.status}).`);
  return data;
}

async function connect() {
  $('#mode-badge').textContent = 'Connecting'; $('#retry-health').disabled = true;
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error(`Health check failed (${response.status}).`);
    const health = await response.json();
    if (!['local', 'production'].includes(health.mode)) throw new Error('The server returned an unknown authentication mode.');
    if (state.health && health.mode !== state.health.mode) resetIdentity();
    state.health = health;
    $('#mode-badge').textContent = `${health.mode} mode`; $('#mode-badge').classList.remove('error');
    $('#identity-control').hidden = health.mode !== 'local'; $('#token-control').hidden = health.mode !== 'production';
    $('#sidebar-status').textContent = display(health.status); $('#footer-status').textContent = `${health.mode === 'local' ? 'Local' : 'Production'} workspace · ${display(health.provider)}`;
    $('#document-count').textContent = display(health.documents); $('#chunk-count').textContent = display(health.chunks);
    $('#answer-provider').textContent = display(health.provider); $('#embedding-provider').textContent = display(health.embedding_provider);
    $('#global-error').hidden = true;
    if (canRequest()) { loadMemory(); loadCurrentView(); }
    else { empty($('#preferences'), 'Not connected'); renderAccessState(); }
    $('#chat-status').textContent = canRequest() ? 'Ready' : 'Awaiting production token';
  } catch (error) {
    state.health = null; $('#mode-badge').textContent = 'Offline'; $('#mode-badge').classList.add('error');
    $('#identity-control').hidden = true; $('#token-control').hidden = true;
    $('#sidebar-status').textContent = 'Disconnected'; $('#global-error-text').textContent = error.message; $('#global-error').hidden = false;
    $('#chat-status').textContent = 'Disconnected'; empty($('#preferences'), 'Connection unavailable'); renderAccessState();
  } finally { $('#retry-health').disabled = false; }
}

function resetIdentity() {
  state.generation += 1; state.controller?.abort(); state.epochController.abort(); state.epochController = new AbortController();
  state.sessionId = null; state.chatBusy = false; state.documents = [];
  $('#send-question').disabled = false; $('#question').value = ''; $('#source-dialog').close();
  $('#messages').setAttribute('aria-busy', 'false'); renderMessages(); $('#conversation-label').textContent = 'New conversation';
  $('#chat-status').textContent = 'Ready'; $('#memory-status').textContent = '';
  feedback('#index-status', ''); $('#index-report').replaceChildren();
  empty($('#documents'), 'Loading documents…'); empty($('#evaluation-runs'), 'Loading evaluation runs…'); empty($('#evaluation-dataset'), 'Loading dataset…');
}
function renderAccessState() {
  const message = state.health ? 'Production token required.' : 'Connection unavailable.';
  empty($('#documents'), message); empty($('#evaluation-runs'), message); empty($('#evaluation-dataset'), message);
}
function navigate(view) {
  if (!['ask', 'knowledge', 'evaluation'].includes(view)) view = 'ask';
  state.view = view;
  document.querySelectorAll('.view').forEach(element => { element.hidden = element.id !== `view-${view}`; });
  document.querySelectorAll('[data-view]').forEach(element => { const active = element.dataset.view === view; element.classList.toggle('active', active); if (active) element.setAttribute('aria-current', 'page'); else element.removeAttribute('aria-current'); });
  $('#view-name').textContent = { ask: 'Ask', knowledge: 'Knowledge', evaluation: 'Evaluation' }[view];
  closeNavigation(); if (canRequest()) loadCurrentView(); else renderAccessState();
}
function loadCurrentView() { if (state.view === 'knowledge') loadDocuments(); if (state.view === 'evaluation') loadEvaluation(); }
function closeNavigation() { $('#sidebar').classList.remove('open'); $('#nav-scrim').hidden = true; $('#nav-toggle').setAttribute('aria-expanded', 'false'); }

function renderMessages() {
  const container = $('#messages'); container.replaceChildren();
  const messages = history();
  if (!messages.length) {
    const block = node('div', 'empty-chat'); const symbol = node('div', 'empty-symbol'); symbol.append(icon('messages-square'));
    block.append(symbol, node('h2', '', 'What would you like to know?'));
    container.append(block);
  }
  messages.forEach(message => {
    const block = node('article', `message ${message.role}`);
    const heading = node('div', 'message-heading', message.role === 'user' ? 'You' : message.role === 'error' ? 'Request failed' : 'Nexus');
    if (message.data) {
      const metadata = [];
      if (typeof message.data.evidence_support_percent === 'number') metadata.push(`Evidence support ${message.data.evidence_support_percent}%`);
      else if (typeof message.data.confidence === 'number') metadata.push(`Evidence support ${Math.round(message.data.confidence * 100)}%`);
      if (typeof message.data.trace?.latency_ms === 'number') metadata.push(`${Math.round(message.data.trace.latency_ms)} ms`);
      if (metadata.length) heading.append(node('span', 'answer-meta', metadata.join(' · ')));
    }
    block.append(heading, node('div', 'message-text', message.text));
    if (message.data) {
      const citations = list(message.data.citations);
      if (citations.length) {
        const sources = node('div', 'source-cards');
        citations.forEach(citation => {
          const button = node('button', 'source-card'); button.type = 'button'; const description = node('span');
          description.append(node('strong', '', citation.title || 'Untitled source'), node('small', '', sourceLocation(citation) || citation.source || 'Source'));
          button.append(node('span', 'citation-number', `[${citation.number ?? '?'}]`), description); button.addEventListener('click', () => inspectCitation(citation)); sources.append(button);
        }); block.append(sources);
      }
      if (message.data.trace) block.append(renderTrace(message.data.trace));
    }
    container.append(block);
  });
  if (state.chatBusy) container.append(node('div', 'pending', 'Retrieving evidence…'));
  refreshIcons(); container.scrollTop = container.scrollHeight;
}

function renderTrace(trace) {
  const details = node('details', 'trace'); details.append(node('summary', '', 'Why this answer'));
  const content = node('div', 'trace-content');
  if (trace.rewritten_query) content.append(node('h3', '', 'Rewritten query'), node('p', '', trace.rewritten_query));
  if (list(trace.queries).length) { content.append(node('h3', '', 'Retrieval queries')); const queries = node('ul'); trace.queries.forEach(query => queries.append(node('li', '', display(query)))); content.append(queries); }
  const evidence = list(trace.reranked).length ? trace.reranked : trace.retrieved;
  content.append(node('h3', '', list(trace.reranked).length ? 'Reranked evidence' : 'Retrieved evidence'));
  if (!list(evidence).length) content.append(node('p', '', 'No evidence returned.'));
  list(evidence).forEach((item, index) => {
    const row = node('div', 'evidence-row'); row.append(node('p', '', `${index + 1}. ${item.title || item.document_id || 'Source'}`));
    const metadata = [sourceLocation(item), item.score != null ? `Score ${display(item.score)}` : '', item.rerank_score != null ? `Rerank ${display(item.rerank_score)}` : ''].filter(Boolean);
    if (metadata.length) row.append(node('p', 'answer-meta', metadata.join(' · ')));
    if (item.text) row.append(node('p', '', item.text));
    row.append(action('Inspect evidence', 'file-search', () => inspectCitation(item))); content.append(row);
  });
  if (list(trace.claims).length) { content.append(node('h3', '', 'Claim support')); trace.claims.forEach(claim => { const row = node('div', 'claim-row'); const supported = claim.supported === true; row.append(node('span', `claim-status${supported ? '' : ' unsupported'}`, `${supported ? 'Supported' : claim.supported === false ? 'Unsupported' : 'Not verified'}${list(claim.citation_numbers).length ? ` · ${claim.citation_numbers.map(number => `[${number}]`).join(' ')}` : ''}`), node('p', '', claim.claim)); content.append(row); }); }
  if (trace.verification) {
    const verificationStatus = trace.verification.question_fully_answered === false
      ? 'Partial answer; some requested details are missing'
      : trace.verification.supported === true ? 'Supported and complete'
        : trace.verification.supported === false ? 'Not fully supported' : 'Not verified';
    content.append(node('h3', '', 'Verification'), node('p', '', `${verificationStatus} · ${trace.verification.method || 'Method unavailable'}`));
    if (list(trace.verification.missing_information).length) {
      content.append(node('h3', '', 'Missing from draft'));
      const missing = node('ul');
      trace.verification.missing_information.forEach(item => missing.append(node('li', '', item)));
      content.append(missing);
    }
  }
  const timings = Object.entries(trace).filter(([key, value]) => /(?:_ms$|timings)/.test(key) && value != null);
  if (timings.length) { content.append(node('h3', '', 'Timings')); timings.forEach(([key, value]) => content.append(node('p', 'answer-meta', `${key.replaceAll('_', ' ')}: ${display(value)}`))); }
  details.append(content); return details;
}

$('#chat-form').addEventListener('submit', async event => {
  event.preventDefault(); const question = $('#question').value.trim(); if (!question || state.chatBusy) return;
  if (!canRequest()) { feedback('#chat-status', state.health ? 'Production token required.' : 'Connection unavailable.', true); return; }
  const generation = state.generation; const messages = history(); messages.push({ role: 'user', text: question });
  state.chatBusy = true; state.controller = new AbortController(); $('#question').value = ''; $('#send-question').disabled = true;
  $('#messages').setAttribute('aria-busy', 'true'); feedback('#chat-status', 'Working'); renderMessages();
  try {
    const data = await api('/api/chat', { method: 'POST', signal: state.controller.signal, body: JSON.stringify({ question, session_id: state.sessionId, mode: $('#retrieval-mode').value }) });
    if (generation !== state.generation) return;
    const oldKey = historyKey();
    if (data.session_id) { state.sessionId = data.session_id; state.histories.delete(oldKey); state.histories.set(historyKey(), messages); }
    messages.push({ role: 'assistant', text: data.answer ?? 'No answer returned.', data });
    $('#conversation-label').textContent = 'Active conversation'; feedback('#chat-status', 'Ready');
  } catch (error) {
    if (generation !== state.generation || error.name === 'AbortError') return;
    messages.push({ role: 'error', text: error.message }); $('#question').value = question; feedback('#chat-status', 'Request failed', true);
  } finally {
    if (generation === state.generation) { state.chatBusy = false; $('#send-question').disabled = false; $('#messages').setAttribute('aria-busy', 'false'); renderMessages(); $('#question').focus(); }
  }
});

function sourceContent(source) {
  const block = node('div'); const metadata = node('div', 'source-meta');
  [source.section, source.page != null ? `Page ${source.page}` : null, source.paragraph != null ? `Paragraph ${source.paragraph}` : null, source.chunk_id, source.source].filter(Boolean).forEach(value => metadata.append(node('span', '', value)));
  block.append(metadata); const url = safeUrl(source.url) || safeUrl(source.source);
  if (url) { const link = node('a', 'source-link', 'Open original source'); link.href = url; link.target = '_blank'; link.rel = 'noopener noreferrer'; block.append(link); }
  block.append(node('div', 'source-text', source.text ?? 'Source text unavailable.')); return block;
}
let inspectorVersion = 0;
async function inspectCitation(citation) {
  const version = ++inspectorVersion; const generation = state.generation;
  $('#source-title').textContent = citation.title || 'Retrieved source'; $('#source-body').replaceChildren(sourceContent(citation)); $('#source-dialog').showModal();
  if (citation.document_id && citation.chunk_id) {
    const pending = node('div', 'pending', 'Loading full source…'); $('#source-body').append(pending);
    try { const data = await api(`/api/documents/${encodeURIComponent(citation.document_id)}`); if (version !== inspectorVersion || generation !== state.generation) return; const chunk = list(data.chunks).find(item => item.chunk_id === citation.chunk_id); if (chunk) $('#source-body').replaceChildren(sourceContent({ ...data.document, ...citation, ...chunk })); else pending.replaceWith(node('p', 'form-status', 'Full chunk unavailable; showing retrieved text.')); }
    catch (error) { if (version === inspectorVersion && generation === state.generation) pending.replaceWith(node('p', 'form-status error', error.message)); }
  }
}
async function inspectDocument(documentId) {
  const version = ++inspectorVersion; const generation = state.generation;
  $('#source-title').textContent = 'Document'; empty($('#source-body'), 'Loading document…'); $('#source-dialog').showModal();
  try {
    const data = await api(`/api/documents/${encodeURIComponent(documentId)}`); if (version !== inspectorVersion || generation !== state.generation) return;
    $('#source-title').textContent = data.document?.title || 'Document'; const body = $('#source-body'); body.replaceChildren();
    if (!list(data.chunks).length) empty(body, 'No indexed chunks.');
    list(data.chunks).forEach((chunk, index) => { body.append(node('h3', 'chunk-title', `CHUNK ${index + 1}`), sourceContent({ ...data.document, ...chunk })); });
  } catch (error) { if (version === inspectorVersion && generation === state.generation) empty($('#source-body'), error.message, true); }
}

async function loadDocuments() {
  const generation = state.generation; empty($('#documents'), 'Loading documents…');
  try { const data = await api('/api/documents'); if (generation !== state.generation) return; state.documents = list(data.documents); renderDocuments(); }
  catch (error) { if (generation === state.generation) empty($('#documents'), error.message, true, loadDocuments); }
}
function renderDocuments() {
  const container = $('#documents'); container.replaceChildren(); const search = $('#document-search').value.trim().toLowerCase();
  const documents = state.documents.filter(item => `${item.title ?? ''} ${item.department ?? ''}`.toLowerCase().includes(search)); $('#library-count').textContent = `${documents.length} / ${state.documents.length}`;
  if (!documents.length) { empty(container, search ? 'No matching documents.' : 'No documents available.'); return; }
  documents.forEach(item => {
    const card = node('article', 'document-card'); const preview = node('div', 'document-preview');
    preview.append(node('span', 'preview-title', item.title || item.source || 'Untitled document'), node('span', 'preview-type', item.source_type || 'DOCUMENT'));
    const information = node('div', 'document-info'); information.append(node('h3', '', item.title || 'Untitled document'), node('p', '', `${item.department || 'General'} · ${item.chunk_count ?? '—'} chunks`), node('p', '', `Roles: ${list(item.allowed_roles).join(', ') || 'Not specified'}`), node('p', '', date(item.created_at)), node('p', '', item.source || ''));
    const actions = node('div', 'document-actions'); actions.append(action('Inspect', 'file-search', () => inspectDocument(item.document_id)), action('Remove', 'trash-2', async event => {
      if (!window.confirm(`Remove "${item.title || 'this document'}"?`)) return;
      const button = event.currentTarget; button.disabled = true; const generation = state.generation;
      try { await api(`/api/documents/${encodeURIComponent(item.document_id)}`, { method: 'DELETE' }); if (generation === state.generation) { await loadDocuments(); await refreshHealthCounts(); } }
      catch (error) { if (generation === state.generation) { const alert = node('p', 'form-status error', error.message); alert.setAttribute('role', 'alert'); information.append(alert); } }
      finally { button.disabled = false; }
    }, 'button danger compact'));
    information.append(actions); card.append(preview, information); container.append(card);
  }); refreshIcons();
}
async function refreshHealthCounts() {
  const generation = state.generation;
  try { const response = await fetch('/api/health'); if (!response.ok) return; const data = await response.json(); if (generation !== state.generation) return; $('#document-count').textContent = display(data.documents); $('#chunk-count').textContent = display(data.chunks); }
  catch { $('#document-count').textContent = '—'; $('#chunk-count').textContent = '—'; }
}
function renderIndexReport(report) {
  const container = $('#index-report'); container.replaceChildren();
  for (const [key, label] of [['indexed', 'Indexed'], ['skipped', 'Skipped'], ['failed', 'Failed']]) {
    const entries = list(report[key]); const section = node('div');
    section.append(node('h3', '', `${label} (${entries.length})`));
    if (entries.length) {
      const items = node('ul');
      entries.forEach(entry => {
        const detail = key === 'indexed' ? `: ${display(entry.chunk_count)} chunks` : key === 'failed' ? `: ${display(entry.error)}` : '';
        items.append(node('li', key === 'failed' ? 'form-status error' : '', `${display(entry.source)}${detail}`));
      });
      section.append(items);
    }
    container.append(section);
  }
}
$('#index-pdfs').addEventListener('click', async event => {
  const button = event.currentTarget; if (button.disabled) return;
  const identitySignal = state.epochController.signal;
  button.disabled = true; $('#index-pdfs-label').textContent = 'Indexing PDFs...';
  $('#index-report').replaceChildren(); $('#index-report').setAttribute('aria-busy', 'true');
  feedback('#index-status', 'Indexing PDFs...');
  try {
    const started = await api('/api/index-documents', { method: 'POST', signal: identitySignal });
    if (identitySignal.aborted) return;
    feedback('#index-status', 'Indexing is running in the background.');
    let job;
    do {
      await new Promise(resolve => setTimeout(resolve, 1500));
      if (identitySignal.aborted) return;
      job = await api(`/api/index-documents/${encodeURIComponent(started.job_id)}`, { signal: identitySignal });
    } while (job.status === 'running');
    if (job.status === 'failed') throw new Error(job.error || 'PDF indexing failed.');
    const report = job.report || { indexed: [], skipped: [], failed: [] };
    renderIndexReport(report);
    feedback('#index-status', list(report.failed).length ? 'Indexing finished with errors.' : 'Indexing complete.', list(report.failed).length > 0);
  } catch (error) {
    if (!identitySignal.aborted) feedback('#index-status', error.message, true);
  } finally {
    if (!identitySignal.aborted) await Promise.all([loadDocuments(), refreshHealthCounts()]);
    button.disabled = false; $('#index-pdfs-label').textContent = 'Index PDFs';
    $('#index-report').setAttribute('aria-busy', 'false');
  }
});

async function loadMemory() {
  const generation = state.generation; empty($('#preferences'), 'Loading preferences…');
  try {
    const data = await api('/api/memory'); if (generation !== state.generation) return; const container = $('#preferences'); container.replaceChildren();
    if (!list(data.preferences).length) { empty(container, 'No saved preferences.'); return; }
    data.preferences.forEach(preference => {
      const row = node('div', 'preference'); const description = node('span'); description.append(node('strong', '', preference.key), node('span', '', display(preference.value)));
      row.append(description, action('Remove', 'x', async event => {
        const button = event.currentTarget; button.disabled = true;
        try { await api(`/api/memory/${encodeURIComponent(preference.key)}`, { method: 'DELETE' }); if (generation === state.generation) loadMemory(); }
        catch (error) { if (generation === state.generation) feedback('#memory-status', error.message, true); }
        finally { button.disabled = false; }
      }, 'button secondary compact')); container.append(row);
    }); refreshIcons();
  } catch (error) { if (generation === state.generation) empty($('#preferences'), error.message, true, loadMemory); }
}
$('#memory-form').addEventListener('submit', async event => {
  event.preventDefault(); const key = $('#preference-key').value.trim(); const value = $('#preference-value').value.trim(); if (!key || !value) return;
  const generation = state.generation; const button = $('button', event.currentTarget); button.disabled = true; feedback('#memory-status', 'Saving…');
  try { await api('/api/memory', { method: 'PUT', body: JSON.stringify({ key, value }) }); if (generation !== state.generation) return; feedback('#memory-status', 'Saved'); $('#preference-key').value = ''; $('#preference-value').value = ''; loadMemory(); }
  catch (error) { if (generation === state.generation) feedback('#memory-status', error.message, true); }
  finally { button.disabled = false; }
});

function dataTable(cases) {
  const wrapper = node('div', 'data-table-wrap'); if (!list(cases).length) { empty(wrapper, 'No cases returned.'); return wrapper; }
  const rows = cases.map(item => typeof item === 'object' && item !== null ? item : { case: item });
  const keys = [...new Set(rows.flatMap(item => Object.keys(item)))]; const table = node('table', 'data-table'); const header = node('thead'); const headerRow = node('tr');
  keys.forEach(key => headerRow.append(node('th', '', key.replaceAll('_', ' ')))); header.append(headerRow); const body = node('tbody');
  rows.forEach(item => { const row = node('tr'); keys.forEach(key => row.append(node('td', '', display(item[key])))); body.append(row); }); table.append(header, body); wrapper.append(table); return wrapper;
}
function renderRuns(runs) {
  const container = $('#evaluation-runs'); container.replaceChildren(); if (!list(runs).length) { empty(container, 'No evaluation runs yet.'); return; }
  runs.forEach(run => {
    const block = node('article', 'run'); const heading = node('div', 'run-header'); heading.append(node('h3', '', modes[run.mode] || run.mode || 'Evaluation'), node('span', '', `${date(run.created_at)} · ${run.run_id || 'Run ID unavailable'}`));
    const metrics = node('div', 'metrics'); Object.entries(metricLabels).forEach(([key, label]) => {
      const metric = node('div', 'metric'); const value = run.metrics?.[key]; const formatted = typeof value === 'number' && Number.isFinite(value) ? key === 'avg_latency_ms' ? `${Math.round(value)} ms` : value.toFixed(3) : '—';
      metric.append(node('span', '', label), node('strong', '', formatted)); metrics.append(metric);
    }); const cases = node('details', 'case-details'); cases.append(node('summary', '', `Cases (${list(run.cases).length})`), dataTable(run.cases)); block.append(heading, metrics, cases); container.append(block);
  });
}
async function loadEvaluation() {
  const generation = state.generation; empty($('#evaluation-runs'), 'Loading evaluation runs…'); empty($('#evaluation-dataset'), 'Loading dataset…');
  await Promise.all([
    api('/api/evaluation').then(data => { if (generation === state.generation) renderRuns(list(data.runs)); }).catch(error => { if (generation === state.generation) empty($('#evaluation-runs'), error.message, true, loadEvaluation); }),
    api('/api/evaluation/dataset').then(data => { if (generation === state.generation) $('#evaluation-dataset').replaceChildren(dataTable(data.cases)); }).catch(error => { if (generation === state.generation) empty($('#evaluation-dataset'), error.message, true, loadEvaluation); })
  ]);
}
$('#evaluation-form').addEventListener('submit', async event => {
  event.preventDefault(); const button = $('button[type=submit]', event.currentTarget); const generation = state.generation; button.disabled = true; feedback('#evaluation-status', 'Evaluation running…');
  try { await api('/api/evaluation/run', { method: 'POST', body: JSON.stringify({ mode: $('#evaluation-mode').value, k: Number($('#evaluation-k').value) }) }); if (generation !== state.generation) return; feedback('#evaluation-status', 'Evaluation complete'); await loadEvaluation(); }
  catch (error) { if (generation === state.generation) feedback('#evaluation-status', error.message, true); }
  finally { button.disabled = false; }
});

document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => { location.hash = button.dataset.view; }));
window.addEventListener('hashchange', () => navigate(location.hash.slice(1)));
$('#identity').addEventListener('change', event => { state.role = event.target.value; resetIdentity(); state.histories.set(historyKey(), []); renderMessages(); loadMemory(); loadCurrentView(); });
$('#token-control').addEventListener('submit', event => { event.preventDefault(); state.token = $('#token').value.trim(); $('#token').value = ''; resetIdentity(); state.histories.clear(); renderMessages(); if (canRequest()) { loadMemory(); loadCurrentView(); feedback('#chat-status', 'Ready'); } else { renderAccessState(); empty($('#preferences'), 'Production token required.'); feedback('#chat-status', 'Production token required.', true); } });
$('#new-session').addEventListener('click', () => { state.generation += 1; state.controller?.abort(); state.sessionId = null; state.histories.set(historyKey(), []); state.chatBusy = false; $('#send-question').disabled = false; $('#messages').setAttribute('aria-busy', 'false'); $('#conversation-label').textContent = 'New conversation'; feedback('#chat-status', canRequest() ? 'Ready' : 'Not connected'); renderMessages(); $('#question').value = ''; $('#question').focus(); });
$('#document-search').addEventListener('input', renderDocuments);
$('#refresh-documents').addEventListener('click', loadDocuments);
$('#refresh-evaluation').addEventListener('click', loadEvaluation);
$('#refresh-memory').addEventListener('click', loadMemory);
$('#retry-health').addEventListener('click', connect);
$('#close-source').addEventListener('click', () => $('#source-dialog').close());
$('#source-dialog').addEventListener('close', () => { inspectorVersion += 1; });
$('#source-dialog').addEventListener('click', event => { if (event.target === event.currentTarget) { const rect = event.currentTarget.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.currentTarget.close(); } });
$('#nav-toggle').addEventListener('click', () => { const open = $('#sidebar').classList.toggle('open'); $('#nav-scrim').hidden = !open; $('#nav-toggle').setAttribute('aria-expanded', String(open)); });
$('#nav-scrim').addEventListener('click', closeNavigation);
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeNavigation(); });
refreshIcons(); renderMessages(); navigate(location.hash.slice(1)); connect();