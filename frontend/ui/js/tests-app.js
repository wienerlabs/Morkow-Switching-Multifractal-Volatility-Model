/**
 * Cortex Test Runner — frontend application logic.
 * Connects to FastAPI test runner API at /api/tests/*.
 */
(function () {
    'use strict';

    // ── Config ──
    var API_BASE = window.location.hostname === 'localhost'
        ? 'http://localhost:8000'
        : '';
    var WS_BASE = window.location.hostname === 'localhost'
        ? 'ws://localhost:8000'
        : ('wss://' + window.location.host);
    var DEBOUNCE_MS = 300;

    // ── State ──
    var state = {
        files: [],
        totalFiles: 0,
        totalTests: 0,
        expanded: {},       // file -> bool
        filter: 'all',
        search: '',
        currentRunId: null,
        ws: null,
        running: false,
        progress: { completed: 0, total: 0, passed: 0, failed: 0 },
        lastSummary: null,
        fileResults: {},    // file -> { passed: 0, failed: 0, ... }
        testResults: {},    // "file::test" -> { status, duration }
        history: [],
    };

    // ── DOM refs ──
    var els = {};
    function $(id) { return document.getElementById(id); }

    function cacheDom() {
        els.totalFiles = $('metricTotalFiles');
        els.totalTests = $('metricTotalTests');
        els.lastRun = $('metricLastRun');
        els.suiteStatus = $('suiteStatusDot');
        els.duration = $('metricDuration');
        els.passRate = $('metricPassRate');
        els.btnRunAll = $('btnRunAll');
        els.btnRunFailed = $('btnRunFailed');
        els.btnCancel = $('btnCancel');
        els.searchInput = $('searchInput');
        els.progressSection = $('progressSection');
        els.progressLabel = $('progressLabel');
        els.progressPassed = $('progressPassed');
        els.progressFailed = $('progressFailed');
        els.progressRunning = $('progressRunning');
        els.progressFill = $('progressFill');
        els.progressFailBar = $('progressFailBar');
        els.fileList = $('testFileList');
        els.fileCountLabel = $('fileCountLabel');
        els.emptyState = $('testEmptyState');
        els.output = $('testOutput');
        els.historyBody = $('historyBody');
        els.panelCurrent = $('panelCurrentRun');
        els.panelHistory = $('panelHistory');
        els.tabCurrent = $('tabCurrentRun');
        els.tabHistory = $('tabHistory');
    }

    // ── API helpers ──
    async function apiGet(path) {
        try {
            var res = await fetch(API_BASE + '/api/tests' + path);
            if (!res.ok) throw new Error('HTTP ' + res.status);
            return await res.json();
        } catch (e) {
            console.warn('[TestRunner] GET ' + path + ' failed:', e.message);
            return null;
        }
    }

    async function apiPost(path, body) {
        try {
            var res = await fetch(API_BASE + '/api/tests' + path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!res.ok) throw new Error('HTTP ' + res.status);
            return await res.json();
        } catch (e) {
            console.warn('[TestRunner] POST ' + path + ' failed:', e.message);
            return null;
        }
    }

    // ── Utility ──
    function esc(s) {
        var d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    function cssId(s) {
        return s.replace(/[^a-zA-Z0-9]/g, '-');
    }

    function createEl(tag, className, textContent) {
        var el = document.createElement(tag);
        if (className) el.className = className;
        if (textContent !== undefined) el.textContent = textContent;
        return el;
    }

    // ── Discover ──
    async function discoverTests() {
        var data = await apiGet('/discover');
        if (!data) {
            showEmptyState('API server not available', 'Start the API with: uvicorn cortex.api.test_runner:router');
            return;
        }
        state.files = data.test_files || [];
        state.totalFiles = data.total_files || 0;
        state.totalTests = data.total_tests || 0;

        els.totalFiles.textContent = state.totalFiles;
        els.totalTests.textContent = state.totalTests.toLocaleString();
        els.fileCountLabel.textContent = state.totalFiles + ' files';

        renderFileList();
    }

    function showEmptyState(title, sub) {
        els.emptyState.style.display = '';
        els.emptyState.textContent = '';
        var icon = createEl('div', 'empty-icon', '\u25A0');
        var titleEl = createEl('div', 'empty-title', title);
        var subEl = createEl('div', 'empty-sub', sub);
        els.emptyState.appendChild(icon);
        els.emptyState.appendChild(titleEl);
        els.emptyState.appendChild(subEl);
    }

    // ── Render file list (DOM-based, no innerHTML) ──
    function renderFileList() {
        var filtered = getFilteredFiles();
        els.fileCountLabel.textContent = filtered.length + ' / ' + state.totalFiles + ' files';

        // Clear existing content
        els.fileList.textContent = '';

        if (filtered.length === 0) {
            var empty = createEl('div', 'test-empty-state');
            if (state.search) {
                empty.appendChild(createEl('div', 'empty-title', 'No matching tests'));
                empty.appendChild(createEl('div', 'empty-sub', 'Try a different search query'));
            } else {
                empty.appendChild(createEl('div', 'empty-title', 'No tests match filter'));
                empty.appendChild(createEl('div', 'empty-sub', 'Change the filter to see tests'));
            }
            els.fileList.appendChild(empty);
            return;
        }

        for (var i = 0; i < filtered.length; i++) {
            var f = filtered[i];
            var shortName = f.file.replace('tests/', '');
            var isExpanded = !!state.expanded[f.file];
            var fileStatus = getFileStatus(f.file);

            // File row
            var row = createEl('div', 'test-file-row' + (state.running && isFileRunning(f.file) ? ' running' : ''));
            row.dataset.file = f.file;
            row.addEventListener('click', (function (file) {
                return function () { window.onFileClick(file); };
            })(f.file));

            var chevron = createEl('div', 'test-file-chevron', isExpanded ? '\u25BC' : '\u25B6');
            var nameEl = createEl('div', 'test-file-name', shortName);
            nameEl.title = f.file;
            var countEl = createEl('div', 'test-file-count', f.test_count + ' tests');
            var statusEl = createEl('div', 'test-file-status');
            statusEl.appendChild(makeBadgeEl(fileStatus));
            var actionEl = createEl('div', 'test-file-action');
            var runBtn = createEl('button', 'btn', 'Run');
            runBtn.addEventListener('click', (function (file) {
                return function (e) { e.stopPropagation(); window.onRunFile(file); };
            })(f.file));
            actionEl.appendChild(runBtn);

            row.appendChild(chevron);
            row.appendChild(nameEl);
            row.appendChild(countEl);
            row.appendChild(statusEl);
            row.appendChild(actionEl);
            els.fileList.appendChild(row);

            // Expanded section
            var expanded = createEl('div', 'test-file-expanded' + (isExpanded ? ' open' : ''));
            expanded.id = 'expanded-' + cssId(f.file);
            if (isExpanded && f.tests) {
                for (var j = 0; j < f.tests.length; j++) {
                    var t = f.tests[j];
                    var testKey = f.file + '::' + t;
                    var tr = state.testResults[testKey];
                    var tStatus = tr ? tr.status : 'not_run';
                    var tDur = tr && tr.duration != null ? tr.duration.toFixed(3) + 's' : '';

                    var funcRow = createEl('div', 'test-function-row');
                    funcRow.appendChild(createEl('div', ''));
                    var dotEl = createEl('div', 'test-function-status');
                    dotEl.appendChild(makeDotEl(tStatus));
                    funcRow.appendChild(dotEl);
                    funcRow.appendChild(createEl('div', 'test-function-name', t));
                    funcRow.appendChild(createEl('div', 'test-function-duration', tDur));
                    funcRow.appendChild(createEl('div', ''));
                    expanded.appendChild(funcRow);
                }
            }
            els.fileList.appendChild(expanded);
        }
    }

    function getFilteredFiles() {
        var out = state.files;
        if (state.search) {
            var q = state.search.toLowerCase();
            out = out.filter(function (f) { return f.file.toLowerCase().indexOf(q) >= 0; });
        }
        if (state.filter !== 'all') {
            out = out.filter(function (f) {
                var s = getFileStatus(f.file);
                if (state.filter === 'passed') return s === 'passed';
                if (state.filter === 'failed') return s === 'failed' || s === 'error';
                if (state.filter === 'skipped') return s === 'skipped';
                return true;
            });
        }
        return out;
    }

    function getFileStatus(file) {
        var r = state.fileResults[file];
        if (!r) return 'not_run';
        if (r.failed > 0 || r.error > 0) return 'failed';
        if (r.passed > 0) return 'passed';
        if (r.skipped > 0) return 'skipped';
        return 'not_run';
    }

    function isFileRunning() {
        return false;
    }

    function makeBadgeEl(status) {
        var span = document.createElement('span');
        var map = {
            passed: ['badge badge-green', 'PASS'],
            failed: ['badge badge-red', 'FAIL'],
            error: ['badge badge-red', 'ERR'],
            skipped: ['badge badge-yellow', 'SKIP'],
            not_run: ['badge badge-dim', '\u2014'],
            running: ['badge badge-yellow dot-running', 'RUN'],
        };
        var cfg = map[status] || map.not_run;
        span.className = cfg[0];
        span.textContent = cfg[1];
        return span;
    }

    function makeDotEl(status) {
        var span = document.createElement('span');
        var colors = { passed: 'var(--green)', failed: 'var(--red)', error: 'var(--red)', skipped: 'var(--yellow)' };
        var c = colors[status];
        if (c) {
            span.className = 'dot';
            span.style.background = c;
        } else {
            span.className = 'dot dot-dim';
        }
        return span;
    }

    // ── Run tests ──
    window.onRunAll = async function () {
        await startRun('all');
    };

    window.onRunFailed = async function () {
        var failedFiles = state.files.filter(function (f) {
            return getFileStatus(f.file) === 'failed';
        });
        if (failedFiles.length === 0) return;
        for (var i = 0; i < failedFiles.length; i++) {
            await startRun(failedFiles[i].file);
        }
    };

    window.onRunFile = async function (file) {
        await startRun(file);
    };

    window.onCancel = function () {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ action: 'cancel' }));
        }
    };

    async function startRun(target) {
        if (state.running) return;

        var data = await apiPost('/run', { target: target });
        if (!data || !data.run_id) {
            appendOutput('Failed to start test run \u2014 is the API server running?', 'line-fail');
            return;
        }

        state.currentRunId = data.run_id;
        state.running = true;
        state.progress = { completed: 0, total: 0, passed: 0, failed: 0 };
        state.testResults = {};
        state.fileResults = {};

        els.btnCancel.classList.remove('hidden');
        els.btnRunAll.disabled = true;
        els.btnRunFailed.disabled = true;
        els.progressSection.classList.remove('hidden');
        clearOutput();
        appendOutput('Starting: pytest ' + target, 'line-info');
        updateProgress();

        connectWebSocket(data.run_id);
    }

    // ── WebSocket ──
    function connectWebSocket(runId) {
        if (state.ws) {
            state.ws.close();
        }

        var url = WS_BASE + '/api/tests/stream/' + runId;
        var ws = new WebSocket(url);
        state.ws = ws;

        ws.onopen = function () {
            appendOutput('Connected to test stream', 'line-info');
        };

        ws.onmessage = function (evt) {
            var msg;
            try { msg = JSON.parse(evt.data); } catch (e) { return; }

            switch (msg.type) {
                case 'output':
                    var cls = 'line-info';
                    if (msg.line.indexOf('PASSED') >= 0) cls = 'line-pass';
                    else if (msg.line.indexOf('FAILED') >= 0 || msg.line.indexOf('ERROR') >= 0) cls = 'line-fail';
                    else if (msg.line.indexOf('SKIPPED') >= 0) cls = 'line-skip';
                    appendOutput(msg.line, cls);
                    break;

                case 'test_result':
                    updateTestResult(msg);
                    break;

                case 'progress':
                    state.progress.completed = msg.completed;
                    state.progress.total = msg.total || state.progress.total;
                    state.progress.passed = msg.passed;
                    state.progress.failed = msg.failed;
                    updateProgress();
                    break;

                case 'complete':
                    onRunComplete(msg.summary);
                    break;
            }
        };

        ws.onclose = function () {
            if (state.running) {
                onRunComplete(state.progress);
            }
        };

        ws.onerror = function () {
            appendOutput('WebSocket connection error', 'line-fail');
        };
    }

    function updateTestResult(msg) {
        var testKey = msg.test;
        state.testResults[testKey] = { status: msg.status, duration: msg.duration };

        var filePart = testKey.split('::')[0];
        if (!state.fileResults[filePart]) {
            state.fileResults[filePart] = { passed: 0, failed: 0, error: 0, skipped: 0 };
        }
        var key = msg.status === 'error' ? 'error' : (msg.status === 'skipped' ? 'skipped' : (msg.status === 'passed' ? 'passed' : 'failed'));
        state.fileResults[filePart][key]++;

        renderFileList();
    }

    function onRunComplete(summary) {
        state.running = false;
        state.lastSummary = summary;
        state.ws = null;

        els.btnCancel.classList.add('hidden');
        els.btnRunAll.disabled = false;
        els.btnRunFailed.disabled = false;

        var passed = summary.passed || 0;
        var failed = summary.failed || 0;
        var total = (summary.total || passed + failed) || 1;
        var dur = summary.duration || 0;

        els.lastRun.textContent = passed + '/' + total + (failed > 0 ? ' (' + failed + ' failed)' : '');
        els.duration.textContent = dur.toFixed(2) + 's';
        els.passRate.textContent = Math.round(passed / total * 100) + '%';

        els.suiteStatus.className = failed > 0 ? 'dot dot-red' : 'dot dot-green';

        appendOutput('', 'line-info');
        appendOutput('Completed: ' + passed + ' passed, ' + failed + ' failed in ' + dur.toFixed(2) + 's', failed > 0 ? 'line-fail' : 'line-pass');

        renderFileList();
        loadHistory();
    }

    // ── Progress ──
    function updateProgress() {
        var p = state.progress;
        var total = p.total || 1;
        var pct = Math.round(p.completed / total * 100);

        els.progressLabel.textContent = p.completed + ' / ' + total + ' (' + pct + '%)';
        els.progressPassed.textContent = p.passed + ' passed';
        els.progressFailed.textContent = p.failed + ' failed';
        els.progressRunning.textContent = (p.completed < total ? '1 running' : '0 running');

        var passPct = (p.passed / total * 100);
        var failPct = (p.failed / total * 100);
        els.progressFill.style.width = passPct + '%';
        els.progressFailBar.style.width = failPct + '%';
        els.progressFailBar.style.left = passPct + '%';
    }

    // ── Output terminal ──
    function clearOutput() {
        els.output.textContent = '';
    }

    function appendOutput(text, cls) {
        var div = document.createElement('div');
        if (cls) div.className = cls;
        div.textContent = text;
        els.output.appendChild(div);
        els.output.scrollTop = els.output.scrollHeight;
    }

    // ── Filter / Search ──
    var searchTimer = null;
    window.onSearchInput = function (e) {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(function () {
            state.search = e.target.value;
            renderFileList();
        }, DEBOUNCE_MS);
    };

    window.onFilterClick = function (filter, btn) {
        state.filter = filter;
        var pills = document.querySelectorAll('.filter-pill');
        pills.forEach(function (p) { p.classList.remove('active'); });
        btn.classList.add('active');
        renderFileList();
    };

    window.onFileClick = function (file) {
        state.expanded[file] = !state.expanded[file];
        renderFileList();
    };

    // ── Tabs ──
    window.switchOutputTab = function (tab) {
        if (tab === 'current') {
            els.panelCurrent.classList.remove('hidden');
            els.panelHistory.classList.add('hidden');
            els.tabCurrent.classList.add('active');
            els.tabHistory.classList.remove('active');
        } else {
            els.panelCurrent.classList.add('hidden');
            els.panelHistory.classList.remove('hidden');
            els.tabCurrent.classList.remove('active');
            els.tabHistory.classList.add('active');
            loadHistory();
        }
    };

    // ── History ──
    async function loadHistory() {
        var data = await apiGet('/history?limit=20');
        if (!data) return;
        state.history = data;
        renderHistory();
    }

    function renderHistory() {
        els.historyBody.textContent = '';

        if (!state.history.length) {
            var empty = createEl('div', 'test-empty-state');
            empty.appendChild(createEl('div', 'empty-sub', 'No test history yet'));
            els.historyBody.appendChild(empty);
            return;
        }

        for (var i = 0; i < state.history.length; i++) {
            var h = state.history[i];
            var ts = new Date(h.started_at * 1000);
            var timeStr = ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' +
                ts.toLocaleDateString([], { month: 'short', day: 'numeric' });
            var dur = h.summary.duration ? h.summary.duration.toFixed(2) + 's' : '--';

            var row = createEl('div', 'test-history-row');
            row.appendChild(createEl('div', '', timeStr));
            row.appendChild(createEl('div', '', dur));

            var passedEl = createEl('div', '', String(h.summary.passed || 0));
            passedEl.style.color = 'var(--green)';
            row.appendChild(passedEl);

            var failedEl = createEl('div', '', String(h.summary.failed || 0));
            failedEl.style.color = 'var(--red)';
            row.appendChild(failedEl);

            var skippedEl = createEl('div', '', String(h.summary.skipped || 0));
            skippedEl.style.color = 'var(--yellow)';
            row.appendChild(skippedEl);

            row.appendChild(createEl('div', 'test-file-name', h.target));

            var badge = document.createElement('span');
            badge.className = h.summary.failed > 0 ? 'badge badge-red' : 'badge badge-green';
            badge.textContent = h.summary.failed > 0 ? 'FAIL' : 'PASS';
            var badgeWrap = createEl('div', '');
            badgeWrap.appendChild(badge);
            row.appendChild(badgeWrap);

            els.historyBody.appendChild(row);
        }
    }

    // ── Keyboard shortcuts ──
    document.addEventListener('keydown', function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (!state.running) window.onRunAll();
        }
        if (e.key === 'Escape' && state.running) {
            e.preventDefault();
            window.onCancel();
        }
    });

    // ── Init ──
    document.addEventListener('DOMContentLoaded', async function () {
        cacheDom();
        await discoverTests();
        await loadHistory();
    });
})();
