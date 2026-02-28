/**
 * Cortex Test Runner v2 — complete rewrite.
 *
 * Features: sessionStorage persistence, WebSocket reconnection, live feed,
 * animations, toast notifications, ETA calculation, per-file mini progress.
 *
 * Exposes: window.TestRunner = { runAll, runFailed, cancel, onSearch, setFilter, switchTab }
 */
(function () {
    'use strict';

    /* ════════════════════════════════════════════════
       §1  CONFIG
       ════════════════════════════════════════════════ */
    var API_BASE  = window.location.origin;
    var WS_BASE   = window.location.origin.replace(/^http/, 'ws');
    var STORAGE_KEY    = 'cortex_test_state';
    var DEBOUNCE_MS    = 300;
    var RING_R         = 52;
    var RING_C         = 2 * Math.PI * RING_R;          // ≈ 326.73
    var MAX_FEED       = 200;
    var MAX_TERM       = 2000;
    var PERSIST_TERM   = 500;
    var PERSIST_FEED   = 100;
    var RECONNECT_MAX  = 5;
    var RECONNECT_MS   = 2000;
    var TOAST_MS       = 4000;
    var COUNTER_MS     = 400;

    /* ════════════════════════════════════════════════
       §2  STATE
       ════════════════════════════════════════════════ */
    var state = {
        files:          [],
        totalFiles:     0,
        totalTests:     0,
        expanded:       {},          // file -> bool
        filter:         'all',
        search:         '',
        currentRunId:   null,
        running:        false,
        progress:       { completed: 0, total: 0, passed: 0, failed: 0, skipped: 0 },
        lastSummary:    null,
        fileResults:    {},          // file -> { passed, failed, error, skipped, total }
        testResults:    {},          // "file::class::test" -> { status, duration }
        terminalLines:  [],          // { text, cls }
        feedItems:      [],          // { test, status, duration, file, name, ts }
        startTime:      null,
        historyLoaded:  false,
        history:        [],
    };

    var ws           = null;
    var reconnects   = 0;
    var searchTimer  = null;

    /* ════════════════════════════════════════════════
       §3  DOM CACHE
       ════════════════════════════════════════════════ */
    var els = {};
    function $(id) { return document.getElementById(id); }

    function cacheDom() {
        var ids = [
            'heroRing', 'ringPass', 'ringFail', 'ringCenter', 'ringPct',
            'metricTotalFiles', 'metricTotalTests', 'metricPassed', 'metricFailed',
            'metricDuration', 'metricETA',
            'heroStatus', 'statusDot', 'statusText',
            'btnRunAll', 'btnRunFailed', 'btnCancel',
            'progressWrap', 'progressTitle', 'progressCounter',
            'progressPassBar', 'progressFailBar', 'progressActiveBar',
            'psPass', 'psFail', 'psSkip', 'psRemaining',
            'searchInput', 'filterBar',
            'testFileList', 'fileCountLabel', 'testEmptyState',
            'panelFeed', 'feedEmpty', 'feedList', 'feedCount',
            'panelTerminal', 'terminalOutput',
            'panelHistory', 'historyBody',
            'tabLiveFeed', 'tabOutput', 'tabHistory',
            'toastContainer',
            'footerTestCount'
        ];
        for (var i = 0; i < ids.length; i++) {
            els[ids[i]] = $(ids[i]);
        }
    }

    /* ════════════════════════════════════════════════
       §4  PERSISTENCE  (sessionStorage)
       ════════════════════════════════════════════════ */
    function saveState() {
        try {
            var snap = {
                currentRunId:  state.currentRunId,
                running:       state.running,
                progress:      state.progress,
                testResults:   state.testResults,
                fileResults:   state.fileResults,
                lastSummary:   state.lastSummary,
                filter:        state.filter,
                search:        state.search,
                expanded:      state.expanded,
                startTime:     state.startTime,
                terminalLines: state.terminalLines.slice(-PERSIST_TERM),
                feedItems:     state.feedItems.slice(-PERSIST_FEED),
            };
            sessionStorage.setItem(STORAGE_KEY, JSON.stringify(snap));
        } catch (_) { /* quota */ }
    }

    function loadState() {
        try {
            var raw = sessionStorage.getItem(STORAGE_KEY);
            if (!raw) return false;
            var snap = JSON.parse(raw);
            if (!snap) return false;
            state.currentRunId  = snap.currentRunId  || null;
            state.running       = !!snap.running;
            state.progress      = snap.progress      || state.progress;
            state.testResults   = snap.testResults   || {};
            state.fileResults   = snap.fileResults   || {};
            state.lastSummary   = snap.lastSummary   || null;
            state.filter        = snap.filter        || 'all';
            state.search        = snap.search        || '';
            state.expanded      = snap.expanded      || {};
            state.startTime     = snap.startTime     || null;
            state.terminalLines = snap.terminalLines || [];
            state.feedItems     = snap.feedItems     || [];
            return true;
        } catch (_) { return false; }
    }

    /* ════════════════════════════════════════════════
       §5  API HELPERS
       ════════════════════════════════════════════════ */
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

    /* ════════════════════════════════════════════════
       §6  DOM UTILITIES
       ════════════════════════════════════════════════ */
    function createEl(tag, className, textContent) {
        var el = document.createElement(tag);
        if (className) el.className = className;
        if (textContent !== undefined && textContent !== null) el.textContent = textContent;
        return el;
    }

    function removeChildren(el) {
        while (el.firstChild) el.removeChild(el.firstChild);
    }

    function cssId(s) {
        return s.replace(/[^a-zA-Z0-9]/g, '-');
    }

    function formatTime(sec) {
        if (sec == null || isNaN(sec)) return '\u2014';
        if (sec < 60) return sec.toFixed(1) + 's';
        var m = Math.floor(sec / 60);
        var s = Math.round(sec % 60);
        return m + 'm ' + s + 's';
    }

    function formatETA(sec) {
        if (sec == null || isNaN(sec) || sec <= 0) return '\u2014';
        if (sec < 60) return '~' + Math.round(sec) + 's';
        var m = Math.floor(sec / 60);
        var s = Math.round(sec % 60);
        return '~' + m + 'm ' + (s > 0 ? s + 's' : '');
    }

    function timestamp() {
        var d = new Date();
        var pad = function (n) { return n < 10 ? '0' + n : '' + n; };
        return '[' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds()) + ']';
    }

    /* ════════════════════════════════════════════════
       §7  COUNTER ANIMATION
       ════════════════════════════════════════════════ */
    function animateCounter(el, from, to) {
        if (from === to) return;
        var start = performance.now();
        var diff = to - from;

        function step(now) {
            var t = Math.min((now - start) / COUNTER_MS, 1);
            // ease-out quad
            var ease = t * (2 - t);
            var val = Math.round(from + diff * ease);
            el.textContent = val;
            if (t < 1) {
                requestAnimationFrame(step);
            } else {
                el.textContent = to;
                el.classList.add('tr-counter-pop');
                setTimeout(function () { el.classList.remove('tr-counter-pop'); }, 300);
            }
        }
        requestAnimationFrame(step);
    }

    /* ════════════════════════════════════════════════
       §8  TOAST NOTIFICATIONS
       ════════════════════════════════════════════════ */
    function showToast(message, type) {
        var container = els.toastContainer;
        if (!container) return;

        var toast = createEl('div', 'tr-toast tr-toast-' + (type || 'success'));
        var icon = createEl('span', 'tr-toast-icon', type === 'error' ? '\u2717' : '\u2713');
        var msg  = createEl('span', 'tr-toast-msg', message);
        toast.appendChild(icon);
        toast.appendChild(msg);
        container.appendChild(toast);

        // trigger entrance
        requestAnimationFrame(function () {
            toast.classList.add('tr-toast-enter');
        });

        setTimeout(function () {
            toast.classList.remove('tr-toast-enter');
            toast.classList.add('tr-toast-exit');
            setTimeout(function () {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 400);
        }, TOAST_MS);
    }

    /* ════════════════════════════════════════════════
       §9  RING CHART (SVG DONUT)
       ════════════════════════════════════════════════ */
    function updateRing(passed, failed, total) {
        if (!total || total === 0) {
            if (els.ringPass) els.ringPass.setAttribute('stroke-dasharray', '0 ' + RING_C);
            if (els.ringFail) els.ringFail.setAttribute('stroke-dasharray', '0 ' + RING_C);
            if (els.ringPct)  els.ringPct.textContent = '--';
            return;
        }

        var passLen = (passed / total) * RING_C;
        var failLen = (failed / total) * RING_C;
        var pct     = Math.round((passed / total) * 100);

        if (els.ringPass) {
            els.ringPass.setAttribute('stroke-dasharray', passLen + ' ' + RING_C);
        }
        if (els.ringFail) {
            els.ringFail.setAttribute('stroke-dasharray', failLen + ' ' + RING_C);
            // offset fail arc after pass arc (CSS rotate(-90deg) handles 12 o'clock start)
            var failOffset = -passLen;
            els.ringFail.setAttribute('stroke-dashoffset', failOffset);
        }

        if (els.ringPct) {
            var prev = parseInt(els.ringPct.textContent, 10);
            if (isNaN(prev)) prev = 0;
            animateCounter(els.ringPct, prev, pct);
        }
    }

    /* ════════════════════════════════════════════════
       §10  STATUS INDICATOR
       ════════════════════════════════════════════════ */
    function setStatus(label, dotClass) {
        if (els.statusDot) {
            els.statusDot.className = 'tr-status-dot ' + (dotClass || 'tr-status-idle');
        }
        if (els.statusText) {
            els.statusText.textContent = label;
        }
    }

    /* ════════════════════════════════════════════════
       §11  PROGRESS BAR
       ════════════════════════════════════════════════ */
    function showProgress() {
        if (els.progressWrap) els.progressWrap.classList.remove('hidden');
    }

    function hideProgress() {
        if (els.progressWrap) els.progressWrap.classList.add('hidden');
    }

    function updateProgress() {
        var p     = state.progress;
        var total = p.total || 1;
        var done  = p.completed || 0;
        var pass  = p.passed || 0;
        var fail  = p.failed || 0;
        var skip  = p.skipped || 0;
        var rem   = Math.max(0, total - done);

        // header
        if (els.progressTitle) {
            els.progressTitle.textContent = state.running ? 'Running tests\u2026' : 'Tests complete';
        }
        if (els.progressCounter) {
            els.progressCounter.textContent = done + ' / ' + total;
        }

        // bar widths
        var passPct   = (pass / total * 100).toFixed(2);
        var failPct   = (fail / total * 100).toFixed(2);
        var activePct = state.running ? Math.max(0, ((done - pass - fail - skip) / total * 100) + 1).toFixed(2) : '0';
        // clamp active to not overflow
        var usedPct   = parseFloat(passPct) + parseFloat(failPct);
        if (parseFloat(activePct) + usedPct > 100) {
            activePct = (100 - usedPct).toFixed(2);
        }

        if (els.progressPassBar)   els.progressPassBar.style.width   = passPct + '%';
        if (els.progressFailBar)   els.progressFailBar.style.width   = failPct + '%';
        if (els.progressActiveBar) els.progressActiveBar.style.width = (state.running ? activePct : '0') + '%';

        // stats
        if (els.psPass)      els.psPass.textContent      = pass;
        if (els.psFail)      els.psFail.textContent      = fail;
        if (els.psSkip)      els.psSkip.textContent      = skip;
        if (els.psRemaining) els.psRemaining.textContent = rem;

        // ETA
        if (state.running && state.startTime && done > 0) {
            var elapsed    = (Date.now() - state.startTime) / 1000;
            var avgPerTest = elapsed / done;
            var etaSec     = avgPerTest * rem;
            if (els.metricETA) els.metricETA.textContent = formatETA(etaSec);
        } else if (!state.running) {
            if (els.metricETA) els.metricETA.textContent = '\u2014';
        }

        saveState();
    }

    /* ════════════════════════════════════════════════
       §12  HERO METRICS
       ════════════════════════════════════════════════ */
    function updateHeroMetrics() {
        var p = state.progress;
        var s = state.lastSummary;

        if (els.metricTotalFiles) els.metricTotalFiles.textContent = state.totalFiles || '--';
        if (els.metricTotalTests) els.metricTotalTests.textContent = state.totalTests ? state.totalTests.toLocaleString() : '--';
        if (els.footerTestCount)  els.footerTestCount.textContent  = state.totalTests ? state.totalTests.toLocaleString() : '--';

        if (s) {
            var prevP = parseInt(els.metricPassed.textContent, 10) || 0;
            var prevF = parseInt(els.metricFailed.textContent, 10) || 0;
            animateCounter(els.metricPassed, prevP, s.passed || 0);
            animateCounter(els.metricFailed, prevF, s.failed || 0);
            if (els.metricDuration) els.metricDuration.textContent = formatTime(s.duration);
            updateRing(s.passed || 0, s.failed || 0, s.total || 0);
        } else if (state.running) {
            var prevP2 = parseInt(els.metricPassed.textContent, 10) || 0;
            var prevF2 = parseInt(els.metricFailed.textContent, 10) || 0;
            animateCounter(els.metricPassed, prevP2, p.passed || 0);
            animateCounter(els.metricFailed, prevF2, p.failed || 0);
            updateRing(p.passed || 0, p.failed || 0, p.total || 0);
        }
    }

    /* ════════════════════════════════════════════════
       §13  TERMINAL OUTPUT
       ════════════════════════════════════════════════ */
    function clearTerminal() {
        state.terminalLines = [];
        if (els.terminalOutput) removeChildren(els.terminalOutput);
    }

    function appendTerminal(text, cls) {
        var entry = { text: text, cls: cls || '' };
        state.terminalLines.push(entry);

        // cap
        if (state.terminalLines.length > MAX_TERM) {
            state.terminalLines = state.terminalLines.slice(-MAX_TERM);
        }

        if (els.terminalOutput) {
            // remove overflow DOM nodes
            while (els.terminalOutput.childNodes.length >= MAX_TERM) {
                els.terminalOutput.removeChild(els.terminalOutput.firstChild);
            }

            var line = createEl('div', 'tr-term-line' + (cls ? ' ' + cls : ''));
            var ts   = createEl('span', 'tr-term-ts', timestamp() + ' ');
            line.appendChild(ts);
            var txt = document.createTextNode(text);
            line.appendChild(txt);
            els.terminalOutput.appendChild(line);
            els.terminalOutput.scrollTop = els.terminalOutput.scrollHeight;
        }
    }

    function restoreTerminal() {
        if (!els.terminalOutput) return;
        removeChildren(els.terminalOutput);
        for (var i = 0; i < state.terminalLines.length; i++) {
            var entry = state.terminalLines[i];
            var line  = createEl('div', 'tr-term-line' + (entry.cls ? ' ' + entry.cls : ''));
            var ts    = createEl('span', 'tr-term-ts', '          ');
            line.appendChild(ts);
            var txt = document.createTextNode(entry.text);
            line.appendChild(txt);
            els.terminalOutput.appendChild(line);
        }
        els.terminalOutput.scrollTop = els.terminalOutput.scrollHeight;
    }

    /* ════════════════════════════════════════════════
       §14  LIVE FEED
       ════════════════════════════════════════════════ */
    function clearFeed() {
        state.feedItems = [];
        if (els.feedList)  removeChildren(els.feedList);
        if (els.feedEmpty) els.feedEmpty.classList.remove('hidden');
        if (els.feedCount) els.feedCount.textContent = '';
    }

    function addFeedItem(testKey, status, duration) {
        // parse test key: "tests/file.py::TestClass::test_name"
        var parts    = testKey.split('::');
        var file     = parts[0] || '';
        var funcName = parts[parts.length - 1] || testKey;

        var item = {
            test:     testKey,
            status:   status,
            duration: duration,
            file:     file,
            name:     funcName,
            ts:       Date.now()
        };

        state.feedItems.unshift(item);
        if (state.feedItems.length > MAX_FEED) {
            state.feedItems = state.feedItems.slice(0, MAX_FEED);
        }

        // hide empty
        if (els.feedEmpty) els.feedEmpty.classList.add('hidden');

        // build card (DOM only, no innerHTML)
        var card = createEl('div', 'tr-feed-item tr-feed-enter');
        if (status === 'passed') {
            card.classList.add('tr-feed-pass');
        } else if (status === 'failed' || status === 'error') {
            card.classList.add('tr-feed-fail');
        } else if (status === 'skipped') {
            card.classList.add('tr-feed-skip');
        }

        var iconDiv = createEl('div', 'tr-feed-item-icon');
        if (status === 'passed') {
            iconDiv.textContent = '\u2713';
        } else if (status === 'failed' || status === 'error') {
            iconDiv.textContent = '\u2717';
        } else {
            iconDiv.textContent = '\u2014';
        }

        var bodyDiv = createEl('div', 'tr-feed-item-body');
        var nameDiv = createEl('div', 'tr-feed-item-name', funcName);
        var metaDiv = createEl('div', 'tr-feed-item-meta',
            file + (duration != null ? ' \u00b7 ' + duration.toFixed(3) + 's' : ''));
        bodyDiv.appendChild(nameDiv);
        bodyDiv.appendChild(metaDiv);

        card.appendChild(iconDiv);
        card.appendChild(bodyDiv);

        // prepend (newest first)
        if (els.feedList) {
            if (els.feedList.firstChild) {
                els.feedList.insertBefore(card, els.feedList.firstChild);
            } else {
                els.feedList.appendChild(card);
            }

            // cap DOM nodes
            while (els.feedList.childNodes.length > MAX_FEED) {
                els.feedList.removeChild(els.feedList.lastChild);
            }
        }

        // update count badge
        if (els.feedCount) {
            els.feedCount.textContent = state.feedItems.length;
        }

        // trigger animation — remove the enter class after frame
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                card.classList.remove('tr-feed-enter');
            });
        });

        saveState();
    }

    function restoreFeed() {
        if (!els.feedList) return;
        removeChildren(els.feedList);

        if (state.feedItems.length === 0) {
            if (els.feedEmpty) els.feedEmpty.classList.remove('hidden');
            if (els.feedCount) els.feedCount.textContent = '';
            return;
        }

        if (els.feedEmpty) els.feedEmpty.classList.add('hidden');

        for (var i = 0; i < state.feedItems.length; i++) {
            var item = state.feedItems[i];
            var card = createEl('div', 'tr-feed-item');

            if (item.status === 'passed') {
                card.classList.add('tr-feed-pass');
            } else if (item.status === 'failed' || item.status === 'error') {
                card.classList.add('tr-feed-fail');
            } else if (item.status === 'skipped') {
                card.classList.add('tr-feed-skip');
            }

            var iconDiv = createEl('div', 'tr-feed-item-icon');
            if (item.status === 'passed') {
                iconDiv.textContent = '\u2713';
            } else if (item.status === 'failed' || item.status === 'error') {
                iconDiv.textContent = '\u2717';
            } else {
                iconDiv.textContent = '\u2014';
            }

            var bodyDiv = createEl('div', 'tr-feed-item-body');
            var nameDiv = createEl('div', 'tr-feed-item-name', item.name);
            var metaDiv = createEl('div', 'tr-feed-item-meta',
                item.file + (item.duration != null ? ' \u00b7 ' + item.duration.toFixed(3) + 's' : ''));
            bodyDiv.appendChild(nameDiv);
            bodyDiv.appendChild(metaDiv);

            card.appendChild(iconDiv);
            card.appendChild(bodyDiv);
            els.feedList.appendChild(card);
        }

        if (els.feedCount) {
            els.feedCount.textContent = state.feedItems.length;
        }
    }

    /* ════════════════════════════════════════════════
       §15  FILE LIST RENDERING (DOM-based, NO innerHTML)
       ════════════════════════════════════════════════ */
    function getFilteredFiles() {
        var out = state.files;
        if (state.search) {
            var q = state.search.toLowerCase();
            out = out.filter(function (f) { return f.file.toLowerCase().indexOf(q) >= 0; });
        }
        if (state.filter !== 'all') {
            out = out.filter(function (f) {
                var s = getFileStatus(f.file);
                if (state.filter === 'passed')  return s === 'passed';
                if (state.filter === 'failed')  return s === 'failed' || s === 'error';
                if (state.filter === 'skipped') return s === 'skipped';
                if (state.filter === 'not_run') return s === 'not_run';
                return true;
            });
        }
        return out;
    }

    function getFileStatus(file) {
        var r = state.fileResults[file];
        if (!r) return 'not_run';
        if ((r.failed || 0) > 0 || (r.error || 0) > 0) return 'failed';
        if ((r.passed || 0) > 0) return 'passed';
        if ((r.skipped || 0) > 0) return 'skipped';
        return 'not_run';
    }

    function getFileTotalRun(file) {
        var r = state.fileResults[file];
        if (!r) return 0;
        return (r.passed || 0) + (r.failed || 0) + (r.error || 0) + (r.skipped || 0);
    }

    function renderFileList() {
        if (!els.testFileList) return;
        var filtered = getFilteredFiles();

        if (els.fileCountLabel) {
            els.fileCountLabel.textContent = filtered.length + ' / ' + state.totalFiles + ' files';
        }

        removeChildren(els.testFileList);

        if (filtered.length === 0) {
            var empty = createEl('div', 'tr-empty');
            if (state.totalFiles === 0) {
                // still loading or no tests
                var emptyIcon = createEl('div', 'tr-empty-icon');
                var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                svg.setAttribute('width', '48');
                svg.setAttribute('height', '48');
                svg.setAttribute('viewBox', '0 0 48 48');
                svg.setAttribute('fill', 'none');
                var rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', '8'); rect.setAttribute('y', '6');
                rect.setAttribute('width', '32'); rect.setAttribute('height', '36');
                rect.setAttribute('rx', '2');
                rect.setAttribute('stroke', 'currentColor'); rect.setAttribute('stroke-width', '1.5');
                rect.setAttribute('fill', 'none');
                svg.appendChild(rect);
                emptyIcon.appendChild(svg);
                empty.appendChild(emptyIcon);
                empty.appendChild(createEl('div', 'tr-empty-title', 'Discovering tests\u2026'));
                empty.appendChild(createEl('div', 'tr-empty-sub', 'Connecting to API server'));
                empty.appendChild(createEl('div', 'tr-spinner'));
            } else if (state.search) {
                empty.appendChild(createEl('div', 'tr-empty-title', 'No matching tests'));
                empty.appendChild(createEl('div', 'tr-empty-sub', 'Try a different search query'));
            } else {
                empty.appendChild(createEl('div', 'tr-empty-title', 'No tests match filter'));
                empty.appendChild(createEl('div', 'tr-empty-sub', 'Change the filter to see tests'));
            }
            els.testFileList.appendChild(empty);
            return;
        }

        // find max duration across all tests for relative bar sizing
        var maxDur = 0;
        var allKeys = Object.keys(state.testResults);
        for (var k = 0; k < allKeys.length; k++) {
            var d = state.testResults[allKeys[k]].duration;
            if (d != null && d > maxDur) maxDur = d;
        }

        for (var i = 0; i < filtered.length; i++) {
            var f = filtered[i];
            var shortName  = f.file.replace(/^tests\//, '');
            var isExpanded = !!state.expanded[f.file];
            var fileStatus = getFileStatus(f.file);
            var fileTotalRun = getFileTotalRun(f.file);
            var fr = state.fileResults[f.file] || {};

            // ── file row ──
            var row = createEl('div', 'tr-file-row');
            row.setAttribute('data-file', f.file);

            // chevron
            var chevron = createEl('div', 'tr-file-chevron', isExpanded ? '\u25BC' : '\u25B6');
            if (isExpanded) chevron.classList.add('tr-chevron-open');

            // name
            var nameEl = createEl('div', 'tr-file-name', shortName);
            nameEl.title = f.file;

            // count
            var countEl = createEl('div', 'tr-file-count', f.test_count + ' tests');

            // mini progress bar
            var miniBar  = createEl('div', 'tr-file-mini-bar');
            var miniPass = createEl('div', 'tr-file-mini-pass');
            var miniFail = createEl('div', 'tr-file-mini-fail');
            if (fileTotalRun > 0) {
                var pPct = ((fr.passed || 0) / fileTotalRun * 100).toFixed(1);
                var fPct = (((fr.failed || 0) + (fr.error || 0)) / fileTotalRun * 100).toFixed(1);
                miniPass.style.width = pPct + '%';
                miniFail.style.width = fPct + '%';
            } else {
                miniPass.style.width = '0%';
                miniFail.style.width = '0%';
            }
            miniBar.appendChild(miniPass);
            miniBar.appendChild(miniFail);

            // status badge
            var statusEl = createEl('div', 'tr-file-status');
            statusEl.appendChild(makeBadge(fileStatus));

            // run button
            var actionEl = createEl('div', 'tr-file-action');
            var runBtn   = createEl('button', 'btn tr-file-run-btn', 'Run');
            (function (filePath) {
                runBtn.addEventListener('click', function (e) {
                    e.stopPropagation();
                    runFile(filePath);
                });
            })(f.file);
            actionEl.appendChild(runBtn);

            row.appendChild(chevron);
            row.appendChild(nameEl);
            row.appendChild(countEl);
            row.appendChild(miniBar);
            row.appendChild(statusEl);
            row.appendChild(actionEl);

            // row click → expand
            (function (filePath) {
                row.addEventListener('click', function () {
                    state.expanded[filePath] = !state.expanded[filePath];
                    saveState();
                    renderFileList();
                });
            })(f.file);

            els.testFileList.appendChild(row);

            // ── expanded section ──
            var expDiv = createEl('div', 'tr-file-expanded' + (isExpanded ? ' open' : ''));
            expDiv.id = 'expanded-' + cssId(f.file);

            if (isExpanded && f.tests) {
                for (var j = 0; j < f.tests.length; j++) {
                    var t       = f.tests[j];
                    var testKey = f.file + '::' + t;
                    var tr      = state.testResults[testKey];
                    var tStatus = tr ? tr.status : 'not_run';
                    var tDur    = tr && tr.duration != null ? tr.duration : null;

                    var funcRow = createEl('div', 'tr-func-row');

                    // colored dot
                    var dotEl = createEl('div', 'tr-func-dot');
                    dotEl.appendChild(makeDot(tStatus));

                    // name
                    var fnName = createEl('div', 'tr-func-name', t);

                    // duration text
                    var durText = createEl('div', 'tr-func-duration',
                        tDur != null ? tDur.toFixed(3) + 's' : '');

                    // duration bar
                    var durBar  = createEl('div', 'tr-func-dur-bar');
                    var durFill = createEl('div', 'tr-func-dur-fill');
                    if (tDur != null && maxDur > 0) {
                        var fillPct = Math.min(100, (tDur / maxDur) * 100);
                        durFill.style.width = fillPct + '%';
                        if (tStatus === 'passed') durFill.classList.add('tr-dur-pass');
                        else if (tStatus === 'failed' || tStatus === 'error') durFill.classList.add('tr-dur-fail');
                        else durFill.classList.add('tr-dur-skip');
                    }
                    durBar.appendChild(durFill);

                    funcRow.appendChild(dotEl);
                    funcRow.appendChild(fnName);
                    funcRow.appendChild(durText);
                    funcRow.appendChild(durBar);
                    expDiv.appendChild(funcRow);
                }
            }

            els.testFileList.appendChild(expDiv);
        }
    }

    function makeBadge(status) {
        var span = document.createElement('span');
        var map = {
            passed:  ['tr-badge tr-badge-green', 'PASS'],
            failed:  ['tr-badge tr-badge-red',   'FAIL'],
            error:   ['tr-badge tr-badge-red',   'ERR'],
            skipped: ['tr-badge tr-badge-yellow', 'SKIP'],
            not_run: ['tr-badge tr-badge-dim',   '\u2014'],
            running: ['tr-badge tr-badge-yellow tr-badge-pulse', 'RUN'],
        };
        var cfg = map[status] || map.not_run;
        span.className = cfg[0];
        span.textContent = cfg[1];
        return span;
    }

    function makeDot(status) {
        var span = document.createElement('span');
        var colorMap = {
            passed:  'tr-dot tr-dot-pass',
            failed:  'tr-dot tr-dot-fail',
            error:   'tr-dot tr-dot-fail',
            skipped: 'tr-dot tr-dot-skip',
        };
        span.className = colorMap[status] || 'tr-dot tr-dot-dim';
        return span;
    }

    /* ════════════════════════════════════════════════
       §16  DISCOVER TESTS
       ════════════════════════════════════════════════ */
    async function discoverTests() {
        var data = await apiGet('/discover');
        if (!data) {
            if (els.testEmptyState) {
                removeChildren(els.testEmptyState);
                els.testEmptyState.appendChild(createEl('div', 'tr-empty-title', 'API server not available'));
                els.testEmptyState.appendChild(createEl('div', 'tr-empty-sub', 'Start with: python -m uvicorn cortex.api.test_runner:app'));
            }
            return;
        }

        state.files      = data.test_files || [];
        state.totalFiles  = data.total_files || state.files.length;
        state.totalTests  = data.total_tests || 0;

        updateHeroMetrics();
        renderFileList();
    }

    /* ════════════════════════════════════════════════
       §17  RUN TESTS
       ════════════════════════════════════════════════ */
    async function runAll() {
        await startRun('all');
    }

    async function runFailed() {
        var failedFiles = state.files.filter(function (f) {
            var s = getFileStatus(f.file);
            return s === 'failed' || s === 'error';
        });
        if (failedFiles.length === 0) {
            showToast('No failed tests to re-run', 'error');
            return;
        }
        // run all — backend handles filtering, or run individually
        await startRun('all');
    }

    async function runFile(file) {
        await startRun(file);
    }

    async function startRun(target) {
        if (state.running) return;

        var data = await apiPost('/run', { target: target });
        if (!data || !data.run_id) {
            appendTerminal('Failed to start test run \u2014 is the API server running?', 'tr-term-fail');
            showToast('Failed to start test run', 'error');
            return;
        }

        state.currentRunId = data.run_id;
        state.running      = true;
        state.startTime    = Date.now();
        state.progress     = { completed: 0, total: state.totalTests || 0, passed: 0, failed: 0, skipped: 0 };
        state.lastSummary  = null;
        state.testResults  = {};
        state.fileResults  = {};

        // UI transitions
        if (els.btnCancel)    els.btnCancel.classList.remove('hidden');
        if (els.btnRunAll)    els.btnRunAll.classList.add('hidden');
        if (els.btnRunFailed) els.btnRunFailed.classList.add('hidden');

        showProgress();
        clearFeed();
        clearTerminal();
        appendTerminal('Starting: pytest ' + target, 'tr-term-info');
        setStatus('Running', 'tr-status-running');
        updateProgress();
        saveState();
        renderFileList();

        reconnects = 0;
        connectWebSocket(data.run_id);
    }

    /* ════════════════════════════════════════════════
       §18  WEBSOCKET STREAMING
       ════════════════════════════════════════════════ */
    function connectWebSocket(runId) {
        if (ws) {
            try { ws.close(); } catch (_) {}
            ws = null;
        }

        var url = WS_BASE + '/api/tests/stream/' + runId;
        ws = new WebSocket(url);

        ws.onopen = function () {
            reconnects = 0;
            appendTerminal('Connected to test stream [' + runId + ']', 'tr-term-info');
        };

        ws.onmessage = function (evt) {
            var msg;
            try { msg = JSON.parse(evt.data); } catch (_) { return; }

            switch (msg.type) {
                case 'output':
                    handleOutput(msg);
                    break;
                case 'test_result':
                    handleTestResult(msg);
                    break;
                case 'progress':
                    handleProgress(msg);
                    break;
                case 'complete':
                    handleComplete(msg);
                    break;
            }
        };

        ws.onclose = function () {
            if (state.running) {
                // attempt reconnect
                if (reconnects < RECONNECT_MAX) {
                    reconnects++;
                    appendTerminal('Connection lost. Reconnecting (' + reconnects + '/' + RECONNECT_MAX + ')\u2026', 'tr-term-skip');
                    setTimeout(function () {
                        if (state.running && state.currentRunId) {
                            connectWebSocket(state.currentRunId);
                        }
                    }, RECONNECT_MS);
                } else {
                    appendTerminal('Max reconnection attempts reached. Checking final state\u2026', 'tr-term-fail');
                    checkRunStatus(state.currentRunId);
                }
            }
        };

        ws.onerror = function () {
            appendTerminal('WebSocket connection error', 'tr-term-fail');
        };
    }

    function handleOutput(msg) {
        var cls = 'tr-term-info';
        if (msg.line && msg.line.indexOf('PASSED') >= 0)   cls = 'tr-term-pass';
        else if (msg.line && (msg.line.indexOf('FAILED') >= 0 || msg.line.indexOf('ERROR') >= 0)) cls = 'tr-term-fail';
        else if (msg.line && msg.line.indexOf('SKIPPED') >= 0) cls = 'tr-term-skip';
        appendTerminal(msg.line || '', cls);
    }

    function handleTestResult(msg) {
        var testKey = msg.test;
        state.testResults[testKey] = { status: msg.status, duration: msg.duration };

        // update per-file aggregation
        var filePart = testKey.split('::')[0];
        if (!state.fileResults[filePart]) {
            state.fileResults[filePart] = { passed: 0, failed: 0, error: 0, skipped: 0 };
        }
        if (msg.status === 'passed')       state.fileResults[filePart].passed++;
        else if (msg.status === 'failed')  state.fileResults[filePart].failed++;
        else if (msg.status === 'error')   state.fileResults[filePart].error++;
        else if (msg.status === 'skipped') state.fileResults[filePart].skipped++;

        // live feed card
        addFeedItem(testKey, msg.status, msg.duration);

        // re-render file list (throttled by browser paint)
        renderFileList();
        updateHeroMetrics();
        saveState();
    }

    function handleProgress(msg) {
        state.progress.completed = msg.completed || state.progress.completed;
        state.progress.total     = msg.total     || state.progress.total;
        state.progress.passed    = msg.passed    != null ? msg.passed : state.progress.passed;
        state.progress.failed    = msg.failed    != null ? msg.failed : state.progress.failed;
        // derive skipped from testResults
        var skipCount = 0;
        var keys = Object.keys(state.testResults);
        for (var i = 0; i < keys.length; i++) {
            if (state.testResults[keys[i]].status === 'skipped') skipCount++;
        }
        state.progress.skipped = skipCount;
        updateProgress();
        updateHeroMetrics();
    }

    function handleComplete(msg) {
        var summary = msg.summary || {};
        state.running     = false;
        state.lastSummary = summary;

        if (ws) {
            try { ws.close(); } catch (_) {}
            ws = null;
        }

        // UI transitions
        if (els.btnCancel)    els.btnCancel.classList.add('hidden');
        if (els.btnRunAll)    els.btnRunAll.classList.remove('hidden');
        if (els.btnRunFailed) els.btnRunFailed.classList.remove('hidden');

        var passed = summary.passed  || 0;
        var failed = summary.failed  || 0;
        var total  = summary.total   || (passed + failed) || 1;
        var dur    = summary.duration || 0;

        // final progress
        state.progress.completed = total;
        state.progress.total     = total;
        state.progress.passed    = passed;
        state.progress.failed    = failed;
        state.progress.skipped   = summary.skipped || 0;
        updateProgress();
        updateHeroMetrics();

        var statusLabel = failed > 0 ? 'Failed' : 'Passed';
        var statusClass = failed > 0 ? 'tr-status-fail' : 'tr-status-pass';
        setStatus(statusLabel, statusClass);

        appendTerminal('', '');
        appendTerminal('Completed: ' + passed + ' passed, ' + failed + ' failed in ' + formatTime(dur),
            failed > 0 ? 'tr-term-fail' : 'tr-term-pass');

        // toast
        if (failed > 0) {
            showToast(failed + ' test' + (failed > 1 ? 's' : '') + ' failed', 'error');
        } else {
            showToast('All tests passed \u2014 ' + passed + '/' + total + ' in ' + formatTime(dur), 'success');
        }

        renderFileList();
        saveState();

        // refresh history if loaded
        if (state.historyLoaded) {
            loadHistory();
        }
    }

    function cancel() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'cancel' }));
        }
        appendTerminal('Cancelling test run\u2026', 'tr-term-skip');
    }

    /* ════════════════════════════════════════════════
       §19  RECONNECTION / STATE RECOVERY
       ════════════════════════════════════════════════ */
    async function checkRunStatus(runId) {
        var data = await apiGet('/results/' + runId);
        if (!data) {
            // 404 or error — clear stale state
            appendTerminal('Run not found — clearing stale state', 'tr-term-skip');
            resetRunState();
            return;
        }

        if (data.status === 'running') {
            // still going — reconnect
            reconnects = 0;
            connectWebSocket(runId);
        } else {
            // completed or failed — restore final state
            if (data.summary) {
                handleComplete({ summary: data.summary });
            } else {
                resetRunState();
            }
        }
    }

    function resetRunState() {
        state.running      = false;
        state.currentRunId = null;
        state.startTime    = null;

        if (ws) { try { ws.close(); } catch (_) {} ws = null; }

        if (els.btnCancel)    els.btnCancel.classList.add('hidden');
        if (els.btnRunAll)    els.btnRunAll.classList.remove('hidden');
        if (els.btnRunFailed) els.btnRunFailed.classList.remove('hidden');

        setStatus('Idle', 'tr-status-idle');
        if (els.metricETA) els.metricETA.textContent = '\u2014';
        saveState();
    }

    /* ════════════════════════════════════════════════
       §20  HISTORY TAB
       ════════════════════════════════════════════════ */
    async function loadHistory() {
        var data = await apiGet('/history?limit=20');
        if (!data) return;
        state.history = Array.isArray(data) ? data : [];
        state.historyLoaded = true;
        renderHistory();
    }

    function renderHistory() {
        if (!els.historyBody) return;
        removeChildren(els.historyBody);

        if (!state.history || state.history.length === 0) {
            var empty = createEl('div', 'tr-empty');
            empty.appendChild(createEl('div', 'tr-empty-sub', 'No test history yet'));
            els.historyBody.appendChild(empty);
            return;
        }

        for (var i = 0; i < state.history.length; i++) {
            var h   = state.history[i];
            var ts  = new Date((h.started_at || 0) * 1000);
            var timeStr = ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' +
                ts.toLocaleDateString([], { month: 'short', day: 'numeric' });
            var sum = h.summary || {};
            var dur = sum.duration ? formatTime(sum.duration) : '\u2014';

            var row = createEl('div', 'tr-history-row');

            // time
            row.appendChild(createEl('div', 'tr-history-cell', timeStr));

            // target
            var targetText = h.target || 'all';
            if (targetText.length > 30) targetText = '\u2026' + targetText.slice(-28);
            row.appendChild(createEl('div', 'tr-history-cell tr-history-target', targetText));

            // duration
            row.appendChild(createEl('div', 'tr-history-cell', dur));

            // result counts
            var resultDiv = createEl('div', 'tr-history-cell tr-history-result');
            var passSpan  = createEl('span', 'tr-hist-pass', String(sum.passed || 0));
            var sep       = document.createTextNode(' / ');
            var failSpan  = createEl('span', 'tr-hist-fail', String(sum.failed || 0));
            resultDiv.appendChild(passSpan);
            resultDiv.appendChild(sep);
            resultDiv.appendChild(failSpan);
            row.appendChild(resultDiv);

            // status badge
            var badgeDiv = createEl('div', 'tr-history-cell');
            var badge    = document.createElement('span');
            badge.className = (sum.failed || 0) > 0 ? 'tr-badge tr-badge-red' : 'tr-badge tr-badge-green';
            badge.textContent = (sum.failed || 0) > 0 ? 'FAIL' : 'PASS';
            badgeDiv.appendChild(badge);
            row.appendChild(badgeDiv);

            els.historyBody.appendChild(row);
        }
    }

    /* ════════════════════════════════════════════════
       §21  TAB SWITCHING
       ════════════════════════════════════════════════ */
    function switchTab(tabName) {
        // panels
        var panels = {
            feed:    els.panelFeed,
            output:  els.panelTerminal,
            history: els.panelHistory,
        };
        var tabs = {
            feed:    els.tabLiveFeed,
            output:  els.tabOutput,
            history: els.tabHistory,
        };

        var names = Object.keys(panels);
        for (var i = 0; i < names.length; i++) {
            var n = names[i];
            if (panels[n]) {
                if (n === tabName) {
                    panels[n].classList.remove('hidden');
                } else {
                    panels[n].classList.add('hidden');
                }
            }
            if (tabs[n]) {
                if (n === tabName) {
                    tabs[n].classList.add('active');
                } else {
                    tabs[n].classList.remove('active');
                }
            }
        }

        // lazy-load history
        if (tabName === 'history' && !state.historyLoaded) {
            loadHistory();
        }
    }

    /* ════════════════════════════════════════════════
       §22  FILTER & SEARCH
       ════════════════════════════════════════════════ */
    function onSearch(e) {
        clearTimeout(searchTimer);
        var input = e.target || e.srcElement;
        searchTimer = setTimeout(function () {
            state.search = input.value || '';
            saveState();
            renderFileList();
        }, DEBOUNCE_MS);
    }

    function setFilter(filter, btn) {
        state.filter = filter;
        saveState();

        // update active class on filter buttons
        if (els.filterBar) {
            var buttons = els.filterBar.querySelectorAll('.tr-filter');
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].classList.remove('active');
            }
        }
        if (btn) btn.classList.add('active');

        renderFileList();
    }

    /* ════════════════════════════════════════════════
       §23  KEYBOARD SHORTCUTS
       ════════════════════════════════════════════════ */
    function setupKeyboard() {
        document.addEventListener('keydown', function (e) {
            // Cmd/Ctrl + Enter → Run All
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                if (!state.running) runAll();
            }
            // Escape → Cancel
            if (e.key === 'Escape' && state.running) {
                e.preventDefault();
                cancel();
            }
        });
    }

    /* ════════════════════════════════════════════════
       §24  CLEANUP
       ════════════════════════════════════════════════ */
    function setupCleanup() {
        window.addEventListener('beforeunload', function () {
            saveState();
            if (ws) {
                try { ws.close(); } catch (_) {}
            }
        });
    }

    /* ════════════════════════════════════════════════
       §25  RESTORE UI FROM PERSISTED STATE
       ════════════════════════════════════════════════ */
    function restoreUI() {
        // search input
        if (els.searchInput && state.search) {
            els.searchInput.value = state.search;
        }

        // filter buttons
        if (els.filterBar && state.filter !== 'all') {
            var buttons = els.filterBar.querySelectorAll('.tr-filter');
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].classList.remove('active');
                if (buttons[i].getAttribute('data-filter') === state.filter) {
                    buttons[i].classList.add('active');
                }
            }
        }

        // restore terminal
        restoreTerminal();

        // restore feed
        restoreFeed();

        // restore progress section visibility
        if (state.running || state.lastSummary) {
            showProgress();
            updateProgress();
        }

        // restore metrics from last summary or progress
        updateHeroMetrics();

        // restore status indicator
        if (state.running) {
            setStatus('Running', 'tr-status-running');
            if (els.btnCancel)    els.btnCancel.classList.remove('hidden');
            if (els.btnRunAll)    els.btnRunAll.classList.add('hidden');
            if (els.btnRunFailed) els.btnRunFailed.classList.add('hidden');
        } else if (state.lastSummary) {
            var hasFail = (state.lastSummary.failed || 0) > 0;
            setStatus(hasFail ? 'Failed' : 'Passed', hasFail ? 'tr-status-fail' : 'tr-status-pass');
        }
    }

    /* ════════════════════════════════════════════════
       §26  INITIALIZATION
       ════════════════════════════════════════════════ */
    async function init() {
        cacheDom();
        setupKeyboard();
        setupCleanup();

        var hadState = loadState();

        if (hadState) {
            restoreUI();
        }

        // always discover tests to refresh file list
        await discoverTests();

        // if was running, attempt to reconnect
        if (state.running && state.currentRunId) {
            appendTerminal('Reconnecting to run [' + state.currentRunId + ']\u2026', 'tr-term-info');
            reconnects = 0;
            await checkRunStatus(state.currentRunId);
        }

        renderFileList();
    }

    // ── DOMContentLoaded ──
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    /* ════════════════════════════════════════════════
       §27  PUBLIC API
       ════════════════════════════════════════════════ */
    window.TestRunner = {
        runAll:    runAll,
        runFailed: runFailed,
        cancel:    cancel,
        onSearch:  onSearch,
        setFilter: setFilter,
        switchTab: switchTab,
    };

})();
