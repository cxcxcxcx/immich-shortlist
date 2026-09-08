/* Home page: inline range-calendar picker + live run progress polling. */
(function () {
    'use strict';

    /* ================= Range calendar ================= */
    const calEl = document.getElementById('calendar');
    const summaryEl = document.getElementById('range-summary');
    const startInput = document.getElementById('start_date');
    const endInput = document.getElementById('end_date');
    const runBtn = document.getElementById('run-btn');

    const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'];
    const DOW = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'];
    const DAY_MS = 86400000;
    const today = startOfDay(new Date());
    let start = null;   // Date | null
    let end = null;     // Date | null
    let hoverDate = null;
    let view = { y: today.getFullYear(), m: today.getMonth() };

    function startOfDay(d) { const x = new Date(d); x.setHours(0, 0, 0, 0); return x; }
    function iso(d) {
        const p = (n) => String(n).padStart(2, '0');
        return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate());
    }
    function fromIso(s) { return new Date(s + 'T00:00:00'); }
    function addDays(d, n) { return new Date(d.getTime() + n * DAY_MS); }
    function sameDay(a, b) { return a !== null && b !== null && iso(a) === iso(b); }
    function daysCount(a, b) { return Math.round((startOfDay(b) - startOfDay(a)) / DAY_MS) + 1; }
    function fmtHuman(d) { return MONTHS[d.getMonth()].slice(0, 3) + ' ' + d.getDate() + ', ' + d.getFullYear(); }
    function monthEnd(y, m) { return new Date(y, m + 1, 0); }

    function applyRange(s, e) {
        start = s; end = e; hoverDate = null;
        if (s) view = { y: s.getFullYear(), m: s.getMonth() };
        sync(); render();
    }

    function sync() {
        startInput.value = start ? iso(start) : '';
        endInput.value = end ? iso(end) : '';
        if (start && end) {
            summaryEl.textContent = fmtHuman(start) + ' → ' + fmtHuman(end) + ' · ' + daysCount(start, end) + ' days';
            summaryEl.classList.remove('empty');
            runBtn.disabled = false;
        } else if (start) {
            summaryEl.textContent = fmtHuman(start) + ' → pick an end date';
            summaryEl.classList.remove('empty');
            runBtn.disabled = true;
        } else {
            summaryEl.textContent = 'Pick a start date to begin';
            summaryEl.classList.add('empty');
            runBtn.disabled = true;
        }
    }

    function rangeEnd() { return end || hoverDate; }

    function cellClass(d, first, last) {
        const cls = ['cal-day'];
        if (d < first || d > last) cls.push('is-outside');
        if (sameDay(d, today)) cls.push('is-today');
        const re = rangeEnd();
        if (start && re) {
            const lo = start < re ? start : re;
            const hi = start < re ? re : start;
            if (d >= lo && d <= hi) cls.push('in-range');
        }
        if (sameDay(d, start) || sameDay(d, end)) cls.push('is-edge');
        return cls.join(' ');
    }

    function monthGrid(y, m) {
        const first = new Date(y, m, 1);
        const gridStart = addDays(first, -((first.getDay() + 6) % 7)); // back to Monday
        let html = '<div class="cal-month"><div class="cal-month-label">' + MONTHS[m] + ' ' + y + '</div><div class="cal-grid">';
        for (const d of DOW) html += '<div class="cal-dow">' + d + '</div>';
        for (let i = 0; i < 42; i++) {
            const d = addDays(gridStart, i);
            html += '<button type="button" class="' + cellClass(d, first, monthEnd(y, m)) +
                '" data-iso="' + iso(d) + '">' + d.getDate() + '</button>';
        }
        return html + '</div></div>';
    }

    function render() {
        const twoMonths = window.innerWidth >= 720;
        let html = '<div class="cal-toolbar">' +
            '<button type="button" class="cal-btn" data-nav="-1" aria-label="Previous month">‹</button>' +
            '<div class="cal-title">' + MONTHS[view.m] + ' ' + view.y + '</div>' +
            '<button type="button" class="cal-btn" data-nav="1" aria-label="Next month">›</button>' +
            '</div><div class="cal-months">' + monthGrid(view.y, view.m);
        if (twoMonths) {
            const nm = view.m + 1;
            html += monthGrid(view.y + (nm > 11 ? 1 : 0), nm % 12);
        }
        html += '</div>';
        calEl.innerHTML = html;
    }

    function shiftMonth(n) {
        view.m += n;
        if (view.m < 0) { view.m = 11; view.y--; }
        if (view.m > 11) { view.m = 0; view.y++; }
        render();
    }

    calEl.addEventListener('click', (ev) => {
        const nav = ev.target.closest('[data-nav]');
        if (nav) { shiftMonth(parseInt(nav.dataset.nav, 10)); return; }
        const btn = ev.target.closest('[data-iso]');
        if (!btn) return;
        const d = fromIso(btn.dataset.iso);
        if (!start || (start && end) || d < start) applyRange(d, null);
        else { end = d; hoverDate = null; sync(); render(); }
    });
    calEl.addEventListener('mouseover', (ev) => {
        const btn = ev.target.closest('[data-iso]');
        if (!btn || !start || end) return;
        const d = fromIso(btn.dataset.iso);
        if (d < start || sameDay(d, start)) { if (hoverDate) { hoverDate = null; render(); } return; }
        if (!sameDay(d, hoverDate)) { hoverDate = d; render(); }
    });
    calEl.addEventListener('mouseleave', () => {
        if (start && !end && hoverDate) { hoverDate = null; render(); }
    });

    document.getElementById('presets').addEventListener('click', (ev) => {
        const chip = ev.target.closest('[data-preset]');
        if (!chip) return;
        const p = chip.dataset.preset;
        if (p === 'last7') applyRange(addDays(today, -6), today);
        else if (p === 'last30') applyRange(addDays(today, -29), today);
        else if (p === 'thisMonth') {
            applyRange(new Date(today.getFullYear(), today.getMonth(), 1), monthEnd(today.getFullYear(), today.getMonth()));
        } else if (p === 'lastMonth') {
            const nm = today.getMonth() - 1;
            const y = today.getFullYear() + (nm < 0 ? -1 : 0);
            applyRange(new Date(y, (nm + 12) % 12, 1), monthEnd(y, (nm + 12) % 12));
        }
    });

    let resizeTimer = null;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(render, 150);
    });

    /* ================= Run progress polling ================= */
    const panel = document.getElementById('progress-panel');
    const form = document.getElementById('curation-form');
    const bar = document.getElementById('bar');
    const textEl = document.getElementById('progress-text');
    const rangeEl = document.getElementById('progress-range');
    const cancelBtn = document.getElementById('cancel-btn');
    let armed = false;

    function fmtSecs(s) {
        const m = Math.floor(s / 60), r = s % 60;
        return m > 0 ? m + 'm ' + r + 's' : r + 's';
    }

    cancelBtn.addEventListener('click', async () => {
        cancelBtn.disabled = true;
        textEl.textContent = 'Stopping after the current day…';
        try { await fetch('/cancel', { method: 'POST' }); } catch (e) { /* server gone; poll will report */ }
    });

    form.addEventListener('submit', (ev) => {
        const s = startInput.value, e = endInput.value;
        if (s && e) {
            const days = daysCount(fromIso(s), fromIso(e));
            if (days > 40 && !confirm('About to curate ' + days + ' days — this can take a long time. Continue?')) {
                ev.preventDefault();
            }
        }
    });

    async function poll() {
        let s;
        try { s = await (await fetch('/progress')).json(); } catch (e) { setTimeout(poll, 4000); return; }
        if (s.state === 'running') {
            armed = true;
            cancelBtn.disabled = false;
            panel.style.display = 'block';
            form.style.display = 'none';
            rangeEl.textContent = (s.start_date || '') + ' → ' + (s.end_date || '');
            const pct = s.days_total ? Math.round(100 * s.days_done / s.days_total) : 0;
            bar.style.width = pct + '%';
            const assets = s.assets_total ? ' · photo ' + s.assets_done + '/' + s.assets_total : '';
            textEl.textContent = 'Day ' + Math.min(s.days_done + 1, s.days_total) + '/' + s.days_total +
                ' — ' + (s.current_date || '') + assets + ' · ' + fmtSecs(s.elapsed_sec || 0) + ' elapsed';
            setTimeout(poll, 2000);
        } else if (s.state === 'done') {
            if (armed) {
                textEl.textContent = 'Done — opening top picks…';
                setTimeout(() => { location.href = '/top'; }, 800);
            }
        } else if (s.state === 'cancelled') {
            if (armed) {
                textEl.textContent = 'Stopped. Days completed so far are kept.';
                panel.style.display = 'none';
                form.style.display = 'block';
            }
        } else if (s.state === 'error') {
            panel.style.display = 'block';
            form.style.display = 'none';
            bar.style.width = '100%';
            textEl.textContent = 'Run failed: ' + s.error;
        } else {
            setTimeout(poll, 4000);
        }
    }

    sync();
    render();
    poll();
})();
