// Top picks page: live progress for a background re-score; reload when it finishes.
(function () {
    const panel = document.getElementById('rescore-progress');
    if (!panel) return;
    const bar = document.getElementById('rescore-bar');
    const text = document.getElementById('rescore-text');

    function fmtSecs(s) {
        const m = Math.floor(s / 60), r = s % 60;
        return m > 0 ? m + 'm ' + r + 's' : r + 's';
    }

    async function poll() {
        let s;
        try { s = await (await fetch('/progress')).json(); } catch (e) { setTimeout(poll, 4000); return; }
        if (s.state === 'running' && s.phase === 'rescore') {
            panel.style.display = 'block';
            const pct = s.days_total ? Math.round(100 * s.days_done / s.days_total) : 0;
            bar.style.width = pct + '%';
            text.textContent = 'Re-scoring day ' + Math.min(s.days_done + 1, s.days_total) + '/' + s.days_total +
                ' — ' + (s.current_date || '') + ' · ' + fmtSecs(s.elapsed_sec || 0) + ' elapsed';
            setTimeout(poll, 1500);
        } else if (s.state === 'running') {
            // A full curation took over; the home page shows its progress.
            setTimeout(poll, 4000);
        } else if (s.state === 'done' || s.state === 'cancelled') {
            if (panel.style.display === 'block') {
                text.textContent = s.state === 'done' ? 'Re-score finished — refreshing…' : 'Re-score stopped — refreshing…';
                setTimeout(() => location.reload(), 600);
            }
        } else if (s.state === 'error') {
            panel.style.display = 'block';
            bar.style.width = '100%';
            text.textContent = 'Re-score failed: ' + s.error;
        } else {
            setTimeout(poll, 4000);
        }
    }

    poll();
})();
