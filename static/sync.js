// Sync page: live progress for the album -> pixtale reconcile (polls /sync/status).
(function () {
    const form = document.getElementById('sync-form');
    const formCard = document.getElementById('sync-form-card');
    const panel = document.getElementById('sync-panel');
    const dot = document.getElementById('sync-dot');
    const title = document.getElementById('sync-title');
    const bar = document.getElementById('sync-bar');
    const text = document.getElementById('sync-text');
    const planEl = document.getElementById('sync-plan');
    const detail = document.getElementById('sync-detail');
    const cancelBtn = document.getElementById('sync-cancel');
    const againBtn = document.getElementById('sync-again');
    const summaryEl = document.getElementById('sync-summary');
    const failWrap = document.getElementById('sync-failures');
    const failList = document.getElementById('sync-failure-list');

    function fmtSecs(s) {
        const m = Math.floor(s / 60), r = s % 60;
        return m > 0 ? m + 'm ' + r + 's' : r + 's';
    }

    // The delete path is the pipeline's only destructive step — ask before starting.
    form.addEventListener('submit', (ev) => {
        const deleteChecked = form.querySelector('input[name="delete"]').checked;
        const isPreview = ev.submitter && ev.submitter.name === 'dry_run';
        if (!isPreview && deleteChecked && !confirm('Delete photos in Pixtale that are no longer in the album?\nThis is permanent.')) {
            ev.preventDefault();
        }
    });

    cancelBtn.addEventListener('click', async () => {
        cancelBtn.disabled = true;
        text.textContent = 'Stopping — in-flight uploads will finish first…';
        try { await fetch('/sync/cancel', { method: 'POST' }); } catch (e) { /* the poll will reflect it */ }
    });

    againBtn.addEventListener('click', () => {
        formCard.style.display = 'block';
        againBtn.style.display = 'none';
    });

    const PHASE_LABEL = {
        fetching_album: 'Listing album assets from Immich…',
        fetching_pixtale: 'Listing the current gallery…',
        planned: 'Plan ready',
        deleting: 'Deleting',
        adding: 'Uploading',
    };

    function renderFailures(fails) {
        if (fails && fails.length) {
            failList.innerHTML = fails.map(f =>
                '<li><code>' + String(f[0]).slice(0, 8) + '…</code> ' + String(f[1]).replace(/[<>&]/g, '') + '</li>'
            ).join('');
            failWrap.style.display = 'block';
        } else {
            failWrap.style.display = 'none';
        }
    }

    function showSummary(s) {
        const c = s.counters || {};
        summaryEl.textContent = (s.dry_run ? 'Preview — nothing was changed. ' : '') +
            'added ' + (c.ok || 0) + ' · duplicates ' + (c.duplicate || 0) +
            ' · failed ' + (c.error || 0) + ' · deleted ' + (s.deleted || 0);
        summaryEl.style.display = 'block';
        renderFailures(s.failures);
    }

    async function poll() {
        let s;
        try { s = await (await fetch('/sync/status')).json(); } catch (e) { setTimeout(poll, 4000); return; }
        if (s.state === 'running') {
            panel.style.display = 'block';
            formCard.style.display = 'none';
            againBtn.style.display = 'none';
            cancelBtn.disabled = false;
            title.textContent = s.dry_run ? 'Preview running' : 'Sync running';
            bar.style.width = (s.pct || 0) + '%';
            const c = s.counters || {};
            const bits = [PHASE_LABEL[s.phase] || s.phase];
            if (s.phase === 'deleting') bits.push(Math.min(s.deleted, s.plan_delete) + '/' + s.plan_delete);
            if (s.phase === 'adding') {
                bits.push(s.add_done + '/' + s.add_total +
                    ' · ok ' + (c.ok || 0) + ' · dup ' + (c.duplicate || 0) + ' · failed ' + (c.error || 0));
                detail.textContent = s.current_asset ? 'last: ' + s.current_asset : '';
            } else {
                detail.textContent = '';
            }
            if (s.dry_run) bits.push('DRY RUN');
            bits.push(fmtSecs(s.elapsed_sec || 0) + ' elapsed');
            text.textContent = bits.join(' — ');
            planEl.textContent = (s.plan_add || s.plan_delete)
                ? 'Plan: add ' + s.plan_add + ', delete ' + s.plan_delete +
                  (s.delete ? ' (deletes ON)' : ' (deletes off)') + ' · storage ' + s.storage_id
                : '';
            setTimeout(poll, 1500);
        } else if (s.state === 'done' || s.state === 'cancelled') {
            dot.classList.remove('running');
            cancelBtn.disabled = true;
            bar.style.width = '100%';
            title.textContent = s.state === 'cancelled' ? 'Sync stopped'
                : (s.dry_run ? 'Preview finished' : 'Sync finished');
            showSummary(s);
            againBtn.style.display = 'inline-block';
            setTimeout(poll, 4000);
        } else if (s.state === 'error') {
            dot.classList.remove('running');
            cancelBtn.disabled = true;
            bar.style.width = '100%';
            title.textContent = 'Sync failed';
            text.textContent = s.error || 'Unknown error';
            showSummary(s);
            againBtn.style.display = 'inline-block';
            setTimeout(poll, 4000);
        } else {  // idle
            panel.style.display = 'none';
            formCard.style.display = 'block';
            setTimeout(poll, 4000);
        }
    }

    poll();
})();
