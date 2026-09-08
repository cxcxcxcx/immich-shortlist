/* Gallery pages: live selection count + per-day all/none toggles. */
(function () {
    'use strict';
    const form = document.querySelector('form[data-gallery-form]');
    if (!form) return;
    const countEl = document.getElementById('sel-count');
    const submitBtn = document.getElementById('submit-selected');
    const submitLabel = submitBtn ? submitBtn.dataset.label : null;

    function boxesInSection(section) {
        return Array.from(section.querySelectorAll('input[name="asset_ids"]'));
    }

    function refresh() {
        const n = Array.from(form.querySelectorAll('input[name="asset_ids"]:checked')).length;
        if (countEl) countEl.textContent = String(n);
        if (submitBtn) {
            submitBtn.disabled = n === 0;
            if (submitLabel) submitBtn.textContent = n === 0 ? 'Select photos first' : submitLabel;
        }
    }

    form.addEventListener('change', refresh);

    document.addEventListener('click', (ev) => {
        const all = ev.target.closest('.js-all');
        const none = ev.target.closest('.js-none');
        if (!all && !none) return;
        const section = (all || none).closest('.day-section');
        if (!section) return;
        boxesInSection(section).forEach((b) => { b.checked = !!all; });
        refresh();
    });

    refresh();
})();
