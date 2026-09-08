// Settings page: All/None helpers for the face pickers.
(function () {
    document.addEventListener('click', (ev) => {
        const btn = ev.target.closest('.js-face-all, .js-face-none');
        if (!btn) return;
        const group = document.querySelector('[data-face-group="' + btn.dataset.for + '"]');
        if (!group) return;
        const check = btn.classList.contains('js-face-all');
        group.querySelectorAll('input[type="checkbox"]').forEach((box) => { box.checked = check; });
    });
})();
