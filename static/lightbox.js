// Gallery lightbox: the card's zoom button opens the 1440px preview without
// toggling selection. Arrow keys / on-screen arrows move through the page's
// photos in DOM order; Esc or the backdrop closes.
(function () {
    let items = [];          // [{full, caption, thumb}] in DOM order
    let index = -1;

    const root = document.createElement('div');
    root.className = 'lightbox';
    root.innerHTML =
        '<button type="button" class="lb-close" aria-label="Close">✕</button>' +
        '<button type="button" class="lb-nav lb-prev" aria-label="Previous">‹</button>' +
        '<button type="button" class="lb-nav lb-next" aria-label="Next">›</button>' +
        '<figure><img alt="Preview"><figcaption></figcaption></figure>';
    document.body.appendChild(root);

    const img = root.querySelector('img');
    const caption = root.querySelector('figcaption');

    function collect(btn) {
        items = Array.from(document.querySelectorAll('.zoom-btn')).map((b) => ({
            btn: b,
            full: b.dataset.full,
            caption: b.dataset.caption || '',
            thumb: b.closest('.photo-card')?.querySelector('img')?.src || '',
        }));
        index = items.findIndex((it) => it.btn === btn);
    }

    function render() {
        const it = items[index];
        if (!it) return close();
        if (!img.src.startsWith(it.full)) {
            img.src = it.full;
            if (it.thumb) img.style.backgroundImage = 'url("' + it.thumb + '")';
        }
        caption.textContent = it.caption;
        root.querySelector('.lb-prev').style.visibility = index > 0 ? 'visible' : 'hidden';
        root.querySelector('.lb-next').style.visibility = index < items.length - 1 ? 'visible' : 'hidden';
        [index - 1, index + 1].forEach((i) => {
            if (items[i]) { const pre = new Image(); pre.src = items[i].full; }
        });
    }

    function open(btn) {
        collect(btn);
        if (index < 0) return;
        render();
        root.classList.add('open');
        document.body.classList.add('lb-lock');
    }

    function close() {
        root.classList.remove('open');
        document.body.classList.remove('lb-lock');
        index = -1;
    }

    function step(delta) {
        if (index < 0) return;
        const next = index + delta;
        if (next >= 0 && next < items.length) { index = next; render(); }
    }

    // The button lives inside the selection <label>: swallow the click so the
    // checkbox does not flip, then open.
    document.addEventListener('click', (ev) => {
        const btn = ev.target.closest('.zoom-btn');
        if (!btn) return;
        ev.preventDefault();
        ev.stopPropagation();
        open(btn);
    }, true);

    root.addEventListener('click', (ev) => {
        if (ev.target === root || ev.target.closest('.lb-close')) return close();
        if (ev.target.closest('.lb-prev')) return step(-1);
        if (ev.target.closest('.lb-next')) return step(1);
    });

    document.addEventListener('keydown', (ev) => {
        if (!root.classList.contains('open')) return;
        if (ev.key === 'Escape') close();
        else if (ev.key === 'ArrowLeft') step(-1);
        else if (ev.key === 'ArrowRight') step(1);
    });

    // Swipe between photos on touch screens.
    let touchX = null;
    root.addEventListener('touchstart', (ev) => { touchX = ev.touches[0].clientX; }, { passive: true });
    root.addEventListener('touchend', (ev) => {
        if (touchX === null) return;
        const dx = ev.changedTouches[0].clientX - touchX;
        if (Math.abs(dx) > 48) step(dx < 0 ? 1 : -1);
        touchX = null;
    }, { passive: true });
})();
