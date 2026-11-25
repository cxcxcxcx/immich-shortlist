# Shortlist

**Automated family-photo curation for [Immich](https://immich.app) — final say yours.**
[中文说明](README.zh-CN.md)

Shortlist is a companion app for Immich that ranks the month's thousands of
frames into a shortlist of keepers per day and hands you the checkbox. It sits
beside your existing Immich server (never replaces or forks it), writes your
selections **back into an Immich album**, and can optionally sync that album to
a shareable [Pixtale](https://github.com/aslost/pixtale) gallery.

![Daily top picks — scored & clustered shortlist](docs/top.png)

## Why

Immich is superb at keeping and finding photos once you know what you're
looking for. It is not built to answer the question that actually hurts:

> *"I took 3,000 photos of the kids this month — which 20 are worth keeping and
> sharing?"*

Immich de-duplicates and recognizes faces, but it won't rank a burst by who's
smiling and looking at the camera, collapse 30 near-identical frames into the
one good moment, or hand you a curated shortlist to say yes/no on. Google
Photos does surface highlights, but its reels don't know *our* kids, don't
score the faces we care about, and — for the family members we most want to
share with, back in China — **can't be opened at all** without a Google account
and a VPN.

So we built the missing layer: **automated curation with a human in the loop.**
The machine does the tedious part (search, score, cluster, propose); you do the
one only you can (pick the shots with meaning). The results flow back into
Immich, which stays the single source of truth — a Shortlist album is just
an Immich album, so nothing is locked away. And when you do want relatives to
see it, a push to a self-hosted Pixtale gallery gives them a link that works in
Shanghai or Shenzhen.

Concretely, it aims to:

1. **know whose faces matter** — your kids first, the rest of the family second,
2. **propose one great frame per moment**, not 30 look-alikes,
3. **keep you in charge** — it shortlists, you approve, and it writes back to Immich,
4. **publish somewhere family in China can actually open** — no Google, no VPN.

## How it works

You run an analysis over a date range (pick it on a calendar):

![Curation run — the date picker](docs/home.png)

For each day:

- **Search** — every photo Immich has face-tagged as one of your *primary
  people* (your kids) on that date becomes a candidate.
- **Score** — three local ML models (nothing leaves your machine):
  - *Quality*: MUSIQ no-reference image quality (sharpness, exposure, composition)
  - *Expression*: MediaPipe Face Landmarker reads eyes-open, smile, and
    camera-facing per face; kids looking at the camera and smiling score highest
  - *Togetherness*: a person-boost for shots with multiple family members, plus
    a family-expression boost for group warmth
- **Cluster** — CLIP embeddings group near-duplicate shots into *moments*, so a
  burst of 30 frames collapses to its best one (cluster-top bonus).
- **Rank & select** — `total = w_q·quality + w_e·expression + person_boost +
  family_boost + cluster_top_bonus`. Per-day defaults pre-select the top picks;
  every weight and threshold is editable in the settings UI.

Then you **review**: a per-day top-picks page, an all-clusters page, and deep
per-photo views. Tick what you like, submit — selections are added to your
Immich album. The album is the source of truth.

### Re-scoring, not re-running

All expensive per-photo features (quality, expression, CLIP vectors) are
cached to disk keyed by asset. Tuning the weights — "expression matters more,
cluster tighter" — triggers an instant re-score over the cache; no GPU
re-inference, and your manual selections survive. Settings changes that
outdate a run show a "re-score to apply" banner.

### Publishing to a shareable gallery (optional)

A built-in **Sync** page reconciles the album into a
[Pixtale](https://github.com/aslost/pixtale) gallery: live progress, dry-run
preview, opt-in deletes, and a storage-backend picker (Pixtale's S3 storage —
we use a Qiniu Kodo bucket, which is why this setup works well for family in
mainland China). Pixtale generates its own previews/thumbnails and serves a
lightweight public or password-protected web gallery. Photos already in the
gallery are deduped by content hash, so syncing is safe to re-run any time.

## Privacy

All analysis runs locally: candidate search talks only to *your* Immich server,
and scoring only to CPU-resident models. Photos leave your machine only if you
choose the optional pixtale sync, to a gallery and storage you operate. No
third-party API keys, no telemetry.

## Setup

Requires a running Immich server with face recognition enabled and your family's
people created (the `/settings` page lets you browse and pick them by name —
person UUIDs never need to be typed by hand).

### Option 1: Docker Compose (recommended)

```bash
git clone https://github.com/cxcxcxcx/immich-shortlist.git
cd immich-shortlist
# Edit docker-compose.yml with your IMMICH_URL, IMMICH_API_KEY, IMMICH_ALBUM_ID, and LOCAL_TZ
docker compose up -d --build
```

Open `http://localhost:5724`, visit **Settings** to pick your primary & family
faces, and start your first curation run. The `./data` volume persists your
settings, feature cache, checkpoints, and downloaded model weights across
restarts. The image uses CPU-only PyTorch and pre-bakes MediaPipe's
`face_landmarker.task` model.

### Option 2: Local Python (`uv`)

Requires Python ≥ 3.13:

```bash
# 1. Install dependencies
uv sync

# 2. Configure — copy the example and fill in your values
cp config.local.json.example config.local.json

# 3. Run (face_landmarker.task is downloaded automatically on first use)
uv run python daily_highlights.py     # UI on http://localhost:5724
```

Everything site-specific lives in `config.local.json` (gitignored) or the
matching environment variables (which win): Immich URL / API key / album ID,
the face-set seed lists, timezone, and the optional Pixtale endpoint. The first
curation run pays a one-time ~10 s ML import; each job afterwards runs in a
short-lived worker subprocess so gigabytes of model memory return to the OS
when it exits.

Headless mode is available for cron-style monthly runs:

```bash
uv run python daily_highlights.py --month 2026-04 --skip-in-album
# or via Docker:
docker compose run --rm shortlist python daily_highlights.py --month 2026-04 --skip-in-album
```

## Tuning what "best" means

`/settings` exposes the whole ranking function: weight of quality vs
expression vs person boost, per-component expression weights (eyes / smile /
facing, separately for primary and family faces), cluster similarity
threshold, max moments per day, and the default-selection floor. If the
tool's taste and your taste diverge, that's a settings conversation, not a
code change.

## Project status

Shortlist is a personal tool built for my own family's Immich library and shared
as-is under the MIT license. Issues and PRs are welcome, though maintenance
follows my own family's needs.

## License

[MIT](LICENSE)
