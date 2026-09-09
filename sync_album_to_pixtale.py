#!/usr/bin/env python3
"""
Reconcile an Immich album into a Pixtale gallery (the album is the source of truth).

Pixtale cannot scan an object store: photos must enter through its own
``/api/photo/add`` pipeline. This script makes Pixtale mirror the album exactly:

  * ADD    album assets missing from Pixtale -> download the Immich preview,
           inject EXIF in-memory (same mapping as sync_immich_album_previews.py),
           upload as ``<asset_id>.<ext>``.
  * DELETE Pixtale photos whose name stem is not an album asset id, via
           ``/api/photo/delete`` (permanent: removes storage objects + records).

Videos are never synced (matching sync_immich_album_previews.py), so they are
also never deleted. Matching is by asset-id stem, so a preview that Immich
serves as a different format (jpg vs webp) does not cause a duplicate.

Safety:
  * Deletes only happen with ``--delete``.
  * ``--dry-run`` prints the plan (adds, deletes) and touches nothing.

Unlike upload_to_pixtale.py there is no local state file: the album and the
live ``/photo/list`` enumeration are both queried on every run, so re-adding a
previously deleted photo works without clearing any state.

The orchestration lives in ``run_sync()``, which is also imported by the web UI
(daily_highlights.py) — it takes a ``progress`` callback (phase events) and a
``cancelled`` predicate so a run can be watched and stopped from the browser.
The ``main()`` CLI below is a thin wrapper that prints the same stderr output
as before.

Configure via args / env: IMMICH_URL, IMMICH_API_KEY, IMMICH_ALBUM_ID,
PIXTALE_URL, PIXTALE_USER, PIXTALE_PASSWORD, PIXTALE_STORAGE_ID.
"""
from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from config import ALBUM_ID as DEFAULT_ALBUM_ID, API_KEY as DEFAULT_API_KEY, IMMICH_URL as DEFAULT_IMMICH_URL
from sync_immich_album_previews import (
    _asset_datetime,
    _attach_exif_json,
    _download_preview,
    _fetch_album_assets,
    _is_video,
)
from upload_to_pixtale import PixtaleClient, _read_default_password

DELETE_BATCH = 50
LIST_PAGE_SIZE = 500
SYNC_CONCURRENCY = 4


def _with_exif(content: bytes, asset_id: str, exif_info: dict, ext: str) -> bytes:
    """
    Attach EXIF using the same path-based routine as the NAS-tree sync (piexif's
    insert() has no pure in-memory mode), via a temp file.
    """
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"{asset_id}.{ext}"
        p.write_bytes(content)
        _attach_exif_json(p, exif_info, asset_id=asset_id, ext=ext)
        return p.read_bytes()


def _capture_ms(asset: dict) -> str:
    """Capture time as epoch ms, used as Pixtale's lastModified EXIF fallback."""
    return str(int(_asset_datetime(asset).timestamp() * 1000))


def list_pixtale_photos(client: PixtaleClient) -> dict[str, str]:
    """
    Enumerate all normal-status Pixtale photos as {name_stem: photoId}.
    Cursor pagination: the list is ordered by takenTime desc, so the last row
    of each page provides the next (cursorTime, cursorPhotoId).
    """
    stems: dict[str, str] = {}
    cursor_time: str | None = None
    cursor_photo_id: str | None = None
    while True:
        payload: dict = {"size": LIST_PAGE_SIZE}
        if cursor_time and cursor_photo_id:
            payload["cursorTime"] = cursor_time
            payload["cursorPhotoId"] = cursor_photo_id
        r = client.session.post(f"{client.base}/api/photo/list", json=payload, timeout=60)
        r.raise_for_status()
        body = r.json()
        if body.get("code") != 200:
            raise RuntimeError(f"/api/photo/list -> {body.get('message') or body}")
        rows = (body.get("data") or {}).get("list") or []
        if not rows:
            break
        for row in rows:
            name = str(row.get("name") or "")
            photo_id = str(row.get("photoId") or "")
            if name and photo_id:
                stems[name.rsplit(".", 1)[0]] = photo_id
        if len(rows) < LIST_PAGE_SIZE:
            break
        cursor_time = str(rows[-1].get("takenTime"))
        cursor_photo_id = str(rows[-1].get("photoId"))
    return stems


def upload_asset(client: PixtaleClient, asset: dict) -> tuple[str, str]:
    """
    Download one album asset's preview and push it through /api/photo/add.
    Returns (status, detail) with status in {ok, duplicate, error}.
    """
    asset_id = str(asset["id"])
    try:
        content, ext = _download_preview(client.immich_url, client.immich_api_key, asset_id)
        content = _with_exif(content, asset_id, asset.get("exifInfo") or {}, ext)
        mime = mimetypes.guess_type(f"x.{ext}")[0] or "image/webp"
        data = {"storageId": client.storage_id, "lastModified": _capture_ms(asset)}
        r = client.session.post(
            f"{client.base}/api/photo/add",
            files={"file": (f"{asset_id}.{ext}", content, mime)},
            data=data,
            timeout=300,
        )
        body = r.json()
        if r.status_code == 200 and body.get("code") == 200:
            payload = body.get("data") or {}
            if payload.get("duplicate") and not payload.get("photo"):
                return "duplicate", "identical bytes already in gallery"
            return "ok", "added"
        return "error", str(body.get("message") or body)[:200]
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"[:200]


def _delete_batch(client: PixtaleClient, batch: list[str]) -> None:
    """POST /api/photo/delete for one batch (permanent: storage objects + records)."""
    r = client.session.post(f"{client.base}/api/photo/delete", json={"photoIds": batch}, timeout=120)
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 200:
        raise RuntimeError(f"/api/photo/delete -> {body.get('message') or body}")


def run_sync(
    immich_url: str,
    api_key: str,
    album_id: str,
    base_url: str,
    username: str,
    password: str,
    storage_id: str | None = None,
    *,
    delete: bool = False,
    dry_run: bool = False,
    limit: int = 0,
    concurrency: int = SYNC_CONCURRENCY,
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """
    Reconcile one album into pixtale and return a JSON-serializable summary.

    ``progress`` receives flat phase events (``phase`` key), in order:
      fetching_album | fetching_pixtale (album/image counts, resolved storage_id)
      | planned (plan_add, plan_delete, samples_add/samples_delete <=20, flags)
      | deleting (per DELETE_BATCH: deleted, plan_delete)
      | adding (phase start with add_total, then one event per completed asset:
        add_done, add_total, last_status, last_asset, last_detail, counters, rate)
    Events are emitted from the calling thread only, and a raising callback is
    swallowed (a UI must never kill a sync).

    ``cancelled`` is polled between phases, between delete batches and at each
    refill of the upload window; already-submitted uploads always drain and are
    counted. Raises on unrecoverable errors (Immich, login, storage config);
    per-photo failures are reported in the summary, not raised.
    """
    t0 = time.time()

    def emit(ev: dict) -> None:
        if progress is None:
            return
        try:
            progress(ev)
        except Exception as e:  # a UI callback must never kill a sync
            print(f"progress callback failed: {e}", file=sys.stderr)

    def stop() -> bool:
        return bool(cancelled()) if cancelled else False

    # Filled in as phases complete, so even a very early cancel returns a full shape.
    result: dict = {
        "cancelled": False,
        "dry_run": dry_run,
        "delete_requested": delete,
        "storage_id": "",
        "album_assets": 0,
        "image_assets": 0,
        "pixtale_photos": 0,
        "plan_add": 0,
        "plan_delete": 0,
        "add_total": 0,
        "add_done": 0,
        "deleted": 0,
        "counters": {"ok": 0, "duplicate": 0, "error": 0},
        "failures": [],
        "samples_add": [],
        "samples_delete": [],
        "current_asset": "",
    }

    def summary() -> dict:
        result["elapsed_sec"] = round(time.time() - t0, 1)
        return result

    if stop():
        result["cancelled"] = True
        return summary()

    emit({"phase": "fetching_album"})
    assets = _fetch_album_assets(immich_url, api_key, album_id)
    image_assets = {str(a["id"]): a for a in assets if not _is_video(a)}
    result["album_assets"] = len(assets)
    result["image_assets"] = len(image_assets)

    client = PixtaleClient(base_url, username, password, storage_id)
    client.immich_url = immich_url  # duck-typed attrs relied on by upload_asset
    client.immich_api_key = api_key
    result["storage_id"] = client.storage_id
    emit({
        "phase": "fetching_pixtale",
        "album_assets": len(assets),
        "image_assets": len(image_assets),
        "storage_id": client.storage_id,
    })

    existing = list_pixtale_photos(client)
    result["pixtale_photos"] = len(existing)
    to_add = [aid for aid in image_assets if aid not in existing]
    to_delete = [(stem, pid) for stem, pid in existing.items() if stem not in image_assets]
    result["plan_add"] = len(to_add)
    result["plan_delete"] = len(to_delete)
    result["samples_add"] = [
        {"id": aid, "taken": _asset_datetime(image_assets[aid]).strftime("%Y-%m-%d %H:%M")}
        for aid in to_add[:20]
    ]
    result["samples_delete"] = [[stem, pid] for stem, pid in to_delete[:20]]
    emit({
        "phase": "planned",
        "pixtale_photos": len(existing),
        "plan_add": len(to_add),
        "plan_delete": len(to_delete),
        "dry_run": dry_run,
        "delete": delete,
        "samples_add": result["samples_add"],
        "samples_delete": result["samples_delete"],
    })

    if dry_run:
        return summary()
    if stop():
        result["cancelled"] = True
        return summary()

    if delete and to_delete:
        ids = [pid for _stem, pid in to_delete]
        for i in range(0, len(ids), DELETE_BATCH):
            if stop():
                result["cancelled"] = True
                return summary()
            _delete_batch(client, ids[i : i + DELETE_BATCH])
            result["deleted"] = min(i + DELETE_BATCH, len(ids))
            emit({"phase": "deleting", "deleted": result["deleted"], "plan_delete": len(ids)})

    todo = to_add[:limit] if limit else to_add
    result["add_total"] = len(todo)
    result["add_done"] = 0
    emit({"phase": "adding", "add_done": 0, "add_total": len(todo), "rate": 0.0})

    done = 0
    if todo:
        t_add = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            # Bounded in-flight window: cancel takes effect within ~2 batches and
            # already-submitted work always drains before the run ends.
            pending: dict = {}
            next_i = 0

            def refill() -> None:
                nonlocal next_i
                while len(pending) < concurrency * 2 and next_i < len(todo) and not stop():
                    aid = todo[next_i]
                    next_i += 1
                    pending[pool.submit(upload_asset, client, image_assets[aid])] = aid

            refill()
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    aid = pending.pop(fut)
                    status, detail = fut.result()  # upload_asset never raises
                    done += 1
                    result["counters"][status] += 1
                    if status == "error":
                        result["failures"].append([aid, detail])
                    result["current_asset"] = aid
                    emit({
                        "phase": "adding",
                        "add_done": done,
                        "add_total": len(todo),
                        "last_status": status,
                        "last_asset": aid,
                        "last_detail": detail,
                        "counters": dict(result["counters"]),
                        "rate": round(done / max(time.time() - t_add, 0.01), 1),
                    })
                refill()

    result["add_done"] = done
    result["cancelled"] = stop()
    return summary()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--immich-url", default=os.environ.get("IMMICH_URL", DEFAULT_IMMICH_URL))
    ap.add_argument("--api-key", default=os.environ.get("IMMICH_API_KEY", DEFAULT_API_KEY))
    ap.add_argument("--album-id", default=os.environ.get("IMMICH_ALBUM_ID", DEFAULT_ALBUM_ID))
    ap.add_argument("--base-url", default=os.environ.get("PIXTALE_URL", "http://localhost:8083"))
    ap.add_argument("--username", default=os.environ.get("PIXTALE_USER", "admin"))
    ap.add_argument("--password", default=os.environ.get("PIXTALE_PASSWORD") or _read_default_password())
    ap.add_argument("--storage-id", default=os.environ.get("PIXTALE_STORAGE_ID"))
    ap.add_argument("--concurrency", type=int, default=SYNC_CONCURRENCY)
    ap.add_argument("--limit", type=int, default=0, help="only add N missing photos (0 = all)")
    ap.add_argument("--delete", action="store_true", help="permanently delete Pixtale photos not in the album")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.password:
        raise SystemExit("Missing password: pass --password, set PIXTALE_PASSWORD, or populate ~/pixtale/.env")

    # CLI formatter: turns run_sync's phase events into the traditional stderr lines.
    shown = {"deleting_header": False, "adding_header": False}

    def _show(ev: dict) -> None:
        phase = ev.get("phase")
        if phase == "fetching_album":
            print("Fetching album assets from Immich...", file=sys.stderr)
        elif phase == "fetching_pixtale":
            print(f"  album assets={ev['album_assets']} images={ev['image_assets']} (videos skipped)", file=sys.stderr)
            print(f"  pixtale storage={ev['storage_id']} at {args.base_url}", file=sys.stderr)
        elif phase == "planned":
            print(f"  pixtale photos={ev['pixtale_photos']}", file=sys.stderr)
            print(f"\nPlan: add={ev['plan_add']} delete={ev['plan_delete']}"
                  f" (deletes {'ENABLED' if ev['delete'] else 'skipped, pass --delete'})", file=sys.stderr)
            for stem, pid in ev["samples_delete"]:
                print(f"  would delete photoId={pid} stem={stem}", file=sys.stderr)
            if ev["plan_delete"] > 20:
                print(f"  ... and {ev['plan_delete'] - 20} more", file=sys.stderr)
            if ev["dry_run"]:
                for s in ev["samples_add"]:
                    print(f"  would add {s['id']} taken={s['taken']}", file=sys.stderr)
                if ev["plan_add"] > 20:
                    print(f"  ... and {ev['plan_add'] - 20} more", file=sys.stderr)
        elif phase == "deleting":
            if not shown["deleting_header"]:
                shown["deleting_header"] = True
                print(f"\nDeleting {ev['plan_delete']} photos not in album...", file=sys.stderr)
            print(f"  deleted {ev['deleted']}/{ev['plan_delete']}", file=sys.stderr)
        elif phase == "adding":
            if not shown["adding_header"]:
                shown["adding_header"] = True
                if ev["add_total"] == 0:
                    print("\nNothing to add.", file=sys.stderr)
                elif args.limit:
                    print(f"--limit {args.limit}: adding {ev['add_total']} this run", file=sys.stderr)
            elif ev["add_done"] and (ev["add_done"] % 25 == 0 or ev["last_status"] == "error"):
                print(f"[{ev['add_done']}/{ev['add_total']}] {ev['last_status']} {ev['last_asset']}"
                      f" {str(ev['last_detail'])[:60]} ({ev['rate']:.1f}/s)", file=sys.stderr)

    summary = run_sync(
        args.immich_url, args.api_key, args.album_id, args.base_url,
        args.username, args.password, args.storage_id,
        delete=args.delete, dry_run=args.dry_run, limit=args.limit,
        concurrency=args.concurrency, progress=_show,
    )
    if args.dry_run:
        print("\nDry run: nothing touched.", file=sys.stderr)
        return
    if summary["plan_add"] == 0:  # "\nNothing to add." was printed by the formatter
        return

    print(
        f"\nDone. added={summary['counters']['ok']} duplicate={summary['counters']['duplicate']} "
        f"failed={summary['counters']['error']} deleted={summary['deleted']}",
        file=sys.stderr,
    )
    if summary["failures"]:
        print("Failures:", file=sys.stderr)
        for aid, detail in summary["failures"][:30]:
            print(f"  {aid}: {detail}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
