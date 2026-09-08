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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def delete_photos(client: PixtaleClient, photo_ids: list[str]) -> None:
    """Permanently delete photos in batches (storage objects + records)."""
    for i in range(0, len(photo_ids), DELETE_BATCH):
        batch = photo_ids[i : i + DELETE_BATCH]
        r = client.session.post(f"{client.base}/api/photo/delete", json={"photoIds": batch}, timeout=120)
        r.raise_for_status()
        body = r.json()
        if body.get("code") != 200:
            raise RuntimeError(f"/api/photo/delete -> {body.get('message') or body}")
        print(f"  deleted {min(i + DELETE_BATCH, len(photo_ids))}/{len(photo_ids)}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--immich-url", default=os.environ.get("IMMICH_URL", DEFAULT_IMMICH_URL))
    ap.add_argument("--api-key", default=os.environ.get("IMMICH_API_KEY", DEFAULT_API_KEY))
    ap.add_argument("--album-id", default=os.environ.get("IMMICH_ALBUM_ID", DEFAULT_ALBUM_ID))
    ap.add_argument("--base-url", default=os.environ.get("PIXTALE_URL", "http://localhost:8083"))
    ap.add_argument("--username", default=os.environ.get("PIXTALE_USER", "admin"))
    ap.add_argument("--password", default=os.environ.get("PIXTALE_PASSWORD") or _read_default_password())
    ap.add_argument("--storage-id", default=os.environ.get("PIXTALE_STORAGE_ID"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only add N missing photos (0 = all)")
    ap.add_argument("--delete", action="store_true", help="permanently delete Pixtale photos not in the album")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.password:
        raise SystemExit("Missing password: pass --password, set PIXTALE_PASSWORD, or populate ~/pixtale/.env")

    print("Fetching album assets from Immich...", file=sys.stderr)
    assets = _fetch_album_assets(args.immich_url, args.api_key, args.album_id)
    image_assets = {str(a["id"]): a for a in assets if not _is_video(a)}
    print(f"  album assets={len(assets)} images={len(image_assets)} (videos skipped)", file=sys.stderr)

    client = PixtaleClient(args.base_url, args.username, args.password, args.storage_id)
    client.immich_url = args.immich_url
    client.immich_api_key = args.api_key
    print(f"  pixtale storage={client.storage_id} at {args.base_url}", file=sys.stderr)

    existing = list_pixtale_photos(client)
    print(f"  pixtale photos={len(existing)}", file=sys.stderr)

    to_add = [aid for aid in image_assets if aid not in existing]
    to_delete = [(stem, pid) for stem, pid in existing.items() if stem not in image_assets]

    print(f"\nPlan: add={len(to_add)} delete={len(to_delete)} (deletes {'ENABLED' if args.delete else 'skipped, pass --delete'})", file=sys.stderr)
    for stem, pid in to_delete[:20]:
        print(f"  would delete photoId={pid} stem={stem}", file=sys.stderr)
    if len(to_delete) > 20:
        print(f"  ... and {len(to_delete) - 20} more", file=sys.stderr)
    if args.dry_run:
        for aid in to_add[:20]:
            taken = _asset_datetime(image_assets[aid]).strftime("%Y-%m-%d %H:%M")
            print(f"  would add {aid} taken={taken}", file=sys.stderr)
        if len(to_add) > 20:
            print(f"  ... and {len(to_add) - 20} more", file=sys.stderr)
        print("\nDry run: nothing touched.", file=sys.stderr)
        return

    if to_delete and args.delete:
        print(f"\nDeleting {len(to_delete)} photos not in album...", file=sys.stderr)
        delete_photos(client, [pid for _stem, pid in to_delete])

    if not to_add:
        print("\nNothing to add.", file=sys.stderr)
        return
    if args.limit:
        to_add = to_add[: args.limit]
        print(f"--limit {args.limit}: adding {len(to_add)} this run", file=sys.stderr)

    counters = {"ok": 0, "duplicate": 0, "error": 0}
    lock = threading.Lock()
    failures: list[tuple[str, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(upload_asset, client, image_assets[aid]): aid for aid in to_add}
        for i, fut in enumerate(as_completed(futures), start=1):
            status, detail = fut.result()
            aid = futures[fut]
            with lock:
                counters[status] += 1
                if status == "error":
                    failures.append((aid, detail))
            if i % 25 == 0 or status == "error":
                rate = i / max(time.time() - t0, 0.01)
                print(f"[{i}/{len(to_add)}] {status} {aid} {detail[:60]} ({rate:.1f}/s)", file=sys.stderr)

    print(
        f"\nDone. added={counters['ok']} duplicate={counters['duplicate']} "
        f"failed={counters['error']} deleted={len(to_delete) if args.delete else 0}",
        file=sys.stderr,
    )
    if failures:
        print("Failures:", file=sys.stderr)
        for aid, detail in failures[:30]:
            print(f"  {aid}: {detail}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
