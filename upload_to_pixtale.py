#!/usr/bin/env python3
"""
Bulk-import local photos into a Pixtale gallery.

Pixtale cannot scan an object store: photos must enter through its own
``/api/photo/add`` pipeline (which generates previews/thumbnails, parses EXIF,
and uploads original + derived images to the configured storage backend — here
a Qiniu Kodo S3-compatible bucket). This script walks a local photo directory
(the Immich preview sync tree) and feeds every image to that API.

The synced ``.webp`` previews already carry EXIF (see
``sync_immich_album_previews.py``), so Pixtale reads capture date / GPS / camera
directly; we also pass the file mtime as ``lastModified`` as a fallback for any
image whose EXIF is missing.

Resume / idempotency:
  * A local state file records every file (relative path + size + mtime) that
    was already sent, so re-runs skip it without resending bytes.
  * Pixtale itself also dedupes by SHA-1, so a photo already present is reported
    as ``duplicate`` and treated as success.

Configure via args / env:
  --base-url   PIXTALE_URL     (default http://localhost:8083)
  --username   PIXTALE_USER    (default admin)
  --password   PIXTALE_PASSWORD (default: read from ~/pixtale/.env PASSWORD)
  --root       SYNC_OUT_ROOT   (default from config.py, the kids_photos tree)
  --storage-id PIXTALE_STORAGE_ID (default: auto-detect the S3/object storage)
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from config import SYNC_OUT_ROOT as DEFAULT_ROOT

IMAGE_EXTS = {".webp", ".jpg", ".jpeg", ".png", ".gif"}
DEFAULT_STATE = Path.home() / ".pixtale_upload_state.json"

# Terminal colour helpers (no-op when not a TTY).
def _c(code: str) -> str:
    return code if sys.stderr.isatty() else ""


GREEN, RED, YELLOW, DIM, RESET = _c("\033[32m"), _c("\033[31m"), _c("\033[33m"), _c("\033[2m"), _c("\033[0m")


def _read_default_password() -> str | None:
    """Read the admin password from the pixtale compose .env if present."""
    env_file = Path.home() / "pixtale" / ".env"
    if not env_file.is_file():
        return None
    for line in env_file.read_text().splitlines():
        if line.startswith("PASSWORD="):
            return line.split("=", 1)[1].strip() or None
    return None


class State:
    """JSON file recording which source files have already been uploaded."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._done: dict[str, int] = {}
        if path.is_file():
            try:
                self._done = json.loads(path.read_text()).get("done", {})
            except Exception:
                self._done = {}

    @staticmethod
    def key_for(root: Path, f: Path) -> str:
        st = f.stat()
        rel = f.relative_to(root).as_posix()
        # Include size + mtime so a changed file is re-uploaded.
        return f"{rel}|{st.st_size}|{int(st.st_mtime)}"

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._done

    def mark(self, key: str) -> None:
        with self._lock:
            self._done[key] = int(time.time())
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps({"done": self._done}))
            tmp.replace(self.path)


class PixtaleClient:
    def __init__(self, base_url: str, username: str, password: str, storage_id: str | None):
        self.base = base_url.rstrip("/")
        self.storage_id = storage_id
        self.session = requests.Session()
        self._login(username, password)
        if not self.storage_id:
            self.storage_id = self._detect_storage()

    def _post_json(self, path: str, payload: dict, timeout: int = 30) -> dict:
        r = self.session.post(f"{self.base}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != 200:
            raise RuntimeError(f"{path} -> {data.get('message') or data}")
        return data

    def _login(self, username: str, password: str) -> None:
        data = self._post_json("/api/login", {"username": username, "password": password})
        if not (data.get("data") or {}).get("token"):
            raise RuntimeError("login succeeded but no token returned")

    def _detect_storage(self) -> str:
        """Pick the first enabled object-storage (S3) backend; fall back to first."""
        data = self._post_json("/api/storage/select", {})
        options = data.get("data") or []
        if not options:
            raise RuntimeError("no storage configured in pixtale")
        # type 2 == object storage (S3); type 1 == local.
        for opt in options:
            if opt.get("type") == 2:
                return opt["storageId"]
        return options[0]["storageId"]

    def upload(self, f: Path) -> tuple[str, str]:
        """Upload one file. Returns (status, detail). status in {ok, duplicate, error}."""
        mime = mimetypes.guess_type(f.name)[0] or "image/webp"
        last_modified = int(f.stat().st_mtime * 1000)
        with open(f, "rb") as fh:
            files = {"file": (f.name, fh, mime)}
            data = {"storageId": self.storage_id, "lastModified": str(last_modified)}
            r = self.session.post(f"{self.base}/api/photo/add", files=files, data=data, timeout=300)
        try:
            body = r.json()
        except Exception:
            return "error", f"HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code == 200 and body.get("code") == 200:
            payload = body.get("data") or {}
            if payload.get("duplicate") and not payload.get("photo"):
                return "duplicate", "already in gallery"
            photo = payload.get("photo") or {}
            return "ok", str(photo.get("takenTime") or "")
        return "error", f"{body.get('message') or body}"


def iter_images(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in IMAGE_EXTS:
                yield p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("SYNC_OUT_ROOT", DEFAULT_ROOT)),
                    help="photo tree to import (default: SYNC_OUT_ROOT / kids_photos)")
    ap.add_argument("--base-url", default=os.environ.get("PIXTALE_URL", "http://localhost:8083"))
    ap.add_argument("--username", default=os.environ.get("PIXTALE_USER", "admin"))
    ap.add_argument("--password", default=os.environ.get("PIXTALE_PASSWORD") or _read_default_password())
    ap.add_argument("--storage-id", default=os.environ.get("PIXTALE_STORAGE_ID"),
                    help="pixtale storageId (default: auto-detect S3 backend)")
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE, help="resume state file")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only process N new files (0 = all)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.password:
        sys.exit("Missing password: pass --password, set PIXTALE_PASSWORD, or populate ~/pixtale/.env")
    if not args.root.is_dir():
        sys.exit(f"Root not found: {args.root}")

    files = sorted(iter_images(args.root))
    print(f"{DIM}Found {len(files)} images under {args.root}{RESET}", file=sys.stderr)

    client = None
    if not args.dry_run:
        client = PixtaleClient(args.base_url, args.username, args.password, args.storage_id)
        print(f"{DIM}Using storage {client.storage_id} at {args.base_url}{RESET}", file=sys.stderr)

    state = State(args.state)
    pending = [f for f in files if not state.has(State.key_for(args.root, f))]
    print(
        f"{DIM}Pending upload: {len(pending)} (already done: {len(files) - len(pending)}){RESET}",
        file=sys.stderr,
    )
    todo = pending[: args.limit] if args.limit else pending
    if args.limit:
        print(f"{DIM}--limit {args.limit}: processing {len(todo)} this run{RESET}", file=sys.stderr)

    if args.dry_run:
        for f in todo[:20]:
            print(f"  would upload {f.relative_to(args.root)}")
        print(f"... ({len(todo)} total)")
        return

    counters = {"ok": 0, "duplicate": 0, "error": 0}
    lock = threading.Lock()
    failures: list[tuple[str, str]] = []

    def work(f: Path) -> tuple[str, str, Path]:
        key = State.key_for(args.root, f)
        last_err = ""
        for attempt in range(args.retries):
            try:
                status, detail = client.upload(f)
                if status in ("ok", "duplicate"):
                    state.mark(key)
                return status, detail, f
            except Exception as e:  # transient (network/5xx) -> retry
                last_err = str(e)
                time.sleep(min(2 ** attempt, 10))
        return "error", last_err, f

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(work, f) for f in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            status, detail, f = fut.result()
            with lock:
                counters[status] += 1
                if status == "error":
                    failures.append((str(f.relative_to(args.root)), detail))
            tag = {
                "ok": f"{GREEN}ok{RESET}",
                "duplicate": f"{YELLOW}dup{RESET}",
                "error": f"{RED}ERROR{RESET}",
            }[status]
            if i % 25 == 0 or status == "error":
                rate = i / max(time.time() - t0, 0.01)
                print(f"[{i}/{len(todo)}] {tag} {f.relative_to(args.root)} {DIM}{detail[:80]} ({rate:.1f}/s){RESET}",
                      file=sys.stderr)

    print(
        f"\nDone. uploaded={counters['ok']} duplicate={counters['duplicate']} "
        f"failed={counters['error']} total_seen={len(files)}",
        file=sys.stderr,
    )
    if failures:
        print(f"{RED}Failures:{RESET}", file=sys.stderr)
        for path, detail in failures[:30]:
            print(f"  {path}: {detail[:160]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
