#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
from pathlib import Path

import piexif
import requests

from config import (
    API_KEY as DEFAULT_API_KEY,
    ALBUM_ID as DEFAULT_ALBUM_ID,
    IMMICH_URL as DEFAULT_IMMICH_URL,
    SYNC_OUT_ROOT,
)


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "Accept": "application/json",
    }


def _parse_immich_iso(s: str) -> dt.datetime:
    # Immich commonly returns ISO strings like "...Z" or with timezone offsets.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s)


def _asset_date(asset: dict) -> dt.date:
    return _asset_datetime(asset).date()


def _asset_datetime(asset: dict) -> dt.datetime:
    """Best capture/create datetime for this asset (for mtime and ordering)."""
    candidates = [
        asset.get("localDateTime"),
        asset.get("fileCreatedAt"),
        asset.get("exifInfo", {}).get("dateTimeOriginal") if isinstance(asset.get("exifInfo"), dict) else None,
        asset.get("createdAt"),
        asset.get("updatedAt"),
    ]
    for s in candidates:
        if not s:
            continue
        return _parse_immich_iso(str(s))
    raise RuntimeError(f"Could not determine date for asset: keys={sorted(asset.keys())}")


def _resolve_album_id(immich_url: str, api_key: str, album_id: str | None, album_name: str | None) -> str:
    if album_id:
        return album_id
    if not album_name:
        raise SystemExit("Provide --album-id or --album-name (or set IMMICH_ALBUM_ID).")

    url = f"{immich_url.rstrip('/')}/api/albums"
    resp = requests.get(url, headers=_auth_headers(api_key), timeout=30)
    resp.raise_for_status()
    albums = resp.json()
    if not isinstance(albums, list):
        raise RuntimeError(f"Unexpected /api/albums response type: {type(albums)}")

    # Exact match first, then case-insensitive.
    for a in albums:
        if isinstance(a, dict) and a.get("albumName") == album_name:
            aid = a.get("id")
            if aid:
                return str(aid)
    for a in albums:
        if isinstance(a, dict) and str(a.get("albumName", "")).lower() == album_name.lower():
            aid = a.get("id")
            if aid:
                return str(aid)

    names = sorted({str(a.get("albumName")) for a in albums if isinstance(a, dict) and a.get("albumName")})
    raise RuntimeError(f"Album name not found: {album_name!r}. Available names (sample): {names[:30]}")


def _is_video(asset: dict) -> bool:
    """True if the asset is a video (we skip syncing video previews)."""
    if asset.get("type") == "VIDEO":
        return True
    if asset.get("isVideo") is True:
        return True
    return False


def _fetch_album_assets(immich_url: str, api_key: str, album_id: str) -> list[dict]:
    url = f"{immich_url.rstrip('/')}/api/albums/{album_id}"
    resp = requests.get(url, headers=_auth_headers(api_key), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected /api/albums/{{id}} response type: {type(data)}")
    assets = data.get("assets") or []
    if not isinstance(assets, list):
        raise RuntimeError(f"Unexpected assets type: {type(assets)}")
    return [a for a in assets if isinstance(a, dict) and a.get("id")]


def _download_preview(immich_url: str, api_key: str, asset_id: str) -> tuple[bytes, str]:
    """
    Download thumbnail; return (content, extension) with extension from actual format.
    Immich may return JPEG or WebP depending on asset/encoding.
    """
    url = f"{immich_url.rstrip('/')}/api/assets/{asset_id}/thumbnail"
    resp = requests.get(
        url,
        headers={"x-api-key": api_key},
        params={"size": "preview"},
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.content
    # Prefer Content-Type, then magic bytes
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "webp" in ctype:
        return content, "webp"
    if "jpeg" in ctype or "jpg" in ctype:
        return content, "jpg"
    if content[:3] == b"\xff\xd8\xff":
        return content, "jpg"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return content, "webp"
    return content, "jpg"


def _target_path(out_root: Path, taken_date: dt.date, asset_id: str, ext: str) -> Path:
    """
    One directory per month: <out_root>/<YYYY-mm>/<asset_id>.<ext>
    """
    yyyy_mm = taken_date.strftime("%Y-%m")
    return out_root / yyyy_mm / f"{asset_id}.{ext}"


def _preview_exists(out_root: Path, taken_date: dt.date, asset_id: str) -> bool:
    """True if a preview for this asset already exists (any supported extension)."""
    yyyy_mm = taken_date.strftime("%Y-%m")
    month_dir = out_root / yyyy_mm
    if not month_dir.is_dir():
        return False
    for ext in ("jpg", "jpeg", "webp"):
        if (month_dir / f"{asset_id}.{ext}").exists():
            return True
    return False


def _set_file_mtime(path: Path, taken: dt.datetime) -> None:
    """Set file atime/mtime to the asset's capture time so creation date is preserved."""
    ts = taken.timestamp()
    os.utime(path, (ts, ts))


def _maybe_parse_exif_datetime(s: str) -> str:
    """
    Best-effort normalize a datetime string into EXIF format 'YYYY:MM:DD HH:MM:SS'.
    If parsing fails, return the original string.
    """
    if not s:
        return s
    # Already EXIF-like
    if len(s) >= 19 and s[4] == ":" and s[7] == ":" and s[10] == " ":
        return s[:19]
    try:
        # Handle common ISO-8601 variants Immich might return
        iso = s.rstrip("Z")
        dt_obj = dt.datetime.fromisoformat(iso)
        return dt_obj.strftime("%Y:%m:%d %H:%M:%S")
    except Exception:
        return s


def _exif_str_bytes(s: str) -> bytes:
    """Encode as UTF-8 for EXIF text tags; piexif writes bytes as-is (preserves e.g. β)."""
    return s.encode("utf-8")


def _apply_exifinfo_to_dict(exif_dict: dict, exif_info: dict) -> None:
    """
    Map a subset of Immich exifInfo fields onto real EXIF tags.
    This is best-effort; unknown fields are ignored.
    """
    zeroth = exif_dict.setdefault("0th", {})
    exif = exif_dict.setdefault("Exif", {})
    gps = exif_dict.setdefault("GPS", {})

    # Date/time
    dt_raw = (
        exif_info.get("dateTimeOriginal")
        or exif_info.get("dateTime")
        or exif_info.get("DateTimeOriginal")
    )
    if isinstance(dt_raw, str) and dt_raw:
        dt_exif = _maybe_parse_exif_datetime(dt_raw)
        if dt_exif:
            dt_bytes = _exif_str_bytes(dt_exif)
            exif[piexif.ExifIFD.DateTimeOriginal] = dt_bytes
            zeroth[piexif.ImageIFD.DateTime] = dt_bytes

    # Camera make/model
    make = (
        exif_info.get("make")
        or exif_info.get("cameraMake")
        or exif_info.get("Make")
    )
    if isinstance(make, str) and make:
        zeroth[piexif.ImageIFD.Make] = _exif_str_bytes(make)

    model = (
        exif_info.get("model")
        or exif_info.get("cameraModel")
        or exif_info.get("Model")
    )
    if isinstance(model, str) and model:
        zeroth[piexif.ImageIFD.Model] = _exif_str_bytes(model)

    # Lens model
    lens_model = (
        exif_info.get("lensModel")
        or exif_info.get("LensModel")
    )
    if isinstance(lens_model, str) and lens_model:
        exif[piexif.ExifIFD.LensModel] = _exif_str_bytes(lens_model)

    # Exposure settings
    iso = exif_info.get("iso") or exif_info.get("ISO") or exif_info.get("isoSpeed")
    if isinstance(iso, (int, float)) and iso > 0:
        exif[piexif.ExifIFD.ISOSpeedRatings] = int(iso)

    fnumber = exif_info.get("fNumber") or exif_info.get("FNumber")
    if isinstance(fnumber, (int, float)) and fnumber > 0:
        exif[piexif.ExifIFD.FNumber] = (int(round(fnumber * 100)), 100)

    exposure_time = exif_info.get("exposureTime") or exif_info.get("ExposureTime")
    if isinstance(exposure_time, (int, float)) and exposure_time > 0:
        # Store as a rational approximate (1 / exposure_time seconds)
        denom = 1000000
        num = int(round(exposure_time * denom))
        exif[piexif.ExifIFD.ExposureTime] = (num, denom)

    focal_length = exif_info.get("focalLength") or exif_info.get("FocalLength")
    if isinstance(focal_length, (int, float)) and focal_length > 0:
        exif[piexif.ExifIFD.FocalLength] = (int(round(focal_length * 100)), 100)

    # GPS (latitude/longitude in decimal degrees)
    lat = exif_info.get("latitude") or exif_info.get("Latitude")
    lon = exif_info.get("longitude") or exif_info.get("Longitude")

    def _deg_to_dms_rational(deg: float):
        deg_abs = abs(deg)
        d = int(deg_abs)
        m_float = (deg_abs - d) * 60.0
        m = int(m_float)
        s = (m_float - m) * 60.0
        return (
            (d, 1),
            (m, 1),
            (int(round(s * 100)), 100),
        )

    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        gps[piexif.GPSIFD.GPSLatitudeRef] = "N" if lat >= 0 else "S"
        gps[piexif.GPSIFD.GPSLatitude] = _deg_to_dms_rational(lat)
        gps[piexif.GPSIFD.GPSLongitudeRef] = "E" if lon >= 0 else "W"
        gps[piexif.GPSIFD.GPSLongitude] = _deg_to_dms_rational(lon)


def _attach_exif_json(path: Path, exif_info: dict, asset_id: str = "", ext: str = "jpg") -> None:
    """
    Apply exifInfo onto real EXIF tags (DateTimeOriginal, Make/Model, lens, ISO, etc.)
    for JPEG and WebP previews. Fails loudly on error.
    """
    aid = asset_id or path.stem
    if ext == "webp":
        # WebP often has no EXIF; build from scratch and insert.
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
        _apply_exifinfo_to_dict(exif_dict, exif_info)
        exif_bytes = piexif.dump(exif_dict)
        try:
            piexif.insert(exif_bytes, str(path))
        except Exception as e:
            raise RuntimeError(
                f"EXIF attachment failed for file path={path!s} asset_id={aid}: {e}"
            ) from e
        return

    # JPEG: load existing EXIF (if any), merge structured fields, insert.
    try:
        exif_dict = piexif.load(str(path))
    except Exception as e:
        raise RuntimeError(
            f"EXIF attachment failed for file path={path!s} asset_id={aid}: {e}"
        ) from e
    _apply_exifinfo_to_dict(exif_dict, exif_info)
    exif_bytes = piexif.dump(exif_dict)
    try:
        piexif.insert(exif_bytes, str(path))
    except Exception as e:
        raise RuntimeError(
            f"EXIF attachment failed for file path={path!s} asset_id={aid}: {e}"
        ) from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Immich album assets to a folder as preview images (JPEG or WebP per response); download only if missing."
    )
    parser.add_argument("--immich-url", default=os.environ.get("IMMICH_URL", DEFAULT_IMMICH_URL))
    parser.add_argument("--api-key", default=os.environ.get("IMMICH_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--album-id", default=os.environ.get("IMMICH_ALBUM_ID", DEFAULT_ALBUM_ID))
    parser.add_argument("--album-name", default=os.environ.get("IMMICH_ALBUM_NAME"))
    parser.add_argument("--out-root", type=Path, default=SYNC_OUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing --api-key (or set IMMICH_API_KEY).")

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)  # ensure root exists before any downloads

    album_id = _resolve_album_id(args.immich_url, args.api_key, args.album_id, args.album_name)
    assets = _fetch_album_assets(args.immich_url, args.api_key, album_id)

    downloaded = 0
    skipped = 0
    skipped_video = 0
    for i, asset in enumerate(assets, start=1):
        if _is_video(asset):
            skipped_video += 1
            continue
        asset_id = str(asset["id"])
        taken_date = _asset_date(asset)
        if _preview_exists(out_root, taken_date, asset_id):
            skipped += 1
            continue

        if args.dry_run:
            downloaded += 1
            continue

        # Download preview (Immich may return JPEG or WebP); save with correct extension.
        content, ext = _download_preview(args.immich_url, args.api_key, asset_id)
        dst = _target_path(out_root, taken_date, asset_id, ext)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)

        # Preserve creation date on the file (mtime).
        taken_dt = _asset_datetime(asset)
        _set_file_mtime(dst, taken_dt)

        exif_info = asset.get("exifInfo")
        if isinstance(exif_info, dict) and exif_info:
            _attach_exif_json(dst, exif_info, asset_id=asset_id, ext=ext)
        downloaded += 1

        if downloaded <= 3 or (downloaded % 50 == 0):
            print(f"[{i}/{len(assets)}] wrote {dst}")

    print(
        f"Done. album_assets={len(assets)} downloaded={downloaded} skipped_existing={skipped} "
        f"skipped_video={skipped_video} out_root={out_root}"
    )


if __name__ == "__main__":
    main()

