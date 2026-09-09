"""
Shared Immich config for photo_highlights scripts.
Override via environment: IMMICH_URL, IMMICH_API_KEY, IMMICH_ALBUM_ID, SYNC_OUT_ROOT.
"""
import os
from pathlib import Path

IMMICH_URL = os.environ.get("IMMICH_URL", "http://localhost:2283")
API_KEY = os.environ.get("IMMICH_API_KEY", "69T5wx39HYcNg2DyfM3Yvz0IQOY5hTtBREGjAzjXNw")
ALBUM_ID = os.environ.get("IMMICH_ALBUM_ID", "fced049d-52dc-4fc0-aa50-8962dc4506d2")
SYNC_OUT_ROOT = Path(
    os.environ.get("SYNC_OUT_ROOT", "/media/NASRoot/homes/admin/.Qsync/kids_photos")
)
LOCAL_TZ_NAME = os.environ.get("LOCAL_TZ", "America/Los_Angeles")

# Pixtale gallery (see upload_to_pixtale.py / sync_album_to_pixtale.py).
# No password here: it is read from PIXTALE_PASSWORD or ~/pixtale/.env at use
# time (config can't import upload_to_pixtale — that would be circular).
PIXTALE_URL = os.environ.get("PIXTALE_URL", "http://localhost:8083")
PIXTALE_USER = os.environ.get("PIXTALE_USER", "admin")
PIXTALE_STORAGE_ID = os.environ.get("PIXTALE_STORAGE_ID", "")
