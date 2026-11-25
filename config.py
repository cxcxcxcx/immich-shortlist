"""
Deployment configuration for Shortlist (immich-shortlist).

No site-specific values live in code. Each setting resolves as:

    environment variable  >  config.local.json  >  default

`config.local.json` is gitignored (see config.local.json.example for keys).
Immich: IMMICH_URL, IMMICH_API_KEY, IMMICH_ALBUM_ID, FACE_IDS, FAMILY_FACES.
Paths:  SYNC_OUT_ROOT, LOCAL_TZ.
Pixtale (see upload_to_pixtale.py / sync_album_to_pixtale.py): PIXTALE_URL,
PIXTALE_USER, PIXTALE_STORAGE_ID (the password is read from PIXTALE_PASSWORD
or ~/pixtale/.env at use time — config can't import upload_to_pixtale, that
would be circular).
"""
import json
import os

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR") or _APP_DIR

_local: dict = {}
for _candidate in (
    os.path.join(_APP_DIR, "config.local.json"),
    os.path.join(DATA_DIR, "config.local.json"),
):
    if os.path.isfile(_candidate):
        with open(_candidate) as f:
            _local = json.load(f)
        break


def _setting(env: str, key: str | None = None, default=None):
    """Env var wins, then config.local.json, then the default."""
    value = os.environ.get(env)
    if value not in (None, ""):
        return value
    return _local.get(key or env, default)


def _list_setting(env: str, key: str | None = None, default=None) -> list:
    """Like _setting, but parses comma-separated env strings into a list."""
    value = _setting(env, key=key, default=default if default is not None else [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value) if value else []


PORT = int(_setting("PORT", default=5724))
IMMICH_URL = _setting("IMMICH_URL", default="http://localhost:2283")
API_KEY = _setting("IMMICH_API_KEY", default="")
ALBUM_ID = _setting("IMMICH_ALBUM_ID", default="")
SYNC_OUT_ROOT = _setting("SYNC_OUT_ROOT", default="")
LOCAL_TZ_NAME = _setting("LOCAL_TZ", "LOCAL_TZ_NAME", default="America/Los_Angeles")

# Immich person UUIDs used to seed the scoring params (the /settings page is
# the editor of record afterwards; these are just the initial face sets).
FACE_IDS = _list_setting("FACE_IDS", default=[])
FAMILY_FACES = _list_setting("FAMILY_FACES", default=[])

# Pixtale gallery (password deliberately absent; see module docstring).
PIXTALE_URL = _setting("PIXTALE_URL", default="http://localhost:8083")
PIXTALE_USER = _setting("PIXTALE_USER", default="admin")
PIXTALE_STORAGE_ID = _setting("PIXTALE_STORAGE_ID", default="")

# Optional EXIF-based source-album rules for sync_album_to_pixtale.py.
# Empty by default (disabled). Example in config.local.json:
# [{"album": "Camera", "make_contains": ["canon"]}, {"album": "Other", "default": true}]
PIXTALE_SOURCE_ALBUMS = _local.get("PIXTALE_SOURCE_ALBUMS", [])
