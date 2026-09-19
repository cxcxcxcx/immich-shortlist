"""
Deployment configuration for photo_highlights scripts.

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

_LOCAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.local.json")
_local: dict = {}
if os.path.isfile(_LOCAL_PATH):
    with open(_LOCAL_PATH) as f:
        _local = json.load(f)


def _setting(env: str, key: str | None = None, default=None):
    """Env var wins, then config.local.json, then the default."""
    value = os.environ.get(env)
    if value not in (None, ""):
        return value
    return _local.get(key or env, default)


IMMICH_URL = _setting("IMMICH_URL", default="http://localhost:2283")
API_KEY = _setting("IMMICH_API_KEY", default="")
ALBUM_ID = _setting("IMMICH_ALBUM_ID", default="")
SYNC_OUT_ROOT = _setting("SYNC_OUT_ROOT", default="")
LOCAL_TZ_NAME = _setting("LOCAL_TZ", "LOCAL_TZ_NAME", default="America/Los_Angeles")

# Immich person UUIDs used to seed the scoring params (the /settings page is
# the editor of record afterwards; these are just the initial face sets).
FACE_IDS = _setting("FACE_IDS", default=[])
FAMILY_FACES = _setting("FAMILY_FACES", default=[])

# Pixtale gallery (password deliberately absent; see module docstring).
PIXTALE_URL = _setting("PIXTALE_URL", default="http://localhost:8083")
PIXTALE_USER = _setting("PIXTALE_USER", default="admin")
PIXTALE_STORAGE_ID = _setting("PIXTALE_STORAGE_ID", default="")
