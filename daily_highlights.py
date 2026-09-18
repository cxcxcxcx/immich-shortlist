"""
Immich Album Curator

This script automates the curation of photos for an Immich album.
It searches for photos containing specific faces within a date range,
evaluates image quality with a CLIP-based aesthetic head and facial
expression/eye state with a MediaPipe Face Landmarker, clusters them
for variety, and presents the top choices in a local web UI for final
confirmation before adding them to an album.
"""

import argparse
import calendar
import copy
import ctypes
import gc
import os
import subprocess
import sys
import io
import json
import re
import time
import base64
import requests
import datetime
import threading
from zoneinfo import ZoneInfo
from PIL import Image

# ML / evaluation. torch / transformers / pyiqa / mediapipe are imported lazily
# inside the functions that use them: importing all four costs ~1.3 GB of RSS
# (mediapipe alone drags in TensorFlow), which the gallery/sync UI never needs
# until a curation or re-score actually runs. numpy/PIL stay top-level (cheap).
import numpy as np

# Web UI
from flask import Flask, render_template, request, redirect, url_for, jsonify

# ==========================================
# CONFIGURATION
# ==========================================
from config import IMMICH_URL, API_KEY, ALBUM_ID, LOCAL_TZ_NAME
from config import PIXTALE_URL, PIXTALE_USER, PIXTALE_STORAGE_ID
from sync_album_to_pixtale import run_sync, SYNC_CONCURRENCY
from upload_to_pixtale import PixtaleClient, _read_default_password, default_storage_id

# Local timezone used to interpret date boundaries (inclusive days).
LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)

# List of Face/Person UUIDs to search for
FACE_IDS = [
    "aa79eb75-13ba-4a9f-8ee6-733606733577",
    "42107bf5-8241-4780-8795-f654176e5b3c",
]

FAMILY_FACES = [
    "3e00abfd-f940-4abb-a0b8-7b4b60a6ce66",
    "2712e983-c68b-44cd-ac25-cea81edff32c",
    "fb5339f7-fe52-4382-97a0-27dff719eae8",
    "ea5e068a-3d4c-4bd6-8035-426749eaa2ed",
    "a23eb31d-3ff8-44c9-a2ea-b90819646d24",
    "8300f245-08e9-422d-8f90-96910cbd04bd",
]

# ==========================================
# TUNABLE PARAMETERS (UI-editable via /settings)
# ==========================================
# All scoring knobs live in one dict, persisted in curation_params.json and
# editable from the settings page. Scoring code must read params from the
# snapshot handed to it (jobs take snapshot_params()), never from here
# directly, so a run keeps one consistent config even if settings change
# mid-run. The old module-level constants were folded into DEFAULT_PARAMS.
DEFAULT_PARAMS: dict = {
    # Combined score: total = w_q·q_norm + w_e·e_norm + person_boost + boosts
    "quality_weight": 0.4,
    "expression_weight": 0.2,
    "person_weight": 0.2,
    # Global normalization (z-score): MUSIQ raw quality is typically ~60–74,
    # expression scores land in roughly [-0.5, 1.0].
    "quality_mean": 65.0,
    "quality_std": 6.0,
    "expression_mean": 0.25,
    "expression_std": 0.3,
    # Expression per-face sub-weights: expr = w_eyes·eyes + w_smile·smile + w_facing·facing
    "expr_w_eyes": 0.5,
    "expr_w_smile": 0.2,
    "expr_w_facing": 0.3,
    # Family boost: per family face with facing >= min, add discount·(w·eyes + w·smile + w·facing)
    "family_facing_min": 0.75,
    "family_boost_discount": 0.5,
    "family_boost_weight_eyes": 0.25,
    "family_boost_weight_smile": 0.25,
    "family_boost_weight_facing": 0.5,
    # Clustering: CLIP cosine threshold (higher = fewer, tighter clusters) and
    # the cluster-top bonus that grows with cluster size and saturates.
    "clip_cluster_sim_threshold": 0.9,
    "cluster_top_weight": 0.5,
    "cluster_top_max_size": 8,
    "max_clusters_per_day": 20,
    # Default selection: which cluster tops are pre-checked when the user has
    # not hand-picked anything.
    "default_min_final_score": 1.0,
    "default_min_top_clusters": 6,
    # Face sets: primary people are searched in Immich and drive the main
    # expression score; family people contribute the family boost.
    "face_ids": list(FACE_IDS),
    "family_faces": list(FAMILY_FACES),
}

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curation_params.json")

# Spec table driving both validation and the settings form rendering.
PARAM_SPECS: dict[str, dict] = {
    name: spec
    for name, spec in {
        "quality_weight": dict(kind="float", min=0.0, max=10.0, step=0.05, label="Quality weight"),
        "expression_weight": dict(kind="float", min=0.0, max=10.0, step=0.05, label="Expression weight"),
        "person_weight": dict(kind="float", min=0.0, max=10.0, step=0.05, label="Person boost weight"),
        "quality_mean": dict(kind="float", min=-1000.0, max=1000.0, step=0.5, label="Quality mean"),
        "quality_std": dict(kind="float", min=0.05, max=1000.0, step=0.05, label="Quality std (> 0)"),
        "expression_mean": dict(kind="float", min=-10.0, max=10.0, step=0.05, label="Expression mean"),
        "expression_std": dict(kind="float", min=0.02, max=10.0, step=0.02, label="Expression std (> 0)"),
        "expr_w_eyes": dict(kind="float", min=0.0, max=10.0, step=0.05, label="Expr weight: eyes"),
        "expr_w_smile": dict(kind="float", min=0.0, max=10.0, step=0.05, label="Expr weight: smile"),
        "expr_w_facing": dict(kind="float", min=0.0, max=10.0, step=0.05, label="Expr weight: facing"),
        "family_facing_min": dict(kind="float", min=0.0, max=1.0, step=0.05, label="Family facing min"),
        "family_boost_discount": dict(kind="float", min=0.0, max=2.0, step=0.05, label="Family boost discount"),
        "family_boost_weight_eyes": dict(kind="float", min=0.0, max=2.0, step=0.05, label="Family weight: eyes"),
        "family_boost_weight_smile": dict(kind="float", min=0.0, max=2.0, step=0.05, label="Family weight: smile"),
        "family_boost_weight_facing": dict(kind="float", min=0.0, max=2.0, step=0.05, label="Family weight: facing"),
        "clip_cluster_sim_threshold": dict(kind="float", min=0.0, max=1.0, step=0.01, label="Cluster similarity threshold"),
        "cluster_top_weight": dict(kind="float", min=0.0, max=10.0, step=0.05, label="Cluster-top bonus weight"),
        "cluster_top_max_size": dict(kind="int", min=1, max=1000, step=1, label="Cluster-top saturating size"),
        "max_clusters_per_day": dict(kind="int", min=1, max=500, step=1, label="Max clusters per day"),
        "default_min_final_score": dict(kind="float", min=-100.0, max=100.0, step=0.05, label="Default min final score"),
        "default_min_top_clusters": dict(kind="int", min=0, max=200, step=1, label="Default min top clusters"),
    }.items()
}

# Settings-page sections: (heading, hint, field names in display order).
PARAM_GROUPS: list[tuple[str, str, list[str]]] = [
    ("Score weights",
     "total = w_q·q_norm + w_e·e_norm + person_boost + family_boost + cluster_top_bonus",
     ["quality_weight", "expression_weight", "person_weight"]),
    ("Normalization",
     "Quality and expression are z-scored with these mean/std before weighting",
     ["quality_mean", "quality_std", "expression_mean", "expression_std"]),
    ("Expression sub-weights",
     "Per-face expression = w_eyes·eyes + w_smile·smile + w_facing·facing",
     ["expr_w_eyes", "expr_w_smile", "expr_w_facing"]),
    ("Family boost",
     "Each family face with facing ≥ min adds discount·(w_eyes·eyes + w_smile·smile + w_facing·facing)",
     ["family_facing_min", "family_boost_discount", "family_boost_weight_eyes",
      "family_boost_weight_smile", "family_boost_weight_facing"]),
    ("Clustering",
     "CLIP cosine similarity groups near-duplicates; top of each cluster gets a saturating bonus",
     ["clip_cluster_sim_threshold", "cluster_top_weight", "cluster_top_max_size", "max_clusters_per_day"]),
    ("Default selection",
     "Cluster tops pre-checked when nothing is hand-picked",
     ["default_min_final_score", "default_min_top_clusters"]),
]

# Face-set pickers: (params key, label, description).
FACE_SETS: tuple[tuple[str, str, str], ...] = (
    ("face_ids", "Primary faces",
     "Searched in Immich and drive the main expression score"),
    ("family_faces", "Family faces",
     "Contribute the family boost"),
)

_FACE_PARAM_NAMES = ("face_ids", "family_faces")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

_params: dict = copy.deepcopy(DEFAULT_PARAMS)
_last_scored_params: dict | None = None  # snapshot used by the most recent run/rescore


def snapshot_params() -> dict:
    """Deep copy of the live settings; jobs score against an immutable snapshot."""
    return copy.deepcopy(_params)


def params_fingerprint(params: dict | None = None) -> str:
    """Stable string for comparing two param sets (used by the re-score banner)."""
    return json.dumps(params if params is not None else _params, sort_keys=True)


def _coerce_field(name: str, raw) -> tuple[bool, object, str]:
    """Parse one numeric field per its spec. Returns (ok, value, error)."""
    spec = PARAM_SPECS[name]
    try:
        if spec["kind"] == "float":
            value = float(raw)
        else:
            value = int(float(raw))
    except (TypeError, ValueError):
        return False, None, f"{spec['label']}: '{raw}' is not a number"
    if not (spec["min"] <= value <= spec["max"]):
        return False, None, f"{spec['label']} must be between {spec['min']} and {spec['max']}"
    return True, value, ""


def _clean_face_ids(ids) -> tuple[list[str], list[str]]:
    """Lowercase, validate and de-duplicate person ids. Returns (clean, errors)."""
    errors: list[str] = []
    out: list[str] = []
    for raw in ids or []:
        pid = str(raw).strip().lower()
        if not pid:
            continue
        if not _UUID_RE.match(pid):
            errors.append(f"'{pid}' is not a valid person id")
            continue
        if pid not in out:
            out.append(pid)
    return out, errors


def sanitize_faces(face_ids, family_faces) -> tuple[list[str], list[str], list[str]]:
    """Clean the two face sets. Returns (face_ids, family_faces, errors).

    An id in both sets stays primary (matching the classifier's precedence) and
    is dropped from family. At least one primary face is required; family may be
    empty. Unknown-but-well-formed ids are kept: Immich may know people we did
    not page through.
    """
    face_ids, errors = _clean_face_ids(face_ids)
    family_faces, fam_errors = _clean_face_ids(family_faces)
    errors.extend(fam_errors)
    family_faces = [pid for pid in family_faces if pid not in face_ids]
    if not face_ids:
        errors.append("Pick at least one primary face")
    return face_ids, family_faces, errors


def sanitize_params(raw: dict, face_ids=None, family_faces=None) -> tuple[dict | None, list[str]]:
    """Validate a raw dict (form data or curation_params.json).

    Numeric fields come from `raw`; the face sets from the explicit args when
    given (form checkboxes/text fallback), else from `raw` (settings file).
    Returns (params, errors).
    """
    errors: list[str] = []
    params = dict(DEFAULT_PARAMS)
    for name, spec in PARAM_SPECS.items():
        if name not in raw or raw[name] in ("", None):
            errors.append(f"{spec['label']} is missing")
            continue
        ok, value, err = _coerce_field(name, raw[name])
        if not ok:
            errors.append(err)
        else:
            params[name] = value
    if face_ids is None:
        face_ids = raw.get("face_ids")
    if family_faces is None:
        family_faces = raw.get("family_faces")
    f_ids, fam_ids, face_errors = sanitize_faces(face_ids, family_faces)
    errors.extend(face_errors)
    params["face_ids"] = f_ids
    params["family_faces"] = fam_ids
    if errors:
        return None, errors
    return params, []


def load_params() -> None:
    """Load settings from disk at import; invalid files fall back to defaults."""
    global _params
    if not os.path.exists(PARAMS_PATH):
        return
    try:
        with open(PARAMS_PATH) as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"Could not read {PARAMS_PATH} ({exc}); using default params.")
        return
    params, errors = sanitize_params(data)
    if params is None:
        print(f"Ignoring invalid {PARAMS_PATH}: {errors[:3]}")
        return
    _params = params
    print(f"Loaded scoring params from {os.path.basename(PARAMS_PATH)}.")


def save_params() -> None:
    """Atomically persist the live settings."""
    tmp = PARAMS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_params, f, indent=1, sort_keys=True)
    os.replace(tmp, PARAMS_PATH)


# Flask app configuration
FLASK_PORT = 5724

# MediaPipe face landmarker model (Tasks API). You must download this file
# from the official MediaPipe model zoo and place it here.
# Example URL (float32 variant):
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float32/1/face_landmarker.task
FACE_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "face_landmarker.task"
)

# ==========================================
# GLOBALS & FLASK SETUP
# ==========================================
app = Flask(__name__)
# This will store our processed and selected assets to serve to the UI
curated_assets_by_date = {}
# Tracks which asset IDs are currently selected in the UI
selected_asset_ids = set()
# Lazy-loaded models (loaded on first /process)
_models_cache = None
_musiq_metric = None
# Cache for per-asset face metadata fetched from Immich
_faces_cache: dict[str, list[dict]] = {}
# In-memory thumbnail cache (Immich thumbnails are immutable per asset id).
_thumb_cache: dict[str, tuple[bytes, str]] = {}
_thumb_lock = threading.Lock()
# People list + avatar cache for the settings face pickers.
_people_cache: dict | None = None  # {"fetched_at": float, "people": [{"id","name"}]}
_people_lock = threading.Lock()
_person_thumb_cache: dict[str, tuple[bytes, str]] = {}
# Live progress of a curation run (written by the job thread, read by /progress).
CURATION_LOCK = threading.Lock()
CURATION_STATE = {
    "state": "idle",  # idle | running | done | cancelled | error
    "phase": "",  # curation | rescore
    "days_total": 0,
    "days_done": 0,
    "current_date": "",
    "assets_done": 0,
    "assets_total": 0,
    "started_at": None,
    "finished_at": None,
    "error": "",
    "start_date": "",
    "end_date": "",
}
# Set by /cancel; the job checks it between days.
CURATION_CANCEL = threading.Event()
# Live progress of an album -> pixtale sync (sync_album_to_pixtale.run_sync).
# Deliberately independent of CURATION_STATE: a sync is cheap and network-bound,
# so it may legitimately run alongside a curation or re-score.
SYNC_LOCK = threading.Lock()
SYNC_STATE = {
    "state": "idle",  # idle | running | done | cancelled | error
    "phase": "",  # fetching_album | fetching_pixtale | planned | deleting | adding
    "dry_run": False,
    "delete": False,
    "storage_id": "",
    "plan_add": 0,
    "plan_delete": 0,
    "add_done": 0,
    "add_total": 0,
    "deleted": 0,
    "counters": {"ok": 0, "duplicate": 0, "error": 0},
    "samples_add": [],  # <=5 {"id","taken"}
    "samples_delete": [],  # <=5 [stem, photoId]
    "failures": [],  # <=20 [asset_id, detail]
    "current_asset": "",
    "started_at": None,
    "finished_at": None,
    "error": "",
}
# Set by /sync/cancel; run_sync checks it between phases/batches/uploads.
SYNC_CANCEL = threading.Event()
# Pixtale storage list for the /sync dropdown, TTL-cached like _people_cache.
_pixtale_storage_cache: dict | None = None  # {"fetched_at": float, "storages": [...]}
_pixtale_storage_lock = threading.Lock()
# On-disk checkpoint of curation results, so restarts don't lose them.
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curation_checkpoint.json")
_checkpoint_info: dict | None = None


def save_checkpoint(submitted: bool = False, params: dict | None = None) -> None:
    """Atomically write curated results + selection + the scoring params to disk."""
    global _checkpoint_info
    scored_params = params or _last_scored_params or snapshot_params()
    data = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "submitted": submitted,
        "params": scored_params,
        "curated_assets_by_date": curated_assets_by_date,
        "selected_asset_ids": sorted(selected_asset_ids),
    }
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, default=float)
    os.replace(tmp, CHECKPOINT_PATH)
    _checkpoint_info = {
        "days": len(curated_assets_by_date),
        "saved_at": data["saved_at"],
        "submitted": submitted,
        "params": scored_params,
    }


def load_checkpoint() -> None:
    """Restore curated results from the checkpoint file, if present."""
    global curated_assets_by_date, selected_asset_ids, _checkpoint_info
    if not os.path.exists(CHECKPOINT_PATH):
        return
    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)
    curated_assets_by_date = data["curated_assets_by_date"]
    selected_asset_ids = set(data["selected_asset_ids"])
    _checkpoint_info = {
        "days": len(curated_assets_by_date),
        "saved_at": data.get("saved_at", ""),
        "submitted": bool(data.get("submitted")),
        "params": data.get("params"),  # None for pre-params checkpoints -> defaults
    }
    print(
        f"Restored checkpoint from {_checkpoint_info['saved_at']} "
        f"({len(curated_assets_by_date)} days, submitted={_checkpoint_info['submitted']})."
    )


def discard_checkpoint() -> None:
    """Throw away the checkpoint file and the in-memory results."""
    global curated_assets_by_date, selected_asset_ids, _checkpoint_info
    curated_assets_by_date = {}
    selected_asset_ids = set()
    _checkpoint_info = None
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


# Per-asset feature cache: the expensive, image-derived quantities (MUSIQ
# quality, MediaPipe expression summary, CLIP embedding). Deterministic per
# image, so they are reusable across runs and date ranges; only new assets
# pay inference cost again. Bump the version when models or preprocessing
# change.
FEATURE_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_cache.json")
# v2 adds per-face raw rows ("faces") so expression/family scores can be
# re-aggregated under new params or face sets without re-running MediaPipe.
# v1 files are still accepted: entries without "faces" are backfilled lazily.
FEATURE_CACHE_VERSION = 2
SUPPORTED_CACHE_VERSIONS = (1, 2)
_feature_cache: dict[str, dict] = {}
_feature_lock = threading.Lock()


def load_feature_cache() -> None:
    global _feature_cache
    if not os.path.exists(FEATURE_CACHE_PATH):
        return
    with open(FEATURE_CACHE_PATH) as f:
        data = json.load(f)
    if data.get("version") not in SUPPORTED_CACHE_VERSIONS:
        print(f"Ignoring feature cache version {data.get('version')} (want {SUPPORTED_CACHE_VERSIONS}).")
        return
    _feature_cache = data.get("assets", {})
    print(f"Loaded feature cache: {len(_feature_cache)} assets (v{data.get('version')}).")


def save_feature_cache() -> None:
    """Atomically persist the per-asset feature cache.

    Snapshots under the lock first: the job thread inserts entries while we
    dump, and json.dump over a growing dict raises 'dictionary changed size'.
    """
    with _feature_lock:
        payload = {"version": FEATURE_CACHE_VERSION,
                   "assets": {k: dict(v) for k, v in _feature_cache.items()}}
    tmp = FEATURE_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, FEATURE_CACHE_PATH)


load_checkpoint()
load_feature_cache()
load_params()


def _pil_to_data_url(img: Image.Image) -> str:
    """Encode a PIL image as a base64 data URL for HTML debug views."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
# Headers for all Immich API calls
HEADERS = {
    "x-api-key": API_KEY,
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# ==========================================
# 1. IMMICH API HELPER FUNCTIONS
# ==========================================

def search_assets_for_date(target_date, person_ids):
    """
    Queries Immich for assets on a specific date containing specific people.
    Immich's personIds filter behaves like an AND for multiple IDs, so we
    explicitly OR across IDs by issuing one query per person and unioning results.
    """
    print(f"[{target_date}] Searching for assets...")

    # Interpret the calendar date in the local timezone, then convert to UTC
    # for the Immich API's takenAfter/takenBefore filters.
    local_start = datetime.datetime.combine(
        target_date, datetime.time.min, tzinfo=LOCAL_TZ
    )
    local_end = datetime.datetime.combine(
        target_date, datetime.time.max, tzinfo=LOCAL_TZ
    )
    start_time = (
        local_start.astimezone(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    end_time = (
        local_end.astimezone(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    all_assets: dict[str, dict] = {}
    for pid in person_ids:
        payload = {
            "personIds": [pid],
            "takenAfter": start_time,
            "takenBefore": end_time,
        }
        url = f"{IMMICH_URL}/api/search/metadata"
        response = requests.post(url, headers=HEADERS, json=payload)
        if response.status_code not in (200, 201):
            print(f"  -> API Error for person {pid} ({response.status_code}): {response.text}")
            continue
        data = response.json()
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'assets' in data and 'items' in data['assets']:
            items = data['assets']['items']
        elif isinstance(data, dict) and 'items' in data:
            items = data['items']
        else:
            items = []

        for asset in items:
            asset_id = asset.get("id")
            if not asset_id:
                continue
            if asset_id not in all_assets:
                # Clone and annotate with person match count
                asset_copy = dict(asset)
                asset_copy["_person_match_count"] = 1
                all_assets[asset_id] = asset_copy
            else:
                all_assets[asset_id]["_person_match_count"] += 1

    print(f"  -> Found {len(all_assets)} unique assets across {len(person_ids)} person queries.")
    return list(all_assets.values())

THUMBNAIL_MAX_SHORT_EDGE = 720


def download_thumbnail(asset_id):
    """
    Downloads the Immich thumbnail for quality/expression evaluation.
    Downsizes so the short edge is at most THUMBNAIL_MAX_SHORT_EDGE (720) px.
    """
    url = f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview"
    response = requests.get(url, headers={"x-api-key": API_KEY})
    if response.status_code != 200:
        return None
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    w, h = img.size
    min_side = min(w, h)
    if min_side > THUMBNAIL_MAX_SHORT_EDGE:
        scale = THUMBNAIL_MAX_SHORT_EDGE / float(min_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)
    return img

def add_assets_to_album(album_id, asset_ids):
    """
    Adds a list of asset IDs to the specified Immich album.
    """
    url = f"{IMMICH_URL}/api/albums/{album_id}/assets"
    payload = {"ids": asset_ids}
    response = requests.put(url, headers=HEADERS, json=payload)
    
    return response.status_code in (200, 201)


def get_album_asset_ids(album_id):
    """
    Fetch existing asset IDs for the given Immich album.
    Used when we optionally skip days that are already fully covered.
    """
    url = f"{IMMICH_URL}/api/albums/{album_id}"
    response = requests.get(url, headers=HEADERS, timeout=15)
    if response.status_code != 200:
        return set()
    data = response.json()
    assets = data.get("assets") or []
    return {a.get("id") for a in assets if a.get("id")}

# ==========================================
# 2. QUALITY & EXPRESSION EVALUATION (Aesthetics + MediaPipe Face Mesh)
# ==========================================

def _pil_to_tensor_01(pil_img):
    """PIL Image (RGB) -> tensor (1, 3, H, W) in [0, 1] for PyIQA."""
    import torch
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(arr).unsqueeze(0)

_face_landmarker = None


def _get_face_landmarker():
    """
    Lazily create and cache a MediaPipe Tasks FaceLandmarker instance.
    Uses the face_landmarker.task model at FACE_LANDMARKER_MODEL_PATH.
    """
    global _face_landmarker
    if _face_landmarker is not None:
        return _face_landmarker
    if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
        raise RuntimeError(
            f"Face Landmarker model not found at {FACE_LANDMARKER_MODEL_PATH}. "
            "Download face_landmarker.task from the MediaPipe model zoo and place it there."
        )
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=10,
    )
    _face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


def _lm_dist(lms, i, j):
    dx = lms[i].x - lms[j].x
    dy = lms[i].y - lms[j].y
    return float((dx * dx + dy * dy) ** 0.5)


# Thresholds for continuous eye/smile mapping; at or above these we cap at 1.0.
EYE_OPEN_RATIO_AT_ONE = 0.21
SMILE_WIDTH_RATIO_AT_ONE = 0.35
SMILE_HEIGHT_RATIO_AT_ONE = 0.04

# MediaPipe face landmark index for nose tip (used for facing-camera symmetry).
NOSE_TIP_LANDMARK_INDEX = 1


def _analyze_face_landmarks(lms, expr_weights=None):
    """
    Given MediaPipe face landmarks, compute:
      - eyes_open in [0, 1] (continuous) from eye aspect ratio
      - smile in [0, 1] (continuous) from mouth width/height relative to face width
      - facing in [0, 1] (continuous) from vertical symmetry: nose tip vs face center
      - expression score in [0, 1] (weighted eyes + smile + facing). Weights come
        from the params snapshot (expr_w_*) so debug views agree with scoring.
    """
    w_eyes, w_smile, w_facing = expr_weights or (0.5, 0.2, 0.3)
    xs = [lm.x for lm in lms]
    face_w = max(xs) - min(xs) if xs else 0.0
    if face_w <= 0:
        face_w = 1.0
    center_x = (min(xs) + max(xs)) / 2.0 if xs else 0.5

    # Eye openness: ratio of lid distance to eye width. Map continuously to [0, 1].
    left_open = _lm_dist(lms, 159, 145) / (_lm_dist(lms, 33, 133) + 1e-6)
    right_open = _lm_dist(lms, 386, 374) / (_lm_dist(lms, 362, 263) + 1e-6)
    eye_open_ratio = (left_open + right_open) / 2.0
    eyes_open = min(1.0, eye_open_ratio / EYE_OPEN_RATIO_AT_ONE)

    # Smile: mouth width and height relative to face width. Map continuously to [0, 1].
    mouth_width = _lm_dist(lms, 61, 291)
    mouth_height = _lm_dist(lms, 13, 14)
    width_ratio = mouth_width / (face_w + 1e-6)
    height_ratio = mouth_height / (face_w + 1e-6)
    smile = min(
        1.0,
        (width_ratio / SMILE_WIDTH_RATIO_AT_ONE)
        * (height_ratio / SMILE_HEIGHT_RATIO_AT_ONE),
    )

    # Facing camera: symmetry about a vertical line. Nose tip centered => 1, off to side => 0.
    nose_x = lms[NOSE_TIP_LANDMARK_INDEX].x
    facing = max(0.0, 1.0 - 2.0 * abs(nose_x - center_x) / (face_w + 1e-6))

    # Expression: weighted eyes + smile + facing (primary and family both use this).
    expr_score = w_eyes * eyes_open + w_smile * smile + w_facing * facing
    return eyes_open, smile, facing, expr_score


def _fetch_faces_for_asset(asset_id: str) -> list[dict]:
    """
    Fetch detailed face metadata for a given asset via Immich GET /api/faces?id=...
    Results are cached in _faces_cache.
    """
    global _faces_cache
    if not asset_id:
        return []
    if asset_id in _faces_cache:
        return _faces_cache[asset_id]
    url = f"{IMMICH_URL}/api/faces"
    resp = requests.get(url, headers=HEADERS, params={"id": asset_id}, timeout=10)
    if resp.status_code != 200:
        # Do NOT cache failures: a transient Immich error must not become a
        # permanent "no faces" for this asset.
        return []
    data = resp.json()
    if not isinstance(data, list):
        data = []
    _faces_cache[asset_id] = data
    # Log face/person IDs for debugging
    face_ids = []
    person_ids = []
    family_hits = []
    for f in data:
        fid = f.get("id")
        if fid:
            face_ids.append(fid)
        person = f.get("person") or {}
        if isinstance(person, dict):
            pid = person.get("id") or person.get("personId") or person.get("person_id")
            if pid:
                person_ids.append(pid)
                if pid in _params["family_faces"]:
                    family_hits.append(pid)
    if face_ids or person_ids:
        print(
            f"[immich_faces] asset={asset_id} faces={len(data)} "
            f"face_ids={face_ids} person_ids={person_ids} "
            f"family_hits={list(sorted(set(family_hits)))}"
        )
    return data


def _face_type(pid, params=None):
    """Classify an Immich person id: 'primary' > 'family' > 'other' (None -> other)."""
    p = params or _params
    if pid is not None:
        if pid in p["face_ids"]:
            return "primary"
        if pid in p["family_faces"]:
            return "family"
    return "other"


def _extract_face_boxes_with_pid(asset):
    """
    Extract face bounding boxes for an Immich asset using /api/faces.
    Returns list of (x1, y1, x2, y2, imageWidth, imageHeight, pid_or_None).
    The owning person id is kept (not pre-classified) so faces can be
    re-classified under whatever face sets are active at scoring time.
    """
    asset_id = asset.get("id")
    faces = _fetch_faces_for_asset(asset_id)
    boxes = []
    for f in faces:
        if not isinstance(f, dict):
            continue
        x1 = f.get("boundingBoxX1")
        x2 = f.get("boundingBoxX2")
        y1 = f.get("boundingBoxY1")
        y2 = f.get("boundingBoxY2")
        iw = f.get("imageWidth")
        ih = f.get("imageHeight")
        if None in (x1, x2, y1, y2, iw, ih):
            continue
        if x2 <= x1 or y2 <= y1 or iw <= 0 or ih <= 0:
            continue
        person = f.get("person") or {}
        pid = None
        if isinstance(person, dict):
            pid = person.get("id") or person.get("personId") or person.get("person_id")
        if pid:
            pid = str(pid).lower()
        boxes.append((float(x1), float(y1), float(x2), float(y2), float(iw), float(ih), pid))
    return boxes


# Expand face crop by this fraction in all directions for landmark detection (debug and processing use same value).
# 0.2 = 20% extra margin around the Immich face box.
FACE_CROP_EXPAND_FRAC = 0.2


def _face_box_to_expanded_crop(x1, y1, x2, y2, iw, ih, img_width, img_height, expand_frac=FACE_CROP_EXPAND_FRAC):
    """
    Map Immich face box (in original image coords) to thumbnail pixel rect and expand
    by expand_frac in all directions. Used for both landmark processing and debug crops.
    Returns (ex1, ey1, ex2, ey2) in pixel coords, or None if invalid.
    """
    if iw <= 0 or ih <= 0:
        return None
    w, h = img_width, img_height
    sx = w / float(iw)
    sy = h / float(ih)
    rx1 = max(0, min(w - 1, int(round(x1 * sx))))
    rx2 = max(0, min(w, int(round(x2 * sx))))
    ry1 = max(0, min(h - 1, int(round(y1 * sy))))
    ry2 = max(0, min(h, int(round(y2 * sy))))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    bw = rx2 - rx1
    bh = ry2 - ry1
    pad_x = int(round(bw * expand_frac))
    pad_y = int(round(bh * expand_frac))
    ex1 = max(0, rx1 - pad_x)
    ex2 = min(w, rx2 + pad_x)
    ey1 = max(0, ry1 - pad_y)
    ey2 = min(h, ry2 + pad_y)
    if ex2 <= ex1 or ey2 <= ey1:
        return None
    return (ex1, ey1, ex2, ey2)


def _detect_face_rows(img, asset):
    """
    Detect faces and measure raw per-face signals (MediaPipe only, no scoring).

    If Immich provides face bounding boxes, run the landmarker on each face crop
    and keep the owning person id with every row; otherwise run once on the full
    image with pid=None (those faces can never be classified primary/family).

    Returns {"mode": "boxes"|"full", "rows": [[pid_or_None, eyes, smile, facing], ...]}.
    The rows are cacheable, so aggregate_expression() can re-score under any
    params or face sets without re-running MediaPipe.
    """
    import mediapipe as mp

    landmarker = _get_face_landmarker()

    img_rgb = np.array(img.convert("RGB"))
    # Use same dimension convention as debug: (width, height) from PIL so crop math matches.
    img_width, img_height = img.size

    # Face boxes from Immich, each with its owning Immich person id.
    boxes = _extract_face_boxes_with_pid(asset)
    rows: list[list] = []

    # Helper to run the landmarker on one RGB crop and append rows.
    def detect_crop(crop_rgb: np.ndarray, pid, bbox_desc: str):
        if crop_rgb.size == 0:
            return
        mp_face = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        result = landmarker.detect(mp_face)
        lms_list = result.face_landmarks or []
        print(f"  -> {bbox_desc} {len(lms_list)} faces detected")
        for li, lms in enumerate(lms_list):
            eyes_open, smile, facing, _expr = _analyze_face_landmarks(lms)
            print(
                f"  {bbox_desc} face#{li}: pid={pid} "
                f"eyes={eyes_open:.2f} smile={smile:.2f} facing={facing:.2f}"
            )
            rows.append([pid, eyes_open, smile, facing])

    if boxes:
        # Use Immich's face boxes to crop and run the landmarker per face (same crop logic as debug).
        for bi, (x1, y1, x2, y2, iw, ih, pid) in enumerate(boxes):
            rect = _face_box_to_expanded_crop(x1, y1, x2, y2, iw, ih, img_width, img_height)
            if rect is None:
                continue
            ex1, ey1, ex2, ey2 = rect
            # Contiguous copy so MediaPipe gets a valid buffer (slicing can yield non-contiguous view).
            crop = np.ascontiguousarray(img_rgb[ey1:ey2, ex1:ex2, :])
            detect_crop(crop, pid, f"[box{bi} type={_face_type(pid)}]")
        return {"mode": "boxes", "rows": rows}

    # Fallback: run on the full image once; faces have no person id ("other").
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
    result = landmarker.detect(mp_image)
    faces = result.face_landmarks or []
    if not faces:
        print(f"  -> No faces detected in {asset.get('id')}")
        return {"mode": "full", "rows": []}
    for li, lms in enumerate(faces):
        eyes_open, smile, facing, _expr = _analyze_face_landmarks(lms)
        print(f"  [full] face#{li}: pid=None eyes={eyes_open:.2f} smile={smile:.2f} facing={facing:.2f}")
        rows.append([None, eyes_open, smile, facing])
    return {"mode": "full", "rows": rows}

def aggregate_expression(face_payload, params):
    """
    Pure re-aggregation of expression score + family boost from cached face rows
    (see _detect_face_rows): rows are [pid_or_None, eyes, smile, facing]. The pid
    decides each face's role under `params`, so changing face sets, expression
    sub-weights or family-boost params re-scores without re-running MediaPipe.

    Returns (expression_score, family_boost, summary_dict) with the historical
    summary keys. No rows -> (0.0, 0.0, {}).
    """
    rows = (face_payload or {}).get("rows") or []
    face_ids = params["face_ids"]
    family_ids = params["family_faces"]
    primary = [r for r in rows if r[0] in face_ids]
    family = [r for r in rows if r[0] not in face_ids and r[0] in family_ids]
    other_count = len(rows) - len(primary) - len(family)

    w_eyes = params["expr_w_eyes"]
    w_smile = params["expr_w_smile"]
    w_facing = params["expr_w_facing"]

    def face_expr(r):
        return w_eyes * r[1] + w_smile * r[2] + w_facing * r[3]

    # Aggregate expression:
    # - Primary faces drive the main expression score if present.
    # - Otherwise, family faces drive expression.
    # - Only non-primary, non-family faces -> no reliable expression signal (0.0).
    if primary:
        expr = float(np.mean([face_expr(r) for r in primary]))
    elif family:
        expr = float(np.mean([face_expr(r) for r in family]))
    elif other_count:
        expr = 0.0
    else:
        return 0.0, 0.0, {}

    # Family boost: continuous eyes, smile, facing per face; facing weighted higher.
    # Only faces with facing >= family_facing_min contribute, discounted per face.
    fw_eyes = params["family_boost_weight_eyes"]
    fw_smile = params["family_boost_weight_smile"]
    fw_facing = params["family_boost_weight_facing"]
    facing_min = params["family_facing_min"]
    discount = params["family_boost_discount"]
    family_boost = 0.0
    family_contributing_count = 0
    for r in family:
        if r[3] < facing_min:
            continue
        raw = fw_eyes * r[1] + fw_smile * r[2] + fw_facing * r[3]
        family_boost += discount * raw
        family_contributing_count += 1

    # Summary fractions: show the stats for whoever drove expr (primary or family).
    driver = primary or family
    eyes_open_frac = float(np.mean([r[1] for r in driver])) if driver else 0.0
    smile_frac = float(np.mean([r[2] for r in driver])) if driver else 0.0
    facing_frac = float(np.mean([r[3] for r in driver])) if driver else 0.0

    summary = {
        "eyes_open_fraction": eyes_open_frac * 100.0,
        "smile_fraction": smile_frac * 100.0,
        "facing_fraction": facing_frac * 100.0,
        "primary_faces": len(primary),
        "family_faces": len(family),
        "family_boost": family_boost,
        "family_contributing_count": family_contributing_count,
        "other_faces": other_count,
    }
    print(
        f"    [expr_fam] total_faces={len(rows)} primary={len(primary)} fam={len(family)} "
        f"other={other_count} expr={expr:.3f} smile={smile_frac:.2f} eyes={eyes_open_frac:.2f}"
    )
    return expr, family_boost, summary


def _expression_from_mediapipe(img, asset, params=None):
    """Detect faces, then aggregate under `params` (default: live settings).
    Kept for callers that want (expression_score, summary_dict) directly."""
    payload = _detect_face_rows(img, asset)
    expr, _boost, summary = aggregate_expression(payload, params or _params)
    return expr, summary


def _backfill_faces(asset, entry):
    """Backfill a v1 cache entry with per-face rows (MediaPipe only — no MUSIQ/CLIP).

    Stores the payload on the cache entry and returns it; None when no thumbnail
    is available (the caller scores the asset with a neutral expression).
    """
    img = download_thumbnail(asset.get("id"))
    if img is None:
        return None
    payload = _detect_face_rows(img, asset)
    with _feature_lock:
        entry["faces"] = payload
    return payload


def _expression_summary(face_payload, asset_id, params):
    """Aggregate expression + family boost from face rows under `params`."""
    expr, _boost, summary = aggregate_expression(face_payload, params)
    print(f"    [expr] asset={asset_id} expr={expr:.3f} summary={summary}")
    return expr, summary


@app.route("/debug/crops/<asset_id>")
def debug_crops(asset_id):
    """
    Debug page: show all Immich face crops for a given asset so cropping
    can be visually validated.
    """
    import mediapipe as mp

    img = download_thumbnail(asset_id)
    if img is None:
        return f"<h3>No thumbnail available for asset {asset_id}</h3>", 404

    w, h = img.size
    landmarker = _get_face_landmarker()
    boxes = _extract_face_boxes_with_pid({"id": asset_id})
    if not boxes:
        return f"<h3>No Immich faces for asset {asset_id}</h3>", 200

    expr_weights = (_params["expr_w_eyes"], _params["expr_w_smile"], _params["expr_w_facing"])
    rows = []
    for idx, (x1, y1, x2, y2, iw, ih, pid) in enumerate(boxes):
        rect = _face_box_to_expanded_crop(x1, y1, x2, y2, iw, ih, w, h)
        if rect is None:
            continue
        ex1, ey1, ex2, ey2 = rect
        crop = img.crop((ex1, ey1, ex2, ey2))

        # Run face detection on the crop using the same landmarker and analysis logic
        crop_rgb = np.array(crop.convert("RGB"))
        mp_face = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        result = landmarker.detect(mp_face)
        faces = result.face_landmarks or []
        face_count = len(faces)
        face_details: list[str] = []
        for li, lms in enumerate(faces):
            eyes_open, smile, facing, expr = _analyze_face_landmarks(lms, expr_weights)
            face_details.append(
                f"face#{li}: eyes={eyes_open:.2f} smile={smile:.2f} facing={facing:.2f} expr={expr:.3f}"
            )
        if face_details:
            detect_summary = " | ".join(face_details)
        else:
            detect_summary = "no faces detected"

        data_url = _pil_to_data_url(crop)
        rows.append(
            f"<div style='display:inline-block; margin:10px; text-align:center;'>"
            f"<div>#{idx} type={_face_type(pid)} "
            f"orig=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
            f"crop=({ex1},{ey1},{ex2},{ey2}) "
            f"detect_faces={face_count} [{detect_summary}]</div>"
            f"<img src='{data_url}' style='max-width:240px; max-height:240px; border:1px solid #ccc;'/>"
            f"</div>"
        )

    html = (
        "<!DOCTYPE html><html><head><title>Face crops debug</title></head>"
        "<body style='font-family:sans-serif;'>"
        f"<h2>Face crops for asset {asset_id}</h2>"
        f"<p>Thumbnail size: {w}x{h}</p>"
        "<p style='font-size:0.85rem; color:#555; margin-bottom:1rem;'>"
        "Per-face scores (eyes, smile, facing, expr) are in [0, 1]: same as used for scoring. "
        "On the main cards, <em>eyes_open_fraction</em> and <em>smile_fraction</em> are the "
        "<strong>mean over family faces only</strong> (0–100%). Primary/other faces do not "
        "contribute to those card numbers.</p>"
        + "".join(rows) +
        "</body></html>"
    )
    return html

def combine_and_cluster(assets, quality_scores, expression_scores, clip_feats, params):
    """
    Pure scoring core shared by the run path and re-scores: normalize quality and
    expression, add person/family boosts, cluster in CLIP space, boost cluster
    tops and sort. Mutates and returns `assets` sorted for the cluster view.
    """
    if not assets:
        return []

    # 3. Normalize and combine using global params (so scales are comparable across days)
    def _norm_global(scores, mean, std):
        s = np.array(scores, dtype=np.float64)
        if std <= 0:
            return np.zeros_like(s)
        return (s - mean) / std
    q_norm = _norm_global(quality_scores, params["quality_mean"], params["quality_std"])
    e_norm = _norm_global(expression_scores, params["expression_mean"], params["expression_std"])
    for i in range(len(assets)):
        asset = assets[i]
        person_hits = asset.get("_person_match_count", 1)
        person_boost = params["person_weight"] * (person_hits - 1)

        # Family boost: continuous eyes/smile/facing (precomputed in expression summary).
        summary = asset.get("_expression_emotions") or {}
        smile_boost = float(summary.get("family_boost", 0.0))
        total = float(
            params["quality_weight"] * q_norm[i]
            + params["expression_weight"] * e_norm[i]
            + person_boost
            + smile_boost
        )
        asset["_smile_boost"] = smile_boost
        if smile_boost > 0:
            print(
                f"    [smile_boost] asset={asset.get('id')} "
                f"family_contributing={summary.get('family_contributing_count', 0)} boost={smile_boost:.3f}"
            )
        asset["_total_score"] = total
        asset["_base_score"] = total

    # 4. Threshold-based clustering in CLIP space:
    #    - Iterate assets in descending total score order.
    #    - For each, assign to the first cluster whose centroid has cosine
    #      similarity >= clip_cluster_sim_threshold; otherwise start a new cluster.
    X = np.stack(clip_feats, axis=0).astype(np.float32)
    order = sorted(range(len(assets)), key=lambda i: -assets[i]["_total_score"])
    centroids: list[np.ndarray] = []
    cluster_ids = [-1] * len(assets)

    for idx in order:
        x = X[idx]
        if not centroids:
            centroids.append(x.copy())
            cluster_ids[idx] = 0
            continue
        sims = [float(np.dot(x, c)) for c in centroids]
        best_id = int(np.argmax(sims))
        if sims[best_id] >= params["clip_cluster_sim_threshold"]:
            cluster_ids[idx] = best_id
            # Update centroid with simple running average
            centroids[best_id] = (centroids[best_id] + x) / 2.0
        else:
            new_id = len(centroids)
            centroids.append(x.copy())
            cluster_ids[idx] = new_id

    n_clusters = len(centroids)

    # Optionally recluster with KMeans if too many clusters; cap at max_clusters_per_day.
    if n_clusters > params["max_clusters_per_day"]:
        from sklearn.cluster import KMeans

        k = min(params["max_clusters_per_day"], len(assets))
        print(f"  -> Reclustering {len(assets)} assets into {k} clusters via KMeans (was {n_clusters}).")
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)
        cluster_ids = [int(l) for l in labels]
        n_clusters = k

    # 5. Cluster-top boost: for each cluster, give the top-1 asset an extra bonus
    #    that grows with cluster size but saturates.
    clusters: dict[int, list[int]] = {}
    for i, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(i)

    top_weight = params["cluster_top_weight"]
    top_max_size = params["cluster_top_max_size"]

    def _cluster_top_boost(size: int) -> float:
        capped = min(max(size, 1), top_max_size)
        # Map size 1..cluster_top_max_size -> (0..1] then scale by cluster_top_weight.
        return top_weight * (np.log1p(capped) / np.log1p(top_max_size))

    # Initialize final scores and cluster sizes
    for cid, idxs in clusters.items():
        size = len(idxs)
        for idx in idxs:
            assets[idx]["_cluster_id"] = cid
            assets[idx]["_cluster_size"] = size
            assets[idx]["_final_score"] = assets[idx]["_total_score"]

        # Boost the top-1 asset in this cluster
        if idxs:
            top_idx = max(idxs, key=lambda i: assets[i]["_total_score"])
            boost = _cluster_top_boost(size)
            assets[top_idx]["_final_score"] = assets[top_idx]["_total_score"] + boost
            assets[top_idx]["_cluster_top_boost"] = boost

    # Sort by cluster top score descending for cluster view / ordering.
    cluster_top_score: dict[int, float] = {}
    for cid, idxs in clusters.items():
        if not idxs:
            continue
        top_idx = max(idxs, key=lambda i: assets[i]["_final_score"])
        cluster_top_score[cid] = assets[top_idx]["_final_score"]

    # Sort by (cluster_top_score desc, then per-asset final score desc)
    assets.sort(
        key=lambda a: (
            -cluster_top_score.get(a.get("_cluster_id", -1), float("-inf")),
            -a.get("_final_score", a.get("_total_score", 0.0)),
        )
    )
    scores = [a.get("_final_score", a["_total_score"]) for a in assets]
    print(f"  -> Score range (final): min={min(scores):.3f} max={max(scores):.3f} mean={sum(scores)/len(scores):.3f}")
    print(f"  -> Total assets this day: {len(assets)} (clusters={n_clusters}).")
    return assets


def _compute_asset_features(asset, device, clip_model, clip_processor, params):
    """ML feature path for one asset: MUSIQ quality + face rows + CLIP embedding.
    Returns (raw_q, face_payload, feats), or None when no thumbnail is available."""
    import torch

    asset_id = asset.get("id")
    img = download_thumbnail(asset_id)
    if img is None:
        return None

    # 1. Image quality scores via MUSIQ (PyIQA). Higher = better quality.
    img_t = _pil_to_tensor_01(img)  # (1,3,H,W), CPU tensor
    q_tensor = _musiq_metric(img_t)
    raw_q = float(q_tensor.item())
    if not np.isfinite(raw_q):
        raw_q = params["quality_mean"]

    # 2. Expression via MediaPipe Face Landmarker (rows kept for re-scoring).
    face_payload = _detect_face_rows(img, asset)

    # 3. CLIP embedding for clustering.
    with torch.no_grad():
        clip_inputs = clip_processor(images=img, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**clip_inputs)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    feats = feats.squeeze(0).cpu().numpy().astype(np.float32)
    return raw_q, face_payload, feats


def evaluate_and_cluster(assets, device, clip_model, clip_processor, params):
    """
    Evaluates assets with PyIQA (quality) and MediaPipe (expression), then hands
    the features to combine_and_cluster for scoring. `params` is the job's
    settings snapshot. Returns the scored, clustered, sorted assets.
    """
    import pyiqa

    if not assets:
        return []
    print(f"  -> Processing {len(assets)} assets...")

    valid_assets: list[dict] = []
    quality_scores: list[float] = []
    expression_scores: list[float] = []
    clip_feats: list[np.ndarray] = []

    global _musiq_metric
    if _musiq_metric is None:
        # Use CPU for MUSIQ; it's reasonably fast and avoids GPU dependency.
        _musiq_metric = pyiqa.create_metric("musiq", device="cpu")

    for idx, asset in enumerate(assets):
        with CURATION_LOCK:
            CURATION_STATE["assets_done"] = idx
        asset_id = asset.get("id")

        with _feature_lock:
            cached = _feature_cache.get(asset_id)
        if cached is not None:
            # Cache hit: reuse the stored quality/CLIP features and re-aggregate
            # the expression under the current params (v1 entries get backfilled).
            raw_q = cached["q"]
            feats = np.frombuffer(base64.b64decode(cached["clip"]), dtype=np.float32)
            face_rows = cached.get("faces")
            if face_rows is None:
                print(f"    [backfill] asset={asset_id} missing face rows; running MediaPipe.")
                face_rows = _backfill_faces(asset, cached)
            expr, summary = _expression_summary(face_rows, asset_id, params)
            with _feature_lock:
                cached["e"] = expr
                cached["sum"] = summary or {}
            print(f"    [cache] asset={asset_id} expr={float(expr):.3f} (reused)")
        else:
            computed = _compute_asset_features(asset, device, clip_model, clip_processor, params)
            if computed is None:
                continue
            raw_q, face_payload, feats = computed
            expr, summary = _expression_summary(face_payload, asset_id, params)
            with _feature_lock:
                _feature_cache[asset_id] = {
                    "q": raw_q,
                    "e": expr,
                    "sum": summary or {},
                    "faces": face_payload,
                    "clip": base64.b64encode(feats.tobytes()).decode("ascii"),
                }

        # Stash metrics on asset and in lists.
        asset["_quality_score"] = raw_q
        asset["_expression_score"] = expr
        if summary:
            asset["_expression_emotions"] = summary

        valid_assets.append(asset)
        quality_scores.append(raw_q)
        expression_scores.append(expr)
        clip_feats.append(feats)

    if not valid_assets:
        return []

    # Log expression score summary for this batch
    if expression_scores:
        exp_min = min(expression_scores)
        exp_max = max(expression_scores)
        exp_mean = sum(expression_scores) / len(expression_scores)
        print(f"  -> Expression scores: min={exp_min:.3f} max={exp_max:.3f} mean={exp_mean:.3f}")

    return combine_and_cluster(valid_assets, quality_scores, expression_scores, clip_feats, params)

# ==========================================
# 3. MAIN WORKFLOW
# ==========================================

def load_models():
    """Load CLIP model once for embeddings; return (device, clip_model, clip_processor)."""
    global _models_cache
    if _models_cache is not None:
        return _models_cache
    import torch
    from transformers import CLIPModel, CLIPProcessor

    print("Loading CLIP embedding model (this may take a moment)...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _models_cache = (device, clip_model, clip_processor)
    print(f"Models loaded on {device}.")
    return _models_cache


def recompute_default_selection(params=None):
    """
    Recompute default selected_asset_ids from curated_assets_by_date
    using the rule:
      - select the top photo in each cluster whose final score
        >= default_min_final_score
      - AND always select the top photo from at least the top
        default_min_top_clusters clusters (by best score) for that day.
    """
    p = params or _params
    global selected_asset_ids
    selected = set()
    for _date, assets in curated_assets_by_date.items():
        best_per_cluster = {}
        for a in assets:
            cid = a.get("_cluster_id", -1)
            score = a.get("_final_score", a.get("_total_score", 0.0))
            cur = best_per_cluster.get(cid)
            cur_score = (
                cur.get("_final_score", cur.get("_total_score", 0.0))
                if cur is not None
                else None
            )
            if cur is None or score > cur_score:
                best_per_cluster[cid] = a
        cluster_items = []
        for cid, a in best_per_cluster.items():
            score = a.get("_final_score", a.get("_total_score", 0.0))
            cluster_items.append((cid, score, a))
        cluster_items.sort(key=lambda x: -x[1])
        top_cluster_ids = {
            cid for cid, _score, _a in cluster_items[: p["default_min_top_clusters"]]
        }
        for cid, score, a in cluster_items:
            if score >= p["default_min_final_score"] or cid in top_cluster_ids:
                if "id" in a:
                    selected.add(a["id"])
    selected_asset_ids = selected


def release_memory() -> None:
    """
    Hand freed heap back to the OS after a big job. CPython frees the objects,
    but glibc keeps the arenas, so RSS sits at the job's high-water mark (gigs)
    until something reuses them. gc first so cycles don't pin arenas.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as e:
        print(f"malloc_trim unavailable, skipping: {e}")


def run_curation_for_date_range(start_date, end_date, skip_dates_already_in_album=False, params=None):
    """
    Run search + scoring + clustering for each day in [start_date, end_date]
    and populate curated_assets_by_date. Call this from /process when user submits dates.
    Updates CURATION_STATE for the /progress endpoint and checkpoints to disk per day.
    Scores against a snapshot of the settings taken when the job starts.
    """
    global curated_assets_by_date, selected_asset_ids, _last_scored_params
    params = params or snapshot_params()
    _last_scored_params = copy.deepcopy(params)
    days_total = (end_date - start_date).days + 1
    CURATION_CANCEL.clear()
    with CURATION_LOCK:
        CURATION_STATE.update(
            state="running", phase="curation", days_total=days_total, days_done=0,
            current_date="", assets_done=0, assets_total=0, started_at=time.time(),
            finished_at=None, error="",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
    try:
        curated_assets_by_date = {}
        selected_asset_ids = set()
        device, clip_model, clip_processor = load_models()
        existing_album_ids = set()
        if skip_dates_already_in_album:
            existing_album_ids = get_album_asset_ids(ALBUM_ID)
        current = start_date
        while current <= end_date:
            if CURATION_CANCEL.is_set():
                print("Curation cancelled by user; keeping days completed so far.")
                break
            with CURATION_LOCK:
                CURATION_STATE["current_date"] = current.strftime("%Y-%m-%d")
                CURATION_STATE["assets_done"] = 0
                CURATION_STATE["assets_total"] = 0
            daily_assets = search_assets_for_date(current, params["face_ids"])
            if daily_assets:
                with CURATION_LOCK:
                    CURATION_STATE["assets_total"] = len(daily_assets)
                ranked = evaluate_and_cluster(daily_assets, device, clip_model, clip_processor, params)
                if ranked:
                    date_key = current.strftime("%Y-%m-%d")
                    date_asset_ids = {a.get("id") for a in ranked if a.get("id")}
                    if (
                        skip_dates_already_in_album
                        and existing_album_ids
                        and date_asset_ids
                        and date_asset_ids.issubset(existing_album_ids)
                    ):
                        print(f"Skipping {date_key}: all candidates already in album.")
                    else:
                        curated_assets_by_date[date_key] = ranked
                        # Per-day checkpoint: a crash loses at most the current day.
                        save_checkpoint(params=params)
                        save_feature_cache()
            with CURATION_LOCK:
                CURATION_STATE["days_done"] += 1
                CURATION_STATE["assets_done"] = CURATION_STATE["assets_total"]
            current += datetime.timedelta(days=1)
        recompute_default_selection(params)
        save_checkpoint(params=params)
        save_feature_cache()
        if CURATION_CANCEL.is_set():
            print("Curation cancelled by user; days completed so far are kept.")
        else:
            print("Curation complete for selected date range.")
    except BaseException as e:
        with CURATION_LOCK:
            CURATION_STATE["state"] = "error"
            CURATION_STATE["error"] = f"{type(e).__name__}: {e}"
            CURATION_STATE["finished_at"] = time.time()
        release_memory()
        raise
    with CURATION_LOCK:
        CURATION_STATE["state"] = "cancelled" if CURATION_CANCEL.is_set() else "done"
        CURATION_STATE["finished_at"] = time.time()
    release_memory()


def rescore_date(assets, params):
    """
    Re-score one day's curated assets in place with `params`: re-aggregate the
    expression from cached face rows, then run the shared combine+cluster+sort.
    Only a one-time MediaPipe face backfill (v1 cache entries) touches ML; assets
    missing from the cache entirely go through the full feature path.
    Returns (scored, backfilled).
    """
    import pyiqa

    global _musiq_metric
    scored = backfilled = 0
    valid: list[dict] = []
    quality_scores: list[float] = []
    expression_scores: list[float] = []
    clip_feats: list[np.ndarray] = []
    for asset in assets:
        asset_id = asset.get("id")
        with _feature_lock:
            entry = _feature_cache.get(asset_id)
        if entry is not None:
            raw_q = entry["q"]
            feats = np.frombuffer(base64.b64decode(entry["clip"]), dtype=np.float32)
            face_rows = entry.get("faces")
            if face_rows is None:
                backfilled += 1
                print(f"    [backfill] asset={asset_id} missing face rows; running MediaPipe.")
                face_rows = _backfill_faces(asset, entry)
            expr, summary = _expression_summary(face_rows, asset_id, params)
            with _feature_lock:
                entry["e"] = expr
                entry["sum"] = summary or {}
        else:
            print(f"    [rescore] asset={asset_id} not in feature cache; computing features.")
            if _musiq_metric is None:
                _musiq_metric = pyiqa.create_metric("musiq", device="cpu")
            device, clip_model, clip_processor = load_models()
            computed = _compute_asset_features(asset, device, clip_model, clip_processor, params)
            if computed is None:
                print(f"    [rescore] asset={asset_id} no thumbnail; keeping old scores.")
                continue
            raw_q, face_rows, feats = computed
            expr, summary = _expression_summary(face_rows, asset_id, params)
            with _feature_lock:
                _feature_cache[asset_id] = {
                    "q": raw_q,
                    "e": expr,
                    "sum": summary or {},
                    "faces": face_rows,
                    "clip": base64.b64encode(feats.tobytes()).decode("ascii"),
                }
        asset["_quality_score"] = raw_q
        asset["_expression_score"] = expr
        if summary:
            asset["_expression_emotions"] = summary
        else:
            # Drop stale summaries so combine uses the freshly computed boost.
            asset.pop("_expression_emotions", None)
        valid.append(asset)
        quality_scores.append(raw_q)
        expression_scores.append(expr)
        clip_feats.append(feats)
        scored += 1
    combine_and_cluster(valid, quality_scores, expression_scores, clip_feats, params)
    return scored, backfilled


def rescore_job(params, dates=None):
    """
    Re-score curated days in place with the `params` snapshot, reusing the run
    state machine for /progress. Never touches selected_asset_ids, so manual
    curation survives; only score/cluster fields change.
    """
    global _last_scored_params
    params = params or snapshot_params()
    _last_scored_params = copy.deepcopy(params)
    dates = dates or sorted(curated_assets_by_date)
    CURATION_CANCEL.clear()
    with CURATION_LOCK:
        CURATION_STATE.update(
            state="running", phase="rescore", days_total=len(dates), days_done=0,
            current_date="", assets_done=0, assets_total=0, started_at=time.time(),
            finished_at=None, error="",
            start_date=dates[0] if dates else "",
            end_date=dates[-1] if dates else "",
        )
    try:
        for date in dates:
            if CURATION_CANCEL.is_set():
                print("Re-score cancelled by user; keeping days completed so far.")
                break
            assets = curated_assets_by_date.get(date) or []
            with CURATION_LOCK:
                CURATION_STATE["current_date"] = date
                CURATION_STATE["assets_done"] = 0
                CURATION_STATE["assets_total"] = len(assets)
            if assets:
                _scored, backfilled = rescore_date(assets, params)
                if backfilled:
                    # Crash safety: the face backfill is the slow part, persist as we go.
                    save_checkpoint(params=params)
                    save_feature_cache()
            with CURATION_LOCK:
                CURATION_STATE["days_done"] += 1
                CURATION_STATE["assets_done"] = CURATION_STATE["assets_total"]
        save_checkpoint(params=params)
        save_feature_cache()
        if CURATION_CANCEL.is_set():
            print("Re-score cancelled; days completed so far keep their new scores.")
        else:
            print(f"Re-score complete for {len(dates)} day(s).")
    except BaseException as e:
        with CURATION_LOCK:
            CURATION_STATE["state"] = "error"
            CURATION_STATE["error"] = f"{type(e).__name__}: {e}"
            CURATION_STATE["finished_at"] = time.time()
        release_memory()
        raise
    with CURATION_LOCK:
        CURATION_STATE["state"] = "cancelled" if CURATION_CANCEL.is_set() else "done"
        CURATION_STATE["finished_at"] = time.time()
    release_memory()


# ==========================================
# PIXTALE SYNC (album -> gallery, see sync_album_to_pixtale.py)
# ==========================================
def _pixtale_password() -> str:
    """PIXTALE_PASSWORD env, else the admin password from ~/pixtale/.env."""
    return os.environ.get("PIXTALE_PASSWORD") or (_read_default_password() or "")


def fetch_pixtale_storages(ttl_sec: float = 300.0) -> list[dict]:
    """Pixtale's enabled storages [{storageId, name, type}]; TTL-cached; raises on failure."""
    global _pixtale_storage_cache
    with _pixtale_storage_lock:
        cached = _pixtale_storage_cache
        if cached and time.time() - cached["fetched_at"] < ttl_sec:
            return cached["storages"]
    password = _pixtale_password()
    if not password:
        raise RuntimeError("No pixtale password (set PIXTALE_PASSWORD or populate ~/pixtale/.env)")
    client = PixtaleClient(PIXTALE_URL, PIXTALE_USER, password, storage_id=None)
    storages = client.storages()
    with _pixtale_storage_lock:
        _pixtale_storage_cache = {"fetched_at": time.time(), "storages": storages}
    return storages


def _safe_pixtale_storages() -> tuple[list[dict], str]:
    """fetch_pixtale_storages that degrades to ([], reason) when pixtale is unreachable."""
    try:
        return fetch_pixtale_storages(), ""
    except Exception as exc:
        print(f"Could not list pixtale storages: {exc}")
        return [], f"{type(exc).__name__}: {exc}"


def sync_job(storage_id: str, delete: bool, dry_run: bool) -> None:
    """Background album -> pixtale reconcile; mirrors the curation state machine."""
    SYNC_CANCEL.clear()
    with SYNC_LOCK:
        SYNC_STATE.update(
            state="running", phase="fetching_album", dry_run=dry_run, delete=delete,
            storage_id=storage_id or "(auto-detect)",
            plan_add=0, plan_delete=0, add_done=0, add_total=0, deleted=0,
            counters={"ok": 0, "duplicate": 0, "error": 0},
            samples_add=[], samples_delete=[], failures=[], current_asset="",
            albums={}, albums_error="",
            started_at=time.time(), finished_at=None, error="",
        )

    def on_event(ev: dict) -> None:
        with SYNC_LOCK:
            SYNC_STATE["phase"] = ev.get("phase", SYNC_STATE["phase"])
            for key in ("plan_add", "plan_delete", "add_done", "add_total", "deleted"):
                if key in ev:
                    SYNC_STATE[key] = ev[key]
            if ev.get("storage_id"):
                SYNC_STATE["storage_id"] = ev["storage_id"]
            if "counters" in ev:
                SYNC_STATE["counters"] = dict(ev["counters"])
            if "samples_add" in ev:
                SYNC_STATE["samples_add"] = ev["samples_add"][:5]
            if "samples_delete" in ev:
                SYNC_STATE["samples_delete"] = ev["samples_delete"][:5]
            if "last_asset" in ev:
                SYNC_STATE["current_asset"] = ev["last_asset"]
            if ev.get("last_status") == "error" and len(SYNC_STATE["failures"]) < 20:
                SYNC_STATE["failures"].append([ev.get("last_asset", ""), ev.get("last_detail", "")])

    try:
        summary = run_sync(
            IMMICH_URL, API_KEY, ALBUM_ID, PIXTALE_URL, PIXTALE_USER, _pixtale_password(),
            storage_id or None,  # "" -> PixtaleClient auto-detects (prefers the S3 backend)
            delete=delete, dry_run=dry_run, concurrency=SYNC_CONCURRENCY,
            progress=on_event, cancelled=SYNC_CANCEL.is_set,
        )
        with SYNC_LOCK:
            SYNC_STATE.update(
                counters=dict(summary["counters"]),
                failures=list(summary["failures"]),
                samples_add=summary["samples_add"][:5],
                samples_delete=summary["samples_delete"][:5],
                plan_add=summary["plan_add"], plan_delete=summary["plan_delete"],
                add_done=summary["add_done"], add_total=summary["add_total"],
                deleted=summary["deleted"],
                albums=summary.get("albums") or {},
                albums_error=summary.get("albums_error", ""),
                storage_id=summary.get("storage_id") or SYNC_STATE["storage_id"],
                state="cancelled" if summary["cancelled"] else "done",
                finished_at=time.time(),
            )
    except BaseException as e:
        with SYNC_LOCK:
            SYNC_STATE["state"] = "error"
            SYNC_STATE["error"] = f"{type(e).__name__}: {e}"
            SYNC_STATE["finished_at"] = time.time()
        raise


def _sync_snapshot() -> dict:
    with SYNC_LOCK:
        snap = dict(SYNC_STATE)
        snap["counters"] = dict(SYNC_STATE["counters"])
        snap["failures"] = list(SYNC_STATE["failures"])
        snap["samples_add"] = list(SYNC_STATE["samples_add"])
        snap["samples_delete"] = list(SYNC_STATE["samples_delete"])
    if snap["started_at"]:
        snap["elapsed_sec"] = int((snap["finished_at"] or time.time()) - snap["started_at"])
    if snap["phase"] == "deleting":
        total, done = snap["plan_delete"], snap["deleted"]
    else:
        total, done = snap["add_total"], snap["add_done"]
    if total:
        snap["pct"] = int(round(100 * done / total))
    else:
        snap["pct"] = 100 if snap["state"] != "running" else 0
    return snap


def build_score_line(asset: dict) -> str:
    """
    Build a short HTML snippet describing base score and boosts
    (cluster-top and family-smile). Reused across templates.
    """
    base = asset.get("_base_score", asset.get("_total_score", 0.0))
    parts = [f"Base: {base:.3f}"]
    cluster_boost = asset.get("_cluster_top_boost")
    if cluster_boost is not None:
        parts.append(f"(+{cluster_boost:.3f} top-in-cluster)")
    smile_boost = asset.get("_smile_boost")
    if smile_boost and smile_boost > 0:
        parts.append(f"(+{smile_boost:.3f} family smile)")
    return " ".join(parts)


def _render_photo_card(asset, score_line_fn, cluster_rank=None, show_badges=False, selected_ids=frozenset()):
    """Render one selectable photo card (image + score overlay + details) for all gallery views."""
    return render_template(
        "partials/photo_card.html",
        asset=asset,
        score_line=score_line_fn,
        cluster_rank=cluster_rank,
        show_badges=show_badges,
        selected_ids=selected_ids,
    )


# ==========================================
# 4. FLASK WEB UI ROUTES  (templates live in ./templates, styles/scripts in ./static)
# ==========================================

@app.route('/')
def index():
    """Home: date range form to trigger curation on demand."""
    return render_template("home.html", checkpoint=_checkpoint_info, active_page="home")


@app.route('/checkpoint/discard', methods=['POST'])
def checkpoint_discard():
    """Throw away the restored checkpoint (file + in-memory results)."""
    discard_checkpoint()
    return redirect(url_for('index'))


# ==========================================
# JOB SUBPROCESS: curation/rescore run in a child process that exits afterwards
# ==========================================
# The ML stack costs ~1.25 GB to import (torch, TensorFlow via mediapipe,
# pyiqa) and the cached models another ~0.7 GB, and none of it can be
# unloaded from a live Python process. So instead of running jobs in a
# thread, the routes spawn `python daily_highlights.py --run-job <req.json>`
# — a fresh interpreter that imports the ML stack, does the work, persists
# results through the normal checkpoint + feature-cache files, and exits,
# handing all of that memory back to the OS. The parent keeps a watcher
# thread that mirrors the child's progress file into CURATION_STATE for
# /progress, and reloads results from the checkpoint when it finishes.
JOB_REQUEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curation_job.json")
JOB_STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curation_job_status.json")
JOB_CANCEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curation_job_cancel")
JOB_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curation_job.log")
_CURATION_PROC: subprocess.Popen | None = None
_JOB_START_LOCK = threading.Lock()
_STATUS_LOCK = threading.Lock()  # dumper thread + job finally both write the status file


def _write_job_status() -> None:
    """Atomically dump CURATION_STATE (+ pid) for the parent's watcher."""
    with _STATUS_LOCK, CURATION_LOCK:
        snap = dict(CURATION_STATE)
        snap["pid"] = os.getpid()
    tmp = JOB_STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(snap, f)
    os.replace(tmp, JOB_STATUS_PATH)


def _run_job_child(request_path: str) -> None:
    """Worker entry point (`--run-job`): run one job, persist, exit.

    Progress is published by rewriting the status file every second; a cancel
    is requested by the parent creating the cancel-flag file.
    """
    with open(request_path) as f:
        job = json.load(f)
    os.remove(request_path)
    stop = threading.Event()

    def dumper() -> None:
        while not stop.is_set():
            _write_job_status()
            stop.wait(1.0)
        # The finally block below writes the final state once the job thread is done.

    def cancel_watcher() -> None:
        while not stop.is_set():
            if os.path.exists(JOB_CANCEL_PATH):
                CURATION_CANCEL.set()
                return
            stop.wait(0.5)

    threading.Thread(target=dumper, daemon=True).start()
    threading.Thread(target=cancel_watcher, daemon=True).start()
    code = 0
    try:
        if job["type"] == "curation":
            run_curation_for_date_range(
                datetime.date.fromisoformat(job["start_date"]),
                datetime.date.fromisoformat(job["end_date"]),
                skip_dates_already_in_album=bool(job.get("skip_in_album")),
                params=job.get("params"),
            )
        elif job["type"] == "rescore":
            rescore_job(job.get("params"), job.get("dates"))
        else:
            raise ValueError(f"unknown job type {job['type']!r}")
    except BaseException as e:
        print(f"job failed: {type(e).__name__}: {e}")
        code = 1
    finally:
        stop.set()
        _write_job_status()
        if os.path.exists(JOB_CANCEL_PATH):
            os.remove(JOB_CANCEL_PATH)
    # Hard-exit instead of returning: mediapipe's native dispatcher thread
    # crashes during interpreter teardown (turning a successful job into a
    # nonzero exit), and there is nothing left to clean up anyway — the job
    # already persisted results through the checkpoint + feature-cache files.
    sys.stdout.flush()
    os._exit(code)


def _job_busy() -> bool:
    """True when a job subprocess is alive here OR was left running by a previous server process."""
    if _CURATION_PROC is not None and _CURATION_PROC.poll() is None:
        return True
    try:
        with open(JOB_STATUS_PATH) as f:
            pid = json.load(f).get("pid")
        if pid and pid != os.getpid():
            os.kill(pid, 0)  # raises ProcessLookupError if that process is gone
            return True
    except (OSError, ValueError):
        pass
    return False


def _watch_job(proc: subprocess.Popen) -> None:
    """Mirror the child's status file into CURATION_STATE; absorb results on exit."""
    global _CURATION_PROC
    code = None
    while True:
        code = proc.poll()
        try:
            with open(JOB_STATUS_PATH) as f:
                final = json.load(f)
            final.pop("pid", None)
            with CURATION_LOCK:
                CURATION_STATE.update(final)
        except (OSError, ValueError):
            pass
        if code is not None:
            break
        time.sleep(1.0)
    if code == 0:
        with CURATION_LOCK:
            if CURATION_STATE["state"] == "running":  # child exited without a final write
                CURATION_STATE.update(state="done", finished_at=time.time())
        # Pull the child's results and its feature-cache additions into this process.
        load_checkpoint()
        with _feature_lock:
            load_feature_cache()
    else:
        with CURATION_LOCK:
            CURATION_STATE.update(
                state="error",
                error=CURATION_STATE.get("error") or f"job process exited with code {code} (see curation_job.log)",
                finished_at=time.time(),
            )
    _CURATION_PROC = None


def _start_job(job: dict) -> None:
    """Write the request file, spawn the worker child, start the watcher."""
    global _CURATION_PROC
    with _JOB_START_LOCK:
        tmp = JOB_REQUEST_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(job, f)
        os.replace(tmp, JOB_REQUEST_PATH)
        for stale in (JOB_CANCEL_PATH, JOB_STATUS_PATH):
            if os.path.exists(stale):
                os.remove(stale)
        log = open(JOB_LOG_PATH, "a")
        # start_new_session: a debug-reloader restart kills this server but not a running job.
        _CURATION_PROC = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--run-job", JOB_REQUEST_PATH],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    with CURATION_LOCK:
        CURATION_STATE.update(
            state="running", phase="starting", started_at=time.time(), finished_at=None,
            error="", current_date="", days_done=0, days_total=0, assets_done=0, assets_total=0,
        )
    threading.Thread(target=_watch_job, args=(_CURATION_PROC,), daemon=True).start()


@app.route('/process', methods=['POST'])
def process():
    """Start curation for the submitted date range in a worker subprocess."""
    start_s = request.form.get('start_date')
    end_s = request.form.get('end_date')
    skip_in_album = bool(request.form.get('skip_in_album'))
    if not start_s or not end_s:
        return redirect(url_for('index'))
    # Validate date format without catching (invalid form → let strptime raise).
    start_date = datetime.datetime.strptime(start_s, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_s, '%Y-%m-%d').date()
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    with CURATION_LOCK:
        if CURATION_STATE["state"] == "running":
            return "A curation run is already in progress — watch it on the home page.", 409
    if _job_busy():
        return "A previous curation job process is still running — watch it on the home page.", 409
    _start_job({
        "type": "curation",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "skip_in_album": skip_in_album,
        "params": snapshot_params(),
    })
    return redirect(url_for('index'))


@app.route('/progress')
def progress():
    """JSON snapshot of the curation run; polled by the home page."""
    with CURATION_LOCK:
        snap = dict(CURATION_STATE)
    if snap["started_at"]:
        snap["elapsed_sec"] = int((snap["finished_at"] or time.time()) - snap["started_at"])
    return jsonify(snap)


@app.route('/cancel', methods=['POST'])
def cancel_curation():
    """Ask the running curation job to stop after the day it is on."""
    CURATION_CANCEL.set()
    proc = _CURATION_PROC
    if proc is not None and proc.poll() is None:
        # The child's cancel-watcher polls for this flag file.
        open(JOB_CANCEL_PATH, "w").close()
    with CURATION_LOCK:
        was_running = CURATION_STATE["state"] == "running"
    return jsonify({"ok": was_running})

@app.route('/rescore', methods=['POST'])
def rescore():
    """Re-score the current curated results with the saved settings (no re-search).

    Recomputes expression/scores/clusters from cached features in the background;
    manual selection is preserved. Optional `date` field limits it to one day.
    """
    with CURATION_LOCK:
        busy = CURATION_STATE["state"] == "running"
    if busy:
        return "A curation or re-score is already running.", 409
    if _job_busy():
        return "A previous curation job process is still running — try again in a moment.", 409
    if not curated_assets_by_date:
        return redirect(url_for('top'))
    date = request.form.get("date", "")
    if date in curated_assets_by_date:
        dates = [date]
    else:
        dates = sorted(curated_assets_by_date)
    _start_job({"type": "rescore", "params": snapshot_params(), "dates": dates})
    return redirect(url_for('top'))


@app.route('/sync')
def sync_page():
    """Storage picker + live progress for the album -> pixtale reconcile."""
    storages, storage_error = _safe_pixtale_storages()
    default_id = default_storage_id(storages) or PIXTALE_STORAGE_ID or ""
    with SYNC_LOCK:
        running = SYNC_STATE["state"] == "running"
    return render_template(
        "sync.html",
        storages=storages,
        storage_error=storage_error,
        default_storage_id=default_id,
        snap=_sync_snapshot(),
        running=running,
        active_page="sync",
    )


@app.route('/sync/start', methods=['POST'])
def sync_start():
    """Start a background album -> pixtale sync (single-flight within sync only)."""
    with SYNC_LOCK:
        if SYNC_STATE["state"] == "running":
            return "A pixtale sync is already running — watch it on the sync page.", 409
    storage_id = (request.form.get("storage_id") or "").strip()
    delete = bool(request.form.get("delete"))
    dry_run = bool(request.form.get("dry_run"))  # set by the "Preview" submit button
    if not _pixtale_password():
        return "Missing pixtale password: set PIXTALE_PASSWORD or populate ~/pixtale/.env", 409
    threading.Thread(target=sync_job, args=(storage_id, delete, dry_run), daemon=True).start()
    return redirect(url_for('sync_page'))


@app.route('/sync/status')
def sync_status():
    """JSON snapshot of the sync run; polled by the sync page."""
    return jsonify(_sync_snapshot())


@app.route('/sync/cancel', methods=['POST'])
def sync_cancel():
    """Ask the running sync to stop; in-flight uploads finish and are counted."""
    SYNC_CANCEL.set()
    with SYNC_LOCK:
        was_running = SYNC_STATE["state"] == "running"
    return jsonify({"ok": was_running})


def _rss_kb() -> dict:
    """Current + peak RSS in KB from /proc (Linux)."""
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, kb, _ = line.split()
                    out[key.rstrip(':').lower()] = int(kb)
    except OSError:
        pass
    return out


@app.route('/debug/mem')
def debug_mem():
    """Memory diagnostics: process RSS, model footprint, per-cache sizes. Read-only."""
    def deep_size(o, _seen=None) -> int:
        if _seen is None:
            _seen = set()
        if id(o) in _seen:
            return 0
        _seen.add(id(o))
        total = sys.getsizeof(o)
        if isinstance(o, dict):
            for k, v in o.items():
                total += deep_size(k, _seen) + deep_size(v, _seen)
        elif isinstance(o, (list, tuple, set, frozenset)):
            for v in o:
                total += deep_size(v, _seen)
        return total

    with _thumb_lock:
        thumbs_bytes = sum(len(b) for b, _mime in _thumb_cache.values())
    person_thumbs_bytes = sum(len(b) for b, _mime in _person_thumb_cache.values())
    with _feature_lock:
        feature = {"entries": len(_feature_cache), "bytes": deep_size(_feature_cache)}

    models = {"clip_loaded": _models_cache is not None, "musiq_loaded": _musiq_metric is not None,
              "face_landmarker_loaded": _face_landmarker is not None}
    if _models_cache is not None:
        clip = _models_cache[1]
        models["clip_params_mb"] = round(sum(p.numel() * p.element_size() for p in clip.parameters()) / 2**20, 1)
    if _musiq_metric is not None:
        try:
            models["musiq_params_mb"] = round(
                sum(p.numel() * p.element_size() for p in _musiq_metric.model.parameters()) / 2**20, 1)
        except AttributeError:
            pass

    return jsonify({
        "rss": _rss_kb(),
        "gc": {"counts": gc.get_count(),
               "collected_total": sum(s["collected"] for s in gc.get_stats())},
        "models": models,
        "caches": {
            "thumbs": {"entries": len(_thumb_cache), "bytes": thumbs_bytes},
            "person_thumbs": {"entries": len(_person_thumb_cache), "bytes": person_thumbs_bytes},
            "faces": {"entries": len(_faces_cache)},
            "feature_cache": feature,
            "curated_days": len(curated_assets_by_date),
            "curated_assets": sum(len(v) for v in curated_assets_by_date.values()),
            "selected_ids": len(selected_asset_ids),
        },
        "jobs": {"curation": CURATION_STATE["state"], "sync": SYNC_STATE["state"]},
    })


@app.route('/top')
def top():
    """Show top photo from each cluster per day, plus any selected non-tops."""
    top_by_date = {}
    for date, assets in curated_assets_by_date.items():
        # pick best asset per cluster by final score
        best_per_cluster = {}
        for a in assets:
            cid = a.get("_cluster_id", -1)
            cur = best_per_cluster.get(cid)
            score = a.get("_final_score", a.get("_total_score", 0.0))
            if (cur is None) or (score > cur.get("_final_score", cur.get("_total_score", 0.0))):
                best_per_cluster[cid] = a
        # sort top assets for this day by final score
        tops = sorted(
            best_per_cluster.values(),
            key=lambda x: -(x.get("_final_score", x.get("_total_score", 0.0))),
        )
        top_ids = {a.get("id") for a in tops if a.get("id")}
        # add selected assets that are not cluster tops so they appear and get added to album
        date_asset_ids = {a.get("id") for a in assets if a.get("id")}
        selected_on_date = selected_asset_ids & date_asset_ids
        extra_ids = selected_on_date - top_ids
        id_to_asset = {a.get("id"): a for a in assets if a.get("id")}
        extra = [id_to_asset[aid] for aid in extra_ids if aid in id_to_asset]
        top_by_date[date] = tops + extra
    # Derive selection for top page:
    # - start from global selected_asset_ids (may have been updated via deep view)
    # - if nothing is selected yet, apply the default threshold rule to top photos
    selected_ids_for_top = set(selected_asset_ids)
    if not selected_ids_for_top:
        p = snapshot_params()
        for _date, tops in top_by_date.items():
            for a in tops:
                score = a.get("_final_score", a.get("_total_score", 0.0))
                if score >= p["default_min_final_score"] and "id" in a:
                    selected_ids_for_top.add(a["id"])
    def _card_html(asset, cluster_rank=None, show_badges=False):
        return _render_photo_card(asset, build_score_line, cluster_rank, show_badges, selected_ids_for_top)

    # Re-score panel context: compare the params the scores were computed with
    # against the live settings, and count photos needing a one-time face backfill.
    with CURATION_LOCK:
        running = CURATION_STATE["state"] == "running"
    scored_params = (_checkpoint_info or {}).get("params")
    settings_differ = scored_params is not None and (
        params_fingerprint(scored_params) != params_fingerprint(_params)
    )
    backfill_needed = sum(
        1
        for assets in curated_assets_by_date.values()
        for a in assets
        if (_feature_cache.get(a.get("id")) or {}).get("faces") is None
    )
    return render_template(
        "top.html",
        data=top_by_date,
        card_html=_card_html,
        active_page="top",
        rescore={
            "running": running,
            "params": _params,
            "settings_differ": settings_differ,
            "backfill_needed": backfill_needed,
            "days": sorted(curated_assets_by_date),
        },
    )

@app.route('/clusters')
def clusters():
    """Cluster debug view with all ranked assets. Sorted by cluster then by top score (best first)."""
    def _card_html(asset, cluster_rank=None, show_badges=False):
        return _render_photo_card(asset, build_score_line, cluster_rank, show_badges, selected_asset_ids)

    # Per date: sort by cluster top score (so best clusters first), then by score within cluster
    def _sorted_assets(assets):
        cluster_top = {}
        for a in assets:
            cid = a.get("_cluster_id", -1)
            s = a.get("_final_score", a.get("_total_score", 0.0))
            cluster_top[cid] = max(cluster_top.get(cid, s), s)
        return sorted(
            assets,
            key=lambda a: (
                -cluster_top.get(a.get("_cluster_id", -1), 0.0),
                a.get("_cluster_id", -1),
                -(a.get("_final_score", a.get("_total_score", 0.0))),
            ),
        )

    sorted_data = {
        date: _sorted_assets(assets)
        for date, assets in curated_assets_by_date.items()
    }

    return render_template(
        "clusters.html",
        data=sorted_data,
        card_html=_card_html,
        active_page="clusters",
    )


@app.route('/date/<date_str>', methods=['GET', 'POST'])
def deep_view_date(date_str):
    """Deep view for a single date; updates global selection and returns to clusters."""
    global selected_asset_ids
    assets = curated_assets_by_date.get(date_str)
    if not assets:
        return redirect(url_for('clusters'))
    if request.method == 'POST':
        # Update selection only for this day's assets
        date_ids = {a.get("id") for a in assets if a.get("id")}
        posted = set(request.form.getlist("asset_ids"))
        selected_asset_ids -= date_ids
        selected_asset_ids |= (posted & date_ids)
        save_checkpoint()
        return redirect(url_for('clusters'))

    def _card_html(asset, cluster_rank=None, show_badges=False):
        return _render_photo_card(asset, build_score_line, cluster_rank, show_badges, selected_asset_ids)

    return render_template(
        "date_view.html",
        date=date_str,
        assets=assets,
        card_html=_card_html,
        active_page="",
    )

def _thumb_headers(ctype: str) -> dict:
    # Asset thumbnails never change for a given id, so let the browser cache hard.
    return {"Content-Type": ctype, "Cache-Control": "public, max-age=86400, immutable"}


@app.route('/thumb/<asset_id>')
def serve_thumbnail(asset_id):
    """
    Acts as a proxy to fetch thumbnails securely from Immich to display in our UI
    without leaking the API key to the browser. Served from an in-memory cache on
    repeat requests.
    """
    with _thumb_lock:
        cached = _thumb_cache.get(asset_id)
    if cached is not None:
        content, ctype = cached
        return content, 200, _thumb_headers(ctype)
    url = f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=thumbnail"
    resp = requests.get(url, headers={"x-api-key": API_KEY}, timeout=30)
    ctype = resp.headers.get('Content-Type', 'image/jpeg')
    if resp.status_code == 200:
        with _thumb_lock:
            if len(_thumb_cache) >= 2000:
                _thumb_cache.clear()
            _thumb_cache[asset_id] = (resp.content, ctype)
    return resp.content, resp.status_code, _thumb_headers(ctype)


@app.route('/preview/<asset_id>')
def serve_preview(asset_id):
    """
    Proxy for the large (1440px) Immich preview used by the lightbox. Too big to
    hold in memory per asset, so this streams through — the browser caches it
    (immutable per asset id) so repeated opens don't re-fetch.
    """
    if not _UUID_RE.match(asset_id):
        return "Not found", 404
    resp = requests.get(
        f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail?size=preview",
        headers={"x-api-key": API_KEY}, timeout=60,
    )
    ctype = resp.headers.get('Content-Type', 'image/jpeg')
    return resp.content, resp.status_code, _thumb_headers(ctype)


def fetch_immich_people(max_pages: int = 2, page_size: int = 1000, ttl_sec: float = 300.0) -> list[dict]:
    """
    List named, non-hidden Immich people for the settings face pickers. Immich's
    directory also holds thousands of unnamed detected faces, so we page from the
    front (sorted by asset count) and stop at the first page with no named person.
    Raises requests.RequestException on network/API failure.
    """
    global _people_cache
    with _people_lock:
        cached = _people_cache
        if cached and time.time() - cached["fetched_at"] < ttl_sec:
            return cached["people"]
    people: list[dict] = []
    for page in range(1, max_pages + 1):
        resp = requests.get(
            f"{IMMICH_URL}/api/people",
            headers=HEADERS,
            params={"size": page_size, "page": page},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        named = [
            {"id": p["id"], "name": p["name"]}
            for p in data.get("people", [])
            if p.get("id") and (p.get("name") or "").strip() and not p.get("isHidden")
        ]
        if not named:
            break
        people.extend(named)
        if not data.get("hasNextPage"):
            break
    with _people_lock:
        _people_cache = {"fetched_at": time.time(), "people": people}
    return people


def _safe_people() -> list[dict]:
    """fetch_immich_people that degrades to [] (text-input fallback) on failure."""
    try:
        return fetch_immich_people()
    except requests.RequestException as exc:
        print(f"Could not list Immich people: {exc}")
        return []


@app.route('/api/people')
def api_people():
    """JSON people list for the settings page; 200 with an error field on failure."""
    try:
        return jsonify({"people": fetch_immich_people(), "error": ""})
    except requests.RequestException as exc:
        return jsonify({"people": [], "error": str(exc)})


@app.route('/people/<person_id>/thumb')
def serve_person_thumbnail(person_id: str):
    """Proxy for Immich person avatars (keeps the API key server-side)."""
    if not _UUID_RE.match(person_id):
        return "Not found", 404
    pid = person_id.lower()
    with _thumb_lock:
        cached = _person_thumb_cache.get(pid)
    if cached is not None:
        content, ctype = cached
        return content, 200, _thumb_headers(ctype)
    resp = requests.get(
        f"{IMMICH_URL}/api/people/{pid}/thumbnail",
        headers={"x-api-key": API_KEY},
        timeout=15,
    )
    ctype = resp.headers.get('Content-Type', 'image/jpeg')
    if resp.status_code == 200:
        with _thumb_lock:
            _person_thumb_cache[pid] = (resp.content, ctype)
    return resp.content, resp.status_code, _thumb_headers(ctype)


def _face_ids_from_form(req, name: str) -> list[str]:
    """Face set from the form: checkboxes, unless the comma-separated fallback is used."""
    text = (req.form.get(f"{name}_text") or "").strip()
    if text:
        return [piece for chunk in text.split(",") for piece in chunk.split()]
    return req.form.getlist(name)


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """View and edit the scoring params + face sets used by runs and re-scores."""
    global _params
    with CURATION_LOCK:
        running = CURATION_STATE["state"] == "running"
        phase = CURATION_STATE.get("phase", "")
    if request.method == 'POST':
        raw = {name: request.form.get(name, "") for name in PARAM_SPECS}
        face_ids = _face_ids_from_form(request, "face_ids")
        family_ids = _face_ids_from_form(request, "family_faces")
        params, errors = sanitize_params(raw, face_ids, family_ids)
        if errors or params is None:
            cleaned_ids, cleaned_fam, _extra = sanitize_faces(face_ids, family_ids)
            form_values = {**raw, "face_ids": cleaned_ids, "family_faces": cleaned_fam}
            return render_template(
                "settings.html",
                specs=PARAM_SPECS, groups=PARAM_GROUPS, face_sets=FACE_SETS,
                params=_params, people=_safe_people(), cache_stats=_face_cache_stats(),
                running=running, phase=phase, errors=errors or ["Settings could not be saved"],
                form_values=form_values, saved=None, active_page="settings",
            )
        _params = params
        save_params()
        return redirect(url_for('settings', saved=1))

    with _feature_lock:
        total = len(_feature_cache)
    return render_template(
        "settings.html",
        specs=PARAM_SPECS, groups=PARAM_GROUPS, face_sets=FACE_SETS,
        params=_params, people=_safe_people(), cache_stats=_face_cache_stats(),
        running=running, phase=phase, errors=[], form_values=None,
        saved=request.args.get("saved"), active_page="settings",
    )


def _face_cache_stats() -> dict:
    """How many cached entries carry per-face rows, by detection mode."""
    boxes = full = 0
    with _feature_lock:
        entries = list(_feature_cache.values())
    for entry in entries:
        payload = entry.get("faces")
        if payload is None:
            continue
        if payload.get("mode") == "boxes":
            boxes += 1
        else:
            full += 1
    return {"boxes": boxes, "full": full, "total": len(entries)}

@app.route('/submit', methods=['POST'])
def submit():
    """Handles the form submission and pushes the final IDs to Immich."""
    selected_ids = request.form.getlist('asset_ids')
    
    if not selected_ids:
        return redirect(url_for('confirm', status="empty"))
        
    print(f"\nAdding {len(selected_ids)} assets to Album {ALBUM_ID}...")
    success = add_assets_to_album(ALBUM_ID, selected_ids)
    
    if success:
        print("Success! You can safely terminate this script (Ctrl+C).")
        save_checkpoint(submitted=True)
        return redirect(url_for('confirm', status="ok", count=len(selected_ids)))
    else:
        return redirect(url_for('confirm', status="error"))


@app.route('/confirm')
def confirm():
    """Separate confirmation page after submission."""
    status = request.args.get("status", "ok")
    count = request.args.get("count", "0")
    return render_template("confirm.html", status=status, count=count, active_page="")

# ==========================================
# 5. ENTRY POINT
# ==========================================
def _parse_month(month_str: str) -> tuple[datetime.date, datetime.date]:
    """Parse 'YYYY-MM' into (first_day, last_day) of that month."""
    year, month = map(int, month_str.split("-"))
    first = datetime.date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    last = datetime.date(year, month, last_day)
    return first, last


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Immich Album Curator: curate photos by date range and add to album.",
    )
    parser.add_argument(
        "--month",
        metavar="YYYY-MM",
        help="Process one month of photos and add selected photos to the album (no UI). Example: --month 2026-02",
    )
    parser.add_argument(
        "--skip-in-album",
        action="store_true",
        help="With --month: skip dates whose candidates are already in the album.",
    )
    parser.add_argument(
        "--run-job",
        metavar="REQUEST_JSON",
        help=argparse.SUPPRESS,  # internal: worker entry point used by the web UI
    )
    args = parser.parse_args()

    if args.run_job:
        _run_job_child(args.run_job)  # never returns

    print("========================================")
    print(" Immich AI Album Curator ")
    print("========================================\n")

    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please update the API_KEY, IMMICH_URL, and ALBUM_ID variables inside the script first.")
        exit(1)

    if args.month:
        start_date, end_date = _parse_month(args.month)
        print(f"Processing month {args.month} ({start_date} to {end_date})...")
        run_curation_for_date_range(
            start_date,
            end_date,
            skip_dates_already_in_album=args.skip_in_album,
        )
        selected_ids = list(selected_asset_ids)
        if not selected_ids:
            print("No photos selected. Nothing to add to the album.")
            exit(0)
        print(f"Adding {len(selected_ids)} selected assets to album {ALBUM_ID}...")
        success = add_assets_to_album(ALBUM_ID, selected_ids)
        if success:
            print(f"Successfully added {len(selected_ids)} assets to the album.")
            save_checkpoint(submitted=True)
        else:
            print("Failed to add assets to the album.")
            exit(1)
        exit(0)

    # FLASK_RELOADER=0 runs a single process: the reloader's supervisor process
    # otherwise keeps a full second copy of the imports alive (~0.65 GB idle).
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True,
            use_reloader=os.environ.get("FLASK_RELOADER", "1") != "0")