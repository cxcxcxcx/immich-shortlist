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
import os
import io
import json
import time
import base64
import requests
import datetime
import threading
from zoneinfo import ZoneInfo
from PIL import Image

# ML / evaluation
import torch
import numpy as np
import mediapipe as mp
from transformers import CLIPModel, CLIPProcessor
import pyiqa

# Web UI
from flask import Flask, render_template, request, redirect, url_for, jsonify

# ==========================================
# CONFIGURATION
# ==========================================
from config import IMMICH_URL, API_KEY, ALBUM_ID, LOCAL_TZ_NAME

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

# Global normalization parameters (tunable; initial guesses):
# - MUSIQ quality scores we observe are typically ~60–74
# - expression scores in roughly [-0.5, 1.0]
QUALITY_MEAN = 65.0
QUALITY_STD = 6
EXPRESSION_MEAN = 0.25
EXPRESSION_STD = 0.3

# Weights for combined score: total_base = QUALITY_WEIGHT * quality_norm
#   + EXPRESSION_WEIGHT * expression_norm
#   + PERSON_WEIGHT * (person_match_count - 1)
QUALITY_WEIGHT = 0.4
EXPRESSION_WEIGHT = 0.2
PERSON_WEIGHT = 0.2
# Family boost: continuous smile, eyes_open, facing (facing weighted higher).
# Only count a family face if facing >= FAMILY_FACING_MIN; each face's score is discounted by 0.5.
FAMILY_FACING_MIN = 0.75
FAMILY_BOOST_DISCOUNT = 0.5
FAMILY_BOOST_WEIGHT_EYES = 0.25
FAMILY_BOOST_WEIGHT_SMILE = 0.25
FAMILY_BOOST_WEIGHT_FACING = 0.5

# CLIP clustering similarity threshold (cosine). Higher = fewer, tighter clusters.
CLIP_CLUSTER_SIM_THRESHOLD = 0.9

# Cluster-top boost: extra score for the top-ranked photo in each cluster.
# The boost grows with cluster size but saturates as size increases.
CLUSTER_TOP_WEIGHT = 0.5
CLUSTER_TOP_MAX_SIZE = 8

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
# Default selection behaviour
DEFAULT_MIN_FINAL_SCORE = 1.0
DEFAULT_MIN_TOP_CLUSTERS = 6
# Lazy-loaded models (loaded on first /process)
_models_cache = None
_musiq_metric = None
# Cache for per-asset face metadata fetched from Immich
_faces_cache: dict[str, list[dict]] = {}
# In-memory thumbnail cache (Immich thumbnails are immutable per asset id).
_thumb_cache: dict[str, tuple[bytes, str]] = {}
_thumb_lock = threading.Lock()
# Live progress of a curation run (written by the job thread, read by /progress).
CURATION_LOCK = threading.Lock()
CURATION_STATE = {
    "state": "idle",  # idle | running | done | cancelled | error
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
# On-disk checkpoint of curation results, so restarts don't lose them.
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curation_checkpoint.json")
_checkpoint_info: dict | None = None


def save_checkpoint(submitted: bool = False) -> None:
    """Atomically write curated results + selection to disk."""
    global _checkpoint_info
    data = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "submitted": submitted,
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
FEATURE_CACHE_VERSION = 1
_feature_cache: dict[str, dict] = {}
_feature_lock = threading.Lock()


def load_feature_cache() -> None:
    global _feature_cache
    if not os.path.exists(FEATURE_CACHE_PATH):
        return
    with open(FEATURE_CACHE_PATH) as f:
        data = json.load(f)
    if data.get("version") != FEATURE_CACHE_VERSION:
        print(f"Ignoring feature cache version {data.get('version')} (want {FEATURE_CACHE_VERSION}).")
        return
    _feature_cache = data.get("assets", {})
    print(f"Loaded feature cache: {len(_feature_cache)} assets.")


def save_feature_cache() -> None:
    """Atomically persist the per-asset feature cache."""
    tmp = FEATURE_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": FEATURE_CACHE_VERSION, "assets": _feature_cache}, f)
    os.replace(tmp, FEATURE_CACHE_PATH)


load_checkpoint()
load_feature_cache()


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


def _analyze_face_landmarks(lms):
    """
    Given MediaPipe face landmarks, compute:
      - eyes_open in [0, 1] (continuous) from eye aspect ratio
      - smile in [0, 1] (continuous) from mouth width/height relative to face width
      - facing in [0, 1] (continuous) from vertical symmetry: nose tip vs face center
      - expression score in [0, 1] (eyes + smile + facing), used for primary and family.
    """
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

    # Expression: eyes + smile + facing (primary and family both use this).
    expr_score = 0.5 * eyes_open + 0.2 * smile + 0.3 * facing
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
        _faces_cache[asset_id] = []
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
                if pid in FAMILY_FACES:
                    family_hits.append(pid)
    if face_ids or person_ids:
        print(
            f"[immich_faces] asset={asset_id} faces={len(data)} "
            f"face_ids={face_ids} person_ids={person_ids} "
            f"family_hits={list(sorted(set(family_hits)))}"
        )
    return data


def _extract_face_bboxes_from_asset(asset):
    """
    Extract face bounding boxes for an Immich asset using /api/faces.
    Returns list of (x1, y1, x2, y2, imageWidth, imageHeight, face_type),
    where face_type is one of: \"primary\", \"family\", \"other\".
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
        # Classify face type: primary (search keys) > family (const) > other.
        face_type = "other"
        if pid is not None:
            if pid in FACE_IDS:
                face_type = "primary"
            elif pid in FAMILY_FACES:
                face_type = "family"
        boxes.append((float(x1), float(y1), float(x2), float(y2), float(iw), float(ih), face_type))
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


def _expression_from_mediapipe(img, asset):
    """
    Compute an expression score for an image using MediaPipe Tasks FaceLandmarker.
    If Immich provides face bounding boxes, run the landmarker on each face crop
    (using those boxes) and aggregate; otherwise, fall back to running on the full image.
    Returns (expression_score, summary_dict).
    """
    landmarker = _get_face_landmarker()

    img_rgb = np.array(img.convert("RGB"))
    # Use same dimension convention as debug: (width, height) from PIL so crop math matches.
    img_width, img_height = img.size

    # Face boxes from Immich, annotated with face_type (\"primary\" | \"family\" | \"other\").
    boxes = _extract_face_bboxes_from_asset(asset)
    primary_exprs: list[float] = []
    primary_eyes: list[float] = []
    primary_smiles: list[float] = []
    primary_facing: list[float] = []
    family_exprs: list[float] = []
    family_eyes: list[float] = []
    family_smiles: list[float] = []
    family_facing: list[float] = []
    other_exprs: list[float] = []
    other_eyes: list[float] = []
    other_smiles: list[float] = []
    face_debug_info: list[str] = []

    # Helper to run landmarker on a given RGB crop and accumulate stats.
    def process_crop(crop_rgb: np.ndarray, face_type: str, bbox_desc: str):
        nonlocal primary_exprs, primary_eyes, primary_smiles, primary_facing
        nonlocal family_exprs, family_eyes, family_smiles, family_facing
        nonlocal other_exprs, other_eyes, other_smiles, face_debug_info
        if crop_rgb.size == 0:
            return
        mp_face = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        result = landmarker.detect(mp_face)
        lms_list = result.face_landmarks or []
        print(f"  -> {bbox_desc} {len(lms_list)} faces detected")
        if not lms_list:
            return
        for li, lms in enumerate(lms_list):
            eyes_open, smile, facing, expr = _analyze_face_landmarks(lms)
            info = (
                f"{bbox_desc} face#{li}: type={face_type} "
                f"eyes={eyes_open:.2f} smile={smile:.2f} facing={facing:.2f} expr={expr:.3f}"
            )
            face_debug_info.append(info)
            if face_type == "primary":
                primary_exprs.append(expr)
                primary_eyes.append(eyes_open)
                primary_smiles.append(smile)
                primary_facing.append(facing)
            elif face_type == "family":
                family_exprs.append(expr)
                family_eyes.append(eyes_open)
                family_smiles.append(smile)
                family_facing.append(facing)
            else:
                other_exprs.append(expr)
                other_eyes.append(eyes_open)
                other_smiles.append(smile)

    if boxes:
        # Use Immich's face boxes to crop and run the landmarker per face (same crop logic as debug).
        for bi, (x1, y1, x2, y2, iw, ih, face_type) in enumerate(boxes):
            rect = _face_box_to_expanded_crop(x1, y1, x2, y2, iw, ih, img_width, img_height)
            if rect is None:
                continue
            ex1, ey1, ex2, ey2 = rect
            # Contiguous copy so MediaPipe gets a valid buffer (slicing can yield non-contiguous view).
            crop = np.ascontiguousarray(img_rgb[ey1:ey2, ex1:ex2, :])
            bbox_desc = f"[box{bi} type={face_type}]"
            process_crop(crop, face_type, bbox_desc)
    else:
        # Fallback: run on the full image once, treat all faces as "other".
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb))
        result = landmarker.detect(mp_image)
        faces = result.face_landmarks or []
        if not faces:
            print(f"  -> No faces detected in {asset.get('id')}")
            return 0.0, {}
        for li, lms in enumerate(faces):
            eyes_open, smile, facing, expr = _analyze_face_landmarks(lms)
            info = (
                f"[full] face#{li}: fam=False "
                f"eyes={eyes_open:.2f} smile={smile:.2f} facing={facing:.2f} expr={expr:.3f}"
            )
            face_debug_info.append(info)
            other_exprs.append(expr)
            other_eyes.append(eyes_open)
            other_smiles.append(smile)

    # Aggregate expression:
    # - Primary faces (FACE_IDS) drive the main expression score if present.
    # - Otherwise, family faces drive expression.
    # - Other faces are ignored for expression (expr=0 if only others).
    if primary_exprs:
        expr = float(np.mean(primary_exprs))
    elif family_exprs:
        expr = float(np.mean(family_exprs))
    elif other_exprs:
        # Only non-family, non-primary faces -> treat as no reliable expression signal.
        expr = 0.0
    else:
        return 0.0, {}

    # Family boost: continuous eyes, smile, facing per face; facing has more weight.
    # Only count face if facing >= FAMILY_FACING_MIN; discount each face's contribution by 0.5.
    if family_exprs:
        fam_eyes = float(np.mean(family_eyes)) if family_eyes else 0.0
        fam_smile = float(np.mean(family_smiles)) if family_smiles else 0.0
        fam_count = len(family_exprs)
        family_boost = 0.0
        for e, s, f in zip(family_eyes, family_smiles, family_facing):
            if f < FAMILY_FACING_MIN:
                continue
            raw = (
                FAMILY_BOOST_WEIGHT_EYES * e
                + FAMILY_BOOST_WEIGHT_SMILE * s
                + FAMILY_BOOST_WEIGHT_FACING * f
            )
            family_boost += FAMILY_BOOST_DISCOUNT * raw
        family_contributing_count = sum(
            1 for f in family_facing if f >= FAMILY_FACING_MIN
        )
    else:
        fam_eyes = 0.0
        fam_smile = 0.0
        fam_count = 0
        family_boost = 0.0
        family_contributing_count = 0

    other_count = len(other_exprs)
    primary_count = len(primary_exprs)

    # Summary fractions: show the stats for whoever drove expr (primary or family).
    if primary_exprs:
        eyes_open_frac = float(np.mean(primary_eyes)) if primary_eyes else 0.0
        smile_frac = float(np.mean(primary_smiles)) if primary_smiles else 0.0
        facing_frac = float(np.mean(primary_facing)) if primary_facing else 0.0
    elif family_exprs:
        eyes_open_frac = fam_eyes
        smile_frac = fam_smile
        facing_frac = float(np.mean(family_facing)) if family_facing else 0.0
    else:
        eyes_open_frac = 0.0
        smile_frac = 0.0
        facing_frac = 0.0

    summary = {
        "eyes_open_fraction": eyes_open_frac * 100.0,
        "smile_fraction": smile_frac * 100.0,
        "facing_fraction": facing_frac * 100.0,
        "primary_faces": primary_count,
        "family_faces": fam_count,
        "family_boost": family_boost,
        "family_contributing_count": family_contributing_count,
        "other_faces": other_count,
    }
    if face_debug_info:
        print(
            f"    [faces_detail] asset={asset.get('id')} "
            f"details=" + " | ".join(face_debug_info)
        )
    print(
        f"    [expr_fam] total_faces={fam_count + other_count} fam={fam_count} other={other_count} "
        f"expr={expr:.3f} smile={smile_frac:.2f} eyes={eyes_open_frac:.2f}"
    )
    return expr, summary


@app.route("/debug/crops/<asset_id>")
def debug_crops(asset_id):
    """
    Debug page: show all Immich face crops for a given asset so cropping
    can be visually validated.
    """
    img = download_thumbnail(asset_id)
    if img is None:
        return f"<h3>No thumbnail available for asset {asset_id}</h3>", 404

    w, h = img.size
    landmarker = _get_face_landmarker()
    boxes = _extract_face_bboxes_from_asset({"id": asset_id})
    if not boxes:
        return f"<h3>No Immich faces for asset {asset_id}</h3>", 200

    rows = []
    for idx, (x1, y1, x2, y2, iw, ih, face_type) in enumerate(boxes):
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
            eyes_open, smile, facing, expr = _analyze_face_landmarks(lms)
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
            f"<div>#{idx} type={face_type} "
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

def evaluate_and_cluster(assets, device, clip_model, clip_processor):
    """
    Evaluates assets with PyIQA (quality) and MediaPipe (expression), clusters for variety.
    Returns the top scoring assets per cluster.
    """
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
            # Cache hit: reuse the stored quality/expression/CLIP features.
            raw_q = cached["q"]
            expr = cached["e"]
            summary = cached.get("sum") or {}
            feats = np.frombuffer(base64.b64decode(cached["clip"]), dtype=np.float32)
            print(f"    [cache] asset={asset_id} expr={float(expr):.3f} (reused)")
        else:
            img = download_thumbnail(asset_id)
            if img is None:
                continue

            # 1. Image quality scores via MUSIQ (PyIQA). Higher = better quality.
            img_t = _pil_to_tensor_01(img)  # (1,3,H,W), CPU tensor
            q_tensor = _musiq_metric(img_t)
            raw_q = float(q_tensor.item())
            if not np.isfinite(raw_q):
                raw_q = QUALITY_MEAN

            # 2. Expression via MediaPipe Face Landmarker.
            expr, summary = _expression_from_mediapipe(img, asset)
            print(f"    [expr] asset={asset_id} expr={expr:.3f} summary={summary}")

            # 3. CLIP embedding for clustering.
            with torch.no_grad():
                clip_inputs = clip_processor(images=img, return_tensors="pt").to(device)
                feats = clip_model.get_image_features(**clip_inputs)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            feats = feats.squeeze(0).cpu().numpy().astype(np.float32)
            with _feature_lock:
                _feature_cache[asset_id] = {
                    "q": raw_q,
                    "e": expr,
                    "sum": summary or {},
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

    # 3. Normalize and combine using global params (so scales are comparable across days)
    def _norm_global(scores, mean, std):
        s = np.array(scores, dtype=np.float64)
        if std <= 0:
            return np.zeros_like(s)
        return (s - mean) / std
    q_norm = _norm_global(quality_scores, QUALITY_MEAN, QUALITY_STD)
    e_norm = _norm_global(expression_scores, EXPRESSION_MEAN, EXPRESSION_STD)
    for i in range(len(valid_assets)):
        asset = valid_assets[i]
        person_hits = asset.get("_person_match_count", 1)
        person_boost = PERSON_WEIGHT * (person_hits - 1)

        # Family boost: continuous eyes/smile/facing (precomputed in expression summary).
        summary = asset.get("_expression_emotions") or {}
        smile_boost = float(summary.get("family_boost", 0.0))
        total = float(
            QUALITY_WEIGHT * q_norm[i]
            + EXPRESSION_WEIGHT * e_norm[i]
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

    # 4. Cluster for variety using CLIP image embeddings (semantic diversity)
    X = np.stack(clip_feats, axis=0).astype(np.float32)

    # 4. Threshold-based clustering in CLIP space:
    #    - Iterate assets in descending total score order.
    #    - For each, assign to the first cluster whose centroid has cosine
    #      similarity >= CLIP_CLUSTER_SIM_THRESHOLD; otherwise start a new cluster.
    order = sorted(range(len(valid_assets)), key=lambda i: -valid_assets[i]["_total_score"])
    centroids: list[np.ndarray] = []
    cluster_ids = [-1] * len(valid_assets)

    for idx in order:
        x = X[idx]
        if not centroids:
            centroids.append(x.copy())
            cluster_ids[idx] = 0
            continue
        sims = [float(np.dot(x, c)) for c in centroids]
        best_id = int(np.argmax(sims))
        if sims[best_id] >= CLIP_CLUSTER_SIM_THRESHOLD:
            cluster_ids[idx] = best_id
            # Update centroid with simple running average
            centroids[best_id] = (centroids[best_id] + x) / 2.0
        else:
            new_id = len(centroids)
            centroids.append(x.copy())
            cluster_ids[idx] = new_id

    n_clusters = len(centroids)

    # Optionally recluster with KMeans if too many clusters; cap at 20 per day.
    if n_clusters > 20:
        from sklearn.cluster import KMeans

        k = min(20, len(valid_assets))
        print(f"  -> Reclustering {len(valid_assets)} assets into {k} clusters via KMeans (was {n_clusters}).")
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)
        cluster_ids = [int(l) for l in labels]
        n_clusters = k

    # 5. Cluster-top boost: for each cluster, give the top-1 asset an extra bonus
    #    that grows with cluster size but saturates.
    clusters: dict[int, list[int]] = {}
    for i, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(i)

    def _cluster_top_boost(size: int) -> float:
        capped = min(max(size, 1), CLUSTER_TOP_MAX_SIZE)
        # Map size 1..CLUSTER_TOP_MAX_SIZE -> (0..1] then scale by CLUSTER_TOP_WEIGHT.
        return CLUSTER_TOP_WEIGHT * (np.log1p(capped) / np.log1p(CLUSTER_TOP_MAX_SIZE))

    # Initialize final scores and cluster sizes
    for cid, idxs in clusters.items():
        size = len(idxs)
        for idx in idxs:
            valid_assets[idx]["_cluster_id"] = cid
            valid_assets[idx]["_cluster_size"] = size
            valid_assets[idx]["_final_score"] = valid_assets[idx]["_total_score"]

        # Boost the top-1 asset in this cluster
        if idxs:
            top_idx = max(idxs, key=lambda i: valid_assets[i]["_total_score"])
            boost = _cluster_top_boost(size)
            valid_assets[top_idx]["_final_score"] = valid_assets[top_idx]["_total_score"] + boost
            valid_assets[top_idx]["_cluster_top_boost"] = boost

    # Sort by cluster top score descending for cluster view / ordering.
    cluster_top_score: dict[int, float] = {}
    for cid, idxs in clusters.items():
        if not idxs:
            continue
        top_idx = max(idxs, key=lambda i: valid_assets[i]["_final_score"])
        cluster_top_score[cid] = valid_assets[top_idx]["_final_score"]

    # Sort by (cluster_top_score desc, then per-asset final score desc)
    valid_assets.sort(
        key=lambda a: (
            -cluster_top_score.get(a.get("_cluster_id", -1), float("-inf")),
            -a.get("_final_score", a.get("_total_score", 0.0)),
        )
    )
    scores = [a.get("_final_score", a["_total_score"]) for a in valid_assets]
    print(f"  -> Score range (final): min={min(scores):.3f} max={max(scores):.3f} mean={sum(scores)/len(scores):.3f}")
    print(f"  -> Total assets this day: {len(valid_assets)} (clusters={n_clusters}).")
    return valid_assets

# ==========================================
# 3. MAIN WORKFLOW
# ==========================================

def load_models():
    """Load CLIP model once for embeddings; return (device, clip_model, clip_processor)."""
    global _models_cache
    if _models_cache is not None:
        return _models_cache
    print("Loading CLIP embedding model (this may take a moment)...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _models_cache = (device, clip_model, clip_processor)
    print(f"Models loaded on {device}.")
    return _models_cache


def recompute_default_selection():
    """
    Recompute default selected_asset_ids from curated_assets_by_date
    using the rule:
      - select the top photo in each cluster whose final score
        >= DEFAULT_MIN_FINAL_SCORE
      - AND always select the top photo from at least the top
        DEFAULT_MIN_TOP_CLUSTERS clusters (by best score) for that day.
    """
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
            cid for cid, _score, _a in cluster_items[:DEFAULT_MIN_TOP_CLUSTERS]
        }
        for cid, score, a in cluster_items:
            if score >= DEFAULT_MIN_FINAL_SCORE or cid in top_cluster_ids:
                if "id" in a:
                    selected.add(a["id"])
    selected_asset_ids = selected


def run_curation_for_date_range(start_date, end_date, skip_dates_already_in_album=False):
    """
    Run search + scoring + clustering for each day in [start_date, end_date]
    and populate curated_assets_by_date. Call this from /process when user submits dates.
    Updates CURATION_STATE for the /progress endpoint and checkpoints to disk per day.
    """
    global curated_assets_by_date, selected_asset_ids
    days_total = (end_date - start_date).days + 1
    CURATION_CANCEL.clear()
    with CURATION_LOCK:
        CURATION_STATE.update(
            state="running", days_total=days_total, days_done=0, current_date="",
            assets_done=0, assets_total=0, started_at=time.time(),
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
            daily_assets = search_assets_for_date(current, FACE_IDS)
            if daily_assets:
                with CURATION_LOCK:
                    CURATION_STATE["assets_total"] = len(daily_assets)
                ranked = evaluate_and_cluster(daily_assets, device, clip_model, clip_processor)
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
                        save_checkpoint()
                        save_feature_cache()
            with CURATION_LOCK:
                CURATION_STATE["days_done"] += 1
                CURATION_STATE["assets_done"] = CURATION_STATE["assets_total"]
            current += datetime.timedelta(days=1)
        recompute_default_selection()
        save_checkpoint()
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
        raise
    with CURATION_LOCK:
        CURATION_STATE["state"] = "cancelled" if CURATION_CANCEL.is_set() else "done"
        CURATION_STATE["finished_at"] = time.time()


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


@app.route('/process', methods=['POST'])
def process():
    """Start curation for the submitted date range in a background thread."""
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
    threading.Thread(
        target=run_curation_for_date_range,
        args=(start_date, end_date),
        kwargs={"skip_dates_already_in_album": skip_in_album},
        daemon=True,
    ).start()
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
    with CURATION_LOCK:
        was_running = CURATION_STATE["state"] == "running"
    return jsonify({"ok": was_running})

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
        for _date, tops in top_by_date.items():
            for a in tops:
                score = a.get("_final_score", a.get("_total_score", 0.0))
                if score >= DEFAULT_MIN_FINAL_SCORE and "id" in a:
                    selected_ids_for_top.add(a["id"])
    def _card_html(asset, cluster_rank=None, show_badges=False):
        return _render_photo_card(asset, build_score_line, cluster_rank, show_badges, selected_ids_for_top)

    return render_template(
        "top.html",
        data=top_by_date,
        card_html=_card_html,
        active_page="top",
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
    args = parser.parse_args()

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

    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True, use_reloader=True)