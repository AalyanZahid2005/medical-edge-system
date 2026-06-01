from pathlib import Path
import json
import shutil
import hashlib
import datetime

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = CONFIG_DIR / "model_registry.json"

SUPPORTED_MODEL_EXTS = {".onnx", ".engine", ".pt", ".pth", ".tflite"}


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _slug(value):
    s = str(value or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in [" ", "-", "_", "/", "\\"]:
            out.append("_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or "unknown"


def _rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_registry():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not REGISTRY_PATH.exists():
        data = {
            "schema_version": 1,
            "created_at": _now(),
            "updated_at": _now(),
            "models": {},
            "history": []
        }
        REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return REGISTRY_PATH


def load_registry():
    ensure_registry()
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {
            "schema_version": 1,
            "created_at": _now(),
            "updated_at": _now(),
            "models": {},
            "history": []
        }
        save_registry(data)
        return data


def save_registry(data):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _now()
    REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return REGISTRY_PATH


def import_model(modality, version, model_file, labels_file=None, preprocessing_file=None, metrics_file=None, notes=""):
    modality_key = _slug(modality)
    version_key = _slug(version)

    model_file = Path(model_file)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    if model_file.suffix.lower() not in SUPPORTED_MODEL_EXTS:
        raise ValueError(f"Unsupported model format: {model_file.suffix}. Use ONNX, TensorRT engine, PT/PTH, or TFLite.")

    target_dir = MODELS_DIR / modality_key / version_key
    target_dir.mkdir(parents=True, exist_ok=True)

    model_dst = target_dir / f"model{model_file.suffix.lower()}"
    shutil.copy2(model_file, model_dst)

    labels_dst = ""
    preprocessing_dst = ""
    metrics_dst = ""

    if labels_file and Path(labels_file).exists():
        labels_dst_path = target_dir / "labels.json"
        shutil.copy2(labels_file, labels_dst_path)
        labels_dst = _rel(labels_dst_path)

    if preprocessing_file and Path(preprocessing_file).exists():
        preprocessing_dst_path = target_dir / "preprocessing.json"
        shutil.copy2(preprocessing_file, preprocessing_dst_path)
        preprocessing_dst = _rel(preprocessing_dst_path)

    if metrics_file and Path(metrics_file).exists():
        metrics_dst_path = target_dir / "metrics.json"
        shutil.copy2(metrics_file, metrics_dst_path)
        metrics_dst = _rel(metrics_dst_path)

    metadata = {
        "modality": modality_key,
        "display_modality": str(modality),
        "version": version_key,
        "display_version": str(version),
        "model_path": _rel(model_dst),
        "model_format": model_dst.suffix.lower().replace(".", ""),
        "sha256": _sha256(model_dst),
        "labels_path": labels_dst,
        "preprocessing_path": preprocessing_dst,
        "metrics_path": metrics_dst,
        "status": "available",
        "notes": str(notes or ""),
        "imported_at": _now(),
        "updated_at": _now(),
    }

    metadata_path = target_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    reg = load_registry()

    if modality_key not in reg["models"]:
        reg["models"][modality_key] = {
            "display_modality": str(modality),
            "active_version": "",
            "versions": {}
        }

    previous_active = reg["models"][modality_key].get("active_version", "")

    reg["models"][modality_key]["display_modality"] = str(modality)
    reg["models"][modality_key]["versions"][version_key] = metadata

    if not previous_active:
        reg["models"][modality_key]["active_version"] = version_key
        reg["history"].append({
            "time": _now(),
            "action": "auto_activate_first_model",
            "modality": modality_key,
            "from": "",
            "to": version_key
        })

    reg["history"].append({
        "time": _now(),
        "action": "import_model",
        "modality": modality_key,
        "version": version_key,
        "model_path": _rel(model_dst)
    })

    save_registry(reg)
    return metadata


def validate_model(modality, version):
    reg = load_registry()
    modality_key = _slug(modality)
    version_key = _slug(version)

    if modality_key not in reg["models"]:
        return False, [f"Modality not found: {modality_key}"]

    versions = reg["models"][modality_key].get("versions", {})
    if version_key not in versions:
        return False, [f"Version not found: {version_key}"]

    meta = versions[version_key]
    messages = []
    ok = True

    model_path = ROOT / meta.get("model_path", "")
    if not model_path.exists():
        ok = False
        messages.append(f"Missing model file: {model_path}")
    else:
        messages.append(f"Model file exists: {model_path.name}")

    if model_path.suffix.lower() not in SUPPORTED_MODEL_EXTS:
        ok = False
        messages.append(f"Unsupported model extension: {model_path.suffix}")
    else:
        messages.append(f"Model format OK: {model_path.suffix}")

    labels_path = meta.get("labels_path", "")
    if labels_path:
        lp = ROOT / labels_path
        if not lp.exists():
            ok = False
            messages.append("labels.json missing")
        else:
            try:
                labels = json.loads(lp.read_text(encoding="utf-8"))
                if isinstance(labels, (list, dict)):
                    messages.append("labels.json OK")
                else:
                    ok = False
                    messages.append("labels.json must be list or dict")
            except Exception as e:
                ok = False
                messages.append(f"labels.json invalid: {e}")
    else:
        messages.append("labels.json not provided")

    preprocessing_path = meta.get("preprocessing_path", "")
    if preprocessing_path:
        pp = ROOT / preprocessing_path
        if pp.exists():
            messages.append("preprocessing.json OK")
        else:
            ok = False
            messages.append("preprocessing.json missing")
    else:
        messages.append("preprocessing.json not provided")

    return ok, messages


def activate_model(modality, version):
    reg = load_registry()
    modality_key = _slug(modality)
    version_key = _slug(version)

    if modality_key not in reg["models"]:
        raise KeyError(f"Modality not found: {modality_key}")

    if version_key not in reg["models"][modality_key].get("versions", {}):
        raise KeyError(f"Version not found: {version_key}")

    previous = reg["models"][modality_key].get("active_version", "")
    reg["models"][modality_key]["active_version"] = version_key

    reg["history"].append({
        "time": _now(),
        "action": "activate_model",
        "modality": modality_key,
        "from": previous,
        "to": version_key
    })

    save_registry(reg)
    return True


def rollback_model(modality):
    reg = load_registry()
    modality_key = _slug(modality)

    if modality_key not in reg["models"]:
        raise KeyError(f"Modality not found: {modality_key}")

    active = reg["models"][modality_key].get("active_version", "")

    previous = ""
    for h in reversed(reg.get("history", [])):
        if h.get("modality") == modality_key and h.get("action") == "activate_model":
            old = h.get("from", "")
            if old and old != active:
                previous = old
                break

    if not previous:
        raise RuntimeError("No previous active model found for rollback.")

    reg["models"][modality_key]["active_version"] = previous
    reg["history"].append({
        "time": _now(),
        "action": "rollback_model",
        "modality": modality_key,
        "from": active,
        "to": previous
    })

    save_registry(reg)
    return previous


def list_models():
    reg = load_registry()
    rows = []

    for modality_key, entry in sorted(reg.get("models", {}).items()):
        active = entry.get("active_version", "")
        versions = entry.get("versions", {})

        for version_key, meta in sorted(versions.items()):
            rows.append({
                "modality": modality_key,
                "display_modality": entry.get("display_modality", modality_key),
                "version": version_key,
                "display_version": meta.get("display_version", version_key),
                "active": version_key == active,
                "status": meta.get("status", "available"),
                "model_path": meta.get("model_path", ""),
                "model_format": meta.get("model_format", ""),
                "labels_path": meta.get("labels_path", ""),
                "updated_at": meta.get("updated_at", meta.get("imported_at", "")),
            })

    return rows


def get_active_model(modality):
    reg = load_registry()
    modality_key = _slug(modality)

    entry = reg.get("models", {}).get(modality_key)
    if not entry:
        return None

    active = entry.get("active_version", "")
    if not active:
        return None

    return entry.get("versions", {}).get(active)

# ============================================================
# AALYAN PATCH: Detect currently deployed Jetson/runtime models
# Registers existing .engine/.onnx/.pt/.pth/.tflite files into registry
# so GUI shows active available models already present in system.
# ============================================================
def _guess_runtime_modality(path):
    s = str(path).lower()

    if "ood" in s and ("mri" in s or "brain" in s):
        return "brain_mri_ood_guard", "Brain MRI OOD Guard", "ood_guard"

    if "brain" in s or "mri" in s:
        return "brain_mri", "Brain MRI", "classification"

    if "chest" in s or "xray" in s or "x_ray" in s or "pneumonia" in s:
        return "chest_xray", "Chest X-Ray", "classification"

    if "skin" in s or "derma" in s:
        return "skin", "Skin", "classification"

    if "retina" in s or "retinal" in s:
        return "retinal", "Retinal", "classification"

    if "ecg" in s:
        return "ecg", "ECG", "classification"

    if "bone" in s:
        return "bone", "Bone", "classification"

    return "unknown", "Unknown", "classification"


def _is_runtime_model_file(path):
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_MODEL_EXTS:
        return False

    bad_parts = {
        "venv",
        ".git",
        "__pycache__",
        "reports",
        "data",
        "medai_mobile_unused_backup",
        "backup",
        "backups",
    }

    parts = {p.lower() for p in path.parts}
    if parts.intersection(bad_parts):
        return False

    return True


def discover_runtime_model_files():
    """
    Finds model files already present in the project.
    Priority is given to jetson_deployment_package because that is the current runtime package.
    """
    roots = [
        ROOT / "jetson_deployment_package",
        ROOT / "models",
        ROOT / "src" / "pipeline",
        ROOT,
    ]

    found = []
    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for f in root.rglob("*"):
            try:
                if not f.is_file():
                    continue

                if not _is_runtime_model_file(f):
                    continue

                rp = str(f.resolve())
                if rp in seen:
                    continue

                seen.add(rp)
                found.append(f)
            except Exception:
                pass

    return found


def register_current_runtime_models():
    """
    Idempotently registers current deployed model files into model_registry.json.
    This makes the Models tab show the models already available on Jetson/project.
    """
    ensure_registry()
    reg = load_registry()

    files = discover_runtime_model_files()
    registered = []

    for model_file in files:
        modality_key, display_modality, task = _guess_runtime_modality(model_file)

        # Create a stable version name from real file name.
        # Example: brain_mri_fp16.engine -> current_brain_mri_fp16_engine
        version_key = _slug("current_" + model_file.stem + "_" + model_file.suffix.replace(".", ""))

        model_rel = _rel(model_file)
        model_hash = _sha256(model_file)

        meta = {
            "modality": modality_key,
            "display_modality": display_modality,
            "version": version_key,
            "display_version": version_key,
            "task": task,
            "model_path": model_rel,
            "model_format": model_file.suffix.lower().replace(".", ""),
            "sha256": model_hash,
            "labels_path": "",
            "preprocessing_path": "",
            "metrics_path": "",
            "status": "runtime_detected",
            "notes": "Automatically detected from current Jetson/project runtime files.",
            "imported_at": _now(),
            "updated_at": _now(),
        }

        if modality_key not in reg["models"]:
            reg["models"][modality_key] = {
                "display_modality": display_modality,
                "active_version": "",
                "versions": {}
            }

        reg["models"][modality_key]["display_modality"] = display_modality
        reg["models"][modality_key]["versions"][version_key] = meta

        # Mark current runtime model active if:
        # - no active model exists, OR
        # - it is inside jetson_deployment_package, OR
        # - it is a TensorRT engine
        active_now = reg["models"][modality_key].get("active_version", "")
        prefer_runtime = "jetson_deployment_package" in model_rel or model_file.suffix.lower() == ".engine"

        if not active_now or prefer_runtime:
            reg["models"][modality_key]["active_version"] = version_key

        reg["history"].append({
            "time": _now(),
            "action": "runtime_detect_model",
            "modality": modality_key,
            "version": version_key,
            "model_path": model_rel
        })

        registered.append(meta)

    save_registry(reg)
    return registered

# ============================================================
# END AALYAN RUNTIME MODEL DETECTION
# ============================================================

# ============================================================
# AALYAN PATCH: Six Modalities + Tier/OOD Guard Registry View
# Shows all intended modalities even when physical model file
# is not loaded yet. Removes dummy/unknown demo model rows.
# ============================================================

EXPECTED_MODALITY_SLOTS = [
    {
        "key": "brain_mri",
        "display": "Brain MRI",
        "task": "classification",
        "aliases": ["brain", "mri", "brain_mri"],
        "labels": ["Glioma", "Meningioma", "Pituitary", "No Tumor"],
    },
    {
        "key": "chest_xray",
        "display": "Chest X-Ray",
        "task": "classification",
        "aliases": ["chest", "xray", "x_ray", "x-ray", "pneumonia"],
        "labels": ["Normal", "Pneumonia"],
    },
    {
        "key": "retinal",
        "display": "Retinal",
        "task": "classification",
        "aliases": ["retina", "retinal", "fundus"],
        "labels": ["Normal", "Abnormal"],
    },
    {
        "key": "ecg",
        "display": "ECG",
        "task": "signal_classification",
        "aliases": ["ecg", "ekg"],
        "labels": ["Normal", "Abnormal"],
    },
    {
        "key": "skin",
        "display": "Skin",
        "task": "classification",
        "aliases": ["skin", "derma", "dermatology"],
        "labels": ["Normal", "Abnormal"],
    },
    {
        "key": "bone",
        "display": "Bone",
        "task": "classification",
        "aliases": ["bone", "fracture", "xray_bone"],
        "labels": ["Normal", "Abnormal"],
    },
]


def _aalyan_slot_labels_path(modality_key, labels):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    slot_dir = CONFIG_DIR / "model_slots"
    slot_dir.mkdir(parents=True, exist_ok=True)

    path = slot_dir / f"{modality_key}_labels.json"
    if not path.exists():
        path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    return _rel(path)


def _aalyan_cleanup_dummy_unknown_rows(reg):
    """
    Remove demo/dummy rows like dummy_dual_output.onnx from registry.
    """
    models = reg.get("models", {})

    remove_modalities = []

    for modality_key, entry in list(models.items()):
        versions = entry.get("versions", {})
        new_versions = {}

        for version_key, meta in versions.items():
            joined = " ".join([
                str(modality_key),
                str(version_key),
                str(meta.get("model_path", "")),
                str(meta.get("display_modality", "")),
                str(meta.get("display_version", "")),
            ]).lower()

            if "dummy" in joined or modality_key == "unknown":
                continue

            new_versions[version_key] = meta

        entry["versions"] = new_versions

        if not new_versions and modality_key == "unknown":
            remove_modalities.append(modality_key)

    for k in remove_modalities:
        models.pop(k, None)

    reg["models"] = models
    return reg


def _aalyan_model_file_matches(path, aliases, want_ood=False):
    s = str(path).lower()

    if "dummy" in s:
        return False

    has_alias = any(a.lower() in s for a in aliases)

    if want_ood:
        return has_alias and ("ood" in s or "guard" in s or "anomaly" in s)

    return has_alias and not ("ood" in s or "guard" in s or "anomaly" in s)


def _aalyan_best_model_file_for_slot(files, aliases, want_ood=False):
    matches = []

    for f in files:
        if _aalyan_model_file_matches(f, aliases, want_ood=want_ood):
            matches.append(Path(f))

    if not matches:
        return None

    # Prefer TensorRT engine, then ONNX, then others.
    def score(path):
        s = str(path).lower()
        val = 0
        if "jetson_deployment_package" in s:
            val += 50
        if path.suffix.lower() == ".engine":
            val += 40
        elif path.suffix.lower() == ".onnx":
            val += 30
        elif path.suffix.lower() in [".pt", ".pth"]:
            val += 20
        return val

    matches.sort(key=score, reverse=True)
    return matches[0]


def _is_runtime_model_file(path):
    """
    Override previous detector:
    skip dummy/demo/backup and only keep real model-like files.
    """
    path = Path(path)

    if path.suffix.lower() not in SUPPORTED_MODEL_EXTS:
        return False

    s = str(path).lower()

    bad_terms = [
        "dummy",
        "demo",
        "unused_backup",
        ".git",
        "venv",
        "__pycache__",
        "reports",
        "data/camera_captures",
        "backup",
        "backups",
    ]

    if any(b in s for b in bad_terms):
        return False

    return True


def register_current_runtime_models():
    """
    Final registry builder:
    - detects actual runtime model files
    - removes dummy/unknown rows
    - always shows all 6 modality slots
    - always shows Tier/OOD Guard slot for each modality
    """
    ensure_registry()
    reg = load_registry()
    reg = _aalyan_cleanup_dummy_unknown_rows(reg)

    files = discover_runtime_model_files()
    registered = []

    for slot in EXPECTED_MODALITY_SLOTS:
        modality_key = slot["key"]
        display = slot["display"]
        aliases = slot["aliases"]
        labels = slot["labels"]
        task = slot["task"]

        if modality_key not in reg["models"]:
            reg["models"][modality_key] = {
                "display_modality": display,
                "active_version": "",
                "versions": {}
            }

        reg["models"][modality_key]["display_modality"] = display

        labels_path = _aalyan_slot_labels_path(modality_key, labels)

        # ----------------------------------------------------
        # Main classifier slot
        # ----------------------------------------------------
        main_file = _aalyan_best_model_file_for_slot(files, aliases, want_ood=False)
        main_version = "current_runtime_classifier"

        if main_file:
            main_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": main_version,
                "display_version": "Current Runtime Classifier",
                "task": task,
                "model_path": _rel(main_file),
                "model_format": main_file.suffix.lower().replace(".", ""),
                "sha256": _sha256(main_file),
                "labels_path": labels_path,
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "runtime_detected",
                "notes": "Current classifier model detected from Jetson/project runtime files.",
                "imported_at": _now(),
                "updated_at": _now(),
            }
            reg["models"][modality_key]["active_version"] = main_version
        else:
            main_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": main_version,
                "display_version": "Current Runtime Classifier",
                "task": task,
                "model_path": "Not loaded yet",
                "model_format": "-",
                "sha256": "",
                "labels_path": labels_path,
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "not_loaded_slot",
                "notes": "Slot reserved. Import/activate an improved model from GUI.",
                "imported_at": _now(),
                "updated_at": _now(),
            }

        reg["models"][modality_key]["versions"][main_version] = main_meta
        registered.append(main_meta)

        # ----------------------------------------------------
        # Tier / OOD Guard slot
        # ----------------------------------------------------
        ood_file = _aalyan_best_model_file_for_slot(files, aliases, want_ood=True)
        ood_version = "tier_ood_guard"

        if ood_file:
            ood_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": ood_version,
                "display_version": "Tier / OOD Guard",
                "task": "ood_guard_and_routing",
                "model_path": _rel(ood_file),
                "model_format": ood_file.suffix.lower().replace(".", ""),
                "sha256": _sha256(ood_file),
                "labels_path": "",
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "runtime_detected_ood_guard",
                "notes": "OOD / anomaly guard detected from runtime files.",
                "imported_at": _now(),
                "updated_at": _now(),
            }
        else:
            ood_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": ood_version,
                "display_version": "Tier / OOD Guard",
                "task": "ood_guard_and_routing",
                "model_path": "Not loaded yet",
                "model_format": "-",
                "sha256": "",
                "labels_path": "",
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "ood_guard_slot",
                "notes": "Tier/OOD guard slot reserved for this modality.",
                "imported_at": _now(),
                "updated_at": _now(),
            }

        reg["models"][modality_key]["versions"][ood_version] = ood_meta
        registered.append(ood_meta)

    reg["history"].append({
        "time": _now(),
        "action": "register_six_modalities_with_ood_guard",
        "count": len(registered)
    })

    save_registry(reg)
    return registered

# ============================================================
# END AALYAN PATCH
# ============================================================

# ============================================================
# AALYAN PATCH: Load active/runtime model files from connected Jetson over SSH
# This makes the laptop GUI Models tab show Jetson-side models, not only local files.
# ============================================================
def _aalyan_jetson_hosts():
    import os

    hosts = []

    env_hosts = os.environ.get("MEDAI_JETSON_HOSTS", "").strip()
    if env_hosts:
        hosts.extend([h.strip() for h in env_hosts.split(",") if h.strip()])

    cfg = CONFIG_DIR / "jetson_hosts.txt"
    if cfg.exists():
        for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                hosts.append(line)

    hosts.extend(["fyp", "jetson-nano@10.42.0.51", "jetson-nano@jetsonnano-desktop"])

    out = []
    seen = set()
    for h in hosts:
        if h not in seen:
            seen.add(h)
            out.append(h)

    return out


def discover_jetson_runtime_model_files():
    """
    SSH into Jetson and detect real runtime model files.
    Returns list of dicts:
    {
      remote: True,
      host: "fyp",
      path: "jetson_deployment_package/brain_mri_fp16.engine",
      sha256: "...",
      suffix: ".engine"
    }
    """
    import subprocess
    import json
    import shlex

    remote_cmd = r'''
cd ~/medical-edge-system 2>/dev/null || cd /home/jetson-nano/medical-edge-system 2>/dev/null || exit 2
python3 - <<'PYREMOTE'
from pathlib import Path
import hashlib, json

roots = [
    Path("jetson_deployment_package"),
    Path("models"),
    Path("src/pipeline"),
    Path("."),
]

exts = {".engine", ".onnx", ".pt", ".pth", ".tflite"}
bad_terms = [
    "dummy",
    "demo",
    "unused_backup",
    ".git",
    "venv",
    "__pycache__",
    "reports",
    "data/camera_captures",
    "backup",
    "backups",
]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
seen = set()

for root in roots:
    if not root.exists():
        continue

    for f in root.rglob("*"):
        try:
            if not f.is_file():
                continue

            s = str(f).lower()
            if f.suffix.lower() not in exts:
                continue

            if any(b in s for b in bad_terms):
                continue

            rp = str(f)
            if rp in seen:
                continue

            seen.add(rp)
            rows.append({
                "path": rp,
                "suffix": f.suffix.lower(),
                "sha256": sha256(f),
                "size_bytes": f.stat().st_size,
            })
        except Exception:
            pass

print(json.dumps(rows))
PYREMOTE
'''

    results = []

    for host in _aalyan_jetson_hosts():
        try:
            cmd = [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=5",
                host,
                f"bash -lc {shlex.quote(remote_cmd)}"
            ]

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20
            )

            if proc.returncode != 0:
                continue

            stdout = proc.stdout.strip()
            if not stdout:
                continue

            # Last line should be JSON.
            last = stdout.splitlines()[-1]
            rows = json.loads(last)

            for row in rows:
                row["remote"] = True
                row["host"] = host
                results.append(row)

            if results:
                # Stop at first working Jetson host.
                break

        except Exception:
            continue

    return results


def _aalyan_model_item_string(item):
    if isinstance(item, dict) and item.get("remote"):
        return item.get("path", "")
    return str(item)


def _aalyan_item_suffix(item):
    if isinstance(item, dict):
        return item.get("suffix", Path(item.get("path", "")).suffix.lower())
    return Path(item).suffix.lower()


def _aalyan_item_sha(item):
    if isinstance(item, dict):
        return item.get("sha256", "")
    return _sha256(item)


def _aalyan_item_rel_or_remote(item):
    if isinstance(item, dict) and item.get("remote"):
        return f"JETSON({item.get('host','')}): {item.get('path','')}"
    return _rel(item)


def _aalyan_best_model_item_for_slot(items, aliases, want_ood=False):
    matches = []

    for item in items:
        s = _aalyan_model_item_string(item).lower()

        if "dummy" in s:
            continue

        has_alias = any(a.lower() in s for a in aliases)

        if want_ood:
            ok = has_alias and ("ood" in s or "guard" in s or "anomaly" in s)
        else:
            ok = has_alias and not ("ood" in s or "guard" in s or "anomaly" in s)

        if ok:
            matches.append(item)

    if not matches:
        return None

    def score(item):
        s = _aalyan_model_item_string(item).lower()
        suf = _aalyan_item_suffix(item)

        val = 0

        if isinstance(item, dict) and item.get("remote"):
            val += 100

        if "jetson_deployment_package" in s:
            val += 60

        if suf == ".engine":
            val += 50
        elif suf == ".onnx":
            val += 40
        elif suf in [".pt", ".pth"]:
            val += 20
        elif suf == ".tflite":
            val += 10

        if "fp16" in s:
            val += 10
        if "int8" in s:
            val += 8

        return val

    matches.sort(key=score, reverse=True)
    return matches[0]


def register_current_runtime_models():
    """
    FINAL version:
    - scans connected Jetson through SSH
    - scans local project as fallback
    - always shows 6 modalities
    - always shows Tier/OOD Guard slot
    - marks Jetson files as jetson_runtime_detected
    """
    ensure_registry()
    reg = load_registry()
    reg = _aalyan_cleanup_dummy_unknown_rows(reg)

    local_files = discover_runtime_model_files()
    jetson_files = discover_jetson_runtime_model_files()

    # Prefer Jetson models first because Jetson is the actual running inference device.
    all_items = jetson_files + local_files

    registered = []

    for slot in EXPECTED_MODALITY_SLOTS:
        modality_key = slot["key"]
        display = slot["display"]
        aliases = slot["aliases"]
        labels = slot["labels"]
        task = slot["task"]

        if modality_key not in reg["models"]:
            reg["models"][modality_key] = {
                "display_modality": display,
                "active_version": "",
                "versions": {}
            }

        reg["models"][modality_key]["display_modality"] = display
        labels_path = _aalyan_slot_labels_path(modality_key, labels)

        # Main classifier
        main_item = _aalyan_best_model_item_for_slot(all_items, aliases, want_ood=False)
        main_version = "current_jetson_runtime_classifier" if isinstance(main_item, dict) else "current_runtime_classifier"

        if main_item:
            remote = isinstance(main_item, dict) and main_item.get("remote")
            main_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": main_version,
                "display_version": "Current Jetson Runtime Classifier" if remote else "Current Runtime Classifier",
                "task": task,
                "model_path": _aalyan_item_rel_or_remote(main_item),
                "model_format": _aalyan_item_suffix(main_item).replace(".", ""),
                "sha256": _aalyan_item_sha(main_item),
                "labels_path": labels_path,
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "jetson_runtime_detected" if remote else "runtime_detected",
                "notes": "Detected from connected Jetson runtime." if remote else "Detected from local runtime files.",
                "imported_at": _now(),
                "updated_at": _now(),
            }
            reg["models"][modality_key]["active_version"] = main_version
        else:
            main_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": "current_runtime_classifier",
                "display_version": "Current Runtime Classifier",
                "task": task,
                "model_path": "Not loaded on Jetson yet",
                "model_format": "-",
                "sha256": "",
                "labels_path": labels_path,
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "not_loaded_on_jetson",
                "notes": "Slot reserved. No classifier file detected on connected Jetson.",
                "imported_at": _now(),
                "updated_at": _now(),
            }

        reg["models"][modality_key]["versions"][main_meta["version"]] = main_meta
        registered.append(main_meta)

        # OOD / Tier guard
        ood_item = _aalyan_best_model_item_for_slot(all_items, aliases, want_ood=True)
        ood_version = "jetson_tier_ood_guard" if isinstance(ood_item, dict) else "tier_ood_guard"

        if ood_item:
            remote = isinstance(ood_item, dict) and ood_item.get("remote")
            ood_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": ood_version,
                "display_version": "Jetson Tier / OOD Guard" if remote else "Tier / OOD Guard",
                "task": "ood_guard_and_routing",
                "model_path": _aalyan_item_rel_or_remote(ood_item),
                "model_format": _aalyan_item_suffix(ood_item).replace(".", ""),
                "sha256": _aalyan_item_sha(ood_item),
                "labels_path": "",
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "jetson_runtime_detected_ood_guard" if remote else "runtime_detected_ood_guard",
                "notes": "Detected OOD/Tier guard from connected Jetson." if remote else "Detected OOD/Tier guard locally.",
                "imported_at": _now(),
                "updated_at": _now(),
            }
        else:
            ood_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": "tier_ood_guard",
                "display_version": "Tier / OOD Guard",
                "task": "ood_guard_and_routing",
                "model_path": "Not loaded on Jetson yet",
                "model_format": "-",
                "sha256": "",
                "labels_path": "",
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "ood_guard_not_loaded_on_jetson",
                "notes": "OOD/Tier guard slot reserved. No guard file detected on connected Jetson.",
                "imported_at": _now(),
                "updated_at": _now(),
            }

        reg["models"][modality_key]["versions"][ood_meta["version"]] = ood_meta
        registered.append(ood_meta)

    reg["history"].append({
        "time": _now(),
        "action": "register_jetson_models_with_six_modalities_and_ood",
        "jetson_files_detected": len(jetson_files),
        "local_files_detected": len(local_files),
        "rows": len(registered)
    })

    save_registry(reg)
    return registered

# ============================================================
# END AALYAN JETSON MODEL LOADER
# ============================================================

# ============================================================
# AALYAN FINAL CLEAN MODEL REGISTRY
# Keep only clean Jetson runtime rows in Models tab:
# 6 classifiers + 6 Tier/OOD guards.
# Removes old duplicate not_loaded_slot / ood_guard_slot rows
# when Jetson models are available.
# ============================================================

def _AALYAN_FINAL_expected_slots():
    return [
        {
            "key": "brain_mri",
            "display": "Brain MRI",
            "task": "classification",
            "aliases": ["brain", "mri", "brain_mri"],
            "labels": ["Glioma", "Meningioma", "Pituitary", "No Tumor"],
        },
        {
            "key": "chest_xray",
            "display": "Chest X-Ray",
            "task": "classification",
            "aliases": ["chest", "xray", "x_ray", "x-ray", "pneumonia"],
            "labels": ["Normal", "Pneumonia"],
        },
        {
            "key": "retinal",
            "display": "Retinal",
            "task": "classification",
            "aliases": ["retina", "retinal", "fundus", "eye", "eyes"],
            "labels": ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
        },
        {
            "key": "ecg",
            "display": "ECG",
            "task": "signal_classification",
            "aliases": ["ecg", "ekg"],
            "labels": ["Normal", "Abnormal"],
        },
        {
            "key": "skin",
            "display": "Skin",
            "task": "classification",
            "aliases": ["skin", "derma", "dermatology", "lesion"],
            "labels": ["NV", "MEL", "BCC", "AKIEC", "BKL", "DF", "VASC"],
        },
        {
            "key": "bone",
            "display": "Bone",
            "task": "classification",
            "aliases": ["bone", "fracture", "xray_bone"],
            "labels": ["Normal", "Fracture"],
        },
    ]


def _AALYAN_FINAL_prefer_model_item(items, aliases, want_ood=False):
    matches = []

    for item in items:
        s = _aalyan_model_item_string(item).lower()

        if "dummy" in s or "demo" in s:
            continue

        has_alias = any(a.lower() in s for a in aliases)

        if want_ood:
            ok = has_alias and ("ood" in s or "guard" in s or "anomaly" in s)
        else:
            ok = has_alias and not ("ood" in s or "guard" in s or "anomaly" in s)

        if ok:
            matches.append(item)

    if not matches:
        return None

    def score(item):
        s = _aalyan_model_item_string(item).lower()
        suf = _aalyan_item_suffix(item)
        val = 0

        # Jetson is source of truth.
        if isinstance(item, dict) and item.get("remote"):
            val += 1000

        if "jetson_deployment_package" in s:
            val += 500

        # Runtime preference.
        if suf == ".engine":
            val += 300
        elif suf == ".onnx":
            val += 200
        elif suf in [".pt", ".pth"]:
            val += 100

        # Prefer specific classifier names over generic production weights.
        if "classifier_int8" in s:
            val += 70
        if "fp16" in s:
            val += 65
        if "classifier.onnx" in s:
            val += 50

        # For bone final runtime, bone_fracture_fp16 is acceptable,
        # but if bone_classifier_int8 exists, prefer it for classifier slot.
        if "bone_classifier_int8" in s:
            val += 120
        if "bone_fracture_fp16" in s:
            val += 100

        # For retinal OOD, eyes_ood_guard is the real naming used in pipeline.
        if "eyes_ood_guard_int8" in s:
            val += 120
        if "eyes_ood_guard.onnx" in s:
            val += 90

        # Prefer INT8 OOD guard if present.
        if want_ood and "int8" in s:
            val += 80

        return val

    matches.sort(key=score, reverse=True)
    return matches[0]


def register_current_runtime_models():
    """
    Final clean version for GUI:
    - SSH scans Jetson
    - creates exactly 12 rows:
      6 Current Jetson Runtime Classifier rows
      6 Jetson Tier / OOD Guard rows
    - removes duplicate local placeholder rows
    """
    ensure_registry()

    old_reg = load_registry()
    old_history = old_reg.get("history", [])

    local_files = discover_runtime_model_files()
    jetson_files = discover_jetson_runtime_model_files()

    all_items = jetson_files + local_files

    clean_models = {}
    registered = []

    for slot in _AALYAN_FINAL_expected_slots():
        modality_key = slot["key"]
        display = slot["display"]
        task = slot["task"]
        aliases = slot["aliases"]
        labels = slot["labels"]

        labels_path = _aalyan_slot_labels_path(modality_key, labels)

        clean_models[modality_key] = {
            "display_modality": display,
            "active_version": "",
            "versions": {}
        }

        # Keep manually imported improved versions if they exist.
        old_versions = old_reg.get("models", {}).get(modality_key, {}).get("versions", {})
        for ver, meta in old_versions.items():
            status = str(meta.get("status", "")).lower()
            if status in {"available", "validated", "manually_imported"}:
                clean_models[modality_key]["versions"][ver] = meta

        # -----------------------------
        # Classifier row
        # -----------------------------
        main_item = _AALYAN_FINAL_prefer_model_item(all_items, aliases, want_ood=False)
        classifier_version = "current_jetson_runtime_classifier"

        if main_item:
            classifier_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": classifier_version,
                "display_version": "Current Jetson Runtime Classifier",
                "task": task,
                "model_path": _aalyan_item_rel_or_remote(main_item),
                "model_format": _aalyan_item_suffix(main_item).replace(".", ""),
                "sha256": _aalyan_item_sha(main_item),
                "labels_path": labels_path,
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "jetson_runtime_detected" if isinstance(main_item, dict) else "runtime_detected",
                "notes": "Detected from connected Jetson runtime package.",
                "imported_at": _now(),
                "updated_at": _now(),
            }
            clean_models[modality_key]["active_version"] = classifier_version
        else:
            classifier_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": classifier_version,
                "display_version": "Current Jetson Runtime Classifier",
                "task": task,
                "model_path": "Not loaded on Jetson",
                "model_format": "-",
                "sha256": "",
                "labels_path": labels_path,
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "not_loaded_on_jetson",
                "notes": "No classifier file detected on connected Jetson.",
                "imported_at": _now(),
                "updated_at": _now(),
            }

        clean_models[modality_key]["versions"][classifier_version] = classifier_meta
        registered.append(classifier_meta)

        # -----------------------------
        # OOD / Tier guard row
        # -----------------------------
        ood_item = _AALYAN_FINAL_prefer_model_item(all_items, aliases, want_ood=True)
        ood_version = "jetson_tier_ood_guard"

        if ood_item:
            ood_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": ood_version,
                "display_version": "Jetson Tier / OOD Guard",
                "task": "ood_guard_and_routing",
                "model_path": _aalyan_item_rel_or_remote(ood_item),
                "model_format": _aalyan_item_suffix(ood_item).replace(".", ""),
                "sha256": _aalyan_item_sha(ood_item),
                "labels_path": "",
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "jetson_runtime_detected_ood_guard" if isinstance(ood_item, dict) else "runtime_detected_ood_guard",
                "notes": "Detected OOD/Tier guard from connected Jetson runtime package.",
                "imported_at": _now(),
                "updated_at": _now(),
            }
        else:
            ood_meta = {
                "modality": modality_key,
                "display_modality": display,
                "version": ood_version,
                "display_version": "Jetson Tier / OOD Guard",
                "task": "ood_guard_and_routing",
                "model_path": "Not loaded on Jetson",
                "model_format": "-",
                "sha256": "",
                "labels_path": "",
                "preprocessing_path": "",
                "metrics_path": "",
                "status": "ood_guard_not_loaded_on_jetson",
                "notes": "No OOD/Tier guard detected on connected Jetson.",
                "imported_at": _now(),
                "updated_at": _now(),
            }

        clean_models[modality_key]["versions"][ood_version] = ood_meta
        registered.append(ood_meta)

    reg = {
        "schema_version": 1,
        "created_at": old_reg.get("created_at", _now()),
        "updated_at": _now(),
        "models": clean_models,
        "history": old_history + [{
            "time": _now(),
            "action": "clean_register_jetson_runtime_models",
            "jetson_files_detected": len(jetson_files),
            "local_files_detected": len(local_files),
            "rows": len(registered)
        }]
    }

    save_registry(reg)
    return registered

# ============================================================
# END AALYAN FINAL CLEAN MODEL REGISTRY
# ============================================================
