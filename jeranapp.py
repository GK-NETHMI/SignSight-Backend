# =========================================================
# 1. CRITICAL COMPATIBILITY FIXES (Must be at the very top)
# =========================================================
import os
import importlib.util as _importlib_util
import numpy as np

# TensorFlow 2.12 keeps numpy<1.24 (no numpy._core). SciPy 1.13+ may import numpy._core — fail fast with a clear fix.
if _importlib_util.find_spec("numpy._core") is None:
    try:
        import scipy as _scipy_startup  # noqa: F401
    except ImportError as _scipy_err:
        if "numpy._core" in str(_scipy_err).lower():
            raise RuntimeError(
                f"SciPy failed to import (NumPy {np.__version__} has no numpy._core). "
                "Downgrade SciPy for this TensorFlow 2.12 stack: pip install 'scipy==1.11.4' "
                "(see requirements.txt)."
            ) from _scipy_err
        raise
    else:
        _maj, _min = (int(x) for x in _scipy_startup.__version__.split(".")[:2])
        if (_maj, _min) >= (1, 13):
            raise RuntimeError(
                f"Incompatible stack: SciPy {_scipy_startup.__version__} expects numpy._core, "
                f"but NumPy {np.__version__} is pinned for TensorFlow 2.12. "
                "Fix: pip install 'scipy==1.11.4' (see requirements.txt)."
            )

# Legacy Keras avoids some LSTM ``time_major`` deserialization issues. TF 2.13+ needs ``tf_keras`` when set.
_force_legacy = os.environ.get("JERAN_FORCE_LEGACY_KERAS", "auto").strip().lower()
if _force_legacy in ("1", "true", "yes"):
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
elif _force_legacy in ("0", "false", "no"):
    os.environ.pop("TF_USE_LEGACY_KERAS", None)
else:
    try:
        import tf_keras  # noqa: F401

        os.environ["TF_USE_LEGACY_KERAS"] = "1"
    except ImportError:
        os.environ.pop("TF_USE_LEGACY_KERAS", None)
        print(
            "[startup] tf_keras not installed - unset TF_USE_LEGACY_KERAS (bundled Keras). "
            "For strict legacy mode: pip install tf_keras"
        )

# Fix for NumPy 2.0+ compatibility with older TensorFlow/JAX
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
if not hasattr(np, 'bool_'):
    np.bool_ = np.bool8

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import joblib
import sys
import uuid
import traceback
import base64
from io import BytesIO
from PIL import Image

print("=== SignSight Backend Starting ===")

# =========================
# Imports
# =========================
try:
    import tensorflow as tf
except ImportError as _e:
    msg = str(_e).lower()
    if "numpy._core" in msg or "no module named 'numpy._core'" in msg:
        raise RuntimeError(
            "Import failed: numpy._core is missing. That usually means NumPy is too old (<1.26.1) "
            "while another package (often SciPy) expects NumPy 1.26+ layout. "
            "This project pins TensorFlow 2.12 with numpy<1.24 — align the rest of the env: "
            "pip install -r requirements.txt  (includes scipy<1.13). "
            "Alternatively upgrade TensorFlow to 2.15+ and use numpy>=1.26.1."
        ) from _e
    raise
import keras
import mediapipe as mp

# =========================================================
# 2. LSTM MONKEY PATCH (Second layer of defense)
# =========================================================
orig_lstm_from_config = keras.layers.LSTM.from_config
def new_from_config(cls, config):
    config.pop('time_major', None)
    return orig_lstm_from_config(config)
keras.layers.LSTM.from_config = classmethod(new_from_config)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# =========================
# Model artifact layout — models-jeran/<subdir>/
# Per-request variant via form/JSON field `model_variant` (lazy load + cache).
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_ROOT = os.path.join(BASE_DIR, "models-jeran")
_FALLBACK_ROOT = os.path.join(BASE_DIR, "modeltraining-jeran")
_FALLBACK_H5 = "modeltraining2_finetuned_1.h5"

_VARIANT_SPECS = {
    "bilstm": {
        "label": "biLSTM",
        "subdir": "bilstm",
        "sequence_length": 45,
        "model_candidates": ["bilstm_modeltraining2_finetuned_1.h5"],
        "scaler_candidates": ["scaler.pkl"],
        "pca_candidates": ["pca.pkl"],
        "pca_required": True,
        "scaler_required": True,
        "fallback_training_dir": True,
    },
    "cnn_lstm": {
        "label": "CNN-LSTM",
        "subdir": "cnn-lstm",
        "sequence_length": 45,
        # Prefer HDF5 first: .keras ZIP often hits Keras-3 recursion / Conv2D weight mismatch on TF 2.12.
        "model_candidates": ["cnn_lstm_model.h5", "cnn_lstm_final.keras"],
        "scaler_candidates": ["scaler_cnn_lstm.pkl", "scaler.pkl"],
        "pca_candidates": ["pca.pkl"],
        "pca_required": False,
        "scaler_required": False,
        "fallback_training_dir": False,
    },
    "gru": {
        "label": "GRU",
        "subdir": "gru",
        # GRU_SignLanguage in your checkpoint expects 32 frames (override if model.input differs).
        "sequence_length": 32,
        "model_candidates": ["gru_sign_language_model.h5"],
        "scaler_candidates": ["scaler.pkl"],
        "pca_candidates": ["pca.pkl"],
        "pca_required": True,
        "scaler_required": True,
        "fallback_training_dir": False,
    },
    "tcn": {
        "label": "TCN",
        "subdir": "tcn",
        "sequence_length": 45,
        "model_candidates": [
            "tcn_weights_only.h5",
            "tcn_model_best.h5",
            "tcn_full.keras",
            "tcn_model.h5",
            "tcn_model_best.keras",
        ],
        "scaler_candidates": ["scaler_tcn.pkl"],
        "pca_candidates": ["pca_tcn.pkl"],
        "pca_required": True,
        "scaler_required": True,
        "fallback_training_dir": False,
    },
}

_DEFAULT_VARIANT_ENV = os.environ.get("JERAN_MODEL_VARIANT", "bilstm").strip().lower()

_stack_cache = {}
# Increment when stack dict schema or pipeline detection changes (drops stale cached stacks).
_STACK_CACHE_SCHEMA = 6
_LAST_REQUEST_LOG_VARIANT = {}  # throttle identical logs per-route if needed — kept simple below

FEATURE_DIM = 1662  # Mediapipe holistic vector length; must exist before bootstrap load_stack()


def _normalize_variant(raw):
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        norm = _DEFAULT_VARIANT_ENV if _DEFAULT_VARIANT_ENV in _VARIANT_SPECS else "bilstm"
    else:
        s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        aliased = {"bilstm": "bilstm", "bi_lstm": "bilstm", "cnnslstm": "cnn_lstm", "cnnlstm": "cnn_lstm"}
        norm = aliased.get(s, s)
    if norm not in _VARIANT_SPECS:
        norm = "bilstm"
    return norm


def _first_existing(directory, candidates):
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path, name
    return None, None


def _keras_custom_objects_for_variant(norm):
    """TCN checkpoints serialize a ``TCN`` custom layer — register exactly one class to avoid deserialization bugs."""
    custom = {}
    if norm != "tcn":
        return custom
    pref = os.environ.get("JERAN_TCN_LIBRARY", "").strip().lower().replace("_", "-")

    impls = []
    try:
        from keras_tcn.tcn import TCN as TCN_kt

        impls.append(("keras-tcn", TCN_kt))
    except ImportError:
        pass
    try:
        from tcn import TCN as TCN_pkg

        impls.append(("tcn", TCN_pkg))
    except ImportError:
        pass

    choice = None
    if pref in ("tcn", "tcn-package"):
        choice = next((x for x in impls if x[0] == "tcn"), None)
    elif pref in ("keras-tcn", "kerastcn"):
        choice = next((x for x in impls if x[0] == "keras-tcn"), None)
    if choice is None and impls:
        choice = impls[0]

    if choice:
        label, cls = choice
        custom["TCN"] = cls
        print(f"[config] TCN layer class registered ({label}): {cls.__module__}.{cls.__name__}")
        if len(impls) > 1 and not pref:
            other = [x[0] for x in impls if x[0] != label]
            print(
                f"[config] tip: both {other} and '{label}' are installed; using '{label}' only. "
                "Set JERAN_TCN_LIBRARY=keras-tcn or JERAN_TCN_LIBRARY=tcn to force the other."
            )
    else:
        print(
            "[config] WARN: variant=tcn but neither 'keras-tcn' nor 'tcn' is importable - "
            "pip install keras-tcn"
        )
    return custom


def _infer_sequence_length(model, fallback):
    """Read time dimension from Functional/Sequential ``model.input_shape``."""
    try:
        shp = getattr(model, "input_shape", None)
        if isinstance(shp, list):
            for s in shp:
                if s and isinstance(s, (tuple, list)) and len(s) >= 2 and s[1] is not None:
                    return int(s[1])
        if isinstance(shp, tuple) and len(shp) >= 2 and shp[1] is not None:
            return int(shp[1])
    except Exception:
        pass
    return int(fallback)


def _infer_input_kind(model):
    """Return ``('rgb_stack', (H,W,C))`` when model consumes (batch, time, H, W, channels); else ``('keypoints', None)``.

    Used for CNN+LSTM, TCN-on-frames, etc. Landmark models use ``(batch, time, feats)``.
    Scans every reported input (multi-input models may list a non-image tensor first).
    """
    try:
        shp = getattr(model, "input_shape", None)
        shapes = []
        if isinstance(shp, list):
            for s in shp:
                if s and isinstance(s, (tuple, list)):
                    shapes.append(tuple(s))
        elif isinstance(shp, tuple):
            shapes.append(shp)
        for t in shapes:
            if len(t) == 5:
                _bt, _t, h, w, ch = t
                if h is not None and w is not None and ch is not None:
                    ch = int(ch)
                    if ch in (1, 3, 4):
                        return "rgb_stack", (int(h), int(w), ch)
        return "keypoints", None
    except Exception:
        return "keypoints", None


def _sync_stack_input_from_model(stk):
    """Force ``input_kind`` / ``rgb_hwc`` to match ``model.input_shape`` (stale cache or older server builds)."""
    ik, hwc = _infer_input_kind(stk["model"])
    old_ik, old_hwc = stk.get("input_kind"), stk.get("rgb_hwc")
    if ik != old_ik or (ik == "rgb_stack" and hwc != old_hwc):
        print(f"[MODEL] stk sync variant={stk.get('norm')}: input_kind {old_ik!r} -> {ik!r} rgb_hwc {old_hwc} -> {hwc}")
    stk["input_kind"] = ik
    stk["rgb_hwc"] = hwc


def _resolve_sequence_length(norm, model, spec, paths):
    """Prefer concrete timestep from loaded model; fallback to variant spec (e.g. GRU=32)."""
    hint = int(spec.get("sequence_length") or 45)
    t = _infer_sequence_length(model, fallback=hint)
    if t != hint:
        print(
            f"[MODEL] variant={norm} timestep from model.input_shape={t} "
            f"(spec hint was {hint}) file={paths.get('model_name')}"
        )
    else:
        print(f"[MODEL] variant={norm} sequence_length={t} file={paths.get('model_name')}")
    return t


def _inspect_hdf5_topology(path):
    try:
        import h5py
        with h5py.File(path, "r") as f:
            print(f"[h5inspect] root keys (first 25): {list(f.keys())[:25]}")
            ak = [str(k) for k in list(f.attrs.keys())[:20]]
            has_cfg = "model_config" in f.attrs
            print(f"[h5inspect] sample attr keys: {ak}")
            print(f"[h5inspect] model_config in attrs: {has_cfg}")
    except Exception as ex:
        print(f"[h5inspect] could not inspect: {ex}")


def _keras_h5_file_plausibility(path):
    """Reject Git-LFS pointers, empty HDF5 shells, and non-checkpoint files before confusing load_model errors."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(240)
    except OSError as ex:
        return False, f"Cannot read {path!r}: {ex}"
    if head.startswith(b"version https://git-lfs.github.com") or b"git-lfs.github.com" in head[:120]:
        return (
            False,
            "This file is a Git LFS pointer (text stub), not the trained weights. "
            "In the repo run: git lfs install && git lfs pull — or copy the real checkpoint from your training machine.",
        )
    try:
        sz = os.path.getsize(path)
    except OSError as ex:
        return False, f"Cannot stat {path!r}: {ex}"
    if sz < 16384:
        return (
            False,
            f"The HDF5 at {path!r} is only {sz} bytes. A real Keras export is usually hundreds of KB or more and "
            "contains model_config and/or model_weights groups. Re-export with model.save('tcn_model.keras') after training.",
        )
    try:
        import h5py

        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            ak = list(f.attrs.keys())
            if "model_config" in f.attrs or len(keys) > 0:
                return True, ""
            return (
                False,
                f"{path!r} ({sz} bytes) opens as HDF5 but has no top-level datasets/groups and no model_config "
                f"attribute (root attrs: {ak[:12]}). It is not a loadable Keras checkpoint.",
            )
    except Exception as ex:
        return False, f"{path!r} is not a readable HDF5 training file ({ex!r})."


def _first_plausible_model_file(directory, candidates):
    """First existing candidate that is not an empty HDF5 shell / LFS pointer / tiny stub."""
    for name in candidates:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        lp = path.lower()
        if lp.endswith(".h5"):
            ok, reason = _keras_h5_file_plausibility(path)
            if ok:
                return path, name
            msg = (reason or "").replace("\n", " ")[:220]
            print(f"[MODEL/paths] skipping unusable `{name}` under {directory}: {msg}")
            continue
        if lp.endswith(".keras"):
            sz = os.path.getsize(path)
            if sz < 2048:
                print(f"[MODEL/paths] skipping tiny `.keras` `{name}` ({sz} bytes)")
                continue
            return path, name
        sz = os.path.getsize(path)
        if sz < 1024:
            print(f"[MODEL/paths] skipping tiny artifact `{name}` ({sz} bytes)")
            continue
        return path, name
    return None, None


def _build_tcn_placeholder_functional():
    """Same graph as ``scripts/bootstrap_tcn_placeholder.py`` (45, 256) -> 15-class softmax.

    Used to load ``tcn_weights_only.h5`` or recover when a Keras-3 ``.keras`` file cannot deserialize on TF 2.12.
    """
    SEQ, PCA_DIM, N_CLASSES = 45, 256, 15
    inp = tf.keras.layers.Input(shape=(SEQ, PCA_DIM), name="input_sequence")
    x = inp
    for dil in (1, 2, 4, 8):
        x = tf.keras.layers.Conv1D(
            64,
            3,
            padding="causal",
            dilation_rate=dil,
            activation="relu",
            name=f"dilated_conv_{dil}",
        )(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="predictions")(x)
    return tf.keras.models.Model(inp, out, name="tcn_style_placeholder")


def _load_keras_file(path, norm):
    """Load SavedModel/Keras/ZIP (.keras) / HDF5 (.h5) with fallbacks compatible with TF 2.12.

    Prefer ``tf.keras.models.load_model`` — using ``keras.models.load_model`` alone often triggers
    "No model config found..." on checkpoints saved by the other runtime.
    """
    lp = path.lower()
    if lp.endswith(".h5"):
        ok, diag = _keras_h5_file_plausibility(path)
        if not ok:
            raise RuntimeError(diag) from None

    base = os.path.basename(path)
    if norm == "tcn" and base.lower() == "tcn_weights_only.h5":
        try:
            m = _build_tcn_placeholder_functional()
            m.load_weights(path, by_name=True)
            print(f"[MODEL LOAD OK] tcn: built graph + load_weights from {path!r}")
            return m
        except Exception as ex:
            raise RuntimeError(
                f"TCN weights-only load failed for {path!r}: {ex}. "
                "Regenerate with: python scripts/bootstrap_tcn_placeholder.py"
            ) from ex

    co = dict(_keras_custom_objects_for_variant(norm))
    last_err = None
    attempts = []
    rl_old = sys.getrecursionlimit()
    if norm == "tcn":
        sys.setrecursionlimit(max(rl_old, 10000))

    def _attempt(label, loader_fn):
        nonlocal last_err
        try:
            m = loader_fn()
            if label not in ("tf.keras(load_model)+custom_objects", "tf.keras(load_model)"):
                print(f"[MODEL LOAD OK] fallback loader used: {label}")
            return m
        except Exception as exc:
            last_err = exc
            attempts.append(f"{label}: {exc}")
            return None

    try:
        def _tf_load(with_co):
            kwargs = dict(compile=False)
            if with_co and co:
                kwargs["custom_objects"] = co
            return tf.keras.models.load_model(path, **kwargs)

        def _keras_load(with_co):
            kwargs = dict(compile=False)
            if with_co and co:
                kwargs["custom_objects"] = co
            return keras.models.load_model(path, **kwargs)

        m = _attempt("tf.keras(load_model)+custom_objects", lambda: _tf_load(True))
        if m is not None:
            return m

        m = _attempt("tf.keras(load_model)", lambda: _tf_load(False))
        if m is not None:
            return m

        try:
            import inspect
            sig = inspect.signature(tf.keras.models.load_model)
            if "safe_mode" in sig.parameters:
                m = _attempt(
                    "tf.keras(load_model,safe_mode=False)+custom_objects",
                    lambda: tf.keras.models.load_model(
                        path,
                        compile=False,
                        safe_mode=False,
                        **({"custom_objects": co} if co else {}),
                    ),
                )
                if m is not None:
                    return m
                m = _attempt(
                    "tf.keras(load_model,safe_mode=False)",
                    lambda: tf.keras.models.load_model(path, compile=False, safe_mode=False),
                )
                if m is not None:
                    return m
        except Exception:
            pass

        m = _attempt("keras(load_model)+custom_objects", lambda: _keras_load(True))
        if m is not None:
            return m

        m = _attempt(
            "keras(load_model)",
            lambda: _keras_load(False),
        )
        if m is not None:
            return m

        if norm == "tcn" and path.lower().endswith(".keras"):
            alt = os.path.join(os.path.dirname(path), "tcn_weights_only.h5")
            if os.path.isfile(alt):
                try:
                    m = _build_tcn_placeholder_functional()
                    m.load_weights(alt, by_name=True)
                    print(f"[MODEL LOAD OK] tcn: .keras incompatible; used load_weights from {alt!r}")
                    return m
                except Exception as ex2:
                    attempts.append(f"tcn fallback load_weights({alt}): {ex2}")

        msg = "; ".join(attempts[-4:]) if attempts else repr(last_err)
        if path.lower().endswith(".h5"):
            _inspect_hdf5_topology(path)
        raise RuntimeError(
            f"All load_model attempts failed for {path!r}: {msg}. "
            "If HDF5 lacks model architecture, ModelCheckpoint(save_weights_only=True) was probably used "
            "--- re-save the full model via model.save('full_model.keras') once (or bundle SavedModel)."
        ) from last_err
    finally:
        if norm == "tcn":
            sys.setrecursionlimit(rl_old)


def _resolve_bilstm_with_fallback(spec):
    subdir_path = os.path.join(_MODELS_ROOT, spec["subdir"])
    model_path, model_name = _first_plausible_model_file(subdir_path, spec["model_candidates"])
    if model_path:
        scaler_path, scaler_name = _first_existing(subdir_path, spec["scaler_candidates"])
        pca_path, pca_name = _first_existing(subdir_path, spec["pca_candidates"])
        if scaler_path and pca_path:
            return {
                "model_dir": subdir_path,
                "model_path": model_path, "model_name": model_name,
                "scaler_path": scaler_path, "scaler_name": scaler_name,
                "pca_path": pca_path, "pca_name": pca_name,
            }
    if os.path.isfile(os.path.join(_FALLBACK_ROOT, _FALLBACK_H5)):
        fb = os.path.join(_FALLBACK_ROOT, _FALLBACK_H5)
        sp = os.path.join(_FALLBACK_ROOT, "scaler.pkl")
        pp = os.path.join(_FALLBACK_ROOT, "pca.pkl")
        if os.path.isfile(sp) and os.path.isfile(pp):
            print(f"[startup/paths] biLSTM using fallback bundle: {_FALLBACK_ROOT}")
            return {
                "model_dir": _FALLBACK_ROOT,
                "model_path": fb,
                "model_name": _FALLBACK_H5,
                "scaler_path": sp,
                "scaler_name": "scaler.pkl",
                "pca_path": pp,
                "pca_name": "pca.pkl",
            }
    model_path2, mn = _first_plausible_model_file(subdir_path, spec["model_candidates"])
    scaler_path2, sn = _first_existing(subdir_path, spec["scaler_candidates"]) if model_path2 else (None, None)
    pca_path2, pn = _first_existing(subdir_path, spec["pca_candidates"]) if model_path2 else (None, None)
    return {
        "model_dir": subdir_path,
        "model_path": model_path2,
        "model_name": mn,
        "scaler_path": scaler_path2,
        "scaler_name": sn,
        "pca_path": pca_path2,
        "pca_name": pn,
    }


def resolve_paths_for_variant(norm):
    spec = _VARIANT_SPECS[norm]
    if norm == "bilstm" and spec.get("fallback_training_dir"):
        return _resolve_bilstm_with_fallback(spec)
    root = os.path.join(_MODELS_ROOT, spec["subdir"])
    model_path, model_name = _first_plausible_model_file(root, spec["model_candidates"])
    scaler_path, scaler_name = _first_existing(root, spec["scaler_candidates"]) if model_path else (None, None)
    pca_path, pca_name = None, None
    if model_path:
        pca_path, pca_name = _first_existing(root, spec["pca_candidates"])
    return {
        "model_dir": root,
        "model_path": model_path,
        "model_name": model_name,
        "scaler_path": scaler_path,
        "scaler_name": scaler_name,
        "pca_path": pca_path,
        "pca_name": pca_name,
    }


def _artifact_paths_acceptable(norm, pst):
    """Filesystem layout is sufficient to *attempt* load_stack for this variant (matches load_stack errs)."""
    spec = _VARIANT_SPECS[norm]
    if not pst.get("model_path"):
        return False
    if spec.get("pca_required") and not pst.get("pca_path"):
        return False
    if spec.get("scaler_required", True) and not pst.get("scaler_path"):
        return False
    return True


def load_stack(norm):
    norm = _normalize_variant(norm)
    paths = resolve_paths_for_variant(norm)
    spec = _VARIANT_SPECS[norm]

    errs = []
    if not paths["model_path"]:
        errs.append(
            f"No loadable checkpoint in {paths['model_dir']} (tried {spec['model_candidates']}). "
            "Stub .h5 files were skipped — save a full model after training "
            "(e.g. model.save('models-jeran/tcn/tcn_model_best.keras')) or run `git lfs pull` "
            "if checkpoints are stored with Git LFS."
        )
    if spec["pca_required"] and not paths["pca_path"]:
        errs.append(f"PCA required but missing (looked for {spec['pca_candidates']})")
    if errs:
        return None, "; ".join(errs), paths, spec["label"]

    mdl = scl = pac = spatial_scl = None
    try:
        print(f"[MODEL LOAD TRY] variant={norm} path={paths['model_path']}")
        mdl = _load_keras_file(paths["model_path"], norm)
        scl_loaded = joblib.load(paths["scaler_path"]) if paths["scaler_path"] else None
        pac_loaded = joblib.load(paths["pca_path"]) if paths["pca_path"] else None

        input_kind, rgb_hwc = _infer_input_kind(mdl)
        seq_len = _resolve_sequence_length(norm, mdl, spec, paths)

        if input_kind == "rgb_stack":
            H, W, C = rgb_hwc
            n_pix = int(H * W * C)
            scl = None
            pac = None
            spatial_scl = None
            if scl_loaded is not None:
                nfi = getattr(scl_loaded, "n_features_in_", None)
                if nfi == n_pix:
                    spatial_scl = scl_loaded
                elif nfi == FEATURE_DIM:
                    print(
                        f"[MODEL] variant={norm}: scaler is ({nfi}-D) landmark scaler; CNN expects "
                        f"flat RGB ({n_pix}); running without scaler (divide 255)."
                    )
                else:
                    print(
                        f"[MODEL] variant={norm}: scaler n_features_in_={nfi} != image flat={n_pix}; "
                        "ignoring scaler"
                    )
            print(
                f"[MODEL] variant={norm}: input_pipeline=rgb_stack shape=(T,{H},{W},{C}) spatial_scaler="
                f"{'yes' if spatial_scl is not None else 'no'}"
            )
        else:
            scl = scl_loaded
            pac = pac_loaded
            if spec["pca_required"] and pac is None:
                return None, "PCA required but missing", paths, spec["label"]
            if scl is None:
                return (
                    None,
                    (
                        f"Scaler required for keypoint-input model; add one of {spec['scaler_candidates']} "
                        f"to {paths['model_dir']}"
                    ),
                    paths,
                    spec["label"],
                )
            spatial_scl = None

        msg = (
            f"[MODEL LOAD OK] variant={norm} ({spec['label']}) | "
            f"model_file={paths['model_name']} | path={paths['model_path']} | "
            f"input_kind={input_kind} | "
            f"scaler={paths['scaler_name']} | "
            f"pca={'none' if pac is None else paths['pca_name']} | "
            f"seq_len={seq_len}"
        )
        print(msg)

        meta = {
            "norm": norm,
            "display_label": spec["label"],
            "paths": paths,
            "sequence_length": seq_len,
            "input_kind": input_kind,
            "rgb_hwc": rgb_hwc,
            "spatial_scaler": spatial_scl,
            "_cache_schema": _STACK_CACHE_SCHEMA,
        }
        return {"model": mdl, "scaler": scl, "pca": pac, **meta}, None, paths, spec["label"]
    except Exception as e:
        traceback.print_exc()
        return None, str(e), paths, spec["label"]


def get_stack(norm):
    norm = _normalize_variant(norm)
    if norm in _stack_cache:
        cached = _stack_cache[norm]
        if cached.get("_cache_schema") != _STACK_CACHE_SCHEMA:
            del _stack_cache[norm]
        else:
            return cached, None
    stk, err, paths, lbl = load_stack(norm)
    if stk is None:
        return None, {"error": err or "unknown load error", "variant": norm, "label": lbl, "paths": paths}
    _stack_cache[norm] = stk
    return stk, None


def preprocess_rgb_sequence(seq_hwc, seq_len, spatial_scaler_opt, expected_hwc):
    """Normalize / optionally scale RGB (or grayscale) clips to ``(1, seq_len, H, W, C)``."""
    seq = np.asarray(seq_hwc, dtype=np.float32)
    if seq.ndim != 4:
        raise ValueError(f"Image-sequence model expects (frames, height, width, channels), got shape {seq.shape}")
    Eh, Ew, Ec = expected_hwc
    Hf, Wf, Cf = int(seq.shape[1]), int(seq.shape[2]), int(seq.shape[3])
    if (Hf, Wf, Cf) != (Eh, Ew, Ec):
        raise ValueError(f"Spatial shape mismatch: frames are ({Hf},{Wf},{Cf}), model expects ({Eh},{Ew},{Ec})")
    if seq.shape[0] == 0:
        raise ValueError("No video frames extracted")
    n = int(seq.shape[0])
    if n != seq_len:
        if n > seq_len:
            start = (n - seq_len) // 2
            seq = np.asarray(seq[start : start + seq_len], dtype=np.float32).copy()
        else:
            pad = np.zeros((seq_len - n, Eh, Ew, Ec), dtype=np.float32)
            seq = np.vstack((seq, pad))
    nt = int(seq.shape[0])
    if spatial_scaler_opt is not None:
        flat = seq.reshape(nt, Eh * Ew * Ec)
        seq = spatial_scaler_opt.transform(flat).astype(np.float32).reshape(nt, Eh, Ew, Ec)
    else:
        seq = np.clip(seq, 0.0, 255.0) / 255.0
    return seq.reshape(1, seq_len, Eh, Ew, Ec).astype(np.float32)


def extract_rgb_thumbnail_stack(cap, h, w, c):
    """Read every frame from a capture as resized RGB (``c==3``) or grayscale-with-channel (``c==1``)."""
    stacks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if c == 1:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pix = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
            stacks.append(pix[:, :, np.newaxis].astype(np.float32))
        elif c == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pix = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            stacks.append(pix.astype(np.float32))
        elif c == 4:
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            pix = cv2.resize(bgra, (w, h), interpolation=cv2.INTER_AREA)
            stacks.append(pix.astype(np.float32))
        else:
            raise ValueError(f"Unsupported channel count for CNN input: {c}")
    if not stacks:
        return np.zeros((0, h, w, c), dtype=np.float32)
    return np.stack(stacks, axis=0).astype(np.float32)


def preprocess_sequence(sequence_1662, scaler, pca_opt, seq_len):
    fixed = []
    for frame in sequence_1662:
        if len(frame) != FEATURE_DIM:
            frame = np.zeros(FEATURE_DIM, dtype=np.float32)
        fixed.append(frame)
    seq = np.array(fixed, dtype=np.float32)
    if seq.shape[0] == 0:
        raise ValueError("Sequence has no frames")
    n, feat_d = int(seq.shape[0]), int(seq.shape[1])
    if n != seq_len:
        if n > seq_len:
            start = (n - seq_len) // 2
            seq = np.asarray(seq[start : start + seq_len], dtype=np.float32).copy()
        else:
            pad = np.zeros((seq_len - n, feat_d), dtype=np.float32)
            seq = np.vstack((seq, pad))
            seq = np.asarray(seq, dtype=np.float32)
    X_scaled = scaler.transform(seq)
    if pca_opt is not None:
        X_feat = pca_opt.transform(X_scaled)
    else:
        X_feat = X_scaled
    td = X_feat.shape[1]
    return X_feat.reshape(1, seq_len, td).astype(np.float32)


def predict_with_stack(stk, sequence_np):
    _sync_stack_input_from_model(stk)
    sl = int(stk["sequence_length"])
    if stk.get("input_kind") == "rgb_stack":
        model_input = preprocess_rgb_sequence(
            sequence_np, sl, stk.get("spatial_scaler"), stk["rgb_hwc"]
        )
    else:
        model_input = preprocess_sequence(sequence_np, stk["scaler"], stk["pca"], sl)
    pred = stk["model"].predict(model_input, verbose=0)[0]
    class_idx = int(np.argmax(pred))
    confidence = float(pred[class_idx])
    english_action = actions[class_idx]
    tamil_action = actions_dict[english_action]
    return {
        "action": f"{english_action} / {tamil_action}",
        "action_english": english_action,
        "action_tamil": tamil_action,
        "english": english_action,
        "tamil": tamil_action,
        "confidence": confidence,
        "model_variant": stk["norm"],
        "model_label": stk["display_label"],
    }


def _log_request_inference(route_name, stk):
    p = stk["paths"]
    print(
        f"[RUN] route={route_name} | variant={stk['norm']} ({stk['display_label']}) | "
        f"files: model={p.get('model_name')} scaler={p.get('scaler_name')} pca={p.get('pca_name') or 'none'} | "
        f"paths: {p.get('model_path')}"
    )

# Warm default variant once (optional bootstrap)
_startup_norm = _normalize_variant(_DEFAULT_VARIANT_ENV)
_default_stack_err = None
try:
    _boot, _default_stack_err = get_stack(_startup_norm)
    if _boot:
        print(f"[startup] Default stack ready: {_startup_norm} ({_boot['display_label']})")
    elif _default_stack_err:
        print(f"[startup] Default stack ({_startup_norm}) not loaded: {_default_stack_err.get('error')}")
except Exception as ex:
    print(f"[startup] bootstrap error: {ex}")
    _default_stack_err = {"error": str(ex)}

# =========================
# Constants (labels aligned with softmax order)
# =========================
actions_dict = {
    'Beautiful': 'அழகு',
    'Drink': 'குடி',
    'Eat': 'சாப்பிடு',
    'Five': 'ஐந்து',
    'Good': 'நல்லது',
    'Hello': 'வணக்கம்',
    'House': 'வீடு',
    'Love': 'காதல்',
    'Man': 'ஆண்',
    'Mother': 'அம்மா',
    'Run': 'ஓடு',
    'Thank you': 'நன்றி',
    'White': 'வெள்ளை',
    'Yellow': 'மஞ்சள்',
    'You': 'நீ'
}

actions = list(actions_dict.keys())

# =========================
# Extract Keypoints
# =========================
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(468*3)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

# =========================
# Flask App
# =========================
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    out = []
    boot_ok = False
    for key in sorted(_VARIANT_SPECS.keys()):
        pst = resolve_paths_for_variant(key)
        spec = _VARIANT_SPECS[key]
        ok_paths = _artifact_paths_acceptable(key, pst)
        out.append({
            "variant": key,
            "label": spec["label"],
            "artifact_paths_ok": ok_paths,
            "model_file": pst["model_name"],
            "scaler_file": pst["scaler_name"],
            "pca_file": pst["pca_name"],
            "directory": pst["model_dir"],
            "hints": [] if ok_paths else _variant_setup_hints(key, pst, spec),
        })
    try:
        dbs, _ = get_stack(_startup_norm)
        boot_ok = dbs is not None
    except Exception:
        boot_ok = False
    return jsonify({
        'status': 'healthy' if boot_ok else 'degraded',
        'default_variant': _startup_norm,
        'default_model_loaded': boot_ok,
        'variants': out,
        'lazy_cache_loaded': list(_stack_cache.keys()),
    })


@app.route("/variant_status")
def variant_status():
    """Warm-cache + verify one variant (optional UI call before inference). Query: model_variant=…"""
    vkey = request.args.get("model_variant") or request.args.get("variant")
    norm = _normalize_variant(vkey)
    pst = resolve_paths_for_variant(norm)
    spec = _VARIANT_SPECS[norm]
    if not _artifact_paths_acceptable(norm, pst):
        return jsonify({
            "variant": norm,
            "label": spec["label"],
            "artifact_paths_ok": False,
            "weights_loaded": False,
            "error": "Missing model/scaler/pca files on disk (see /health variants[].hints).",
            "hints": _variant_setup_hints(norm, pst, spec),
        }), 200
    stk, err = get_stack(norm)
    payload = {
        "variant": norm,
        "label": spec["label"],
        "artifact_paths_ok": True,
        "weights_loaded": stk is not None,
        "model_file": pst.get("model_name"),
        "input_kind": stk.get("input_kind") if stk else None,
    }
    if err:
        payload["error"] = err.get("error", "unknown")
        payload["detail"] = err
    return jsonify(payload), 200


def _variant_setup_hints(norm, pst, spec):
    """Short checklist when artifact_paths_ok is false."""
    hints = []
    if not pst.get("model_path"):
        hints.append(
            f"No loadable checkpoint in {pst['model_dir']} (tried {spec['model_candidates']}). "
            "Export after training with model.save('…') or git lfs pull for real weights."
        )
    if spec.get("pca_required") and not pst.get("pca_path"):
        hints.append(f"PCA missing under {pst['model_dir']} (need one of {spec['pca_candidates']}).")
    if spec.get("scaler_required", True) and not pst.get("scaler_path"):
        hints.append(f"Scaler missing under {pst['model_dir']} (need one of {spec['scaler_candidates']}).")
    if norm == "tcn":
        hints.append(
            "TCN: install `pip install keras-tcn`; training must save a FULL model (.keras preferred), "
            "not save_weights_only."
        )
    return hints


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        vkey = data.get('model_variant') or data.get('model') or request.args.get('model_variant')
        stk, err = get_stack(vkey)
        if err:
            return jsonify({'error': err.get('error'), 'detail': err}), 500

        _sync_stack_input_from_model(stk)
        if stk.get("input_kind") == "rgb_stack":
            return jsonify(
                {
                    "error": (
                        "This model expects raw video frames (image pipeline, not MediaPipe landmarks). "
                        "Use POST /predict_video with multipart video, not JSON keypoints."
                    )
                }
            ), 400

        sequence = np.array(data.get('sequence'), dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
            return jsonify({'error': f'Expected shape (n, {FEATURE_DIM}), got {sequence.shape}'}), 400
        if sequence.shape[0] < 1:
            return jsonify({'error': 'Sequence must contain at least one frame'}), 400

        _log_request_inference('/predict', stk)
        result = predict_with_stack(stk, sequence)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_video', methods=['POST'])
def predict_video():
    raw_v = (
        request.form.get('model_variant')
        or request.args.get('model_variant')
        or None
    )
    stk, err = get_stack(raw_v)
    if err:
        return jsonify({'error': err.get('error'), 'detail': err}), 500
    _sync_stack_input_from_model(stk)
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    video_file = request.files['video']
    temp_path = os.path.join("temp", f"temp_{uuid.uuid4().hex[:8]}.webm")
    os.makedirs("temp", exist_ok=True)
    video_file.save(temp_path)

    try:
        cap = cv2.VideoCapture(temp_path)
        try:
            if stk.get("input_kind") == "rgb_stack":
                h, w, c = stk["rgb_hwc"]
                clips = extract_rgb_thumbnail_stack(cap, h, w, c)
                if clips.shape[0] == 0:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return jsonify({'error': 'No frames decoded from video'}), 400
                _log_request_inference('/predict_video', stk)
                result = predict_with_stack(stk, clips)
            else:
                frames_kp = []
                with mp_holistic.Holistic(
                    min_detection_confidence=0.5, min_tracking_confidence=0.5
                ) as holistic:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(image)
                        frames_kp.append(extract_keypoints(results))

                if len(frames_kp) == 0:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return jsonify({'error': 'No keypoints extracted from video'}), 400

                _log_request_inference('/predict_video', stk)
                result = predict_with_stack(stk, np.array(frames_kp))
        finally:
            cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify(result)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/webcam_predict', methods=['POST'])
def webcam_predict():
    try:
        data = request.get_json() or {}
        vkey = data.get('model_variant') or request.args.get('model_variant')
        stk, err = get_stack(vkey)
        if err:
            return jsonify({'error': err.get('error'), 'detail': err}), 500

        _sync_stack_input_from_model(stk)
        if stk.get("input_kind") == "rgb_stack":
            return jsonify(
                {
                    "error": (
                        "Image-based models need full frames resized like training (e.g. 64x64 clips). "
                        "Use POST /predict_video with a recording; webcam keypoints are not supported."
                    )
                }
            ), 400

        frame_data = data.get('frame')
        sequence = data.get('sequence', [])

        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        frame = np.array(image)

        with mp_holistic.Holistic() as holistic:
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            keypoints = extract_keypoints(results)

        sl = int(stk["sequence_length"])
        sequence.append(keypoints.tolist())
        sequence = sequence[-sl:]

        if len(sequence) == sl:
            _log_request_inference('/webcam_predict', stk)
            base = predict_with_stack(stk, np.array(sequence, dtype=np.float32))
            english_action = base["english"]
            tamil_action = base["tamil"]

            return jsonify({
                'prediction': base["action"],
                'prediction_english': base["action_english"],
                'prediction_tamil': base["action_tamil"],
                'english': english_action,
                'tamil': tamil_action,
                'confidence': base["confidence"],
                'sequence': sequence,
                'model_variant': base["model_variant"],
                'model_label': base["model_label"],
            })
        return jsonify({'message': f'Collecting... {len(sequence)}/{sl}', 'sequence': sequence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*70)
    print("SignSight backend running.")
    print(f"Default variant env: {_startup_norm} ({_VARIANT_SPECS.get(_startup_norm, {}).get('label', '?')})")
    print(f"Lazy-load variants (cache): {', '.join(sorted(_VARIANT_SPECS.keys()))}")
    print("="*70)
    # threaded=True helps while one request is loading heavy weights (first hit per variant)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
