"""
Write ``tcn_weights_only.h5`` (weights only) for the TCN-style placeholder graph defined in ``jeranapp._build_tcn_placeholder_functional``.

TensorFlow 2.12 cannot ``load_model`` Keras-3 ``.keras`` archives; weights + in-code graph avoids that.

Run from ``SignSight-Backend`` (prefer the same Python/TensorFlow as ``jeranapp.py``):
  python scripts/bootstrap_tcn_placeholder.py
"""
from __future__ import annotations

import os
import sys

os.environ.pop("TF_USE_LEGACY_KERAS", None)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_WEIGHTS = os.path.join(BASE, "models-jeran", "tcn", "tcn_weights_only.h5")

SEQ = 45
PCA_DIM = 256
N_CLASSES = 15


def _build_functional():
    import tensorflow as tf

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


def main() -> int:
    m = _build_functional()
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    os.makedirs(os.path.dirname(OUT_WEIGHTS), exist_ok=True)
    m.save_weights(OUT_WEIGHTS)
    sz = os.path.getsize(OUT_WEIGHTS)
    print(f"[ok] wrote {OUT_WEIGHTS} ({sz} bytes)")
    print("[note] Random weights - replace after training (keep same layer names for load_weights).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
