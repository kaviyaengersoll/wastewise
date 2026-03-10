"""
WasteWise Model Training Script (Fixed)
========================================
Fixes: class imbalance, truncated images, proper biodegradable classification
Dataset: 2 classes — biodegradable / non_biodegradable
Model: MobileNetV2 + custom head with class weights

Run: python model_training/train.py --data_dir "C:\path\to\Dataset"
"""

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Fix truncated image errors

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE           = (224, 224)
BATCH_SIZE         = 32
EPOCHS_HEAD        = 10
EPOCHS_FINE        = 10
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5
OUTPUT_DIR         = "model_output"

# ── Metadata for 2-class dataset ─────────────────────────────────────────────
CLASS_META = {
    "biodegradable": {
        "display": "Biodegradable Waste",
        "biodegradable": True,
        "category": "Organic / Natural Waste",
        "health_risk": "Low",
        "severity": "Green",
        "score": 2,
        "decompose_time": "Weeks to months",
        "disposal": "Place in wet/green waste bin. Ideal for composting at home.",
        "fun_fact": "Biodegradable waste can be converted into compost that enriches soil and reduces the need for chemical fertilizers.",
        "upcycling": [
            {"title": "🌍 Home Compost", "desc": "Layer with dry leaves and soil to make rich garden compost."},
            {"title": "🌱 Seed Starter", "desc": "Use cardboard/paper pieces as biodegradable seed starter pots."},
            {"title": "🍵 Vegetable Stock", "desc": "Boil vegetable peels with spices for free cooking stock."},
            {"title": "🌻 Plant Fertilizer", "desc": "Bury fruit peels near plants for a natural nutrient boost."}
        ]
    },
    "non_biodegradable": {
        "display": "Non-Biodegradable Waste",
        "biodegradable": False,
        "category": "Plastic / Metal / Synthetic Waste",
        "health_risk": "High",
        "severity": "Red",
        "score": 8,
        "decompose_time": "100 – 1000+ years",
        "disposal": "Separate by material (plastic/glass/metal). Place in dry recycling bin. Never burn.",
        "fun_fact": "Only 9% of all plastic ever produced has been recycled. The rest ends up in landfills or oceans.",
        "upcycling": [
            {"title": "🌿 Bottle Planter", "desc": "Cut plastic bottles in half and use as planters for small herbs."},
            {"title": "🐦 Bird Feeder", "desc": "Cut holes in a plastic bottle, add a perch and fill with birdseed."},
            {"title": "🕯️ Tin Lantern", "desc": "Punch patterns into tin cans and place a tealight inside."},
            {"title": "🎨 Upcycled Art", "desc": "Use bottle caps, wrappers and cans to create mosaic wall art."}
        ]
    }
}


def build_model(num_classes: int):
    """MobileNetV2 transfer learning model."""
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base


def get_data_generators(data_dir: str):
    """Image data generators with augmentation."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.75, 1.25],
        fill_mode="nearest"
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    val_data = val_gen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    test_dir = os.path.join(data_dir, "test")
    test_data = None
    if os.path.exists(test_dir):
        test_data = val_gen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False
        )

    return train_data, val_data, test_data


def main(data_dir: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n📂 Loading dataset...")
    train_data, val_data, test_data = get_data_generators(data_dir)
    class_names = list(train_data.class_indices.keys())
    num_classes = len(class_names)

    print(f"✅ Found {num_classes} classes: {class_names}")
    print(f"   Train samples : {train_data.samples}")
    print(f"   Val samples   : {val_data.samples}")

    # Print per-class image counts
    unique, counts = np.unique(train_data.classes, return_counts=True)
    for idx, cnt in zip(unique, counts):
        print(f"   {class_names[idx]}: {cnt} images")

    # ── Compute class weights to fix imbalance ────────────────────────────
    labels = train_data.classes
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(cw))
    print(f"\n⚖️  Class weights (fixes imbalance): {class_weights}")

    # ── Save class info ───────────────────────────────────────────────────
    class_info = {
        "class_names": class_names,
        "class_indices": train_data.class_indices,
        "img_size": list(IMG_SIZE),
        "metadata": {k: CLASS_META.get(k, {}) for k in class_names}
    }
    with open(os.path.join(OUTPUT_DIR, "class_info.json"), "w") as f:
        json.dump(class_info, f, indent=2)
    print(f"📝 Saved class info → {OUTPUT_DIR}/class_info.json")

    # ── Build model ───────────────────────────────────────────────────────
    print("\n🏗️  Building MobileNetV2 model...")
    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # ── Phase 1: Train head only ──────────────────────────────────────────
    cb1 = [
        callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
    ]

    print(f"\n🚀 Phase 1: Training classification head ({EPOCHS_HEAD} epochs)...")
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_HEAD,
        callbacks=cb1,
        class_weight=class_weights,
        verbose=1
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────────────────
    print(f"\n🔧 Phase 2: Fine-tuning top MobileNetV2 layers ({EPOCHS_FINE} epochs)...")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE_FINE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb2 = [
        callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, verbose=1
        ),
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FINE,
        callbacks=cb2,
        class_weight=class_weights,
        verbose=1
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n📊 Evaluating on validation set...")
    loss, acc = model.evaluate(val_data, verbose=1)
    print(f"✅ Val Accuracy: {acc*100:.2f}%")

    if test_data:
        loss, acc = model.evaluate(test_data, verbose=1)
        print(f"✅ Test Accuracy: {acc*100:.2f}%")

    # ── Save TFLite ───────────────────────────────────────────────────────
    print("\n💾 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(OUTPUT_DIR, "waste_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite model → {tflite_path}")

    model.save(os.path.join(OUTPUT_DIR, "waste_model.keras"))
    print(f"✅ Keras model  → {OUTPUT_DIR}/waste_model.keras")
    print("\n🎉 Training complete! Run: python app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset with train/ and val/ subfolders")
    args = parser.parse_args()
    main(args.data_dir)