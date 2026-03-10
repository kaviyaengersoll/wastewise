"""
WasteWise Flask App — Local Model Inference (No API Key Required)
=================================================================
Loads your trained TFLite or Keras model and classifies waste images.
Run: python app.py
"""

import os
import io
import json
import base64
import re
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Model & Config ──────────────────────────────────────────────────────────
MODEL_DIR    = "model_output"
TFLITE_PATH  = os.path.join(MODEL_DIR, "waste_model.tflite")
KERAS_PATH   = os.path.join(MODEL_DIR, "waste_model.keras")
CLASS_INFO_PATH = os.path.join(MODEL_DIR, "class_info.json")
IMG_SIZE     = (224, 224)

# ── Load class info ──────────────────────────────────────────────────────────
def load_class_info():
    if os.path.exists(CLASS_INFO_PATH):
        with open(CLASS_INFO_PATH) as f:
            return json.load(f)
    # Default fallback (before training)
    return {
        "class_names": [
            "e-waste", "food_waste", "leaf_waste", "metal_waste",
            "paper_waste", "plastic_bags", "plastic_bottles", "wood_waste"
        ],
        "img_size": [224, 224],
        "metadata": {}
    }

CLASS_INFO = load_class_info()
CLASS_NAMES = CLASS_INFO["class_names"]
IMG_SIZE    = tuple(CLASS_INFO.get("img_size", [224, 224]))

# ── Load model ───────────────────────────────────────────────────────────────
model = None
interpreter = None
use_tflite = False

def load_model():
    global model, interpreter, use_tflite

    if os.path.exists(TFLITE_PATH):
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            use_tflite = True
            print(f"✅ Loaded TFLite model: {TFLITE_PATH}")
            return
        except Exception as e:
            print(f"⚠️  TFLite load failed: {e}, trying Keras...")

    if os.path.exists(KERAS_PATH):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(KERAS_PATH)
            use_tflite = False
            print(f"✅ Loaded Keras model: {KERAS_PATH}")
            return
        except Exception as e:
            print(f"⚠️  Keras load failed: {e}")

    print("⚠️  No trained model found! Run train.py first.")
    print(f"   Expected: {TFLITE_PATH} or {KERAS_PATH}")

load_model()

# ── Metadata ────────────────────────────────────────────────────────────────
DEFAULT_META = {
    "e-waste": {
        "display": "E-Waste / Electronics", "biodegradable": False,
        "category": "Electronic Waste", "health_risk": "High",
        "severity": "Red", "score": 9, "decompose_time": "Indefinite (toxic)",
        "disposal": "Take to certified e-waste collection center. Never burn or throw in regular trash.",
        "fun_fact": "A single mobile phone contains up to 60 different elements, many of which are rare and toxic.",
        "upcycling": [
            {"title": "🔧 Donate for Repair", "desc": "Old devices can be refurbished for schools or low-income families."},
            {"title": "🎨 Circuit Board Art", "desc": "Old PCBs make stunning wall art or unique jewellery pieces."},
            {"title": "💻 Parts Recovery", "desc": "Salvage working RAM, fans, or screens for DIY electronics."},
            {"title": "♻️ Certified Recycler", "desc": "E-waste recyclers safely extract gold, silver, and rare metals."}
        ]
    },
    "food_waste": {
        "display": "Food / Organic Waste", "biodegradable": True,
        "category": "Organic Waste", "health_risk": "Low",
        "severity": "Green", "score": 1, "decompose_time": "1–6 months",
        "disposal": "Separate into wet waste bin. Ideal for home composting.",
        "fun_fact": "Food waste in landfills produces methane, a greenhouse gas 25x more potent than CO₂.",
        "upcycling": [
            {"title": "🌍 Compost Heap", "desc": "Layer with dry leaves and soil to make rich garden compost."},
            {"title": "🍵 Vegetable Stock", "desc": "Boil vegetable peels with spices for free cooking stock."},
            {"title": "🌻 Plant Fertilizer", "desc": "Bury banana peels near plants for a natural potassium boost."},
            {"title": "🧴 Natural Cleaner", "desc": "Soak citrus peels in vinegar for a chemical-free cleaner."}
        ]
    },
    "leaf_waste": {
        "display": "Leaf / Garden Waste", "biodegradable": True,
        "category": "Garden Waste", "health_risk": "Low",
        "severity": "Green", "score": 1, "decompose_time": "3 months – 1 year",
        "disposal": "Collect in green waste bin or compost pile. Do not burn.",
        "fun_fact": "Fallen leaves contain up to 80% of the nutrients a tree extracts from the soil during the year.",
        "upcycling": [
            {"title": "🍂 Mulch Layer", "desc": "Spread dry leaves around plants to retain moisture and suppress weeds."},
            {"title": "🌱 Leaf Compost", "desc": "Mix into compost pile to add carbon-rich material for rich humus."},
            {"title": "🎨 Leaf Art", "desc": "Press and dry leaves to create beautiful nature-inspired art prints."},
            {"title": "🏡 Garden Path", "desc": "Layer shredded leaves on garden paths as biodegradable ground cover."}
        ]
    },
    "metal_waste": {
        "display": "Metal / Tin Waste", "biodegradable": False,
        "category": "Metal Waste", "health_risk": "Low",
        "severity": "Green", "score": 4, "decompose_time": "50–200 years",
        "disposal": "Rinse and flatten cans. Place in metal/dry recycling bin.",
        "fun_fact": "Recycling one aluminium can saves enough energy to run a TV for 3 hours.",
        "upcycling": [
            {"title": "🪴 Planter Pot", "desc": "Punch drainage holes in bottom, use as stylish herb planters."},
            {"title": "🖊️ Pencil Holder", "desc": "Decorate with washi tape or paint for a custom desk organizer."},
            {"title": "🕯️ Lantern", "desc": "Punch patterns with a nail, place a tealight inside for mood lighting."},
            {"title": "🍽️ Mini BBQ Grill", "desc": "Large tins can be converted into small charcoal camping grills."}
        ]
    },
    "paper_waste": {
        "display": "Paper / Cardboard", "biodegradable": True,
        "category": "Paper Waste", "health_risk": "Low",
        "severity": "Green", "score": 2, "decompose_time": "2–6 weeks",
        "disposal": "Flatten boxes and place in paper recycling bin. Keep dry.",
        "fun_fact": "Recycling one ton of paper saves 17 trees, 7,000 gallons of water, and 463 gallons of oil.",
        "upcycling": [
            {"title": "🎨 Paper Mache Art", "desc": "Tear into strips, mix with glue and water, sculpt creative shapes."},
            {"title": "🌱 Seed Starter Pots", "desc": "Roll into cylinders, fill with soil, plant seeds directly."},
            {"title": "📦 Eco Gift Wrap", "desc": "Use newspaper or brown paper as eco-friendly gift wrap."},
            {"title": "🗂️ Desk Organizer", "desc": "Roll cardboard tubes into a honeycomb holder for pens."}
        ]
    },
    "plastic_bags": {
        "display": "Plastic Bags", "biodegradable": False,
        "category": "Plastic Waste", "health_risk": "High",
        "severity": "Red", "score": 8, "decompose_time": "500–1000 years",
        "disposal": "Return to plastic bag drop-off points (grocery stores). Never put in curbside recycling.",
        "fun_fact": "Humans use over 5 trillion plastic bags per year — that's 160,000 every second.",
        "upcycling": [
            {"title": "🧵 Plarn Weaving", "desc": "Cut into strips and weave 'plastic yarn' into reusable mats or bags."},
            {"title": "🌿 Pot Liner", "desc": "Use as drainage liner inside flower pots before adding soil."},
            {"title": "📦 Packing Filler", "desc": "Reuse as cushioning for fragile items when shipping packages."},
            {"title": "🏡 Weed Barrier", "desc": "Lay flat under garden mulch to suppress weed growth naturally."}
        ]
    },
    "plastic_bottles": {
        "display": "Plastic Bottles (PET)", "biodegradable": False,
        "category": "Plastic Waste", "health_risk": "Medium",
        "severity": "Yellow", "score": 6, "decompose_time": "400+ years",
        "disposal": "Rinse, crush, and drop in dry waste / recycling bin. Remove cap first.",
        "fun_fact": "Only 9% of all plastic ever produced has been recycled. The rest is in landfills or the ocean.",
        "upcycling": [
            {"title": "🌿 Mini Planter", "desc": "Cut top off, add soil and grow small herbs or succulents inside."},
            {"title": "🐦 Bird Feeder", "desc": "Cut holes on sides, add a stick perch and fill with birdseed."},
            {"title": "💡 Piggy Bank", "desc": "Seal top, cut a coin slot, decorate as a fun savings bank."},
            {"title": "🚿 Drip Irrigator", "desc": "Poke small holes in cap, fill with water, bury near plant roots."}
        ]
    },
    "wood_waste": {
        "display": "Wood / Timber Waste", "biodegradable": True,
        "category": "Wood Waste", "health_risk": "Low",
        "severity": "Green", "score": 2, "decompose_time": "2–10 years",
        "disposal": "Take to wood recycling facility. Untreated wood can be composted.",
        "fun_fact": "Wood waste makes up about 10–30% of construction and demolition debris in most countries.",
        "upcycling": [
            {"title": "🪑 Furniture Repair", "desc": "Small offcuts can fix or reinforce furniture legs and joints."},
            {"title": "🔥 Firewood", "desc": "Dry untreated wood makes excellent firewood for campfires."},
            {"title": "🌱 Garden Mulch", "desc": "Chip or shred into mulch to retain soil moisture in garden beds."},
            {"title": "🎨 Wood Art", "desc": "Sand and paint small pieces as name plaques, signs, or rustic decor."}
        ]
    }
}

def get_meta(class_name: str) -> dict:
    """Get metadata for a class, merging defaults with trained class info."""
    meta = DEFAULT_META.get(class_name, {})
    # Also check class_info metadata from training
    trained_meta = CLASS_INFO.get("metadata", {}).get(class_name, {})
    return {**meta, **trained_meta}


# ── Inference ────────────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img_array: np.ndarray):
    """Run inference. Returns (class_name, confidence, all_probs)."""
    if use_tflite and interpreter:
        inp_det = interpreter.get_input_details()
        out_det = interpreter.get_output_details()
        interpreter.set_tensor(inp_det[0]["index"], img_array)
        interpreter.invoke()
        probs = interpreter.get_tensor(out_det[0]["index"])[0]
    elif model:
        probs = model.predict(img_array, verbose=0)[0]
    else:
        raise RuntimeError("No model loaded. Run model_training/train.py first.")

    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs.tolist()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.json
        image_data = data.get("image", "")

        # Decode base64 image
        if "," in image_data:
            _, b64 = image_data.split(",", 1)
        else:
            b64 = image_data

        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes))
        arr = preprocess_image(img)

        # Predict
        class_name, confidence, all_probs = predict(arr)
        meta = get_meta(class_name)

        # Build top-3 predictions
        sorted_idx = np.argsort(all_probs)[::-1]
        top3 = [
            {"class": CLASS_NAMES[i], "display": DEFAULT_META.get(CLASS_NAMES[i], {}).get("display", CLASS_NAMES[i]), "prob": round(all_probs[i]*100, 1)}
            for i in sorted_idx[:3]
        ]

        result = {
            "item_name":     meta.get("display", class_name),
            "class_key":     class_name,
            "biodegradable": meta.get("biodegradable", False),
            "category":      meta.get("category", "Unknown"),
            "health_risk":   meta.get("health_risk", "Medium"),
            "severity":      meta.get("severity", "Yellow"),
            "score":         meta.get("score", 5),
            "decompose_time":meta.get("decompose_time", "Unknown"),
            "disposal":      meta.get("disposal", "Check local guidelines."),
            "fun_fact":      meta.get("fun_fact", ""),
            "upcycling":     meta.get("upcycling", []),
            "confidence":    f"{confidence*100:.1f}%",
            "confidence_raw":round(confidence*100, 1),
            "top3":          top3,
        }

        return jsonify({"success": True, "result": result})

    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e), "model_missing": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/model-status")
def model_status():
    loaded = (model is not None) or (interpreter is not None)
    return jsonify({
        "loaded": loaded,
        "type": "TFLite" if use_tflite else ("Keras" if model else "None"),
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
