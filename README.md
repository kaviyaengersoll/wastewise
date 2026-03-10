# 🌿 WasteWise — Local AI Waste Classifier (No API Key)

Classifies waste using a **MobileNetV2 model trained on your Kaggle dataset**.  
Runs 100% offline. No Anthropic API. No internet needed after setup.

---

## 📁 Project Structure

```
wastewise/
├── app.py                        ← Flask web server
├── requirements.txt              ← Python packages
├── run.bat                       ← Windows launcher (double-click!)
├── run.sh                        ← Mac/Linux launcher
├── model_training/
│   └── train.py                  ← Training script (MobileNetV2)
├── model_output/                 ← Created after training
│   ├── waste_model.tflite        ← Fast inference model
│   ├── waste_model.keras         ← Keras model backup
│   └── class_info.json           ← Class names + metadata
├── templates/
│   └── index.html                ← Web UI
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## 🗂 Expected Dataset Structure

Your downloaded Kaggle dataset should look like this:

```
your_dataset_folder/
├── train/
│   ├── e-waste/        (images...)
│   ├── food_waste/     (images...)
│   ├── leaf_waste/     (images...)
│   ├── metal_waste/    (images...)
│   ├── paper_waste/    (images...)
│   ├── plastic_bags/   (images...)
│   ├── plastic_bottles/(images...)
│   └── wood_waste/     (images...)
├── val/
│   └── (same 8 subfolders)
└── test/
    └── (same 8 subfolders)
```

> If your dataset uses different folder names, the training script auto-detects them.

---

## 🚀 Step-by-Step Setup in VS Code

### Step 1: Open Project in VS Code
```
File → Open Folder → select the wastewise folder
```

### Step 2: Open Terminal in VS Code
```
Terminal → New Terminal  (or Ctrl + `)
```

### Step 3: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
> ⏳ This installs TensorFlow (~500MB). Takes 2–5 minutes.

### Step 5: Train the Model
```bash
# Windows — replace with YOUR actual dataset path:
python model_training/train.py --data_dir "C:\Users\YourName\Downloads\waste_dataset"

# Mac/Linux:
python model_training/train.py --data_dir "/Users/yourname/Downloads/waste_dataset"
```

**Training takes ~30–90 minutes** depending on your computer.  
You'll see live progress with accuracy scores per epoch.  
When done, `model_output/` folder is created with your trained model.

**Expected accuracy: ~88–95%** (MobileNetV2 is very good for image classification)

### Step 6: Run the Web App
```bash
python app.py
```
Open browser → **http://localhost:5000**

---

## ⚡ Quick Launch (after first setup)

**Windows:** Double-click `run.bat`  
**Mac/Linux:** `bash run.sh`

---

## 🎯 Features

| Feature | Details |
|---|---|
| 📁 Upload Image | Drag & drop or click to browse |
| 📸 Live Camera | Real-time webcam capture |
| 🔗 URL Load | Paste any image URL |
| 🌱 Biodegradable Check | Auto-detected from class |
| ♻️ Upcycling Ideas | 4 creative DIY ideas per class |
| 📊 Confidence Score | Shows top-3 predictions with % |
| 💡 Fun Facts | Environmental facts per waste type |
| 🔒 100% Offline | No API key, no internet after setup |

---

## 🏷 8 Waste Classes (from your dataset)

| Class | Biodegradable | Severity | Risk Score |
|---|---|---|---|
| food_waste | ✅ Yes | 🟢 Green | 1/10 |
| leaf_waste | ✅ Yes | 🟢 Green | 1/10 |
| paper_waste | ✅ Yes | 🟢 Green | 2/10 |
| wood_waste | ✅ Yes | 🟢 Green | 2/10 |
| metal_waste | ❌ No | 🟢 Green | 4/10 |
| plastic_bottles | ❌ No | 🟡 Yellow | 6/10 |
| plastic_bags | ❌ No | 🔴 Red | 8/10 |
| e-waste | ❌ No | 🔴 Red | 9/10 |

---

## 🔧 Troubleshooting

**"No module named tensorflow"**
```bash
pip install tensorflow
```

**"Model not found" on web app**
- Make sure you ran `train.py` first and it completed
- Check `model_output/` folder exists with `.tflite` or `.keras` file

**Camera not working**
- Camera requires HTTPS in production (localhost is fine)
- Allow camera permissions in browser

**Training is slow**
- Install GPU version: `pip install tensorflow[and-cuda]` (needs NVIDIA GPU)
- Or use Google Colab with GPU runtime (free)

**Low accuracy**
- Increase `EPOCHS_HEAD` and `EPOCHS_FINE` in `train.py`
- Ensure dataset images are clean and properly labelled

---

## 🌐 Deploy to Production (optional)

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

For HTTPS (needed for camera): use nginx + certbot, or deploy to Railway/Render.

---

*Built for waste management automation 🌍 · MobileNetV2 + Flask + TFLite*
