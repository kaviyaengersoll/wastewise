#!/bin/bash
echo ""
echo " =================================="
echo "  WasteWise - Setup & Launch"
echo " =================================="
echo ""

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# Install deps
echo "[3/4] Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo " Choose an option:"
echo "  1. Train the model (first time setup)"
echo "  2. Run the web app (after training)"
echo ""
read -p "Enter 1 or 2: " choice

if [ "$choice" = "1" ]; then
    echo ""
    read -p "Enter full path to your dataset folder: " data_dir
    echo ""
    echo "[4/4] Starting training... (this may take 30-90 minutes)"
    python model_training/train.py --data_dir "$data_dir"
    echo ""
    echo "Training done! Now run this script again and choose option 2."
else
    echo "[4/4] Starting WasteWise web app..."
    echo " Open your browser at: http://localhost:5000"
    echo " Press Ctrl+C to stop."
    echo ""
    python app.py
fi
