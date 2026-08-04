# 🌿 Crop & Weed Detection System

An end-to-end computer vision and machine learning platform for precision agriculture. The system automatically detects **sesame crops** and **weeds** in field images, generates targeted spray nozzle mapping, and calculates pesticide savings.

---

## ✨ Features

- **🌿 Smart Classification:** Distinguishes crops from weeds using a Random Forest classifier paired with color features and the Excess Green Index ($\text{ExG} = 2G - R - B$).
- **📦 Dual Detection Modes:**
  - **Dataset Mode:** Evaluates ground truth annotations using YOLO `.txt` files.
  - **Live Mode:** Real-time green vegetation segmentation via HSV filtering and custom image uploads.
- **🎯 Targeted Spray Simulation:** Maps nozzle crosshairs over weeds and calculates chemical savings (up to 70–78% reduction).
- **📊 Interactive Dashboard:** Web UI with live upload, spray visualization, and dataset metrics.

---

## 🛠️ Machine Learning Pipeline

> **Raw Image** ➔ **HSV & ExG Filtering** ➔ **Segmentation** ➔ **Feature Extraction** ➔ **Random Forest Model** ➔ **Spray Map**

### Key Components
1. **Pre-processing:** Converts images to HSV color space and computes ExG to separate plants from soil.
2. **Feature Extraction:** Extracts 16-bin color histograms, ExG statistics, and RGB channel values.
3. **Classification:** Uses a Random Forest classifier (`n_estimators=100`, `max_depth=12`) to label regions as **Crop (0)** or **Weed (1)**.

---

## 📁 Repository Structure

```text
├── app.py                  # Flask web server & API routes
├── detector.py             # Computer vision & WeedDetector engine
├── dataset_loader.py       # Dataset analytics manager
├── train_classifier.py     # Script to train Random Forest model
├── model.pkl               # Trained model file
├── dataset/                # Images and YOLO annotations
├── static/                 # CSS & JavaScript files
└── templates/              # Dashboard HTML template
```
 # Quick Start
 
1. Clone & Setup
Bash
git clone [https://github.com/your-username/crop-weed-detection.git](https://github.com/your-username/crop-weed-detection.git)
cd crop-weed-detection
2. Create Virtual Environment
Bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
3. Install Dependencies
Bash
pip install flask opencv-python numpy scikit-learn pillow
4. Run the App
Bash
python app.py
Open your browser and navigate to http://localhost:5000.

(Optional) Retrain the model anytime with: python train_classifier.py

🌐 API Overview
Method	Endpoint	Description
GET	/	Renders dashboard
GET	/api/stats	Pipeline & dataset overview
GET	/api/samples	Dataset sample metadata
POST	/api/detect	Runs detection on uploaded or dataset image
POST	/api/simulate_spray	Computes nozzle spray logic & savings
🧰 Tech Stack
Backend: Python, Flask

Computer Vision: OpenCV, Pillow, NumPy

Machine Learning: scikit-learn (Random Forest)

Frontend: HTML5, CSS3, JavaScript
