🌿 Crop & Weed Detection System 
PythonFlaskOpenCVscikit-learn
License

An end-to-end computer vision and machine learning platform designed for precision agriculture. The system automatically detects sesame crops and competing weeds in agricultural field imagery, generates targeted spray nozzle mapping, and quantifies pesticide reduction to minimize environmental impact and reduce farming costs.

🌟 Key Features
🌿 Crop & Weed Classification: Classifies green vegetation into Crops (e.g., Sesame) and Weeds using an optimized Random Forest Classifier trained on color histograms, color statistics, and the Excess Green Index (
ExG
=
2
G
−
R
−
B
ExG=2G−R−B).
📦 Dual Detection Mode:
Dataset Mode: Parses YOLO annotation files (.txt) for high-accuracy ground truth bounding box evaluations.
Custom Image Upload: Automatically segments green vegetation via HSV thresholding & morphological filters, extracting features on-the-fly for real-time inference.
🎯 Targeted Spray Simulation:
Calculates target nozzle crosshairs and spray radiuses over detected weeds.
Computes pesticide volume saved (
L/ha
L/ha) compared to traditional blanket spraying (up to 70–78% reduction).
Dynamic parameters for tractor speed (
km/h
km/h), nozzle count, and system response latency (
∼
18
ms
∼18ms).
📊 Interactive Web Dashboard: Modern UI featuring live image uploads, dataset browser, precision spray visualization, and interactive simulation controls.
📁 Data Pipeline Tracker: Integrated dataset analytics showing pipeline stages from raw high-resolution field photos (4000x3000) to augmentation (Keras) and 2,072 YOLO annotations.
📐 Machine Learning & Vision Pipeline
Raw Image ──► HSV Conversion & ExG Calculation ──► Morphological Segmentation ──► ROI Feature Extraction ──► Random Forest Classifier ──► Precision Spray Map
Pre-processing & Color Space Conversion:
Converts RGB images to HSV color space.
Computes Excess Green Index (
ExG
=
2
G
−
R
−
B
ExG=2G−R−B) to isolate plant vegetation from soil backgrounds.
Feature Extraction:
Color Histograms: 16-bin normalized histograms across H, S, and V channels.
ExG Statistics: Mean and standard deviation of ExG values.
Color Statistics: Mean and standard deviation across B, G, and R channels.
Classification:
Random Forest model (n_estimators=100, max_depth=12) classifies bounding boxes/ROIs into Crop (Class 0) or Weed (Class 1).
📁 Repository Structure


├── app.py                  # Flask web server and REST API routes
├── detector.py             # Core vision & detection engine (WeedDetector class)
├── dataset_loader.py       # Dataset analytics and sample manager
├── train_classifier.py     # Script to train and export the Random Forest model
├── model.pkl               # Serialized trained classifier model
├── dataset/
│   ├── classes.txt         # Class labels (0: crop, 1: weed)
│   └── agri_data/data/     # Field images (.jpeg/.jpg) and YOLO annotations (.txt)
├── static/
│   ├── css/                # Custom stylesheets for the web dashboard
│   └── js/                 # Client-side interactions and API fetch logic
└── templates/
    └── index.html          # Dashboard HTML view
🚀 Quick Start Guide
1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

2. Clone the Repository
bash


git clone https://github.com/your-username/crop-weed-detection.git
cd crop-weed-detection
3. Set Up Virtual Environment & Install Dependencies
bash


# Create virtual environment
python -m venv venv
# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
# Install required packages
pip install flask opencv-python numpy scikit-learn pillow
4. (Optional) Train the Machine Learning Model
To retrain the Random Forest model on the dataset:

bash


python train_classifier.py
This will process the annotations in dataset/agri_data/data/, extract feature vectors, train the Random Forest classifier, and update model.pkl.

5. Run the Flask Web Application
bash


python app.py
Open your browser and navigate to: http://localhost:5000

🌐 API Documentation
Endpoint	Method	Description
/	GET	Renders the interactive web dashboard
/api/stats	GET	Returns dataset overview and pipeline step metrics
/api/samples	GET	Fetches sample dataset images with crop/weed counts
/api/image/<filename>	GET	Serves static images from the dataset
/api/detect	POST	Accepts uploaded image file or dataset_filename; returns annotated base64 images & detection metrics
/api/simulate_spray	POST	Accepts nozzles, speed, and savings_percent; returns spray rates, chemical savings, and environmental score
🛠️ Built With
Python - Core application logic
Flask - Web framework & API backend
OpenCV - Computer vision, HSV thresholding, contour segmentation
scikit-learn - Machine learning classification (Random Forest)
HTML5 / CSS3 / JavaScript - Interactive web interface
