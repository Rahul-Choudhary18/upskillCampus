import os
import glob
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def extract_features(roi):
    """Extract color (HSV histogram), ExG (Excess Green), and texture features from an image crop."""
    if roi is None or roi.size == 0:
        return None
    
    # Resize ROI for uniform feature extraction
    roi_resized = cv2.resize(roi, (64, 64))
    
    # 1. Color features in HSV space
    hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    
    # 2. Excess Green Index (ExG = 2G - R - B)
    b, g, r = cv2.split(roi_resized.astype(np.float32))
    exg = 2 * g - r - b
    exg_mean = np.mean(exg)
    exg_std = np.std(exg)
    
    # 3. Color channel means and stds
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    b_std, g_std, r_std = np.std(b), np.std(g), np.std(r)
    
    # Combine feature vector
    features = np.hstack([
        h_hist / (np.sum(h_hist) + 1e-6),
        s_hist / (np.sum(s_hist) + 1e-6),
        v_hist / (np.sum(v_hist) + 1e-6),
        [exg_mean, exg_std, b_mean, g_mean, r_mean, b_std, g_std, r_std]
    ])
    return features

def build_dataset(data_dir):
    images = glob.glob(os.path.join(data_dir, '*.jpeg')) + glob.glob(os.path.join(data_dir, '*.jpg'))
    X = []
    y = []
    
    print(f"Processing {len(images)} dataset images...")
    for img_path in images:
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        if not os.path.exists(txt_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w, _ = img.shape
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x_center, y_center, bw, bh = map(float, parts[1:])
                    
                    # Convert normalized YOLO to pixel bounding box
                    xmin = max(0, int((x_center - bw / 2) * w))
                    ymin = max(0, int((y_center - bh / 2) * h))
                    xmax = min(w, int((x_center + bw / 2) * w))
                    ymax = min(h, int((y_center + bh / 2) * h))
                    
                    roi = img[ymin:ymax, xmin:xmax]
                    feat = extract_features(roi)
                    if feat is not None:
                        X.append(feat)
                        y.append(cls_id)
                        
    return np.array(X), np.array(y)

def train_and_save():
    data_dir = os.path.join('dataset', 'agri_data', 'data')
    X, y = build_dataset(data_dir)
    print(f"Extracted {len(X)} samples. Crop (0): {np.sum(y == 0)}, Weed (1): {np.sum(y == 1)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Crop', 'Weed']))
    
    model_path = 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_and_save()
