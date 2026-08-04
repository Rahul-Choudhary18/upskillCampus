import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from detector import WeedDetector
from dataset_loader import DatasetLoader

app = Flask(__name__, template_folder='templates', static_folder='static')

detector = WeedDetector()
dataset_loader = DatasetLoader()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    return jsonify(dataset_loader.get_stats())

@app.route('/api/samples')
def get_samples():
    limit = request.args.get('limit', default=24, type=int)
    samples = dataset_loader.get_sample_list(limit=limit)
    return jsonify(samples)

@app.route('/api/image/<path:filename>')
def serve_dataset_image(filename):
    data_dir = os.path.join(app.root_path, 'dataset', 'agri_data', 'data')
    return send_from_directory(data_dir, filename)

@app.route('/api/detect', methods=['POST'])
def run_detection():
    try:
        # Check if an uploaded file is present
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            img_bytes = file.read()
            result = detector.process(img_bytes=img_bytes)
            return jsonify(result)
            
        # Check if JSON payload with dataset_filename is present
        data = request.get_json(silent=True)
        if data and 'dataset_filename' in data:
            filename = data['dataset_filename']
            result = detector.process(dataset_img_name=filename)
            return jsonify(result)
            
        # Default sample if nothing sent
        result = detector.process(dataset_img_name='agri_0_1009.jpeg')
        return jsonify(result)
        
    except Exception as e:
        print(f"[App Error] Detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate_spray', methods=['POST'])
def simulate_spray():
    data = request.get_json() or {}
    nozzle_count = data.get('nozzles', 8)
    speed_kmh = data.get('speed', 12.0)
    
    # Calculate spray response time & chemical saved
    savings = data.get('savings_percent', 78.4)
    blanket_rate = 25.0 # Liters per hectare
    targeted_rate = round(blanket_rate * (1.0 - savings / 100.0), 2)
    
    return jsonify({
        'status': 'active',
        'nozzles_active': nozzle_count,
        'tractor_speed_kmh': speed_kmh,
        'response_time_ms': 18,
        'blanket_rate_l_ha': blanket_rate,
        'targeted_rate_l_ha': targeted_rate,
        'chemical_saved_l_ha': round(blanket_rate - targeted_rate, 2),
        'environmental_impact_score': '9.4 / 10'
    })

if __name__ == '__main__':
    print("Starting Crop & Weed Detection System Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
