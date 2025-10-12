from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import torch
import os
import shutil
from threading import Thread, Lock, Event
import time
from collections import deque
import uuid
import atexit
import face_recognition
import numpy as np
import json
from datetime import datetime
import openvino as ov

# --- MODIFICATION START ---
# Point Flask to the 'build' directory for serving the frontend
app = Flask(__name__, static_folder='../build', static_url_path='/')
# --- MODIFICATION END ---

CORS(app)

# --- Configuration ---
MODEL_PATH = r'best.pt'
DEVICE = 'cpu'  # Default to CPU
CONFIDENCE_THRESHOLD = 0.4
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), 'known_faces')
VIOLATIONS_IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'violations_images')
VIOLATION_LOG_FILE = os.path.join(os.path.dirname(__file__), 'violations.json')
VIOLATION_COOLDOWN = 10  # Seconds to wait before logging the same violation for the same person

# --- Global State ---
model = None
active_streams = {}
stream_lock = Lock()
json_lock = Lock()

# Face recognition state
known_face_encodings = []
known_face_names = []
recent_violations = {}

# PPE Class definitions
PPE_CLASSES = {
    0: {"name": "Hardhat", "color": "#3B82F6", "safe": True}, 1: {"name": "Mask", "color": "#10B981", "safe": True},
    2: {"name": "NO-Hardhat", "color": "#EF4444", "safe": False}, 3: {"name": "NO-Mask", "color": "#F59E0B", "safe": False},
    4: {"name": "NO-Safety Vest", "color": "#EC4899", "safe": False}, 5: {"name": "Person", "color": "#FBBF24", "safe": True},
    6: {"name": "Safety Cone", "color": "#8B5CF6", "safe": True}, 7: {"name": "Safety Vest", "color": "#059669", "safe": True},
    8: {"name": "Machinery", "color": "#6366F1", "safe": True}, 9: {"name": "Vehicle", "color": "#14B8A6", "safe": True}
}

# --- MODIFICATION START ---
# --- Serve React App ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')
# --- MODIFICATION END ---


def get_best_device():
    """Detects and returns the best available device for inference."""
    if torch.cuda.is_available():
        return 'cuda'
    try:
        core = ov.Core()
        devices = core.available_devices
        if 'GPU' in devices:
            return 'gpu'
    except Exception:
        pass
    return 'cpu'


# --- Utility & Face Recognition Functions ---
def load_known_faces():
    """Scans the KNOWN_FACES_DIR and loads face encodings."""
    global known_face_encodings, known_face_names
    print("👤 Loading known faces...")
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"⚠️  '{KNOWN_FACES_DIR}' created. Add face images for recognition.")
        return

    known_face_encodings.clear()
    known_face_names.clear()

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            name = os.path.splitext(filename)[0].replace('_', ' ').title()
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"  - Encoded '{name}'")
                else:
                    print(f"  - ⚠️  Warning: No face found in '{filename}'")
            except Exception as e:
                print(f"  - ❌ Error loading '{filename}': {e}")
    print(f"✅ Found and encoded {len(known_face_names)} faces.")


def log_violation(person_name, violation_type, image_of_person):
    """Logs a violation to the JSON file and saves an image of the person."""
    now = datetime.now()
    violation_id = str(uuid.uuid4())
    image_filename = f"{violation_id}.jpg"
    image_path = os.path.join(VIOLATIONS_IMAGE_DIR, image_filename)

    # Save the cropped image of the person
    cv2.imwrite(image_path, image_of_person)

    entry = {
        'id': violation_id, 'timestamp': now.isoformat(), 'name': person_name,
        'violation_type': violation_type, 'image_path': f"/violations/image/{image_filename}"
    }

    with json_lock:
        log_data = []
        if os.path.exists(VIOLATION_LOG_FILE):
            try:
                with open(VIOLATION_LOG_FILE, 'r') as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                pass
        log_data.insert(0, entry)  # Add to the beginning for newest first
        with open(VIOLATION_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=4)

    print(f"🔴 Logged violation for {person_name}: {violation_type}")


def check_association(person_box, violation_box):
    """Check if a violation is associated with a person."""
    px1, py1, px2, py2 = person_box
    vx1, vy1, vx2, vy2 = violation_box

    # Check for horizontal overlap
    if px1 > vx2 or vx1 > px2:
        return False

    # Check if violation is directly above the person (for no-hardhat)
    v_center_x = (vx1 + vx2) / 2
    is_above = py1 > vy2
    is_close_vertically = (py1 - vy2) < (py2 - py1) * 0.5  # Violation is within 50% of person's height above them
    if px1 < v_center_x < px2 and is_above and is_close_vertically:
        return True

    # Check for general intersection (IoU might be too strict, let's use intersection area)
    x_left = max(px1, vx1)
    y_top = max(py1, vy1)
    x_right = min(px2, vx2)
    y_bottom = min(py2, vy2)

    if x_right > x_left and y_bottom > y_top:
        return True

    return False


def cleanup():
    print("🧹 Cleaning up all active streams and temporary files...")
    with stream_lock:
        for stream_id in list(active_streams.keys()):
            stream = active_streams.get(stream_id)
            if stream: stream['stop_event'].set()
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR, ignore_errors=True)
    if os.path.exists(VIOLATIONS_IMAGE_DIR): shutil.rmtree(VIOLATIONS_IMAGE_DIR, ignore_errors=True)
    if os.path.exists(VIOLATION_LOG_FILE): os.remove(VIOLATION_LOG_FILE)
    print("✅ Cleanup complete.")


atexit.register(cleanup)


def stream_worker(stream_id, source_type, source_path, stop_event):
    cap = None
    try:
        cap = cv2.VideoCapture(int(source_path) if source_type == 'webcam' else source_path,
                               cv2.CAP_DSHOW if source_type == 'webcam' else None)
        if not cap.isOpened():
            print(f"❌ Worker {stream_id}: Failed to open source {source_path}")
            return

        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_deque = deque(maxlen=30)

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if source_type == 'video':
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0);
                    continue
                else:
                    break

            start_time = time.time()
            all_detections = []

            if model:
                results = model(frame, device=DEVICE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

                person_detections = []
                violation_detections = []

                for detection in results.boxes.data:
                    x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()
                    cls_info = PPE_CLASSES.get(int(class_id))
                    if not cls_info: continue

                    det_obj = {'class_name': cls_info['name'], 'confidence': float(confidence),
                               'bbox': [float(x1), float(y1), float(x2), float(y2)], 'color': cls_info['color'],
                               'safe': cls_info['safe']}
                    all_detections.append(det_obj)
                    if cls_info['name'] == 'Person':
                        person_detections.append(det_obj)
                    elif not cls_info['safe']:
                        violation_detections.append(det_obj)

                # --- Violation Logging Logic ---
                if violation_detections and person_detections:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for person_det in person_detections:
                        associated_violations = [v for v in violation_detections if
                                                 check_association(person_det['bbox'], v['bbox'])]

                        if not associated_violations: continue

                        p_x1, p_y1, p_x2, p_y2 = map(int, person_det['bbox'])
                        person_crop_image = frame[p_y1:p_y2, p_x1:p_x2]

                        if person_crop_image.size == 0: continue

                        person_crop_rgb = cv2.cvtColor(person_crop_image, cv2.COLOR_BGR2RGB)

                        # Attempt to recognize the person
                        name = "Unknown Person"
                        if known_face_names:
                            face_locations = face_recognition.face_locations(person_crop_rgb)
                            if face_locations:
                                face_encodings = face_recognition.face_encodings(person_crop_rgb, face_locations)
                                if face_encodings:
                                    matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0],
                                                                             tolerance=0.6)
                                    face_distances = face_recognition.face_distance(known_face_encodings,
                                                                                    face_encodings[0])
                                    if len(face_distances) > 0:
                                        best_match_index = np.argmin(face_distances)
                                        if matches[best_match_index]:
                                            name = known_face_names[best_match_index]

                        # Log a violation for this person (known or unknown)
                        for viol in associated_violations:
                            now = time.time()
                            log_key = (name, viol['class_name'])
                            if now - recent_violations.get(log_key, 0) > VIOLATION_COOLDOWN:
                                log_violation(name, viol['class_name'], person_crop_image)
                                recent_violations[log_key] = now

            end_time = time.time()
            fps_deque.append(1 / (end_time - start_time) if end_time > start_time else 0)

            for det in all_detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                color_bgr = tuple(int(det['color'].lstrip('#')[i:i + 2], 16) for i in (4, 2, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color_bgr, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret: continue

            with stream_lock:
                if stream_id in active_streams:
                    stats = active_streams[stream_id]['stats']
                    stats.update({
                        'fps': sum(fps_deque) / len(fps_deque) if fps_deque else 0,
                        'frame_count': stats['frame_count'] + 1,
                        'violation_count': stats['violation_count'] + len([d for d in all_detections if not d['safe']]),
                        'last_detections': all_detections,
                        'original_resolution': [height, width]
                    })
                    active_streams[stream_id]['frame'] = buffer.tobytes()

    except Exception as e:
        import traceback
        print(f"💥 Unhandled exception in worker {stream_id}: {e}\n{traceback.format_exc()}")
    finally:
        if cap: cap.release()
        print(f"✅ Worker {stream_id} finished.")


def frame_generator(stream_id):
    while True:
        with stream_lock:
            stream_data = active_streams.get(stream_id)
            if not stream_data or stream_data['stop_event'].is_set(): break
            frame = stream_data.get('frame')
        if frame: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1 / 60)


# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/upload/video', methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify({'error': 'No video file provided'}), 400
    video_file = request.files['video']
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_filename = str(uuid.uuid4()) + os.path.splitext(video_file.filename)[1]
    video_path = os.path.join(TEMP_DIR, temp_filename)
    video_file.save(video_path)
    return jsonify({'path': video_path})


@app.route('/stream/start', methods=['POST'])
def start_stream():
    data = request.json
    source_type, source_path = data.get('source_type'), data.get('source_path')
    stream_name = data.get('name', 'Unnamed Stream')  # Added name
    if not source_type or source_path is None:
        return jsonify({'error': 'source_type and source_path are required'}), 400

    stream_id = str(uuid.uuid4())
    stop_event = Event()
    with stream_lock:
        active_streams[stream_id] = {
            'thread': None, 'stop_event': stop_event, 'frame': None, 'name': stream_name,  # Added name
            'stats': {'fps': 0, 'frame_count': 0, 'violation_count': 0, 'last_detections': [],
                      'original_resolution': None}
        }
    thread = Thread(target=stream_worker, args=(stream_id, source_type, source_path, stop_event), daemon=True)
    active_streams[stream_id]['thread'] = thread
    thread.start()
    return jsonify({'stream_id': stream_id, 'name': stream_name})


@app.route('/stream/video_feed/<stream_id>')
def video_feed(stream_id):
    if stream_id not in active_streams: return "Stream not found", 404
    return Response(frame_generator(stream_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/detections/<stream_id>', methods=['GET'])
def get_stream_detections(stream_id):
    with stream_lock:
        stream = active_streams.get(stream_id)
    if not stream: return jsonify({'error': 'Stream not found'}), 404
    response_data = stream['stats'].copy()
    response_data['name'] = stream.get('name', 'Unnamed Stream')
    return jsonify(response_data)


@app.route('/streams', methods=['GET'])
def get_all_streams():
    with stream_lock:
        streams_data = []
        for stream_id, stream_info in active_streams.items():
            streams_data.append({
                'stream_id': stream_id,
                'name': stream_info.get('name', 'Unnamed Stream'),
                'stats': stream_info['stats']
            })
    return jsonify(streams_data)


@app.route('/stream/stop', methods=['POST'])
def stop_stream_endpoint():
    stream_id = request.json.get('stream_id')
    if not stream_id: return jsonify({'error': 'stream_id is required'}), 400
    with stream_lock:
        stream = active_streams.pop(stream_id, None)
    if stream:
        stream['stop_event'].set()
        print(f"🛑 Stream {stream_id} stopped.")
    return jsonify({'status': 'stopped'})


@app.route('/violations', methods=['GET'])
def get_violations():
    with json_lock:
        if not os.path.exists(VIOLATION_LOG_FILE): return jsonify([])
        try:
            with open(VIOLATION_LOG_FILE, 'r') as f:
                violations = json.load(f)
            return jsonify(violations)
        except (json.JSONDecodeError, FileNotFoundError):
            return jsonify([])


@app.route('/violations/clear', methods=['POST'])
def clear_violations():
    with json_lock:
        if os.path.exists(VIOLATION_LOG_FILE):
            os.remove(VIOLATION_LOG_FILE)
    return jsonify({'status': 'cleared'})


@app.route('/violators/unknown', methods=['GET'])
def get_unknown_violators():
    unknown = []
    if os.path.exists(VIOLATION_LOG_FILE):
        with open(VIOLATION_LOG_FILE, 'r') as f:
            try:
                violations = json.load(f)
                for v in violations:
                    if v['name'] == 'Unknown Person':
                        unknown.append(v)
            except json.JSONDecodeError:
                pass
    return jsonify(unknown)


@app.route('/face/merge', methods=['POST'])
def merge_faces():
    data = request.json
    name = data.get('name')
    violation_ids = data.get('violation_ids')

    if not name or not violation_ids:
        return jsonify({'error': 'Name and violation_ids are required'}), 400

    with json_lock:
        if not os.path.exists(VIOLATION_LOG_FILE):
            return jsonify({'error': 'Violation log not found'}), 404
        with open(VIOLATION_LOG_FILE, 'r') as f:
            violations = json.load(f)

    image_path_to_copy = None
    for v in violations:
        if v['id'] in violation_ids:
            v['name'] = name
            if not image_path_to_copy:
                image_path_to_copy = v['image_path']

    with open(VIOLATION_LOG_FILE, 'w') as f:
        json.dump(violations, f, indent=4)

    if image_path_to_copy:
        filename = "".join([c for c in name if c.isalpha() or c.isdigit() or c == ' ']).rstrip().replace(' ',
                                                                                                        '_') + '.jpg'
        source_path = os.path.join(VIOLATIONS_IMAGE_DIR, os.path.basename(image_path_to_copy))
        dest_path = os.path.join(KNOWN_FACES_DIR, filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            load_known_faces()

    return jsonify({'status': 'merged'})


@app.route('/violations/image/<filename>')
def get_violation_image(filename):
    return send_from_directory(VIOLATIONS_IMAGE_DIR, filename)


# --- Main Execution ---
if __name__ == '__main__':
    cleanup()
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(VIOLATIONS_IMAGE_DIR, exist_ok=True)
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    load_known_faces()

    DEVICE = get_best_device()

    if os.path.exists(MODEL_PATH):
        try:
            if DEVICE == 'gpu':
                print("✅ Exporting and loading model for OpenVINO...")
                model = YOLO(MODEL_PATH)
                model.export(format='openvino', device='gpu')
                model = YOLO('best_openvino_model/')
            else:
                model = YOLO(MODEL_PATH)
                model.to(DEVICE)
            print(f"✅ Model loaded successfully on {DEVICE}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None
    else:
        print(f"⚠️ Model file not found at '{MODEL_PATH}'")

    print(
        f"\n{'=' * 50}\n🚀 Safion PPE Detection Server Starting...\n🧠 Model Loaded: {'Yes' if model else 'No'}\n{'=' * 50}\n")
    app.run(host='0.0.0.0', port=5000)