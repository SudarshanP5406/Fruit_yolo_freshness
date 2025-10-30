from flask import Flask, render_template, request, Response
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import os
import shutil

app = Flask(__name__)

# ======= PATHS =======
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ======= MODEL =======
MODEL_PATH = 'best (2).pt'
model = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


# ========== HOME PAGE ==========
@app.route('/', methods=['GET', 'POST'])
def index():
    label = None
    result_image = None

    if request.method == 'POST':
        # Clear old images
        for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

        # Get file
        file = request.files.get('file')
        if not file:
            return render_template('index.html', message='No file selected')

        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        if not model:
            return render_template('index.html', message='Model not found')

        # YOLO Prediction
        img = cv2.imread(upload_path)
        results = model(img)
        annotated = results[0].plot()

        result_path = os.path.join(RESULT_FOLDER, f"result_{filename}")
        cv2.imwrite(result_path, annotated)

        label = results[0].names[int(results[0].boxes.cls[0])] if len(results[0].boxes.cls) else "No detection"
        result_image = result_path

    return render_template('index.html', label=label, result_image=result_image)


# ========== CAMERA STREAM ==========
def generate_frames():
    if not model:
        raise RuntimeError("Model not loaded")

    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        annotated = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/camera_feed')
def camera_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ========== CAMERA PAGE ==========
@app.route('/camera')
def camera_page():
    return render_template('camera.html')


if __name__ == '__main__':
    app.run(debug=True)
