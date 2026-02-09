from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
from datetime import datetime
import csv
import os

app = Flask(__name__)

def mark_attendance(name):
    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        now = datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
        print(f"[✔] Attendance marked for {name}")

KNOWN_DIR = "known_faces"
marked_today = set()

# Load known faces (just info message)
known_faces_list = os.listdir(KNOWN_DIR)
print("Loaded faces:", known_faces_list)

# ---------------- CAMERA STREAM ---------------- #

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        cv2.imwrite("temp.jpg", frame)

        try:
            results = DeepFace.find(
                img_path="temp.jpg",
                db_path="known_faces",
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )

            if isinstance(results, list) and len(results) > 0 and len(results[0]) > 0:
                best_match = results[0]["identity"][0]
                name = os.path.basename(best_match).split(".")[0]

                cv2.putText(frame, name, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                if name not in marked_today:
                    mark_attendance(name)
                    marked_today.add(name)

            else:
                cv2.putText(frame, "Unknown", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        except Exception as e:
            print("Error:", e)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
