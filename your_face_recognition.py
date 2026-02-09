import cv2
from deepface import DeepFace
from datetime import datetime
import csv
import os

# -------------------- ATTENDANCE FUNCTION --------------------

def mark_attendance(name):
    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        writer.writerow([name, date, time])
        print(f"[✔] Attendance marked for {name} at {time}")

# -------------------- LOAD KNOWN FACES --------------------

KNOWN_DIR = "known_faces"

if not os.path.exists(KNOWN_DIR):
    print("❌ ERROR: known_faces folder not found!")
    exit()

print("[ℹ] Loading known faces...")

# DeepFace automatically uses them during matching
known_faces_list = os.listdir(KNOWN_DIR)
if len(known_faces_list) == 0:
    print("❌ No images found in known_faces folder!")
    exit()

print("[✔] Faces loaded:", known_faces_list)

# -------------------- START CAMERA --------------------

cap = cv2.VideoCapture(0)
marked_today = set()    # prevent duplicate attendance in one session

print("[ℹ] System Started. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera Not Working!")
        break

    # Save frame temporarily (DeepFace requires file or array)
    cv2.imwrite("temp.jpg", frame)

    try:
        # Face recognition using DeepFace
        results = DeepFace.find(
            img_path="temp.jpg",
            db_path="known_faces",
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False   # prevents detection errors
        )

        if isinstance(results, list) and len(results) > 0 and len(results[0]) > 0:
            # Get the matched identity
            best_match = results[0]["identity"][0]
            name = os.path.basename(best_match).split(".")[0]

            # Display name on screen
            cv2.putText(frame, name, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Mark attendance only once per session
            if name not in marked_today:
                mark_attendance(name)
                marked_today.add(name)

        else:
            cv2.putText(frame, "Unknown", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    except Exception as e:
        print("Error:", e)

    # Show camera
    cv2.imshow("Face Recognition + Attendance", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()