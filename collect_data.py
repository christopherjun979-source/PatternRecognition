import sys, os, time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, FaceLandmarker
from mediapipe.tasks.python.vision import HandLandmarkerOptions, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import numpy as np

GESTURES = ["open_palm", "no_gesture"]
SAMPLES_PER_CLASS = 1000
OUTPUT_DIR = os.path.expanduser("~/gesture_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.expanduser("~/hand_landmarker.task")),
    running_mode=VisionTaskRunningMode.IMAGE, num_hands=2))

face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.expanduser("~/face_landmarker.task")),
    running_mode=VisionTaskRunningMode.IMAGE, num_faces=1))

FACE_DIM = 478 * 3
HAND_DIM = 21 * 3

def extract_features(frame_rgb):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    face_res = face_landmarker.detect(mp_image)
    hands_res = hand_landmarker.detect(mp_image)
    face_vec = np.zeros(FACE_DIM, dtype=np.float32)
    if face_res.face_landmarks:
        for i, p in enumerate(face_res.face_landmarks[0]):
            face_vec[i*3:i*3+3] = [p.x, p.y, p.z]
    hand_vecs = [np.zeros(HAND_DIM, dtype=np.float32), np.zeros(HAND_DIM, dtype=np.float32)]
    if hands_res.hand_landmarks:
        for idx, hand in enumerate(hands_res.hand_landmarks[:2]):
            for i, p in enumerate(hand):
                hand_vecs[idx][i*3:i*3+3] = [p.x, p.y, p.z]
    return np.concatenate([face_vec, hand_vecs[0], hand_vecs[1]])

def augment(vec):
    v = vec.copy()
    v += np.random.normal(0, 0.005, v.shape).astype(np.float32)
    if np.random.rand() > 0.5:
        v[0::3] = 1.0 - v[0::3]
    return v

print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: webcam failed"); sys.exit(1)
print("Webcam open! Starting in 3 seconds...")

for gesture in GESTURES:
    samples = []
    print(f"\n=== Collecting: {gesture} ===")
    print("Get ready...")
    time.sleep(3)
    collected = 0
    while collected < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = extract_features(frame_rgb)
        samples.append(features)
        samples.append(augment(features))
        collected += 1
        progress = int((collected / SAMPLES_PER_CLASS) * 400)
        cv2.rectangle(frame, (20, 20), (420, 55), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, 20), (20 + progress, 55), (80, 200, 120), -1)
        cv2.putText(frame, f"{gesture}  {collected}/{SAMPLES_PER_CLASS}",
                    (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.imshow("Collecting - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    arr = np.array(samples[:SAMPLES_PER_CLASS * 2], dtype=np.float32)
    save_path = os.path.join(OUTPUT_DIR, f"{gesture}.npy")
    np.save(save_path, arr)
    print(f"Saved {len(arr)} samples to {save_path}")

cap.release()
cv2.destroyAllWindows()
print("\nDone! Next step: python3 train.py")
