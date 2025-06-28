import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
from playsound import playsound
import threading
import os
import time

# Function to play alarm sound
def sound_alarm():
    path = os.path.abspath("beep-02.wav")
    playsound(path)

# EAR (Eye Aspect Ratio) calculation
def calculate_ear(landmarks, eye_indices):
    eye = [landmarks[i] for i in eye_indices]
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Video file instead of webcam
cap = cv2.VideoCapture("driver.mp4")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Left and right eye indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25
CLOSED_EYE_TIME_THRESHOLD = 3.0  # 3 seconds

start_time = None
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read.")
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks]

        left_ear = calculate_ear(landmarks, LEFT_EYE)
        right_ear = calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= CLOSED_EYE_TIME_THRESHOLD:
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=sound_alarm).start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            start_time = None
            alarm_on = False

    cv2.imshow("Wake Vision - Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
