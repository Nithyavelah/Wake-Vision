import cv2
import time
import mediapipe as mp
from scipy.spatial import distance as dist
from playsound import playsound
import threading

# Eye aspect ratio calculation function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Alarm thread
def sound_alarm(path="alarm.mp3"):
    playsound(path)

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48

# Init mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Indexes for eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Start video stream
cap = cv2.VideoCapture(0)
COUNTER = 0
ALARM_ON = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        left_eye = [(int(mesh_points[p].x * w), int(mesh_points[p].y * h)) for p in LEFT_EYE]
        right_eye = [(int(mesh_points[p].x * w), int(mesh_points[p].y * h)) for p in RIGHT_EYE]
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        for pt in left_eye + right_eye:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    threading.Thread(target=sound_alarm).start()
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("WakeVision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
