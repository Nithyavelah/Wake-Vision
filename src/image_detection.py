import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.spatial import distance as dist
import math

# === CONFIGURATIONS ===
EAR_THRESHOLD = 0.22
DISPLAY_SIZE = (300, 300)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# === MEDIAPIPE SETUP ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# === EAR CALCULATION ===
def calculate_ear(landmarks, eye_indices, w, h):
    try:
        points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
        p1, p2, p3, p4, p5, p6 = points
        vertical1 = dist.euclidean(p2, p6)
        vertical2 = dist.euclidean(p3, p5)
        horizontal = dist.euclidean(p1, p4)
        return (vertical1 + vertical2) / (2.0 * horizontal)
    except:
        return None

# === IMAGE ANNOTATION ===
def annotate_image(image, label, color, ear_text=""):
    resized = cv2.resize(image, DISPLAY_SIZE)
    cv2.rectangle(resized, (10, 10), (290, 60), (0, 0, 0), -1)
    cv2.putText(resized, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    if ear_text:
        cv2.putText(resized, ear_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return resized

# === PROCESS EACH IMAGE ===
def process_image(path):
    image = cv2.imread(path)
    if image is None:
        return annotate_image(np.zeros((300, 300, 3), dtype=np.uint8), "Invalid image", (0, 255, 255))

    h, w = image.shape[:2]
    if h < 300 or w < 300:
        image = cv2.resize(image, (max(w, 300), max(h, 300)))
        h, w = image.shape[:2]

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    label, color, ear_text = "No face detected", (0, 255, 255), ""

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(landmarks, LEFT_EYE_IDX, w, h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE_IDX, w, h)

        # EAR info text
        if left_ear is not None:
            ear_text += f"L:{left_ear:.2f} "
        if right_ear is not None:
            ear_text += f"R:{right_ear:.2f}"

        # Eye status logic (handles 1 or 2 eyes)
        eye_status = []
        if left_ear is not None:
            eye_status.append(left_ear < EAR_THRESHOLD)
        if right_ear is not None:
            eye_status.append(right_ear < EAR_THRESHOLD)

        if eye_status:
            if all(eye_status):
                label, color = "ALERT: Eyes Closed!", (0, 0, 255)
            else:
                label, color = "Eyes Open", (0, 255, 0)
        else:
            label, color = "Eyes not detected", (0, 255, 255)

    return annotate_image(image, label, color, ear_text)

# === DISPLAY GRID ===
def display_grid(images, cols=3):
    rows = math.ceil(len(images) / cols)
    blank = np.zeros_like(images[0])
    while len(images) < rows * cols:
        images.append(blank)
    grid_rows = [np.hstack(images[i*cols:(i+1)*cols]) for i in range(rows)]
    return np.vstack(grid_rows)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    folder = "images_75"
    image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not image_files:
        print("No images found in the 'images/' folder.")
    else:
        annotated_images = [process_image(os.path.join(folder, file)) for file in image_files]
        final_grid = display_grid(annotated_images, cols=3)

        cv2.imshow("Wake Vision Results", final_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
