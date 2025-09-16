import cv2
import mediapipe as mp
import numpy as np
import pygame 

# --- Initialize Pygame Mixer ---
pygame.mixer.init()
pygame.mixer.music.load('alarm.mp3') 

# --- Constants ---
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 20

# --- Function to Calculate EAR ---
def calculate_ear(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    vertical_dist1 = np.linalg.norm(p2 - p6)
    vertical_dist2 = np.linalg.norm(p3 - p5)
    horizontal_dist = np.linalg.norm(p1 - p4)
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

# --- Landmark Indices ---
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# --- Initialize Webcam and Counter ---
cap = cv2.VideoCapture(0)
FRAME_COUNTER = 0

# --- Initialize Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            avg_ear = (calculate_ear(face_landmarks.landmark, LEFT_EYE_INDICES) +
                       calculate_ear(face_landmarks.landmark, RIGHT_EYE_INDICES)) / 2.0
            
            ear_text = f"EAR: {avg_ear:.2f}"
            cv2.putText(image, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if avg_ear < EAR_THRESHOLD:
                FRAME_COUNTER += 1
                if FRAME_COUNTER >= CONSECUTIVE_FRAMES:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(-1)
                    

                    # Create a red overlay
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), -1)
                    
                    # Blend the overlay with the original image
                    alpha = 0.4 # Transparency factor
                    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                    # Flashing text logic: text appears for 10 frames, then disappears for 10
                    if FRAME_COUNTER % 20 < 10:
                        cv2.putText(image, "!!DANGER!!", (150, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                        cv2.putText(image, "!!WAKE UP!!", (150, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            else:
                FRAME_COUNTER = 0
                pygame.mixer.music.stop()

        cv2.imshow('Drowsiness Detector', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
pygame.quit()
cv2.destroyAllWindows()