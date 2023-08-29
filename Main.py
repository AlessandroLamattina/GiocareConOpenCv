
import os
import cv2
import mediapipe as mp
from datetime import datetime

# Inizializza il modulo MediaPipe per il rilevamento delle mani
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# Inizializza il modulo Face di MediaPipe per il face landmark detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
landmark_indices_to_exclude = list(range(0, 33)) + list(range(61, 76)) + list(range(136, 196))

# Inizializza la webcam
cap = cv2.VideoCapture(0)


if not os.path.exists("video"):
    os.mkdir("video")

# Imposta i parametri del video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Ottieni la data e l'ora corrente per creare un nome univoco per il video
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = os.path.join("mani", f"video_{current_time}.avi")

out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while cap.isOpened():
    # Cattura il frame
    ret, frame = cap.read()

    if not ret:
        break

    # Ottieni le dimensioni del frame
    h, w, _ = frame.shape
    # Converti il frame in RGB (MediaPipe utilizza l'input RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva le mani nel frame RGB
    results_hands = hands.process(rgb_frame)
    results_face = face_mesh.process(rgb_frame)

    if results_hands.multi_hand_landmarks:
        is_hand_open = 0
        for landmarks in results_hands.multi_hand_landmarks:
            # Disegna le connessioni tra i landmarks delle dita
            thumb_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_landmark = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = ((thumb_landmark.x - index_landmark.x) ** 2 + (thumb_landmark.y - index_landmark.y) ** 2) ** 0.5
            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                x0, y0 = int(landmarks.landmark[connection[0]].x * w), int(landmarks.landmark[connection[0]].y * h)
                x1, y1 = int(landmarks.landmark[connection[1]].x * w), int(landmarks.landmark[connection[1]].y * h)
                cv2.line(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)

            for i, landmark in enumerate(landmarks.landmark):
                x, y = int(landmark.x * w), int(landmark.y * h)
                # Disegna il pallino con un contorno
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Colore interno (fill)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), 2)  # Colore contorno
                # Aggiungi il numero del landmark all'interno del pallino
                # cv2.putText(frame, str(i), (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                thumb_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_finger_landmark = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                vertical_distance = thumb_landmark.y - middle_finger_landmark.y

                # Imposta il dito medio come sollevato se la distanza è sufficientemente grande

                if distance > 0.2:
                    is_hand_open = 1
                if vertical_distance > 0.1 and distance < 0.2:
                    is_hand_open = 2
                elif i < 0.2 and vertical_distance < 0.1:
                    is_hand_open = 0
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            # Disegna i pallini dei landmark facciali
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # Disegna linee sottili che collegano i punti dei landmark facciali
            landmarks = face_landmarks.landmark
            for connection in mp_face_mesh.FACEMESH_CONTOURS:
                start_idx, end_idx = connection
                start_point = (
                    int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0]))
                end_point = (int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 1)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results_pose = pose.process(rgb_frame)
        if results_pose.pose_landmarks:
            for landmark_idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
                if landmark_idx not in landmark_indices_to_exclude:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    cv2.imshow('Face Landmark Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia il video writer, la webcam e chiudi tutte le finestre
out.release()
cap.release()
cv2.destroyAllWindows()