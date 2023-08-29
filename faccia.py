import cv2
import mediapipe as mp

# Inizializza il modulo Face di MediaPipe per il face landmark detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Inizializza la cattura video dalla webcam (0 indica la prima webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converti l'immagine in RGB poiché MediaPipe utilizza immagini RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva i landmark facciali nell'immagine
    results = face_mesh.process(rgb_frame)

    # Disegna i landmark sull'immagine originale
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Disegna i pallini dei landmark facciali
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 3, (0, 0, 0), 2)  # Colore contorno

            # Disegna linee sottili che collegano i punti dei landmark facciali
            landmarks = face_landmarks.landmark
            for connection in mp_face_mesh.FACEMESH_CONTOURS:
                start_idx, end_idx = connection
                start_point = (
                int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0]))
                end_point = (int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 1)

    cv2.imshow('Face Landmark Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
