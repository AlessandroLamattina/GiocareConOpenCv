import cv2
import mediapipe as mp

# Inizializza la rilevazione della postura di Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Apri lo streaming video dalla webcam
cap = cv2.VideoCapture(0)

# Verifica se la cattura è stata aperta correttamente
if not cap.isOpened():
    print("Impossibile aprire la cattura video.")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converte il frame in RGB (mediapipe lavora con immagini RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Esegue la rilevazione della postura sul frame RGB
        results = pose.process(rgb_frame)

        # Disegna i punti della postura sul frame originale
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Mostra il frame elaborato in tempo reale
        cv2.imshow('Posture Detection', frame)

        # Interrompi il ciclo quando viene premuto il tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Rilascia la cattura e chiudi le finestre
cap.release()
cv2.destroyAllWindows()
