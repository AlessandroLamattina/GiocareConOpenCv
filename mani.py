import os
import cv2
import mediapipe as mp
from datetime import datetime

# Inizializza il modulo MediaPipe per il rilevamento delle mani
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(10)

# Inizializza la webcam
cap = cv2.VideoCapture(0)  # 0 indica la webcam predefinita

# Crea la cartella "mani" se non esiste
if not os.path.exists("mani"):
    os.mkdir("mani")

# Imposta i parametri del video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Ottieni la data e l'ora corrente per creare un nome univoco per il video
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = os.path.join("mani", f"video_{current_time}.avi")

out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Definisci le sequenze di landmarks delle dita per le singole lettere dell'alfabeto
alphabet_gestures = {
    'a': [0, 5, 8, 11, 12],
    'b': [0, 5, 9, 13, 17],
    'c': [0, 5, 9, 10, 11],
    'd': [0, 5, 8, 11],
    'e': [0, 5, 9, 10],
    'f': [0, 5, 9, 13],
    'g': [0, 5, 9, 10, 17],
    'h': [0, 5, 8, 11, 15],
    'i': [0, 5, 8],
    'j': [0, 5, 8, 12],
    'k': [0, 5, 9, 13, 14],
    'l': [0, 5, 8, 9],
    'm': [0, 5, 8, 11, 15, 18],
    'n': [0, 5, 8, 11, 18],
    'o': [0, 5, 8, 11, 12, 15],
    'p': [0, 5, 9, 10, 13],
    'q': [0, 5, 9, 10, 13, 17],
    'r': [0, 5, 9, 10, 17],
    's': [0, 5, 9, 10, 16],
    't': [0, 5, 8, 11, 17],
    'u': [0, 5, 8, 11, 16],
    'v': [0, 5, 8, 11, 19],
    'w': [0, 5, 8, 11, 14, 17],
    'x': [0, 5, 10, 11, 19],
    'y': [0, 5, 10, 12, 19],
    'z': [0, 5, 6, 7, 8]
}

recognized_word = ""
current_gesture = ""

def recognize_alphabet_gesture(hand_landmarks, alphabet_gestures):
    for letter, gesture in alphabet_gestures.items():
        if all(hand_landmarks[i].y < hand_landmarks[0].y for i in gesture):
            return letter
    return None

while True:
    # Cattura il frame
    ret, frame = cap.read()

    if not ret:
        break

    # Ottieni le dimensioni del frame
    h, w, _ = frame.shape
    # Converti il frame in RGB (MediaPipe utilizza l'input RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva le mani nel frame RGB
    results = hands.process(rgb_frame)

    is_left_fingers_raised = [False] * 5
    is_right_fingers_raised = [False] * 5
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            if results.multi_handedness:
                handness = results.multi_handedness[0].classification[0]
                if handness.label == "Left":
                    finger_landmarks = [
                        landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                        landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                        landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                        landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                        landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    ]
                    is_fingers_raised = is_left_fingers_raised
                else:
                    finger_landmarks = [
                        landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                        landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                        landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                        landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                        landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    ]
                    is_fingers_raised = is_right_fingers_raised

                # Calcola le distanze tra i landmarks delle dita
                for i, finger_landmark in enumerate(finger_landmarks):
                    thumb_landmark = finger_landmarks[0]
                    vertical_distance = thumb_landmark.y - finger_landmark.y

                    # Imposta il dito come sollevato se la distanza è sufficientemente grande
                    if vertical_distance > 0.15:
                        is_fingers_raised[i] = True
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
                #Aggiungi il numero del landmark all'interno del pallino
                #cv2.putText(frame, str(i), (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                thumb_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_finger_landmark = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                vertical_distance = thumb_landmark.y - middle_finger_landmark.y
        """if is_hand_open == 1:
            cv2.putText(frame, "Hand Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_hand_open == 0:
            cv2.putText(frame, "Hand Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif is_hand_open == 2:
            cv2.putText(frame, "Vitale Merda", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)"""

    # Scrivi il frame nel video
    out.write(frame)

    # Mostra il frame con le mani rilevate a schermo intero
    cv2.imshow('Hand Detection', frame)
    # Visualizza il gesto corrente
    #cv2.putText(frame, "Current Gesture: " + current_gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    finger_status = "Finger Status: "
    for is_raised in is_left_fingers_raised + is_right_fingers_raised:
        finger_status += "1" if is_raised else "0"
        print(finger_status)
    cv2.putText(frame, finger_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Esci dal ciclo quando si preme il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia il video writer, la webcam e chiudi tutte le finestre
out.release()
cap.release()
cv2.destroyAllWindows()

"""
 # Disegna landmarks sulle mani rilevate e riconosci la lettera dell'alfabeto
    if results.multi_hand_landmarks:
        is_hand_open=0
        for landmarks in results.multi_hand_landmarks:
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
                #Aggiungi il numero del landmark all'interno del pallino
                #cv2.putText(frame, str(i), (x - 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                thumb_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_finger_landmark = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                vertical_distance = thumb_landmark.y - middle_finger_landmark.y

                # Imposta il dito medio come sollevato se la distanza è sufficientemente grande

                if distance > 0.2:
                    is_hand_open = 1
                if vertical_distance > 0.1 and distance < 0.2:
                    is_hand_open = 2
                elif i < 0.2 and vertical_distance < 0.1 :
                    is_hand_open = 0



            # Riconosci la lettera dell'alfabeto basandoti sulla posizione delle dita
            recognized_letter = recognize_alphabet_gesture(landmarks.landmark, alphabet_gestures)
            if recognized_letter is not None:

                recognized_word += recognized_letter
                current_gesture = recognized_letter
        if is_hand_open == 1:
            cv2.putText(frame, "Hand Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_hand_open == 0:
            cv2.putText(frame, "Hand Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif is_hand_open == 2:
            cv2.putText(frame, "Vitale Merda", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
"""