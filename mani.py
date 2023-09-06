import os
import cv2
import mediapipe as mp
from datetime import datetime
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(10)
cap = cv2.VideoCapture(0)
cap2 =cv2.VideoCapture(1)
if not os.path.exists("mani"):
    os.mkdir("mani")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = os.path.join("mani", f"video_{current_time}.avi")
out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    lis_hand_open=0
    out.write(frame)
    """for is_raised in is_left_fingers_raised + is_right_fingers_raised:
        finger_status += "1" if is_raised else "0"
        print(finger_status)
    cv2.putText(frame, finger_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)"""
    if results.multi_hand_landmarks:
        is_hand_open = 0
        for landmarks in results.multi_hand_landmarks:
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
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Colore interno (fill)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), 2)  # Colore contorno
                thumb_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_towrist = landmarks.landmark[mp_hands.HandLandmark.WRIST].y - thumb_landmark.y
                thumb_to_index_distance = thumb_landmark.y - landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                thumb_to_middle_distance = thumb_landmark.y - landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                thumb_to_ring_distance = thumb_landmark.y - landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                thumb_to_pinky_distance = thumb_landmark.y - landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                center_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                center_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                thumb_tip_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                wrist_landmark = landmarks.landmark[mp_hands.HandLandmark.WRIST]

                wrist_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                thumb_tip_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                wrist_landmark = landmarks.landmark[mp_hands.HandLandmark.WRIST]
                knuckle_landmark = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]  # Puoi scegliere qualsiasi nocca di dito

                # Calcola il centro approssimativo del palmo come punto medio tra il polso e la nocca della mano
                palm_x = (wrist_landmark.x + knuckle_landmark.x) / 2
                palm_y = (wrist_landmark.y + knuckle_landmark.y) / 2
                palm_z = (wrist_landmark.z + knuckle_landmark.z) / 2

                # Calcola la distanza euclidea tra la punta del pollice e il centro approssimativo del palmo
                distanza = math.sqrt((thumb_tip_landmark.x - palm_x) ** 2 +
                                     (thumb_tip_landmark.y - palm_y) ** 2 +
                                     (thumb_tip_landmark.z - palm_z) ** 2)

                if wrist_x > 0.6:
                    if thumb_to_middle_distance > 0.1 and thumb_to_index_distance > 0.1 and thumb_to_ring_distance > 0.1 and thumb_to_pinky_distance > 0.1 and thumb_towrist < 0.2:
                        lis_hand_open = 1  # aperta
                    elif thumb_to_middle_distance < 0.1 and thumb_to_index_distance < 0.1 and thumb_to_ring_distance < 0.1 and thumb_to_pinky_distance < 0.1 and thumb_towrist < 0.2:
                        lis_hand_open = 0  # chiusa
                    elif thumb_to_middle_distance < 0.1 and thumb_to_index_distance > 0.1 and thumb_to_ring_distance < 0.1 and thumb_to_pinky_distance < 0.1 and thumb_towrist < 0.2:
                        lis_hand_open = 3  # indice
                    elif thumb_to_middle_distance > 0.1 and thumb_to_index_distance < 0.1 and thumb_to_ring_distance < 0.1 and thumb_to_pinky_distance < 0.1 and thumb_towrist < 0.2:
                        lis_hand_open = 4  # medio:
                    elif thumb_to_middle_distance < 0.1 and thumb_to_index_distance < 0.1 and thumb_to_ring_distance < 0.1 and thumb_to_pinky_distance < 0.1 and thumb_towrist < 0.2:
                        lis_hand_open = 2  # pollice
                    elif thumb_to_middle_distance < 0.1 and thumb_to_index_distance < 0.1 and thumb_to_ring_distance > 0.1 and thumb_to_pinky_distance < 0.1 and thumb_towrist < 0.2:
                        lis_hand_open = 5  # anulare
                    elif thumb_to_middle_distance < 0.1 and thumb_to_index_distance < 0.1 and thumb_to_ring_distance < 0.1 and thumb_to_pinky_distance > 0.1 and thumb_towrist < 0.2:
                        lis_hand_open = 6  # mignolo
                else:
                    if thumb_to_middle_distance > 0.1 and thumb_to_index_distance > 0.1 and thumb_to_ring_distance > 0.1 and thumb_to_pinky_distance > 0.1 and thumb_towrist < 0.2:
                        is_hand_open = 1  # aperta
                    elif thumb_to_middle_distance < 0.1 and thumb_to_index_distance < 0.1 and thumb_to_ring_distance < 0.1 and thumb_to_pinky_distance < 0.1 and thumb_towrist < 0.2:
                        is_hand_open = 0  # chiusa
                    elif thumb_to_middle_distance < 0.1 and thumb_to_index_distance > 0.1 and thumb_to_ring_distance < 0.1 and thumb_to_pinky_distance < 0.1 and thumb_towrist < 0.2:
                        is_hand_open = 3  # indice
                    elif thumb_to_middle_distance > 0.1 and thumb_to_index_distance < 0.1 and thumb_to_ring_distance < 0.1 and thumb_to_pinky_distance < 0.1 and thumb_towrist < 0.2:
                        is_hand_open = 4  # medio:
                    elif thumb_to_middle_distance < 0.1 and thumb_to_index_distance < 0.1 and thumb_to_ring_distance < 0.1 and thumb_to_pinky_distance < 0.1 and thumb_towrist < 0.2:
                        is_hand_open = 2  # pollice
                    elif thumb_to_middle_distance < 0.1 and thumb_to_index_distance < 0.1 and thumb_to_ring_distance > 0.1 and thumb_to_pinky_distance < 0.1 and thumb_towrist < 0.2:
                        is_hand_open = 5  # anulare
                    elif thumb_to_middle_distance < 0.1 and thumb_to_index_distance < 0.1 and thumb_to_ring_distance < 0.1 and thumb_to_pinky_distance > 0.1 and thumb_towrist < 0.2:
                        is_hand_open = 6  # mignolo
                if distanza > 0.1:
                    print(distanza, "ok")
                else:
                    print(distanza)


            if lis_hand_open == 1:
                cv2.putText(frame, "Mano aperta", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif lis_hand_open == 0:
                cv2.putText(frame, "Mano chiusa", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif lis_hand_open == 3:
                cv2.putText(frame, "Indice", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif lis_hand_open == 4:
                cv2.putText(frame, "Medio", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif lis_hand_open == 5:
                cv2.putText(frame, "Anulare", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif lis_hand_open == 6:
                cv2.putText(frame, "Mignolo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif lis_hand_open == 2:
                cv2.putText(frame, "Pollice", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if is_hand_open == 1:
                cv2.putText(frame, "              Mano aperta", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_hand_open == 0:
                cv2.putText(frame, "              Mano chiusa", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif is_hand_open == 3:
                cv2.putText(frame, "              Indice", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif is_hand_open == 4:
                cv2.putText(frame, "              Medio", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif is_hand_open == 5:
                cv2.putText(frame, "              Anulare", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif is_hand_open == 6:
                cv2.putText(frame, "              Mignolo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif is_hand_open == 2:
                cv2.putText(frame, "              Pollice", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Detection', frame)
    # Esci dal ciclo quando si preme il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap2.release()
cap.release()
cv2.destroyAllWindows()

"""
 # Disegna landmarks sulle mani rilevate e riconosci la lettera dell'alfabeto
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
        if is_hand_open == 1:
            cv2.putText(frame, "Hand Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_hand_open == 0:
            cv2.putText(frame, "Hand Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif is_hand_open == 2:
            cv2.putText(frame, "Vitale Merda", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)"""