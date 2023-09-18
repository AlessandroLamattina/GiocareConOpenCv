import os
import cv2
import mediapipe as mp
from datetime import datetime
import math
import numpy as np
def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
    y_margin = 0.1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        lis_hand_open=0
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
                    conditions_met = 0
                    sconditions_met = 0
                    sum = 0
                    if wrist_x < 0.6:
                        if thumb_to_middle_distance > y_margin:
                            conditions_met += 1
                        if thumb_to_index_distance > y_margin:
                            conditions_met += 1
                        if thumb_to_ring_distance > y_margin:
                            conditions_met += 1
                        if thumb_to_pinky_distance > y_margin:
                            conditions_met += 1
                        if distanza > 0.18:
                            conditions_met +=1
                        cv2.putText(frame, f"Dita su mano dx: {conditions_met}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                    if wrist_x > 0.6:
                        if thumb_to_middle_distance > y_margin:
                            sconditions_met += 1
                        if thumb_to_index_distance > y_margin:
                            sconditions_met += 1
                        if thumb_to_ring_distance > y_margin:
                            sconditions_met += 1
                        if thumb_to_pinky_distance > y_margin:
                            sconditions_met += 1
                        if distanza > 0.18:
                            sconditions_met += 1
                        cv2.putText(frame, f"Dita su mano sx: {sconditions_met}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)

                    if wrist_x > 0.6:
                        if sconditions_met == 0:
                            lis_hand_open = 0  # chiusa
                        else:
                            if thumb_to_middle_distance > y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance > y_margin and thumb_to_pinky_distance > y_margin and distanza > 0.18:
                                lis_hand_open = 1  # aperta
                            elif thumb_to_middle_distance < y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance < y_margin and distanza < 0.18:
                                lis_hand_open = 3  # indice
                            elif thumb_to_middle_distance > y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance < y_margin and distanza < 0.18:
                                lis_hand_open = 4  # medio:
                            elif thumb_to_middle_distance < y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance < y_margin and distanza > 0.18:
                                lis_hand_open = 2  # pollice
                            elif thumb_to_middle_distance < y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance > y_margin and thumb_to_pinky_distance < y_margin and distanza < 0.18:
                                lis_hand_open = 5  # anulare
                            elif thumb_to_middle_distance < y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance > y_margin and distanza < 0.18:
                                lis_hand_open = 6  # mignolo
                            elif thumb_to_middle_distance > y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance < y_margin and distanza < 0.18:
                                lis_hand_open = 7  #pace
                            elif thumb_to_middle_distance < y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance > y_margin and distanza > 0.18:
                                lis_hand_open = 8  #pace
                            else:
                                lis_hand_open = 11

                        if lis_hand_open == 1:
                            cv2.putText(frame, "Mano sinistra aperta", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                        elif lis_hand_open == 0:
                            cv2.putText(frame, "Mano sinistra chiusa", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
                        elif lis_hand_open == 3:
                            cv2.putText(frame, "Indice Sinistro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif lis_hand_open == 4:
                            cv2.putText(frame, "Medio Sinistro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif lis_hand_open == 5:
                            cv2.putText(frame, "Anulare Sinistro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                        elif lis_hand_open == 6:
                            cv2.putText(frame, "Mignolo Sinistro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                        elif lis_hand_open == 2:
                            cv2.putText(frame, "Pollice Sinistro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                        elif lis_hand_open == 7:
                            cv2.putText(frame, "Pace sx", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif lis_hand_open == 8:
                            cv2.putText(frame, "Rock sx", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        elif lis_hand_open == 11:
                                cv2.putText(frame, f"Dita su: {sconditions_met}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)

                    if wrist_x < 0.6:
                        if conditions_met == 0:
                            is_hand_open = 0  # chiusa
                        else:
                            if thumb_to_middle_distance > y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance > y_margin and thumb_to_pinky_distance > y_margin and distanza > 0.18:
                                is_hand_open = 1  # aperta
                            elif thumb_to_middle_distance < y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance < y_margin and distanza < 0.18:
                                is_hand_open = 3  # indice
                            elif thumb_to_middle_distance > y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance < y_margin and distanza < 0.18:
                                is_hand_open = 4  # medio:
                            elif thumb_to_middle_distance < y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance < y_margin and distanza > 0.18:
                                is_hand_open = 2  # pollice
                            elif thumb_to_middle_distance < y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance > y_margin and thumb_to_pinky_distance < y_margin and distanza < 0.18:
                                is_hand_open = 5  # anulare
                            elif thumb_to_middle_distance < y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance > y_margin and distanza < 0.18:
                                is_hand_open = 6  # mignolo
                            elif thumb_to_middle_distance > y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance < y_margin and distanza < 0.18:
                                is_hand_open = 7  #pace
                            elif thumb_to_middle_distance < y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance > y_margin and distanza > 0.18:
                                is_hand_open = 8 #rock
                            else:
                                is_hand_open = 11

                        if is_hand_open == 1:
                            cv2.putText(frame, "Mano destra aperta",(50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif is_hand_open == 0:
                            cv2.putText(frame, "Mano destra chiusa", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        elif is_hand_open == 3:
                            cv2.putText(frame, "Indice Destro", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif is_hand_open == 4:
                            cv2.putText(frame, "Medio Destro", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif is_hand_open == 5:
                            cv2.putText(frame, "Anulare Destro", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif is_hand_open == 6:
                            cv2.putText(frame, "Mignolo Destro", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif is_hand_open == 2:
                            cv2.putText(frame, "Pollice Destro", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif is_hand_open == 7:
                            cv2.putText(frame, "Pace dx", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif is_hand_open == 8:
                            cv2.putText(frame, "Rock dx", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif is_hand_open == 11:
                            cv2.putText(frame, f"Dita aperte: {conditions_met}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Hand Detection', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cap2.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()