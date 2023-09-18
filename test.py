import os
import cv2
import mediapipe as mp
import math

class Main:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.cap = cv2.VideoCapture(0)
        self.y_margin = 0.1

        self.button_coords = {
            "Mani": (100, 60),
            "Contatore": (500, 60),
        }

        self.active_button = None

    def detect_hand_gestures(self, landmarks):
       pass

    def check_button_tap(self, x, y):
        for button, coords in self.button_coords.items():
            bx, by = coords
            if abs(x - bx) < 20 and abs(y - by) < 20:
                return button
        return None

    def run(self):
        while self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    for id, landmark in enumerate(landmarks.landmark):
                        if id == self.mp_hands.HandLandmark.INDEX_FINGER_TIP:
                            height, width, _ = self.frame.shape
                            x, y = int(landmark.x * width), int(landmark.y * height)
                            cv2.circle(self.frame, (x, y), 10, (0, 255, 0), -1)
                            cv2.putText(self.frame, f'({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            # Verifica se il dito tocca un bottone
                            tapped_button = self.check_button_tap(x, y)
                            if tapped_button:
                                self.active_button = tapped_button
                                if self.active_button == "Mani":
                                    self.mani2()
                                if self.active_button == "Contatore":
                                    self.mani()
                            else:
                                self.active_button = None

            # Disegna i bottoni virtuali
            for button, coords in self.button_coords.items():
                cv2.putText(self.frame, button, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Hand Tracking', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def mani2(self):
        while True:
            print("mani2")
            cv2.imshow('Hand Tracking', self.frame)

    def mani(self):
        print("mani")

if __name__ == "__main__":
    Main().run()




"""import os
import cv2
import mediapipe as mp
from datetime import datetime
import math

# Inizializza MediaPipe Hands
class Main:
    def __init__(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()

        # Inizializza la webcam
        self.cap = cv2.VideoCapture(0)

        while self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                continue
            rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    for id, landmark in enumerate(landmarks.landmark):
                        if id == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                            # Ottiene le coordinate del dito indice
                            height, width, _ = self.frame.shape
                            x, y = int(landmark.x * width), int(landmark.y * height)
                            cv2.circle(self.frame, (x, y), 10, (0, 255, 0), -1)  # Disegna un cerchio sul punto del dito
                            cv2.putText(self.frame, f'({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            if x < 100 and y < 60:
                                while  x > 500 and y < 60:
                                    y_margin = 0.1
                                    h, w, _ = self.frame.shape
                                    lis_hand_open = 0
                                    is_hand_open = 0
                                    thumb_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                                    index_landmark = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                                    distance = ((thumb_landmark.x - index_landmark.x) ** 2 + (
                                                thumb_landmark.y - index_landmark.y) ** 2) ** 0.5
                                    for i, landmark in enumerate(landmarks.landmark):
                                        thumb_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                                        thumb_towrist = landmarks.landmark[
                                                            mp_hands.HandLandmark.WRIST].y - thumb_landmark.y
                                        thumb_to_index_distance = thumb_landmark.y - landmarks.landmark[
                                            mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                                        thumb_to_middle_distance = thumb_landmark.y - landmarks.landmark[
                                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                                        thumb_to_ring_distance = thumb_landmark.y - landmarks.landmark[
                                            mp_hands.HandLandmark.RING_FINGER_TIP].y
                                        thumb_to_pinky_distance = thumb_landmark.y - landmarks.landmark[
                                            mp_hands.HandLandmark.PINKY_TIP].y
                                        center_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                                        center_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                                        thumb_tip_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                                        wrist_landmark = landmarks.landmark[mp_hands.HandLandmark.WRIST]

                                        wrist_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                                        thumb_tip_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                                        wrist_landmark = landmarks.landmark[mp_hands.HandLandmark.WRIST]
                                        knuckle_landmark = landmarks.landmark[
                                            mp_hands.HandLandmark.INDEX_FINGER_MCP]  # Puoi scegliere qualsiasi nocca di dito

                                        # Calcola il centro approssimativo del palmo come punto medio tra il polso e la nocca della mano
                                        palm_x = (wrist_landmark.x + knuckle_landmark.x) / 2
                                        palm_y = (wrist_landmark.y + knuckle_landmark.y) / 2
                                        palm_z = (wrist_landmark.z + knuckle_landmark.z) / 2

                                        # Calcola la distanza euclidea tra la punta del pollice e il centro approssimativo del palmo
                                        distanza = math.sqrt((thumb_tip_landmark.x - palm_x) ** 2 + (
                                                    thumb_tip_landmark.y - palm_y) ** 2 + (
                                                                         thumb_tip_landmark.z - palm_z) ** 2)
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
                                                conditions_met += 1
                                            cv2.putText(self.frame, f"Dita su mano dx: {conditions_met}",
                                                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
                                            cv2.putText(self.frame, f"Dita su mano sx: {sconditions_met}",
                                                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
                                                elif thumb_to_middle_distance < y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance > y_margin and distanza < 0.18:
                                                    lis_hand_open = 5  # anulare
                                                elif thumb_to_middle_distance < y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance > y_margin and distanza < 0.18:
                                                    lis_hand_open = 6  # mignolo
                                                elif thumb_to_middle_distance > y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance < y_margin and distanza < 0.18:
                                                    lis_hand_open = 7  # pace
                                                elif thumb_to_middle_distance < y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance > y_margin and distanza > 0.18:
                                                    lis_hand_open = 8  # rock
                                                else:
                                                    lis_hand_open = 11

                                            if lis_hand_open == 1:
                                                cv2.putText(self.frame, "Mano sinistra aperta", (50, 50),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif lis_hand_open == 0:
                                                cv2.putText(self.frame, "Mano sinistra chiusa", (50, 50),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                            elif lis_hand_open == 3:
                                                cv2.putText(self.frame, "Indice Sinistro", (50, 50),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif lis_hand_open == 4:
                                                cv2.putText(self.frame, "Medio Sinistro", (50, 50),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif lis_hand_open == 5:
                                                cv2.putText(self.frame, "Anulare Sinistro", (50, 50),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif lis_hand_open == 6:
                                                cv2.putText(self.frame, "Mignolo Sinistro", (50, 50),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif lis_hand_open == 2:
                                                cv2.putText(self.frame, "Pollice Sinistro", (50, 50),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif lis_hand_open == 7:
                                                cv2.putText(self.frame, "Pace sx", (50, 50),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif lis_hand_open == 8:
                                                cv2.putText(self.frame, "Rock sx", (50, 50),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif lis_hand_open == 11:
                                                cv2.putText(self.frame, f"Dita su: {sconditions_met}", (50, 50),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
                                                elif thumb_to_middle_distance < y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance > y_margin and distanza < 0.18:
                                                    is_hand_open = 5  # anulare
                                                elif thumb_to_middle_distance < y_margin and thumb_to_index_distance < y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance > y_margin and distanza < 0.18:
                                                    is_hand_open = 6  # mignolo
                                                elif thumb_to_middle_distance > y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance < y_margin and distanza < 0.18:
                                                    is_hand_open = 7  # pace
                                                elif thumb_to_middle_distance < y_margin and thumb_to_index_distance > y_margin and thumb_to_ring_distance < y_margin and thumb_to_pinky_distance > y_margin and distanza > 0.18:
                                                    is_hand_open = 8  # rock
                                                else:
                                                    is_hand_open = 11

                                            if is_hand_open == 1:
                                                cv2.putText(self.frame, "Mano destra aperta", (50, 100),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif is_hand_open == 0:
                                                cv2.putText(self.frame, "Mano destra chiusa", (50, 100),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                            elif is_hand_open == 3:
                                                cv2.putText(self.frame, "Indice Destro", (50, 100),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif is_hand_open == 4:
                                                cv2.putText(self.frame, "Medio Destro", (50, 100),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif is_hand_open == 5:
                                                cv2.putText(self.frame, "Anulare Destro", (50, 100),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif is_hand_open == 6:
                                                cv2.putText(self.frame, "Mignolo Destro", (50, 100),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif is_hand_open == 2:
                                                cv2.putText(self.frame, "Pollice Destro", (50, 100),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif is_hand_open == 7:
                                                cv2.putText(self.frame, "Pace dx", (50, 100),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif is_hand_open == 8:
                                                cv2.putText(self.frame, "Rock dx", (50, 100),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            elif is_hand_open == 11:
                                                cv2.putText(self.frame, f"Dita aperte: {conditions_met}",
                                                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                                            2)
                            if x > 500 and y < 60:
                                self.mani2()

            cv2.imshow('Hand Tracking', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()
"""