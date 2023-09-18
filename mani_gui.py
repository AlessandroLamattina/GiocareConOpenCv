import os
import cv2
import mediapipe as mp
from datetime import datetime
import math
import numpy as np
import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtWidgets import QHBoxLayout

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=10)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(10)
        self.cap = cv2.VideoCapture(0)
        if not os.path.exists("mani"):
            os.mkdir("mani")
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join("mani", f"video_{current_time}.avi")
        self.out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
        self.y_margin = 0.1

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setMinimumSize(640, 480)

        start_button = QPushButton("Start")
        start_button.clicked.connect(self.start_capture)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self.stop_capture)

        button_layout = QHBoxLayout()
        button_layout.addWidget(start_button)
        button_layout.addWidget(stop_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(button_layout)

        central_widget.setLayout(main_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # Update the frame every 10 milliseconds

        self.is_capturing = False

    def start_capture(self):
        self.is_capturing = True

    def stop_capture(self):
        self.is_capturing = False

    def blurface(self):
        results2 = self.face_detection.process(self.rgb_frame)
        for detection in results2.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = self.frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face = self.frame[y:y + h, x:x + w]
            face = cv2.resize(face, (w // 15, h // 15))
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_LINEAR)
            self.frame[y:y + h, x:x + w] = face

    def facemap(self):
        results = self.face_mesh.process(self.rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * self.frame.shape[1])
                    y = int(landmark.y * self.frame.shape[0])
                    cv2.circle(self.frame, (x, y), 1, (255, 255, 255), -1)
                    cv2.circle(self.frame, (x, y), 3, (0, 0, 0), 2)
                landmarks = face_landmarks.landmark
                for connection in self.mp_face_mesh.FACEMESH_CONTOURS:
                    start_idx, end_idx = connection
                    start_point = (
                        int(landmarks[start_idx].x * self.frame.shape[1]), int(landmarks[start_idx].y * self.frame.shape[0]))
                    end_point = (int(landmarks[end_idx].x * self.frame.shape[1]), int(landmarks[end_idx].y * self.frame.shape[0]))
                    cv2.line(self.frame, start_point, end_point, (0, 255, 0), 1)

    def pollicione(self):
        pass

    def blur_hand(self):
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                hand_landmarks = landmarks.landmark
                x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
                for landmark in hand_landmarks:
                    x, y = int(landmark.x * self.frame.shape[1]), int(landmark.y * self.frame.shape[0])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                hand_region = self.frame[y_min:y_max, x_min:x_max]
                hand_region = cv2.GaussianBlur(hand_region, (15, 15), 0)
                self.frame[y_min:y_max, x_min:x_max] = hand_region

    def update_frame(self):
        ret, self.frame = self.cap.read()
        if not ret:
            return

        if self.is_capturing:
            h, w, _ = self.frame.shape
            self.rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(self.rgb_frame)
            lis_hand_open = 0

            if self.results.multi_hand_landmarks:
                is_hand_open = 0
                for landmarks in self.results.multi_hand_landmarks:
                    self.landmarks = landmarks
                    thumb_landmark = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                    index_landmark = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    distance = ((thumb_landmark.x - index_landmark.x) ** 2 + (thumb_landmark.y - index_landmark.y) ** 2) ** 0.5
                    self.connections = self.mp_hands.HAND_CONNECTIONS
                    for connection in self.connections:
                        x0, y0 = int(landmarks.landmark[connection[0]].x * w), int(landmarks.landmark[connection[0]].y * h)
                        x1, y1 = int(landmarks.landmark[connection[1]].x * w), int(landmarks.landmark[connection[1]].y * h)
                        cv2.line(self.frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    for i, landmark in enumerate(landmarks.landmark):
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)  # Colore interno (fill)
                        cv2.circle(self.frame, (x, y), 5, (0, 0, 255), 2)  # Colore contorno
                        thumb_landmark = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                        thumb_towrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y - thumb_landmark.y
                        thumb_to_index_distance = thumb_landmark.y - landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                        thumb_to_middle_distance = thumb_landmark.y - landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                        thumb_to_ring_distance = thumb_landmark.y - landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y
                        thumb_to_pinky_distance = thumb_landmark.y - landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y
                        center_x = landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                        center_y = landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y
                        thumb_tip_landmark = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                        wrist_landmark = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

                        wrist_x = landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                        thumb_tip_landmark = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                        wrist_landmark = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        knuckle_landmark = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]  # Puoi scegliere qualsiasi nocca di dito
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
                            conditions_met = 1
                        if wrist_x > 0.6:
                            if thumb_to_middle_distance < self.y_margin and thumb_to_index_distance < self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza < 0.18:
                                lis_hand_open = 0  # aperta
                            elif thumb_to_middle_distance > self.y_margin and thumb_to_index_distance > self.y_margin and thumb_to_ring_distance > self.y_margin and thumb_to_pinky_distance > self.y_margin and distanza > 0.18:
                                lis_hand_open = 1 #aperta
                            elif thumb_to_middle_distance < self.y_margin and thumb_to_index_distance > self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza < 0.18:
                                lis_hand_open = 3  # indice
                            elif thumb_to_middle_distance > self.y_margin and thumb_to_index_distance < self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza < 0.18:
                                lis_hand_open = 4  # medio:
                            elif thumb_to_middle_distance < self.y_margin and thumb_to_index_distance < self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza > 0.18:
                                lis_hand_open = 2  # pollice
                            elif thumb_to_middle_distance < self.y_margin and thumb_to_index_distance < self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza < 0.18:
                                lis_hand_open = 5  # anulare
                            elif thumb_to_middle_distance < self.y_margin and thumb_to_index_distance < self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance > self.y_margin and distanza < 0.18:
                                lis_hand_open = 6  # mignolo
                            elif thumb_to_middle_distance > self.y_margin and thumb_to_index_distance > self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza < 0.18:
                                lis_hand_open = 7  #pace
                            elif thumb_to_middle_distance < self.y_margin and thumb_to_index_distance > self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance > self.y_margin and distanza > 0.18:
                                lis_hand_open = 8  #pace
                            else:
                                lis_hand_open = 11

                            if lis_hand_open == 1:
                                self.blurface()
                            elif lis_hand_open == 0:
                                self.facemap()
                            elif lis_hand_open == 2:
                                self.pollicione()
                            elif lis_hand_open == 3:
                                self.blur_hand()
                        if wrist_x < 0.6:
                            if conditions_met == 0:
                                is_hand_open = 0  # chiusa
                            else:
                                if thumb_to_middle_distance > self.y_margin and thumb_to_index_distance > self.y_margin and thumb_to_ring_distance > self.y_margin and thumb_to_pinky_distance > self.y_margin and distanza > 0.18:
                                    is_hand_open = 1  # aperta
                                elif thumb_to_middle_distance < self.y_margin and thumb_to_index_distance > self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza < 0.18:
                                    is_hand_open = 3  # indice
                                elif thumb_to_middle_distance > self.y_margin and thumb_to_index_distance < self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza < 0.18:
                                    is_hand_open = 4  # medio:
                                elif thumb_to_middle_distance < self.y_margin and thumb_to_index_distance < self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza > 0.18:
                                    is_hand_open = 2  # pollice
                                elif thumb_to_middle_distance < self.y_margin and thumb_to_index_distance < self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza < 0.18:
                                    is_hand_open = 5  # anulare
                                elif thumb_to_middle_distance < self.y_margin and thumb_to_index_distance < self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance > self.y_margin and distanza < 0.18:
                                    is_hand_open = 6  # mignolo
                                elif thumb_to_middle_distance > self.y_margin and thumb_to_index_distance > self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance < self.y_margin and distanza < 0.18:
                                    is_hand_open = 7  #pace
                                elif thumb_to_middle_distance < self.y_margin and thumb_to_index_distance > self.y_margin and thumb_to_ring_distance < self.y_margin and thumb_to_pinky_distance > self.y_margin and distanza > 0.18:
                                    is_hand_open = 8 #rock
                                else:
                                    is_hand_open = 11

                            if is_hand_open == 1:
                                cv2.putText(self.frame, "Mano destra aperta",(50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif is_hand_open == 0:
                                cv2.putText(self.frame, "Mano destra chiusa", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            elif is_hand_open == 3:
                                cv2.putText(self.frame, "Indice Destro", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif is_hand_open == 4:
                                cv2.putText(self.frame, "Medio Destro", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif is_hand_open == 5:
                                cv2.putText(self.frame, "Anulare Destro", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif is_hand_open == 6:
                                cv2.putText(self.frame, "Mignolo Destro", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif is_hand_open == 2:
                                cv2.putText(self.frame, "Pollice Destro", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif is_hand_open == 7:
                                cv2.putText(self.frame, "Pace dx", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif is_hand_open == 8:
                                cv2.putText(self.frame, "Rock dx", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            elif is_hand_open == 11:
                                cv2.putText(self.frame, f"Dita aperte: {conditions_met}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.out.write(self.frame)
        self.display_frame()

    def display_frame(self):
        h, w, ch = self.frame.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(self.frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def closeEvent(self, event):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainApp()
    mainWin.show()
    sys.exit(app.exec_())
