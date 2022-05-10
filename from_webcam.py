#!/venv/bin/python
import sys
from datetime import datetime

import cv2
import numpy as np
import os
import face_recognition

def attandace(name):
    with open("attendace.csv", "r+") as file:
        data_list = file.readlines()
        name_list = []
        for line in data_list:
            entry = line.split(",")
            name_list.append(entry[0])
        if name not in name_list:
            #now = datetime.now().strftime("%H:%M:%S")
            file.writelines(f"{name},{datetime.now()}\n")


known_face_names = []
known_face_encodings = []

for imgLoc in os.listdir(sys.argv[1]): ## argv1 folder with images
    img = face_recognition.load_image_file(f"{sys.argv[1]}/{imgLoc}")
    known_face_names.append(os.path.splitext(imgLoc)[0])
    known_face_encodings.append(face_recognition.face_encodings(img)[0])


video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()

    # Convert the image from BGR(OpenCV) to RGB (face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        name = "Unknown"
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        attandace(name)


    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

