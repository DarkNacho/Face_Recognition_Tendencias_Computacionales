import sys
import cv2
import numpy as np
import os
import face_recognition

known_face_names = []
known_face_encodings = []

for imgLoc in os.listdir(sys.argv[1]): ## argv1 folder with images
    img = face_recognition.load_image_file(f"{sys.argv[1]}/{imgLoc}")
    known_face_names.append(os.path.splitext(imgLoc)[0])
    known_face_encodings.append(face_recognition.face_encodings(img)[0])



frame = face_recognition.load_image_file(sys.argv[2]) ##img source
#rgb_frame = frame[:, :, ::-1]
face_locations = face_recognition.face_locations(frame)
frame_encoding =face_recognition.face_encodings(frame, face_locations)

for face_encoding, (top, right, bottom, left) in zip(frame_encoding, face_locations):
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

cv2.imshow("webcam", frame)
cv2.imwrite(sys.argv[3], frame) #output img
cv2.waitKey(0)
