from curses.ascii import FF

import numpy as np
import face_recognition as fr
import cv2

video_shot = cv2.VideoCapture(0)

biden_pic = fr.load_image_file("Joe_Biden.jpg")
biden_encoded_face = fr.face_encodings(biden_pic)[0]

known_face_encoding = [biden_encoded_face]
known_face_name = ["The President of US Joe Biden"]

while True:
    ret, frame = video_shot.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encoding = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encoding):
        matches = fr.compare_faces(known_face_encoding, face_encoding)

        name = "unknown"

        face_distance = fr.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_face_name[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Camera face recognition', frame)

    if cv2.waitKey(1) & 0 * FF == ord('q'):
        break

video_shot.release()
cv2.destroyAllWindows()
