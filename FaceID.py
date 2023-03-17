import cv2
import face_recognition

# Load images and create face encodings
image_of_person_1 = face_recognition.load_image_file("person_1.jpg")
person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]

image_of_person_2 = face_recognition.load_image_file("person_2.jpg")
person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    person_1_face_encoding,
    person_2_face_encoding
]
known_face_names = [
    "Person 1",
    "Person 2"
]

# Initialize camera module
video_capture = cv2.VideoCapture(0)

# Recognize faces in real-time video stream
while True:
    # Capture each frame of the video
    ret, frame = video_capture.read()

    # Convert the frame from BGR color to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # If a match was found, display the name of the person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Unknown"

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit program if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
