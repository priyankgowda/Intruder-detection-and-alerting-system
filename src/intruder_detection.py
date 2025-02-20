import cv2
import face_recognition
import pickle
import time
import asyncio

# Load known face encodings
def load_encodings(encoding_file="encodings.pkl"):
    try:
        with open(encoding_file, "rb") as f:
            known_encodings = pickle.load(f)
        return known_encodings
    except FileNotFoundError:
        return {}

# Recognize faces in the frame
def recognize_faces_in_frame(frame, known_encodings, resize_factor=0.8, detection_model="hog"):
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    rgb_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame, model=detection_model)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            [enc for enc_list in known_encodings.values() for enc in enc_list],
            face_encoding
        )
        name = "Intruder"
        if matches and any(matches):
            best_match_index = matches.index(True)
            flattened_encodings = [name for name, enc_list in known_encodings.items() for enc in enc_list]
            name = list(known_encodings.keys())[flattened_encodings.index(flattened_encodings[best_match_index])]
        recognized_names.append(name)

    face_locations = [(int(top * 1 / resize_factor), int(right * 1 / resize_factor),
                       int(bottom * 1 / resize_factor), int(left * 1 / resize_factor))
                      for (top, right, bottom, left) in face_locations]
    return face_locations, recognized_names, face_encodings, frame

# Main function to run the system
async def main():
    known_encodings = load_encodings()
    if not known_encodings:
        return

    video_capture = cv2.VideoCapture(0)

    frame_skip = 5
    frame_count = 0
    room_status = "empty"
    last_face_detected_time = None
    no_face_timeout = 2

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        face_locations, recognized_names, face_encodings, processed_frame = recognize_faces_in_frame(frame, known_encodings)

        if recognized_names:
            last_face_detected_time = time.time()
            if room_status == "empty":
                room_status = "occupied"
        else:
            if last_face_detected_time and time.time() - last_face_detected_time > no_face_timeout:
                room_status = "empty"

        for (top, right, bottom, left), name in zip(face_locations, recognized_names):
            color = (0, 255, 0) if name != "Intruder" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Intruder Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
