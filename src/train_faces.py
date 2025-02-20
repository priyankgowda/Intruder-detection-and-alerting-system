import os
import cv2
import face_recognition
import pickle

def add_known_person(name, image_path=None, encoding_file="encodings.pkl"):
    """
    Capture a new face using the webcam or an image and add it to the known encodings.

    Parameters:
        name (str): Name of the person to add.
        image_path (str): Path to the image file (if not training live).
        encoding_file (str): File to store face encodings.
    """
    # Load existing encodings, if available
    try:
        with open(encoding_file, "rb") as f:
            known_encodings = pickle.load(f)
    except FileNotFoundError:
        known_encodings = {}  # Start fresh if no encodings exist
        print("No existing encodings found. Creating a new database.")

    # Check if the name already exists
    if name in known_encodings:
        print(f"The name '{name}' already exists in the database.")
        return  # Exit the function if the name exists

    # Process training (live or image-based)
    face_encoding = None
    if image_path:
        # Training with an image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image from '{image_path}'.")
            return

        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if face_encodings:
            face_encoding = face_encodings[0]
        else:
            print("No face detected in the provided image. Try again with a different image.")
            return
    else:
        # Training with live capture
        video_capture = cv2.VideoCapture(0)
        print("Capturing image. Press 's' to save and 'q' to quit.")

        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error accessing webcam.")
                break

            frame_count += 1
            if frame_count % 5 != 0:  # Skip frames for faster processing
                continue

            # Resize the frame for faster processing
            small_frame = cv2.resize(frame, (640, 480))  # Use a lower resolution

            # Show the video feed
            cv2.imshow("Video", frame)

            # Detect face when 's' key is pressed
            key = cv2.waitKey(1)
            if key == ord('s'):
                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                if face_encodings:
                    face_encoding = face_encodings[0]
                    print(f"Face captured for {name}.")
                    break
                else:
                    print("No face detected. Try again.")
            elif key == ord('q'):
                print("Exiting without saving.")
                break

        # Release webcam
        video_capture.release()
        cv2.destroyAllWindows()

    if face_encoding is not None:
        # Add new face encoding
        known_encodings[name] = [face_encoding]

        # Save updated encodings
        with open(encoding_file, "wb") as f:
            pickle.dump(known_encodings, f)

        print(f"{name} added to known encodings.")

if __name__ == "__main__":
    person_name = input("Enter the name of the person to add: ").strip()

    # Check if the name exists before proceeding
    try:
        with open("encodings.pkl", "rb") as f:
            known_encodings = pickle.load(f)
    except FileNotFoundError:
        known_encodings = {}

    if person_name in known_encodings:
        print(f"The name '{person_name}' already exists in the database.")
    else:
        train_choice = input("How do you want to train the model? Enter 'live' for live capture or 'image' for an image file: ").strip().lower()

        if train_choice == "image":
            image_file = input("Enter the path to the image file: ").strip()
            add_known_person(person_name, image_path=image_file)
        elif train_choice == "live":
            add_known_person(person_name)
        else:
            print("Invalid choice. Please enter 'live' or 'image'.")
