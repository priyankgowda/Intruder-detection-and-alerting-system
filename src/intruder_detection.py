import asyncio
import cv2
import time
import logging
import face_recognition
import io
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Telegram Bot token and chat ID
BOT_TOKEN = "telegram_bot_api_token"
CHAT_ID = "telegram_chat_id"

# Initialize the bot
bot = Bot(token=BOT_TOKEN)


# Function to load known encodings
def load_encodings():
    """
    Load known face encodings from a file or database.
    Returns a dictionary where keys are names, and values are lists of encodings.
    """
    # This is a placeholder. Replace with your logic to load face encodings.
    return {
        "John Doe": [face_recognition.face_encodings(face_recognition.load_image_file("john.jpg"))[0]]
    }


# Function to handle Telegram buttons
async def button_handler(update, context):
    """
    Handles the button interaction for intruder classification.
    """
    query = update.callback_query
    await query.answer()
    decision = query.data  # 'known' or 'unknown'

    if decision == "known":
        logging.info("User classified the person as Known.")
        await query.edit_message_text(text="Person marked as Known. Updating database...")
        # Logic to update known faces goes here
    elif decision == "unknown":
        logging.info("User classified the person as Unknown.")
        await query.edit_message_text(text="Person marked as Unknown. Alert logged.")


# Function to send three snapshots and a button for intruders
async def send_alert_with_snapshots_and_button(intruder_images):
    """
    Sends three snapshots followed by a single message with the "Known" and "Unknown" buttons.
    """
    # Send three snapshots
    for idx, image in enumerate(intruder_images[:3]):  # Send only the first 3 snapshots
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        image_io = io.BytesIO(image_bytes)

        await bot.send_photo(chat_id=CHAT_ID, photo=image_io, caption=f"Snapshot {idx + 1} of intruder.")

    # Create buttons
    keyboard = [
        [InlineKeyboardButton("Known", callback_data="known"), InlineKeyboardButton("Unknown", callback_data="unknown")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send message with buttons
    await bot.send_message(chat_id=CHAT_ID, text="Is this person known or unknown?", reply_markup=reply_markup)
    logging.info("Intruder alert: Three snapshots sent followed by buttons.")


# Function to recognize faces in the frame
async def recognize_faces_in_frame_and_alert(frame, known_encodings, resize_factor=0.8, detection_model="hog"):
    """
    Recognizes faces in the frame, sends snapshots for intruders, and handles alert logic.
    """
    # Process the frame
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    rgb_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame, model=detection_model)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = []
    intruder_images = []  # Collect images of intruders
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
        if name == "Intruder":
            intruder_images.append(frame.copy())  # Store the intruder's snapshot

    # Scale back face locations to match original frame size
    face_locations = [(int(top * 1 / resize_factor), int(right * 1 / resize_factor),
                       int(bottom * 1 / resize_factor), int(left * 1 / resize_factor))
                      for (top, right, bottom, left) in face_locations]

    return face_locations, recognized_names, intruder_images, frame


# Main function to run the system
async def main():
    known_encodings = load_encodings()
    if not known_encodings:
        logging.warning("No known faces loaded. Exiting...")
        return

    video_capture = cv2.VideoCapture(0)
    logging.info("Webcam started.")

    frame_skip = 5
    frame_count = 0
    room_status = "empty"
    last_face_detected_time = None
    no_face_timeout = 2  # Seconds before considering the room empty

    # Start the Telegram bot application for button interaction
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CallbackQueryHandler(button_handler))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Failed to capture frame. Exiting...")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Recognize faces and handle intruders
        face_locations, recognized_names, intruder_images, processed_frame = await recognize_faces_in_frame_and_alert(
            frame, known_encodings
        )

        if recognized_names:
            # If faces are detected, reset the timer and update room status
            last_face_detected_time = time.time()
            if room_status == "empty":
                for name in recognized_names:
                    logging.info(f"{name} entered the room.")
                room_status = "occupied"

            # Handle intruders
            for name in recognized_names:
                if name == "Intruder" and intruder_images:
                    await send_alert_with_snapshots_and_button(intruder_images)  # Send snapshots and button
                    break
        else:
            # If no faces are detected, check if timeout has passed
            if last_face_detected_time and time.time() - last_face_detected_time > no_face_timeout:
                if room_status == "occupied":
                    logging.info("Room is empty.")
                room_status = "empty"

        # Draw rectangles and labels around detected faces
        for (top, right, bottom, left), name in zip(face_locations, recognized_names):
            color = (0, 255, 0) if name != "Intruder" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Intruder Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
    logging.info("Webcam stopped by user.")
    await application.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
