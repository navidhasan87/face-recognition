"""
camera.py

Everything related to the webcam:
- Opening/closing it
- Reading frames
- Drawing a box + name label around each detected face
- Letting the user snap a single reference photo when enrolling

This file does NOT decide attendance logic itself - it just shows what the
camera sees and who it thinks it recognizes, and optionally reports
recognized names through a callback. attendance.py decides what to actually
do with that information (write it to the database).
"""

import cv2

import my_face_recognition as fr


def draw_face_box(frame, name, box):
    """
    Draw a rectangle around a face and print their name in a small label
    under it. `box` is (top, right, bottom, left) - exactly what
    face_recognition gives us.
    """
    top, right, bottom, left = box

    # Green for a recognized person, red for "Not recognized".
    color = (0, 200, 0) if name != "Not recognized" else (0, 0, 255)

    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Small filled bar behind the text so it stays readable on any background.
    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame,
        name,
        (left + 6, bottom - 6),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (255, 255, 255),
        1,
    )


def run_live_recognition(
    on_recognized=None,
    window_name="Attendance - press Q to quit"
):
    """
    Open the webcam and continuously show recognized faces.

    Face detection is performed only every few frames to improve
    webcam performance. The latest recognition results are drawn
    on the frames between detection cycles.
    """

    known_names, known_encodings = fr.load_known_encodings()

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("[camera] Could not open webcam.")
        return

    print("[camera] Webcam started. Press 'q' to quit.")

    # Run face detection once every N frames.
    DETECTION_INTERVAL = 3

    # Store the latest detection results.
    recognized_faces = []

    frame_count = 0

    try:
        while True:
            ok, frame = video_capture.read()

            if not ok:
                print("[camera] Failed to read frame from webcam.")
                break

            frame_count += 1

            # Only run the expensive face recognition operation
            # every 3rd frame.
            if frame_count % DETECTION_INTERVAL == 0:
                recognized_faces = fr.recognize_faces(
                    frame,
                    known_names,
                    known_encodings
                )

                # Report recognized people only when a new detection
                # is performed.
                for name, box in recognized_faces:
                    if (
                        name != "Not recognized"
                        and on_recognized is not None
                    ):
                        on_recognized(name)

            # Draw the most recent recognition results on every frame.
            for name, box in recognized_faces:
                draw_face_box(frame, name, box)

            # Display every frame normally.
            cv2.imshow(window_name, frame)

            # Check whether 'q' was pressed.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        video_capture.release()
        cv2.destroyAllWindows()


def capture_single_photo(save_path, window_name="Press SPACE to capture, ESC to cancel"):
    """
    Open the webcam so the user can take one reference photo (used when
    enrolling a new student). Saves it to `save_path` when SPACE is
    pressed. Returns True if a photo was saved, False if cancelled.
    """
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[camera] Could not open webcam.")
        return False

    saved = False
    try:
        while True:
            ok, frame = video_capture.read()
            if not ok:
                break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE key
                cv2.imwrite(str(save_path), frame)
                saved = True
                break
            elif key == 27:  # ESC key
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

    return saved
