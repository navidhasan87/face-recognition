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

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_SCALE = 0.25
PROCESS_EVERY_N_FRAMES = 2


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


def run_live_recognition(on_recognized=None, window_name="Attendance - press Q to quit"):
    """
    Open the webcam and continuously show recognized faces until the user
    presses 'q' or closes the window.

    `on_recognized(name)` is an optional callback. If provided, it gets
    called every time a *known* person is seen in a frame - attendance.py
    passes attendance.mark_attendance so camera.py never has to know
    anything about the database itself.
    """
    known_names, known_encodings = fr.load_known_encodings()

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[camera] Could not open webcam.")
        return

    # Keep the capture queue short so recognition works on the newest frame,
    # rather than falling behind and displaying stale video.
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print("[camera] Webcam started. Press 'q' to quit.")

    frame_number = 0
    recognized_faces = []
    names_to_report = []
    try:
        while True:
            ok, frame = video_capture.read()
            if not ok:
                print("[camera] Failed to read frame from webcam.")
                break

            frame_number += 1
            if frame_number % PROCESS_EVERY_N_FRAMES == 0:
                small_frame = cv2.resize(
                    frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE,
                    interpolation=cv2.INTER_LINEAR,
                )
                small_faces = fr.recognize_faces(
                    small_frame, known_names, known_encodings
                )
                # Convert detections from the small frame back to display
                # coordinates before drawing them on the full-size frame.
                recognized_faces = [
                    (
                        name,
                        tuple(round(coordinate / PROCESS_SCALE) for coordinate in box),
                    )
                    for name, box in small_faces
                ]
                names_to_report = [
                    name for name, _ in recognized_faces if name != "Not recognized"
                ]

                for name in names_to_report:
                    if on_recognized is not None:
                        on_recognized(name)

            for name, box in recognized_faces:
                draw_face_box(frame, name, box)

            cv2.imshow(window_name, frame)

            # waitKey(1) pauses 1ms and lets the window repaint each loop;
            # it also lets us check whether 'q' was pressed.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always release the camera and close windows, even if something
        # above raised an error - otherwise the webcam can stay "locked".
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
                saved = bool(cv2.imwrite(str(save_path), frame))
                if not saved:
                    print(f"[camera] Could not save photo to {save_path}.")
                break
            elif key == 27:  # ESC key
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

    return saved
