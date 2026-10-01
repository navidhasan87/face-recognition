import os
import cv2
import pickle
from collections import Counter, deque

import numpy as np
import face_recognition
from PIL import Image, ImageTk

# configuration

FACE_DIR = "faces"
DATA_DIR = "data"
ENCODINGS_FILE = os.path.join(DATA_DIR, "face_encodings.pkl")

DETECTION_MODEL = "hog"


# detection threshold. beshi hole loose kom hole strict. 0.50 average bola jay.
RECOGNITION_THRESHOLD = 0.50

CONFIRMATION_WINDOW = 5

CONFIRMATION_REQUIRED = 3

MIN_FACE_SIZE = 70

BLUR_THRESHOLD = 30.0


class FaceCamera:

    def __init__(self, camera_index=0):

        self.camera_index = camera_index
        self.camera = None

        os.makedirs(FACE_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)

        # recognition er kajkam

        self.recognition_history = deque(
            maxlen=CONFIRMATION_WINDOW
        )

        self.last_recognized_id = None
        self.last_confidence = 0.0

    # camera er kajkam

    def start(self):

        if self.camera is not None:
            return True

        self.camera = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():

            self.camera.release()
            self.camera = None
            return False

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        return True

    def stop(self):

        if self.camera is not None:

            self.camera.release()
            self.camera = None

        self.reset_confirmation()

    def read_frame(self):

        if self.camera is None:
            return None

        success, frame = self.camera.read()

        if not success:
            return None

        return frame

    # recognition er restart

    def reset_confirmation(self):

        self.recognition_history.clear()

        self.last_recognized_id = None

        self.last_confidence = 0.0

    # chehara dekha

    def detect_faces(self, frame):

        if frame is None:
            return []

        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.5,
            fy=0.5
        )

        rgb_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        locations = face_recognition.face_locations(
            rgb_frame,
            model="hog"
        )

        scaled_locations = []

        for top, right, bottom, left in locations:

            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            scaled_locations.append(
                (
                    top,
                    right,
                    bottom,
                    left
                )
            )

        return scaled_locations

    def calculate_blur(self, face):

        if face is None or face.size == 0:
            return 0.0

        gray = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )

        return cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

    def is_good_face(
        self,
        frame,
        location
    ):

        top, right, bottom, left = location

        width = right - left
        height = bottom - top

        if width < MIN_FACE_SIZE:
            return False

        if height < MIN_FACE_SIZE:
            return False

        top = max(0, top)
        left = max(0, left)

        bottom = min(
            frame.shape[0],
            bottom
        )

        right = min(
            frame.shape[1],
            right
        )

        face = frame[
            top:bottom,
            left:right
        ]

        if face.size == 0:
            return False

        blur_value = self.calculate_blur(
            face
        )

        if blur_value < BLUR_THRESHOLD:
            return False

        return True

    def draw_faces(
        self,
        frame,
        locations,
        names=None
    ):

        output = frame.copy()

        if names is None:

            names = [
                "Face"
                for _ in locations
            ]

        for location, name in zip(
            locations,
            names
        ):

            top, right, bottom, left = location

            cv2.rectangle(
                output,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            label_height = 28
            label_top = max(
                0,
                bottom - label_height
            )

            cv2.rectangle(
                output,
                (left, label_top),
                (right, bottom),
                (0, 255, 0),
                cv2.FILLED
            )

            cv2.putText(
                output,
                str(name),
                (left + 5, bottom - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )

        return output

    # tkinter er image

    def get_tk_image(
        self,
        frame,
        width=640,
        height=480
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        image.thumbnail(
            (width, height),
            Image.Resampling.LANCZOS
        )

        return ImageTk.PhotoImage(
            image=image
        )

    # registration er chobi

    def capture_face_image(
        self,
        frame,
        student_id,
        image_number
    ):

        student_dir = os.path.join(
            FACE_DIR,
            str(student_id)
        )

        os.makedirs(
            student_dir,
            exist_ok=True
        )

        filename = os.path.join(
            student_dir,
            f"{image_number}.jpg"
        )

        success = cv2.imwrite(
            filename,
            frame
        )

        if not success:
            return None

        return filename

    # image encode

    def encode_image(
        self,
        image_path
    ):
        try:
            image = face_recognition.load_image_file(
                image_path
            )
            locations = face_recognition.face_locations(
                image,
                model="hog"
            )
            if not locations:
                return None

            #face onek thakle largest face detect hobe
            largest_face = max(
                locations,
                key=lambda box:
                (box[2] - box[0])
                *
                (box[1] - box[3])
            )

            encodings = face_recognition.face_encodings(
                image,
                known_face_locations=[
                    largest_face
                ],
                num_jitters=2
            )
            if not encodings:
                return None
            return encodings[0]

        except Exception as error:
            print(
                f"Error encoding {image_path}:"
            )

            print(error)

            return None

    # proti student er encoding kora

    def create_student_encoding(
        self,
        student_id
    ):
        student_id = str(student_id)
        student_dir = os.path.join(
            FACE_DIR,
            student_id
        )
        if not os.path.isdir(student_dir):
            return False, 0

        image_files = []

        for filename in os.listdir(
            student_dir
        ):
            if filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ):
                image_files.append(
                    os.path.join(
                        student_dir,
                        filename
                    )
                )

        image_files.sort()

        if not image_files:
            return False, 0

        # encoding er dictionary khola

        all_encodings = self.load_encodings()

        all_encodings[student_id] = []

        successful = 0

        for image_path in image_files:

            encoding = self.encode_image(
                image_path
            )

            if encoding is not None:

                all_encodings[
                    student_id
                ].append(
                    encoding
                )

                successful += 1

        # image payni
        if successful == 0:

            all_encodings.pop(
                student_id,
                None
            )

            self.save_encodings(
                all_encodings
            )

            return False, 0

        # save kora

        self.save_encodings(
            all_encodings
        )

        return True, successful

    # encoding load kora

    def load_encodings(self):

        if not os.path.exists(
            ENCODINGS_FILE
        ):
            return {}

        try:
            with open(
                ENCODINGS_FILE,
                "rb"
            ) as file:
                data = pickle.load(
                    file
                )
            if not isinstance(
                data,
                dict
            ):
                return {}
            return data

        except Exception as error:
            print(
                "Error loading encodings:"
            )
            print(error)
            return {}

    # save encoding

    def save_encodings(
        self,
        encodings
    ):
        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )
        temp_file = (
            ENCODINGS_FILE
            + ".tmp"
        )
        try:
            with open(
                temp_file,
                "wb"
            ) as file:
                pickle.dump(
                    encodings,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL
                )
            os.replace(
                temp_file,
                ENCODINGS_FILE
            )
            return True

        except Exception as error:
            print(
                "Error saving encodings:"
            )
            print(error)
            if os.path.exists(
                temp_file
            ):
                try:
                    os.remove(
                        temp_file
                    )
                except OSError:
                    pass
            return False

    # chena encoding ber kora

    def get_known_encodings(self):

        stored = self.load_encodings()

        student_ids = []
        encodings = []

        for student_id, student_encodings in stored.items():

            if isinstance(
                student_encodings,
                np.ndarray
            ):
                student_encodings = [
                    student_encodings
                ]
            if not isinstance(
                student_encodings,
                list
            ):
                continue

            for encoding in student_encodings:
                try:
                    vector = np.asarray(
                        encoding,
                        dtype=np.float64
                    )
                    if vector.shape != (128,):
                        continue
                    student_ids.append(
                        str(student_id)
                    )
                    encodings.append(
                        vector
                    )
                except Exception:
                    continue
        return (
            student_ids,
            encodings
        )

    def calculate_confidence(
        self,
        distance,
        threshold
    ):
        if distance >= threshold:
            return 0.0

        confidence = (1.0-(distance / threshold))* 100.0

        return max(
            0.0,
            min(
                100.0,
                confidence
            )
        )

    # multiframe confirmation

    def confirm_identity(
        self,
        student_id
    ):
        self.recognition_history.append(
            str(student_id)
        )
        if not self.recognition_history:
            return None

        counts = Counter(
            self.recognition_history
        )

        candidate, count = (
            counts.most_common(1)[0]
        )

        if candidate == "Unknown":
            return None

        if candidate == "Poor quality":
            return None

        if count >= CONFIRMATION_REQUIRED:
            self.last_recognized_id = (
                candidate
            )
            return candidate
        return None

    # face chena

    def recognize_faces(
        self,
        frame,
        tolerance=RECOGNITION_THRESHOLD
        ):
        if frame is None:
            return [], []

        locations = self.detect_faces(
            frame
        )

        if not locations:
            self.recognition_history.append(
                "Unknown"
            )
            return [], []

        valid_locations = []

        for location in locations:

            if self.is_good_face(
                frame,
                location
            ):
                valid_locations.append(
                    location
                )

        if not valid_locations:

            self.recognition_history.append(
                "Poor quality"
            )

            return (
                locations,
                [
                    "Poor quality"
                    for _ in locations
                ]
            )

        #image conversion

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # live encoding

        live_encodings = (
            face_recognition.face_encodings(
                rgb_frame,
                known_face_locations=valid_locations,
                num_jitters=1
            )
        )

        if not live_encodings:
            self.recognition_history.append(
                "Unknown"
            )
            return (
                locations,
                [
                    "Unknown"
                    for _ in locations
                ]
            )

        known_ids, known_vectors = (
            self.get_known_encodings()
        )

        if not known_vectors:
            return (
                locations,
                [
                    "No registered faces"
                    for _ in locations
                ]
            )

        results = []

        live_index = 0

        #detected face check

        for location in locations:
            if location not in valid_locations:
                results.append(
                    "Poor quality"
                )
                continue

            if live_index >= len(
                live_encodings
            ):
                results.append(
                    "Unknown"
                )
                continue

            live_encoding = (
                live_encodings[
                    live_index
                ]
            )

            live_index += 1

            # comparison of encodings

            distances = (
                face_recognition.face_distance(
                    known_vectors,
                    live_encoding
                )
            )

            if len(distances) == 0:
                results.append(
                    "Unknown"
                )
                continue

            best_index = int(
                np.argmin(
                    distances
                )
            )

            best_distance = float(
                distances[
                    best_index
                ]
            )

            candidate_id = (
                known_ids[
                    best_index
                ]
            )

            confidence = (
                self.calculate_confidence(
                    best_distance,
                    tolerance
                )
            )

            self.last_confidence = (
                confidence
            )

            # decision

            if best_distance <= tolerance:
                confirmed_id = (
                    self.confirm_identity(
                        candidate_id
                    )
                )

                if confirmed_id is not None:
                    results.append(
                        confirmed_id
                    )
                else:
                    results.append(
                        "Unknown"
                    )

            else:
                self.confirm_identity(
                    "Unknown"
                )

                results.append(
                    "Unknown"
                )

        return (
            locations,
            results
        )

    # encoding rebuild

    def rebuild_all_encodings(self):
        if not os.path.isdir(
            FACE_DIR
        ):

            return 0

        all_encodings = {}

        students_processed = 0

        for student_id in os.listdir(
            FACE_DIR
        ):
            student_dir = os.path.join(
                FACE_DIR,
                student_id
            )
            if not os.path.isdir(
                student_dir
            ):
                continue

            image_files = []

            for filename in os.listdir(
                student_dir
            ):
                if filename.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                ):
                    image_files.append(
                        os.path.join(
                            student_dir,
                            filename
                        )
                    )

            student_encodings = []

            for image_path in image_files:
                encoding = self.encode_image(
                    image_path
                )

                if encoding is not None:
                    student_encodings.append(
                        encoding
                    )

            if student_encodings:
                all_encodings[
                    str(student_id)
                ] = student_encodings

                students_processed += 1

        self.save_encodings(
            all_encodings
        )

        return students_processed