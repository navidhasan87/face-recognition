"""
my_face_recognition.py

This module handles the "AI" part of the project:
- Turning a photo of a face into a list of numbers (an "encoding")
- Comparing a face seen by the camera to the encodings we already know
- Deciding whether a face is "recognized" or "not recognized"

We use the `face_recognition` library (built on top of dlib). It converts a
detected face into a 128-number vector called an "encoding". Two encodings
belonging to the same person will be close together in that 128-dimensional
space (small "distance" between them); different people will be far apart.
"""

import pickle
from pathlib import Path

import face_recognition
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# BASE_DIR = the folder this file lives in (the project root).
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
ENCODINGS_DIR = BASE_DIR / "encodings"

# How strict face matching should be. This is a "distance" threshold, not a
# percentage - lower means stricter (fewer wrong matches, but might miss a
# real match too). face_recognition's own default is 0.6; we use a slightly
# stricter 0.5 to reduce false positives for an attendance system.
MATCH_TOLERANCE = 0.5


def ensure_dirs():
    """Create dataset/ and encodings/ if they don't exist yet."""
    DATASET_DIR.mkdir(exist_ok=True)
    ENCODINGS_DIR.mkdir(exist_ok=True)


def encode_face_from_image(image_path):
    """
    Given the path to a photo that should contain exactly one face,
    return that face's 128-number encoding (a numpy array).

    Returns None if no face, or more than one face, is found - in either
    case we can't be confident which face to enroll.
    """
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) != 1:
        return None

    # We already know exactly where the one face is, so we pass that
    # location in instead of letting face_encodings() search again.
    encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    return encodings[0]


def save_encoding(name, encoding):
    """
    Save one person's encoding to encodings/<name>.pkl using pickle.
    `name` should already be a filesystem-safe string (no spaces, etc.).
    """
    ensure_dirs()
    file_path = ENCODINGS_DIR / f"{name}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(encoding, f)


def enroll_student_from_dataset(name):
    """
    Look for dataset/<name>.jpg, compute its encoding, and save it to
    encodings/<name>.pkl. Returns True on success, False if the photo is
    missing or doesn't contain exactly one clear face.
    """
    image_path = DATASET_DIR / f"{name}.jpg"
    if not image_path.exists():
        print(f"[my_face_recognition] No photo found at {image_path}")
        return False

    encoding = encode_face_from_image(str(image_path))
    if encoding is None:
        print(f"[my_face_recognition] Could not find exactly one face in {image_path}")
        return False

    save_encoding(name, encoding)
    print(f"[my_face_recognition] Enrolled '{name}' successfully.")
    return True


def load_known_encodings():
    """
    Load every .pkl file inside encodings/ and return two matching lists:
        known_names:      ["alice", "bob", ...]
        known_encodings:  [encoding_for_alice, encoding_for_bob, ...]

    On a brand-new project (nobody enrolled yet) both lists come back
    empty - callers should handle that gracefully instead of crashing.
    """
    ensure_dirs()
    known_names = []
    known_encodings = []

    for pkl_file in ENCODINGS_DIR.glob("*.pkl"):
        name = pkl_file.stem  # filename without the ".pkl" extension
        with open(pkl_file, "rb") as f:
            encoding = pickle.load(f)
        known_names.append(name)
        known_encodings.append(encoding)

    return known_names, known_encodings


def recognize_faces(frame, known_names, known_encodings):
    """
    Given one video frame (a raw OpenCV/numpy image) plus the known
    people's names/encodings, find every face in the frame and try to
    recognize each one.

    Returns a list of (name_or_"Not recognized", box) tuples, one per
    face found. `box` is a (top, right, bottom, left) tuple - the same
    format camera.py uses to draw rectangles.
    """
    # OpenCV reads frames in BGR color order, but face_recognition expects
    # RGB, so we reverse the color channel order before doing any face work.
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for face_encoding, location in zip(face_encodings, face_locations):
        name = "Not recognized"

        if known_encodings:  # nothing to compare against if nobody is enrolled
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances)

            if distances[best_match_index] <= MATCH_TOLERANCE:
                name = known_names[best_match_index]

        results.append((name, location))

    return results
