"""
test_setup.py

A quick sanity check to run BEFORE using the real app - confirms every
required library is installed/importable, and that basics like the
webcam and sqlite3 work on this machine.

Usage:
    python test_setup.py
"""

import importlib
import sys

# Standard-library modules (sqlite3, tkinter) are included too, since a
# stripped-down Python install can sometimes be missing tkinter.
REQUIRED_MODULES = [
    "cv2",               # opencv-python
    "face_recognition",  # face-recognition (needs dlib under the hood)
    "dlib",
    "numpy",
    "PIL",                # pillow
    "sqlite3",             # standard library
    "tkinter",              # standard library (GUI)
]


def check_imports():
    print("Checking required libraries...")
    all_ok = True

    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown version")
            print(f"  [OK]   {module_name} ({version})")
        except ImportError as e:
            print(f"  [FAIL] {module_name} - {e}")
            all_ok = False

    return all_ok


def check_webcam():
    print("\nChecking webcam access...")
    try:
        import cv2
    except ImportError:
        print("  [SKIP] opencv-python isn't installed, can't test webcam.")
        return False

    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        print("  [OK]   Webcam opened successfully.")
        cam.release()
        return True

    print("  [FAIL] Could not open webcam (index 0). This is expected on a")
    print("         headless machine / server without a camera attached.")
    return False


def check_sqlite():
    print("\nChecking sqlite3...")
    import sqlite3
    print(f"  [OK]   sqlite3 library version: {sqlite3.sqlite_version}")
    return True


def main():
    print(f"Python version: {sys.version}\n")

    imports_ok = check_imports()
    webcam_ok = check_webcam()
    sqlite_ok = check_sqlite()

    print("\n--- Summary ---")
    print(f"Libraries: {'OK' if imports_ok else 'MISSING SOME - see [FAIL] lines above'}")
    print(f"Webcam:    {'OK' if webcam_ok else 'NOT AVAILABLE'}")
    print(f"sqlite3:   {'OK' if sqlite_ok else 'FAIL'}")

    if imports_ok and sqlite_ok:
        print("\nSetup looks good! (A webcam failure here is fine if you're")
        print("running this on a machine without a camera - main.py just")
        print("needs a webcam at the point you actually use it.)")
    else:
        print("\nSome required libraries are missing - install them (e.g. via")
        print("`uv sync` or `pip install -e .`) before running main.py.")


if __name__ == "__main__":
    main()
