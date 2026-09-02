"""
main.py

Entry point for the whole application. Running this file:
1. Makes sure the dataset/, encodings/, and attendance/ folders exist
2. Initializes the attendance database (creates the table if needed)
3. Launches the GUI

Usage:
    python main.py
"""

import attendance
import my_face_recognition as fr
from gui import launch_gui


def main():
    fr.ensure_dirs()
    attendance.init_db()
    launch_gui()


if __name__ == "__main__":
    main()
