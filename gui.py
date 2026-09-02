"""
gui.py

The user-facing window, built with tkinter. From here the user can:
- Start a live attendance session (opens the webcam window from camera.py)
- Add a new student: type their name, take their photo, and enroll them
  so the camera can recognize them in future sessions
- View who has already been marked present today

This file is kept deliberately "dumb" - it only calls functions from
attendance.py, camera.py and my_face_recognition.py and displays the
results. All the real logic lives in those modules.
"""

import tkinter as tk
from tkinter import messagebox, ttk

import attendance
import camera
import my_face_recognition as fr


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance")
        self.root.geometry("420x320")
        self.root.resizable(False, False)

        # Make sure dataset/encodings folders and the database table exist
        # before the user can click anything.
        fr.ensure_dirs()
        attendance.init_db()

        self._build_widgets()

    def _build_widgets(self):
        title = tk.Label(
            self.root, text="Face Recognition Attendance", font=("Segoe UI", 14, "bold")
        )
        title.pack(pady=15)

        tk.Button(
            self.root,
            text="Start Attendance (Camera)",
            width=30,
            height=2,
            command=self.start_attendance,
        ).pack(pady=8)

        tk.Button(
            self.root,
            text="Add New Student",
            width=30,
            height=2,
            command=self.open_add_student_window,
        ).pack(pady=8)

        tk.Button(
            self.root,
            text="View Today's Attendance",
            width=30,
            height=2,
            command=self.show_today_attendance,
        ).pack(pady=8)

        tk.Button(self.root, text="Quit", width=30, command=self.root.destroy).pack(pady=8)

    # ------------------------------------------------------------------
    # Start Attendance
    # ------------------------------------------------------------------
    def start_attendance(self):
        """
        Opens the webcam window. Every time a known face is recognized,
        camera.py calls attendance.mark_attendance(name) for us - that's
        the `on_recognized` callback wired up below.
        """
        messagebox.showinfo(
            "Attendance",
            "The camera window will open now.\nPress 'q' in that window to stop.",
        )
        camera.run_live_recognition(on_recognized=attendance.mark_attendance)

    # ------------------------------------------------------------------
    # Add New Student
    # ------------------------------------------------------------------
    def open_add_student_window(self):
        add_window = tk.Toplevel(self.root)
        add_window.title("Add New Student")
        add_window.geometry("320x180")
        add_window.resizable(False, False)

        tk.Label(add_window, text="Student name:").pack(pady=(20, 5))
        name_entry = tk.Entry(add_window, width=30)
        name_entry.pack()

        status_label = tk.Label(add_window, text="", fg="red")
        status_label.pack(pady=5)

        def handle_capture():
            raw_name = name_entry.get().strip()
            if not raw_name:
                status_label.config(text="Please enter a name first.")
                return

            # Turn "John Doe" into a filesystem-safe "john_doe" so it works
            # as both a filename (dataset/encodings) and a stored name.
            safe_name = raw_name.lower().replace(" ", "_")
            photo_path = fr.DATASET_DIR / f"{safe_name}.jpg"
            fr.ensure_dirs()

            status_label.config(text="Opening camera... press SPACE to capture.", fg="blue")
            add_window.update()  # repaint the label now, before the camera call blocks

            saved = camera.capture_single_photo(photo_path)
            if not saved:
                status_label.config(text="Cancelled - no photo taken.", fg="red")
                return

            success = fr.enroll_student_from_dataset(safe_name)
            if success:
                status_label.config(text=f"'{raw_name}' enrolled successfully!", fg="green")
            else:
                status_label.config(
                    text="Couldn't detect exactly one clear face. Try again.", fg="red"
                )

        tk.Button(add_window, text="Take Photo & Enroll", command=handle_capture).pack(pady=15)

    # ------------------------------------------------------------------
    # View Today's Attendance
    # ------------------------------------------------------------------
    def show_today_attendance(self):
        records = attendance.get_today_attendance()

        view_window = tk.Toplevel(self.root)
        view_window.title("Today's Attendance")
        view_window.geometry("300x300")

        if not records:
            tk.Label(view_window, text="No one has been marked present yet today.").pack(pady=20)
            return

        tree = ttk.Treeview(view_window, columns=("name", "time"), show="headings")
        tree.heading("name", text="Name")
        tree.heading("time", text="Time")
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        for name, time_str in records:
            tree.insert("", "end", values=(name, time_str))


def launch_gui():
    root = tk.Tk()
    AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
