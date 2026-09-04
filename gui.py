from datetime import datetime
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk
import cv2
from PIL import Image, ImageTk

import attendance
import camera
import my_face_recognition as fr


class AttendanceApp:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title("Offline Face Recognition Attendance System")

        self.root.geometry("1100x800")

        self.root.minsize(900, 650)

        # -------------------------
        # Application State
        # -------------------------

        self.cap = None

        self.registration_running = False
        self.attendance_running = False

        self.capture_count = 0
        self.max_captures = 10

        self.registration_student_id = None

        # -------------------------
        # Database & Dirs
        # -------------------------

        fr.ensure_dirs()
        attendance.init_db()

        # -------------------------
        # GUI
        # -------------------------

        self.setup_style()

        self.create_layout()

        self.show_page("Dashboard")

        # -------------------------
        # Closing
        # -------------------------

        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

    # =====================================================
    # STYLE
    # =====================================================

    def setup_style(self):

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"))

        style.configure("Subtitle.TLabel", font=("Segoe UI", 12))

        style.configure("Menu.TButton", font=("Segoe UI", 11), padding=10)

    # =====================================================
    # MAIN LAYOUT
    # =====================================================

    def create_layout(self):

        self.sidebar = ttk.Frame(self.root, width=230)

        self.sidebar.pack(side="left", fill="y")

        self.sidebar.pack_propagate(False)

        self.content = ttk.Frame(self.root)

        self.content.pack(side="right", fill="both", expand=True)

        self.pages = {}

        self.create_sidebar()

        self.create_dashboard()

        self.create_registration()

        self.create_students()

        self.create_attendance()

        self.create_reports()

    # =====================================================
    # SIDEBAR
    # =====================================================

    def create_sidebar(self):

        title = ttk.Label(
            self.sidebar,
            text="Attendance\nSystem",
            font=("Segoe UI", 18, "bold"),
            justify="center",
        )

        title.pack(pady=30)

        buttons = [
            ("Dashboard", "Dashboard"),
            ("Register Student", "Registration"),
            ("Registered Students", "Students"),
            ("Start Attendance", "Attendance"),
            ("View Reports", "Reports"),
        ]

        for text, page in buttons:

            button = ttk.Button(
                self.sidebar,
                text=text,
                style="Menu.TButton",
                command=lambda p=page: self.show_page(p),
            )

            button.pack(fill="x", padx=20, pady=5)

        exit_button = ttk.Button(
            self.sidebar,
            text="Exit",
            style="Menu.TButton",
            command=self.close_application,
        )

        exit_button.pack(side="bottom", fill="x", padx=20, pady=20)

    # =====================================================
    # PAGE SWITCHING
    # =====================================================

    def show_page(self, page_name):

        if page_name != "Registration" and self.registration_running:
            self.stop_camera()

        if page_name != "Attendance" and self.attendance_running:
            self.stop_attendance_camera()

        for page in self.pages.values():

            page.pack_forget()

        page = self.pages.get(page_name)

        if page:

            page.pack(fill="both", expand=True)

        if page_name == "Students":

            self.load_students()

        if page_name == "Reports":

            self.load_attendance()

    # =====================================================
    # DASHBOARD
    # =====================================================

    def create_dashboard(self):

        page = ttk.Frame(self.content, padding=30)

        self.pages["Dashboard"] = page

        ttk.Label(
            page, text="Face Recognition Attendance System", style="Title.TLabel"
        ).pack(anchor="w", pady=(10, 5))

        ttk.Label(
            page,
            text="Offline AI-powered attendance management system",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(0, 40))

        ttk.Button(
            page,
            text="Register New Student",
            style="Menu.TButton",
            command=lambda: self.show_page("Registration"),
        ).pack(fill="x", pady=10)

        ttk.Button(
            page,
            text="Registered Students",
            style="Menu.TButton",
            command=lambda: self.show_page("Students"),
        ).pack(fill="x", pady=10)

        ttk.Button(
            page,
            text="Start Attendance",
            style="Menu.TButton",
            command=lambda: self.show_page("Attendance"),
        ).pack(fill="x", pady=10)

        ttk.Button(
            page,
            text="View Reports",
            style="Menu.TButton",
            command=lambda: self.show_page("Reports"),
        ).pack(fill="x", pady=10)

    # =====================================================
    # REGISTRATION
    # =====================================================

    def create_registration(self):

        page = ttk.Frame(self.content, padding=25)

        self.pages["Registration"] = page

        ttk.Label(page, text="Register Student", style="Title.TLabel").pack(
            anchor="w", pady=(5, 15)
        )

        form = ttk.Frame(page)

        form.pack(anchor="w")

        ttk.Label(form, text="Student Name:").grid(
            row=0, column=0, padx=10, pady=5, sticky="w"
        )

        self.student_name_entry = ttk.Entry(form, width=30)

        self.student_name_entry.grid(row=0, column=1, padx=10, pady=5)

        camera_frame = ttk.LabelFrame(page, text="Camera Preview")

        camera_frame.pack(pady=10)

        self.camera_label = ttk.Label(
            camera_frame, text="Camera not started", width=50
        )

        self.camera_label.pack(padx=10, pady=10)

        self.capture_status = ttk.Label(page, text="Captured: 0 / 10")

        self.capture_status.pack(pady=5)

        buttons = ttk.Frame(page)

        buttons.pack(pady=10)

        ttk.Button(
            buttons, text="Start Camera", command=self.start_registration_camera
        ).pack(side="left", padx=5)

        ttk.Button(
            buttons, text="Capture Face", command=self.capture_registration_face
        ).pack(side="left", padx=5)

        ttk.Button(buttons, text="Save Student", command=self.save_student).pack(
            side="left", padx=5
        )

        ttk.Button(buttons, text="Stop Camera", command=self.stop_camera).pack(
            side="left", padx=5
        )

    # =====================================================
    # START REGISTRATION CAMERA
    # =====================================================

    def start_registration_camera(self):

        name = self.student_name_entry.get().strip()

        if not name:

            messagebox.showwarning(
                "Missing Information", "Enter the Student Name first."
            )

            return

        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():

            messagebox.showerror("Camera Error", "Could not open the webcam.")

            return

        self.registration_student_id = name.lower().replace(" ", "_")

        self.registration_running = True

        self.capture_count = 0

        self.capture_status.configure(text="Captured: 0 / 10")

        self.update_registration_camera()

    # =====================================================
    # REGISTRATION CAMERA UPDATE
    # =====================================================

    def update_registration_camera(self):

        if not self.registration_running or self.cap is None:

            return

        ok, frame = self.cap.read()

        if not ok:

            self.stop_camera()

            messagebox.showerror("Camera Error", "Could not read from webcam.")

            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame).resize((480, 360))
        tk_img = ImageTk.PhotoImage(image=img)

        self.camera_label.configure(image=tk_img, text="")

        self.camera_label.image = tk_img

        self.root.after(30, self.update_registration_camera)

    # =====================================================
    # CAPTURE FACE
    # =====================================================

    def capture_registration_face(self):

        if not self.registration_running or self.cap is None:

            messagebox.showwarning("Camera", "Start the camera first.")

            return

        if self.capture_count >= self.max_captures:

            messagebox.showinfo("Complete", "Maximum number of images captured.")

            return

        ok, frame = self.cap.read()

        if not ok:

            messagebox.showerror("Camera Error", "Could not capture frame.")

            return

        self.capture_count += 1

        # Saves dataset picture in your dataset folder
        photo_path = fr.DATASET_DIR / f"{self.registration_student_id}.jpg"
        cv2.imwrite(str(photo_path), frame)

        self.capture_status.configure(
            text=f"Captured: {self.capture_count} / {self.max_captures}"
        )

        if self.capture_count == self.max_captures:

            messagebox.showinfo("Complete", "All 10 images captured.")

    # =====================================================
    # STOP REGISTRATION CAMERA
    # =====================================================

    def stop_camera(self):

        self.registration_running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.camera_label.configure(image="", text="Camera stopped")

        self.camera_label.image = None

    # =====================================================
    # SAVE STUDENT
    # =====================================================

    def save_student(self):

        name = self.student_name_entry.get().strip()

        if not name:
            messagebox.showwarning("Missing Information", "Enter Student Name.")
            return

        if self.capture_count == 0:
            messagebox.showwarning(
                "Face Required", "Capture at least one face image first."
            )
            return

        safe_name = name.lower().replace(" ", "_")

        # Encode dataset picture to encodings/ folder
        encoding_success = fr.enroll_student_from_dataset(safe_name)

        if not encoding_success:
            messagebox.showerror(
                "Encoding Error",
                "Photo captured in dataset, but no clear face was detected to create encoding.",
            )
            self.stop_camera()
            return

        self.stop_camera()

        messagebox.showinfo(
            "Registration Complete", f"{name} registered successfully."
        )

        self.student_name_entry.delete(0, tk.END)

        self.capture_count = 0
        self.registration_student_id = None

        self.capture_status.configure(text="Captured: 0 / 10")

    # =====================================================
    # REGISTERED STUDENTS
    # =====================================================

    def create_students(self):

        page = ttk.Frame(self.content, padding=30)

        self.pages["Students"] = page

        ttk.Label(page, text="Registered Students", style="Title.TLabel").pack(
            anchor="w", pady=(5, 10)
        )

        ttk.Label(
            page,
            text="Students currently registered in the system (dataset & encodings).",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(0, 15))

        table_frame = ttk.Frame(page)

        table_frame.pack(fill="both", expand=True)

        self.students_table = ttk.Treeview(
            table_frame, columns=("student_id", "name"), show="headings"
        )

        self.students_table.heading("student_id", text="Student Dataset Name")

        self.students_table.heading("name", text="Encoding File")

        self.students_table.column("student_id", width=300, anchor="center")

        self.students_table.column("name", width=300)

        scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.students_table.yview
        )

        self.students_table.configure(yscrollcommand=scrollbar.set)

        self.students_table.pack(side="left", fill="both", expand=True)

        scrollbar.pack(side="right", fill="y")

        buttons = ttk.Frame(page)

        buttons.pack(fill="x", pady=15)

        ttk.Button(buttons, text="Refresh", command=self.load_students).pack(
            side="left", padx=5
        )

        ttk.Button(
            buttons, text="Back", command=lambda: self.show_page("Dashboard")
        ).pack(side="right", padx=5)

    def load_students(self):

        for item in self.students_table.get_children():

            self.students_table.delete(item)

        known_names, _ = fr.load_known_encodings()

        for name in known_names:

            self.students_table.insert("", "end", values=(name, f"{name}.pkl"))

    # =====================================================
    # ATTENDANCE
    # =====================================================

    def create_attendance(self):

        page = ttk.Frame(self.content, padding=30)

        self.pages["Attendance"] = page

        ttk.Label(page, text="Live Attendance", style="Title.TLabel").pack(
            anchor="w", pady=(5, 15)
        )

        camera_frame = ttk.LabelFrame(page, text="Camera")

        camera_frame.pack(fill="both", expand=True)

        self.attendance_camera_label = ttk.Label(
            camera_frame, text="Webcam feed will appear here"
        )

        self.attendance_camera_label.pack(expand=True)

        self.status_label = ttk.Label(
            page, text="Status: Waiting", font=("Segoe UI", 12)
        )

        self.status_label.pack(pady=5)

        buttons = ttk.Frame(page)

        buttons.pack(pady=10)

        ttk.Button(
            buttons, text="Start Camera", command=self.start_attendance_camera
        ).pack(side="left", padx=5)

        ttk.Button(
            buttons, text="Stop Camera", command=self.stop_attendance_camera
        ).pack(side="left", padx=5)

    def start_attendance_camera(self):

        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():

            messagebox.showerror("Camera Error", "Could not open the webcam.")

            return

        self.known_names, self.known_encodings = fr.load_known_encodings()

        self.attendance_running = True

        self.update_attendance_camera()

    def update_attendance_camera(self):

        if not self.attendance_running or self.cap is None:
            return

        ok, frame = self.cap.read()

        if not ok:

            self.stop_attendance_camera()
            return

        recognized_faces = fr.recognize_faces(
            frame, self.known_names, self.known_encodings
        )

        for name, box in recognized_faces:

            camera.draw_face_box(frame, name, box)

            if name != "Not recognized":

                is_new = attendance.mark_attendance(name)

                if is_new:
                    self.status_label.configure(
                        text=f"Attendance recorded: {name}"
                    )
                else:
                    self.status_label.configure(
                        text=f"Already marked today: {name}"
                    )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(rgb_frame).resize((640, 480))

        tk_img = ImageTk.PhotoImage(image=img)

        self.attendance_camera_label.configure(image=tk_img, text="")

        self.attendance_camera_label.image = tk_img

        self.root.after(30, self.update_attendance_camera)

    def stop_attendance_camera(self):

        self.attendance_running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.attendance_camera_label.configure(
            image="", text="Camera stopped"
        )

        self.attendance_camera_label.image = None

    # =====================================================
    # REPORTS
    # =====================================================

    def create_reports(self):

        page = ttk.Frame(self.content, padding=30)

        self.pages["Reports"] = page

        ttk.Label(page, text="Attendance Reports", style="Title.TLabel").pack(
            anchor="w", pady=(5, 15)
        )

        columns = ("name", "date", "time")

        self.attendance_table = ttk.Treeview(
            page, columns=columns, show="headings"
        )

        self.attendance_table.heading("name", text="Name")

        self.attendance_table.heading("date", text="Date")

        self.attendance_table.heading("time", text="Time")

        self.attendance_table.column("name", width=250)

        self.attendance_table.column("date", width=150)

        self.attendance_table.column("time", width=150)

        self.attendance_table.pack(fill="both", expand=True)

        ttk.Button(
            page, text="Refresh", command=self.load_attendance
        ).pack(anchor="e", pady=10)

    def load_attendance(self):

        for item in self.attendance_table.get_children():

            self.attendance_table.delete(item)

        records = attendance.get_all_attendance()

        for record in records:

            self.attendance_table.insert("", "end", values=record)

    # =====================================================
    # CLOSE APPLICATION
    # =====================================================

    def close_application(self):

        self.registration_running = False

        self.attendance_running = False

        if self.cap is not None:
            self.cap.release()

        self.root.destroy()

    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        self.root.mainloop()


def launch_gui():
    app = AttendanceApp()
    app.run()


if __name__ == "__main__":
    launch_gui()