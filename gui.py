import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk

from recognition import FaceCamera

from database import (
    initialize_database,
    add_student,
    get_students,
    get_attendance,
    attendance_exists,
    add_attendance
)


class AttendanceApp:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            "Offline Face Recognition Attendance System"
        )

        # fullscreen e korlam
        self.root.state("zoomed")
        self.root.minsize(1280,720)

        self.face_camera = FaceCamera()

        self.registration_running = False
        self.attendance_running = False

        self.capture_count = 0
        self.max_captures = 10

        self.registration_student_id = None

        initialize_database()

        self.setup_style()

        self.create_layout()

        self.show_page(
            "Dashboard"
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )


    # tstyle set kortesi
    def setup_style(self):

        style = ttk.Style()

        try:
            style.theme_use(
                "equilux" #theme ta use korlam
            )
        except tk.TclError:
            pass

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 22)
        )

        style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI", 12)
        )

        style.configure(
            "Menu.TButton",
            font=("Segoe UI", 11),
            padding=5
        )

    # layout banailam

    def create_layout(self):
        self.content = ttk.Frame(self.root)

        self.content.pack(
            fill="both",
            expand=True
        )

        # shobgula page

        self.pages = {}

        self.create_dashboard()

        self.create_registration()

        self.create_students()

        self.create_attendance()

        self.create_reports()

    # page paltanor system

    def show_page(self, page_name):

        # registration er camera bondho by default
        if (
            page_name != "Registration"
            and self.registration_running
        ):
            self.stop_camera()

        # attendance er camera bondho by default
        if (
            page_name != "Attendance"
            and self.attendance_running
        ):
            self.stop_attendance_camera()

        for page in self.pages.values():

            page.pack_forget()

        page = self.pages.get(
            page_name
        )

        if page:

            page.pack(
                fill="both",
                expand=True
            )

        # page e dhukle refresh
        if page_name == "Students":

            self.load_students()

        # pgae e dhukle refresh
        if page_name == "Reports":

            self.load_attendance()

    # dashboard banabo

    def create_dashboard(self):

        page = ttk.Frame(
            self.content,
            padding=30
        )

        self.pages["Dashboard"] = page

        ttk.Label(
            page,
            text="Offline Attendance System using Face Recognition",
            style="Title.TLabel"
        ).pack(
            pady=(10, 8)
        )

        ttk.Label(
            page,
            text="Powered by AI",
            style="Subtitle.TLabel"
        ).pack(
            pady=(0, 50)
        )

        # 4 ta page er button banalam with dui row dui column
        button_grid = ttk.Frame(page)
        button_grid.pack()

        buttons = [
            ("Take Attendance", "Attendance"),
            ("View Reports", "Reports"),
            ("Registered Students", "Students"),
            ("Register Student", "Registration")
        ]

        for index, (button_text, target_page) in enumerate(buttons):
            row = index // 2
            column = index % 2

            ttk.Button(
                button_grid,
                text=button_text,
                style="Menu.TButton",
                width=20,
                command=lambda p=target_page: self.show_page(p)
            ).grid(
                row=row,
                column=column,
                padx=5,
                pady=5,
                ipadx=30,
                ipady=5
            )

        # exit button banailam
        ttk.Button(
            page,
            text="Exit",
            width=10,
            style="Menu.TButton",
            command=self.close_application
        ).pack(
            pady=(5, 5),
            ipadx=30,
            ipady=5
        )

    # registration er page banailam

    def create_registration(self):

        page = ttk.Frame(
            self.content,
            padding=30
        )

        self.pages["Registration"] = page

        ttk.Label(
            page,
            text="Register Student",
            style="Title.TLabel"
        ).pack(
            anchor="center",
            pady=(5, 15)
        )

        # form jekhane input jabe

        form = ttk.Frame(
            page
        )

        form.pack(
            anchor="center"
        )

        ttk.Label(
            form,
            text="Student ID:"
        ).grid(
            row=0,
            column=0,
            padx=10,
            pady=5,
            sticky="w"
        )

        self.student_id_entry = ttk.Entry(
            form,
            width=30
        )

        self.student_id_entry.grid(
            row=0,
            column=1,
            padx=10,
            pady=5
        )

        ttk.Label(
            form,
            text="Student Name:"
        ).grid(
            row=1,
            column=0,
            padx=10,
            pady=5,
            sticky="w"
        )

        self.student_name_entry = ttk.Entry(
            form,
            width=30
        )

        self.student_name_entry.grid(
            row=1,
            column=1,
            padx=10,
            pady=5
        )

        # camera er frame
        camera_frame_title = ttk.Label(
            page,
            text="Camera Preview",
        )
        camera_frame_title.pack(anchor="center", pady=(10,0))

        camera_frame = ttk.LabelFrame(
            page
        )

        camera_frame.pack(
            pady=10
        )

        self.camera_label = ttk.Label(
            camera_frame,
            text="Camera not started",
            width=50
        )

        self.camera_label.pack(
            padx=10,
            pady=10
        )

        self.capture_status = ttk.Label(
            page,
            text="Captured: 0 / 10"
        )

        self.capture_status.pack(
            pady=5
        )

        # camera er button

        buttons = ttk.Frame(
            page
        )

        buttons.pack(
            pady=10
        )

        ttk.Button(
            buttons,
            text="Start Camera",
            command=self.start_registration_camera
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Capture Face",
            command=self.capture_registration_face
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Save Student",
            command=self.save_student
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Stop Camera",
            command=self.stop_camera
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            page,
            text="Back",
            command=lambda: self.show_page("Dashboard")
        ).pack(
            side="bottom",
            padx=5,
            pady=(0,5)
        )

    # camera chalu

    def start_registration_camera(self):

        student_id = (
            self.student_id_entry
            .get()
            .strip()
        )

        if not student_id:
            messagebox.showwarning(
                "Missing Information",
                "Enter the Student ID first."
            )
            return

        if not self.face_camera.start():
            messagebox.showerror(
                "Camera Error",
                "Could not open the webcam."
            )
            return

        self.registration_student_id = student_id

        self.registration_running = True

        self.capture_count = 0

        self.capture_status.configure(
            text="Captured: 0 / 10"
        )

        self.update_registration_camera()

    # camera updates

    def update_registration_camera(self):

        if not self.registration_running:
            return

        frame = self.face_camera.read_frame()

        if frame is None:
            self.stop_camera()
            messagebox.showerror(
                "Camera Error",
                "Could not read from webcam."
            )
            return

        faces = self.face_camera.detect_faces(
            frame
        )

        frame = self.face_camera.draw_faces(
            frame,
            faces
        )

        image = self.face_camera.get_tk_image(
            frame,
            width=480,
            height=360
        )

        self.camera_label.configure(
            image=image,
            text=""
        )

        self.camera_label.image = image

        self.camera_label.after(
            30,
            self.update_registration_camera
        )

    # chobi tulo register koro

    def capture_registration_face(self):

        if not self.registration_running:
            messagebox.showwarning(
                "Camera",
                "Start the camera first."
            )
            return

        if self.capture_count >= self.max_captures:
            messagebox.showinfo(
                "Complete",
                "Maximum number of images captured."
            )
            return

        frame = self.face_camera.read_frame()
        if frame is None:
            messagebox.showerror(
                "Camera Error",
                "Could not capture frame."
            )
            return

        faces = self.face_camera.detect_faces(
            frame
        )

        if len(faces) == 0:
            messagebox.showwarning(
                "No Face",
                "No face was detected."
            )
            return

        if len(faces) > 1:
            messagebox.showwarning(
                "Multiple Faces",
                "Only one face should be visible."
            )
            return

        self.capture_count += 1

        filename = (
            self.face_camera
            .capture_face_image(
                frame,
                self.registration_student_id,
                self.capture_count
            )
        )

        self.capture_status.configure(
            text=(
                f"Captured: "
                f"{self.capture_count} / "
                f"{self.max_captures}"
            )
        )

        print(
            f"Saved: {filename}"
        )

        if (
            self.capture_count
            == self.max_captures
        ):

            messagebox.showinfo(
                "Complete",
                "All 10 images captured."
            )

    # camera bondho

    def stop_camera(self):

        self.registration_running = False

        self.face_camera.stop()

        self.camera_label.configure(
            image="",
            text="Camera stopped"
        )

        self.camera_label.image = None

    # student ke save koro

    def save_student(self):
        student_id = self.student_id_entry.get().strip()
        name = self.student_name_entry.get().strip()

        if not student_id or not name:
            messagebox.showwarning(
                "Missing Information",
                "Enter Student ID and Name."
            )
            return

        if self.capture_count == 0:
            messagebox.showwarning(
                "Face Required",
                "Capture at least one face image first."
            )
            return

        # sqlite e student ke add korlam
        success = add_student(student_id, name)

        if not success:
            messagebox.showerror(
                "Duplicate Student",
                "This Student ID already exists."
            )
            return

        # tola chobi diye face encoding banailam
        encoding_success, count = (
            self.face_camera.create_student_encoding(
                student_id
            )
        )

        if not encoding_success:
            messagebox.showerror(
                "Encoding Error",
                "The student was added to the database, "
                "but no valid facial encoding could be created."
            )
            self.stop_camera()
            return

        self.stop_camera()

        messagebox.showinfo(
            "Registration Complete",
            f"{name} registered successfully.\n\n"
            f"{count} facial encodings created."
        )

        self.student_id_entry.delete(
            0,
            tk.END
        )

        self.student_name_entry.delete(
            0,
            tk.END
        )

        self.capture_count = 0
        self.registration_student_id = None

        self.capture_status.configure(
            text="Captured: 0 / 10"
        )

    # registered student page

    def create_students(self):

        page = ttk.Frame(
            self.content,
            padding=30
        )

        self.pages["Students"] = page

        ttk.Label(
            page,
            text="Registered Students",
            style="Title.TLabel"
        ).pack(
            anchor="center",
            pady=(10, 5)
        )

        ttk.Label(
            page,
            text="Students currently registered in the system.",
            style="Subtitle.TLabel"
        ).pack(
            anchor="center",
            pady=(5, 10)
        )

        table_frame = ttk.Frame(
            page
        )

        table_frame.pack(
            fill="both",
            expand=True
        )

        self.students_table = ttk.Treeview(
            table_frame,
            columns=(
                "student_id",
                "name"
            ),
            show="headings"
        )

        self.students_table.heading(
            "student_id",
            text="Student ID"
        )

        self.students_table.heading(
            "name",
            text="Student Name"
        )

        self.students_table.column(
            "student_id",
            width=200,
            anchor="center"
        )

        self.students_table.column(
            "name",
            width=400,
            anchor="center"
        )

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.students_table.yview
        )

        self.students_table.configure(
            yscrollcommand=scrollbar.set
        )

        self.students_table.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )


        ttk.Button(
            page,
            text="Refresh",
            command=self.load_students
        ).pack(
            anchor="center",
            pady=10
        )

        ttk.Button(
            page,
            text="Back",
            command=lambda:
            self.show_page("Dashboard")
        ).pack(
            side="bottom",
            padx=5,
            pady=(0,5)
        )
        
        

    def load_students(self):

        for item in self.students_table.get_children():
            self.students_table.delete(
                item
            )

        students = get_students()

        for student_id, name in students:
            self.students_table.insert(
                "",
                "end",
                values=(
                    student_id,
                    name
                )
            )

    # attendance 

    def create_attendance(self):

        page = ttk.Frame(
            self.content,
            padding=30
        )

        self.pages["Attendance"] = page

        ttk.Label(
            page,
            text="Live Attendance",
            style="Title.TLabel"
        ).pack(
            anchor="center",
            pady=(5, 15)
        )

        camera_frame_title = ttk.Label(
            page,
            text="Camera Preview",
        )
        camera_frame_title.pack(anchor="center", pady=(10,0))

        camera_frame = ttk.LabelFrame(
            page,
        )

        camera_frame.pack(
            fill="both",
            expand=True
        )

        self.attendance_camera_label = ttk.Label(
            camera_frame,
            text="Webcam feed will appear here"
        )

        self.attendance_camera_label.pack(
            expand=True
        )

        self.recognized_label = ttk.Label(
            page,
            text="Recognized Student: —",
            font=("Segoe UI", 14)
        )

        self.recognized_label.pack(
            pady=8
        )

        self.status_label = ttk.Label(
            page,
            text="Status: Waiting",
            font=("Segoe UI", 12)
        )

        self.status_label.pack(
            pady=5
        )

        buttons = ttk.Frame(
            page
        )

        buttons.pack(
            pady=10
        )

        ttk.Button(
            buttons,
            text="Start Camera",
            command=self.start_attendance_camera
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            buttons,
            text="Stop Camera",
            command=self.stop_attendance_camera
        ).pack(
            side="left",
            padx=5
        )

        ttk.Button(
            page,
            text="Back",
            command=lambda: self.show_page("Dashboard")
        ).pack(
            side="bottom",
            pady=(0,5)
        )

    def start_attendance_camera(self):

        if not self.face_camera.start():
            messagebox.showerror(
                "Camera Error",
                "Could not open the webcam."
            )
            return

        self.attendance_running = True

        self.update_attendance_camera()

    def update_attendance_camera(self):

        if not self.attendance_running:
            return

        frame = self.face_camera.read_frame()

        if frame is None:
            self.stop_attendance_camera()
            return

        # face recognize kore

        locations, student_ids = (
            self.face_camera.recognize_faces(
                frame
            )
        )

        display_names = []

        for student_id in student_ids:
            if student_id == "Unknown":
                display_names.append(
                    "Unknown"
                )
                continue

            # student khuija bair koro
            students = get_students()

            student_name = None

            for db_student_id, name in students:
                if str(db_student_id) == str(student_id):
                    student_name = name
                    break

            if student_name is None:
                display_names.append(
                    "Unknown"
                )
                continue

            display_names.append(
                student_name
            )

            # attendance add maro

            from datetime import datetime

            now = datetime.now()

            date = now.strftime(
                "%Y-%m-%d"
            )

            time = now.strftime(
                "%H:%M:%S"
            )

            if not attendance_exists(
                student_id,
                date
            ):
                add_attendance(
                    student_id,
                    student_name,
                    date,
                    time
                )

                self.status_label.configure(
                    text=(
                        f"Attendance recorded: "
                        f"{student_name}"
                    )
                )

            else:
                self.status_label.configure(
                    text=(
                        f"Already marked today: "
                        f"{student_name}"
                    )
                )

        # result bair korilam

        frame = self.face_camera.draw_faces(
            frame,
            locations,
            display_names
        )

        image = self.face_camera.get_tk_image(
            frame,
            width=480,
            height=360
        )

        self.attendance_camera_label.configure(
            image=image,
            text=""
        )

        self.attendance_camera_label.image = image

        self.attendance_camera_label.after(
            100,
            self.update_attendance_camera
        )

    def stop_attendance_camera(self):

        self.attendance_running = False

        self.face_camera.stop()

        self.attendance_camera_label.configure(
            image="",
            text="Camera stopped"
        )

        self.attendance_camera_label.image = None

    # report page

    def create_reports(self):

        page = ttk.Frame(
            self.content,
            padding=30
        )

        self.pages["Reports"] = page

        ttk.Label(
            page,
            text="Attendance Reports",
            style="Title.TLabel"
        ).pack(
            anchor="center",
            pady=(5, 15)
        )

        columns = (
            "student_id",
            "name",
            "date",
            "time"
        )

        self.attendance_table = ttk.Treeview(
            page,
            columns=columns,
            show="headings"
        )

        self.attendance_table.heading(
            "student_id",
            text="Student ID"
        )

        self.attendance_table.heading(
            "name",
            text="Name"
        )

        self.attendance_table.heading(
            "date",
            text="Date"
        )

        self.attendance_table.heading(
            "time",
            text="Time"
        )

        self.attendance_table.column(
            "student_id",
            width=150,
            anchor="center"
        )

        self.attendance_table.column(
            "name",
            width=250,
            anchor="center"
        )

        self.attendance_table.column(
            "date",
            width=150,
            anchor="center"
        )

        self.attendance_table.column(
            "time",
            width=150,
            anchor="center"
        )

        self.attendance_table.pack(
            fill="both",
            expand=True
        )

        ttk.Button(
            page,
            text="Refresh",
            command=self.load_students
        ).pack(
            anchor="center",
            pady=10
        )

        ttk.Button(
            page,
            text="Back",
            command=lambda:
            self.show_page("Dashboard")
        ).pack(
            side="bottom",
            padx=5,
            pady=(0,5)
        )

    def load_attendance(self):

        for item in self.attendance_table.get_children():

            self.attendance_table.delete(
                item
            )

        records = get_attendance()

        for record in records:

            self.attendance_table.insert(
                "",
                "end",
                values=record
            )

    # application bondho

    def close_application(self):

        self.registration_running = False

        self.attendance_running = False

        self.face_camera.stop()

        self.root.destroy()

    # jate chole

    def run(self):
        
        self.root.mainloop()