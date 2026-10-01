from datetime import datetime

from database import (
    get_students,
    attendance_exists,
    add_attendance
)

def mark_attendance(name):
    students = get_students()

    student = None

    for student_id, student_name in students:
        if student_name.lower() == name.lower():
            student = (student_id, student_name)
            break

    if student is None:
        return False, "Student not registered"

    student_id, student_name = student

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if attendance_exists(student_id, date):
        return False, f"{student_name} is already marked present"

    add_attendance(
        student_id,
        student_name,
        date,
        time
    )

    return True, f"Attendance marked for {student_name}"