import sqlite3
import os

DATABASE_DIR = "data"
DATABASE_FILE = os.path.join(
    DATABASE_DIR,
    "attendance.db"
)

def initialize_database():
    """Create the database and required tables."""

    os.makedirs(
        DATABASE_DIR,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    # student er table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL
        )
    """)

    # attendance er table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def add_student(student_id, name):
    """Add a new student to the database."""

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO students (student_id, name)
            VALUES (?, ?)
            """,
            (student_id, name)
        )

        connection.commit()

        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        connection.close()

def get_students():
    """Return all registered students."""

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute("""
        SELECT student_id, name
        FROM students
        ORDER BY student_id
    """)

    students = cursor.fetchall()

    connection.close()

    return students


def student_exists(student_id):
    """Check whether a student is registered."""

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT 1
        FROM students
        WHERE student_id = ?
        """,
        (student_id,)
    )

    result = cursor.fetchone()

    connection.close()

    return result is not None

def add_attendance(student_id, name, date, time):
    """Record attendance."""

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO attendance
        (student_id, name, date, time)
        VALUES (?, ?, ?, ?)
        """,
        (student_id, name, date, time)
    )

    connection.commit()
    connection.close()

def attendance_exists(student_id, date):
    """Check whether attendance was already recorded today."""

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT 1
        FROM attendance
        WHERE student_id = ?
        AND date = ?
        """,
        (student_id, date)
    )

    result = cursor.fetchone()

    connection.close()

    return result is not None

def get_attendance():
    """Return all attendance records."""

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute("""
        SELECT student_id, name, date, time
        FROM attendance
        ORDER BY date DESC, time DESC
    """)

    records = cursor.fetchall()

    connection.close()

    return records