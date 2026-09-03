"""
attendance.py

Manages the attendance database (SQLite, via the standard-library sqlite3
module). Responsible for:
- Creating the database/table the first time the app runs
- Marking someone present (only once per day, even if the camera sees
  their face across many frames in one session)
- Reading back attendance records (e.g. for the GUI to display)
"""

import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ATTENDANCE_DIR = BASE_DIR / "attendance"
DB_PATH = ATTENDANCE_DIR / "attendance.db"


def _get_connection():
    """Open the attendance database, creating the folder/file if needed."""
    ATTENDANCE_DIR.mkdir(exist_ok=True)
    # sqlite3.connect() creates the .db file automatically if it's missing.
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Create the attendance table if it doesn't exist yet. Safe to call
    every time the app starts - it's a no-op once the table is already
    there.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            UNIQUE(name, date)
        )
        """
    )
    # UNIQUE(name, date) above stops the same person from getting two rows
    # on the same day - see mark_attendance() for how we rely on that.
    conn.commit()
    conn.close()


def mark_attendance(name):
    """
    Insert an attendance row for `name` with today's date/time.

    If they're already marked present today, the UNIQUE(name, date)
    constraint makes the INSERT a no-op.

    Returns True if a brand-new record was inserted, False if they were
    already marked present today.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Attendance name must be a non-empty string.")

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    conn = _get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)
            ON CONFLICT(name, date) DO NOTHING
            """,
            (name.strip(), today_str, time_str),
        )
        conn.commit()
        marked_new = cursor.rowcount == 1
    finally:
        conn.close()

    return marked_new


def get_today_attendance():
    """Return [(name, time), ...] for everyone marked present today."""
    today_str = datetime.now().strftime("%Y-%m-%d")

    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, time FROM attendance WHERE date = ? ORDER BY time",
        (today_str,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_all_attendance():
    """Return every attendance record ever recorded, most recent first."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, date, time FROM attendance ORDER BY date DESC, time DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
