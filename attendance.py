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

# People already marked present during this application session.
# This prevents unnecessary SQLite operations for repeated detections.
marked_today_cache = set()

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

    If the person has already been marked during this application
    session, skip the database operation completely.

    The database UNIQUE(name, date) constraint is still kept as a
    safety net in case the database already contains today's record.

    Returns True if a brand-new record was inserted,
    False otherwise.
    """
    # Fast path:
    # If we already marked this person during this session,
    # there is no reason to touch SQLite again.
    if name in marked_today_cache:
        return False

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    conn = _get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
            (name, today_str, time_str),
        )

        conn.commit()
        marked_new = True

        # Only add to the cache after a successful database insert.
        marked_today_cache.add(name)

    except sqlite3.IntegrityError:
        # The person may already have an attendance record from
        # earlier in the day, before this application was started.
        marked_new = False

        # Cache them so we don't query SQLite again during this session.
        marked_today_cache.add(name)

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
