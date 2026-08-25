import csv
from datetime import datetime

from database import get_attendance


def get_report():
    return get_attendance()


def export_csv(filename=None):

    records = get_attendance()

    if filename is None:
        date = datetime.now().strftime("%Y-%m-%d")
        filename = f"Attendance_Report_{date}.csv"

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Student ID",
            "Name",
            "Date",
            "Time"
        ])

        writer.writerows(records)

    return filename