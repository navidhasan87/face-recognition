# Offline Face Recognition using Face Recognition

This App has 4 functions-
- Register students for attendance 
- Take attendance
- See who are registered
- See who have given attendance
#
- Student database is created at `/data/face_encodings.pkl` and `/faces/*`
- Attendance database is created at /data/attendance.database
#
- gui.py holds the container for everything.
- database.py creates the attendance database.
- recognition.py handles the facial recognition using cv2 hog model.
- attendance.py marks attendance for students.
- main.py runs gui.py which runs other modules.