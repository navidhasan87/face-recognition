# face-recognition
File/Folder	Purpose
dataset/	Store pictures of people whose faces will be recognized as studentName.jpg
encodings/	Store the facial encodings generated from the dataset
attendance/	Store attendance records, e.g. studentName.db (using sqlite3: 3.53.4) 
my_face_recognition.py	Detect faces and compare them with known faces and if recognized  then write the name on the top of the box else write "not recognized" during live camera session
attendance.py	Record recognized people's attendance
camera.py	Open webcam and capture frames box the faces 
main.py	Run the whole application
test_setup.py	Check whether required libraries/environment work
gui.py	Give user an interface to interact, give attendance and if not in database then give option to add into database with necessary info using tkinter 8.6

at first there shall be no info in database and no encodings as well
project structure:
face-recognition/
├── attendance/
│   └── attendance.db
├── dataset/
│   └── student_name.jpg
├── encodings/
│   └── student_name.pkl
├── .gitignore
├── attendance.py
├── camera.py
├── gui.py
├── main.py
├── my_face_recognition.py
├── pyproject.toml
├── README.md
└── test_setup.py