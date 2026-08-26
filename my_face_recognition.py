import os
import pickle
import face_recognition


# Folders
DATASET_FOLDER = "dataset"
ENCODINGS_FOLDER = "encodings"


def create_encodings():
    # Create encodings folder if it does not exist
    os.makedirs(ENCODINGS_FOLDER, exist_ok=True)

    # Student images
    students = [
        "student1.jpg",
        "student2.jpg",
        "student3.jpg"
    ]

    for student in students:

        image_path = os.path.join(DATASET_FOLDER, student)

        # Load image
        image = face_recognition.load_image_file(image_path)

        # Find face locations
        face_locations = face_recognition.face_locations(image)

        # Get face encoding
        encoding = face_recognition.face_encodings(
            image,
            face_locations
        )[0]

        # Student name
        student_name = student.replace(".jpg", "")

        # Save encoding
        encoding_path = os.path.join(
            ENCODINGS_FOLDER,
            student_name + ".pkl"
        )

        with open(encoding_path, "wb") as file:
            pickle.dump(encoding, file)

        print(f"Encoding created for {student_name}")


if __name__ == "__main__":
    create_encodings()