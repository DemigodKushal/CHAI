import face_recognition
import os
from db_manager import session, Student
from pinecone_manager import add_student_vector


def enroll_student(name, roll_number, class_name, ref_image_path):
    """
    Enroll student:
    1. Extract embedding from reference image
    2. Add metadata to SQLite
    3. Store embedding in Pinecone
    """
    if not os.path.exists(ref_image_path):
        print("Reference image not found!")
        return

    image = face_recognition.load_image_file(ref_image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        print("No face detected!")
        return

    embedding = encodings[0]

    # SQLite
    student = Student(
        name=name,
        roll_number=roll_number,
        class_name=class_name,
        ref_image_path=ref_image_path
    )
    session.add(student)
    session.commit()

    # Pinecone
    add_student_vector(student.student_id, embedding)
