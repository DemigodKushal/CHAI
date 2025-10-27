import cv2
import face_recognition
from datetime import datetime
import os
from db_manager import session, Attendance
from pinecone_manager import query_student


def mark_attendance(frame, threshold=0.4):
    """
    Mark attendance:
    1. Compute embedding from frame
    2. Query Pinecone for nearest student
    3. Save snapshot + record in SQLite
    """
    rgb_frame = frame[:, :, ::-1]
    encodings = face_recognition.face_encodings(rgb_frame)
    if not encodings:
        print("No face detected!")
        return

    embedding = encodings[0]
    student_id, score = query_student(embedding, threshold)
    if not student_id:
        print("No matching student found")
        return

    # Save snapshot
    date_str = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(f"storage/attendance/{date_str}", exist_ok=True)
    time_str = datetime.now().strftime("%H-%M-%S")
    snapshot_path = f"storage/attendance/{date_str}/{student_id}_{time_str}.jpg"
    cv2.imwrite(snapshot_path, frame)

    # Save attendance in SQLite
    record = Attendance(student_id=int(student_id), snapshot_path=snapshot_path)
    session.add(record)
    session.commit()

    print(f"Attendance marked for Student ID {student_id} (Score: {score:.2f})")
