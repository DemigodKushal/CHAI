# main.py
import os
import cv2
import time
import json
from datetime import datetime

# services (make sure these files exist with the implementations you provided)
from services.face_recognition_service import FaceRecognitionService
from services.flash_liveness_service import FlashLivenessService
from services.database_service import DatabaseService

# constants
IMAGE_DIR = "data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def safe_cast_int(x):
    try:
        return int(x)
    except Exception:
        return x

def compute_confidence_from_distance(dist):
    """
    Convert an L2 distance (InsightFace embedding distance) to a confidence score in (0,1].
    This is a heuristic: confidence = 1 / (1 + dist). Adjust if you prefer a different mapping.
    """
    try:
        d = float(dist)
        return float(1.0 / (1.0 + d))
    except Exception:
        return 0.0

def enroll_flow(cap, face_service: FaceRecognitionService, db_service: DatabaseService):
    """
    Enroll student from the current camera frame.
    Steps:
      - Capture current frame
      - Ensure face is present (via face_service.extract_embedding_from_frame)
      - Ask for name, roll_no, class_name via terminal input
      - Save image to data/images/{roll_no}.jpg
      - Extract embedding and add to FAISS index
      - Add student to DB
    """
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera read failed. Cannot enroll.")
        return

    # Quick check for face
    embeddings = face_service.extract_embedding_from_frame(frame)
    if not embeddings:
        print("⚠️ No face detected in the current frame. Please position face clearly and try again.")
        return

    # Ask user for details
    print("\n--- Enroll New Student ---")
    name = input("Full name: ").strip()
    if not name:
        print("Name required, aborting enroll.")
        return
    roll = input("Roll number (unique): ").strip()
    if not roll:
        print("Roll number required, aborting enroll.")
        return
    class_name = input("Class name: ").strip() or ""

    # Check for existing student with same roll
    existing = db_service.get_student_by_roll(roll)
    if existing:
        print(f"⚠️ Student with roll '{roll}' already exists (ID={existing.id}, name={existing.name}). Aborting.")
        return

    # Save image file
    image_path = os.path.join(IMAGE_DIR, f"{roll}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"💾 Captured image saved to: {image_path}")

    # Extract embedding (use image_path to ensure same pipeline as enrollment)
    emb = face_service.extract_embedding(image_path)
    if emb is None:
        print("❌ Failed to extract embedding from saved image. Enrollment aborted. (No face detected?)")
        try:
            os.remove(image_path)
        except Exception:
            pass
        return

    # Enroll in DB
    student = db_service.enroll_student(name=name, roll_no=roll, class_name=class_name, image_path=image_path)

    # Add to FAISS
    face_service.add_to_index(emb, student.id)
    print(f"✅ Enrollment successful: {name} (ID={student.id})\n")

def attendance_flow(cap, face_service: FaceRecognitionService, liveness_service: FlashLivenessService, db_service: DatabaseService):
    """
    Run liveness test then recognition and mark attendance.
    """
    print("\n⚡ Running flash-based liveness check (will open a fullscreen flash window)...")
    is_live = liveness_service.verify_liveness(cap)

    if not is_live:
        print("❌ Liveness failed. Attendance not marked.")
        return

    # read a fresh frame (liveness routine may have consumed frames)
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame after liveness check.")
        return

    # Extract embeddings from frame
    embeddings = face_service.extract_embedding_from_frame(frame)
    if not embeddings:
        print("⚠️ No face detected after liveness check. Try again.")
        return

    emb = embeddings[0]
    student_id, distance = face_service.find_match(emb)

    if student_id is None:
        print("❌ No match found in database.")
        # Optionally: offer quick enroll here
        choice = input("Do you want to enroll this face now? (y/N): ").strip().lower()
        if choice == 'y':
            enroll_flow(cap, face_service, db_service)
        return

    # student_id might be stored as string in id_map; convert if possible
    student_key = safe_cast_int(student_id)
    student = db_service.get_student_by_id(student_key)
    if student is None:
        # if lookup failed with casted id, try raw value
        student = db_service.get_student_by_id(student_id)

    confidence = compute_confidence_from_distance(distance)
    db_service.mark_attendance(student.id if student else student_id, confidence)

    name = student.name if student else str(student_id)
    print(f"✅ Attendance marked for {name} (id={student.id if student else student_id}, confidence={confidence:.3f})")

def main():
    print("Initializing services...")
    face_service = FaceRecognitionService()         # uses your original implementation with automatic .bin creation
    liveness_service = FlashLivenessService()       # uses your exact flash implementation (fullscreen)
    db_service = DatabaseService()                  # SQLAlchemy-backed DB service (SessionLocal)

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam. Exiting.")
        return

    print("\n=== CLI Attendance System ===")
    print("Press E to enroll a student from current frame")
    print("Press A or SPACE to run liveness + mark attendance")
    print("Press Q to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Camera read failed (frame not available). Retrying...")
                time.sleep(0.1)
                continue

            # Show live feed
            display = frame.copy()
            cv2.putText(display, "E:Enroll  A/Space:Attendance  Q:Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Live Feed", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord('e'):
                enroll_flow(cap, face_service, db_service)
            elif key == ord('a') or key == ord(' '):
                attendance_flow(cap, face_service, liveness_service, db_service)

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")

    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
            db_service.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
