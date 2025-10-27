import os
import cv2
from services.face_recognition_service import FaceRecognitionService
from services.database_service import DatabaseService
from services.attendance_service import AttendanceService

# ------------------ IMAGE CAPTURE FUNCTION ------------------
def capture_image(roll_no):
    """Capture an image via webcam and save to /data/images."""
    os.makedirs("data/images", exist_ok=True)
    cap = cv2.VideoCapture(0)
    print("\n📸 Press 's' to capture and save the image, or 'q' to quit without saving.")

    saved_path = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Unable to access camera.")
            break

        cv2.imshow("Enrollment Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            saved_path = os.path.join("data", "images", f"{roll_no}.jpg")
            cv2.imwrite(saved_path, frame)
            print(f"✅ Image saved: {saved_path}")
            break
        elif key == ord('q'):
            print("🚫 Enrollment cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_path

# ------------------ MAIN FLOW ------------------
def main():
    face_service = FaceRecognitionService(threshold=0.55)
    db_service = DatabaseService()
    attendance_service = AttendanceService(face_service, db_service)

    print("\n==== FACE ATTENDANCE SYSTEM ====\n")
    print("1️⃣ Enroll new student")
    print("2️⃣ Take attendance")
    choice = input("\nEnter choice: ")

    if choice == "1":
        name = input("Enter name: ")
        roll_no = input("Enter roll number: ")
        class_name = input("Enter class name: ")

        print("\nOpening camera for image capture...")
        img_path = capture_image(roll_no)
        if not img_path:
            print("❌ Image not captured. Enrollment aborted.")
            return

        embedding = face_service.extract_embedding(img_path)
        if embedding is None:
            print("❌ Could not extract embedding. Enrollment failed.")
            return

        student = db_service.enroll_student(name, roll_no, class_name, img_path)
        face_service.add_to_index(embedding, student.id)
        print("Index total:", face_service.index.ntotal)

    elif choice == "2":
        print("📷 Starting attendance process...")
        attendance_service.take_attendance()

    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
