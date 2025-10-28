import cv2

class AttendanceService:
    def __init__(self, face_service, db_service):
        self.face_service = face_service
        self.db_service = db_service

    def take_attendance(self):
        cap = cv2.VideoCapture(0)
        print("📷 Camera started... Press 's' to capture, 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame.")
                break

            # Display live feed
            cv2.imshow("Attendance", frame)
            key = cv2.waitKey(1) & 0xFF

            # Capture and process when 's' pressed
            if key == ord("s"):
                print("📸 Capturing frame...")

                embeddings = self.face_service.extract_embedding_from_frame(frame)

                if not embeddings:
                    print("⚠️ No face detected. Try again.")
                    continue

                for emb in embeddings:
                    student_id, similarity = self.face_service.find_match(emb)

                    if student_id is not None:
                        confidence = 1 - (similarity/2)  # cosine similarity directly
                        student = self.db_service.get_student_by_id(student_id)
                        self.db_service.mark_attendance(student_id, confidence)
                        print(f"✅ {student.name} recognized (similarity={confidence:.2f})")
                    else:
                        print(f"❌ Unknown face detected (similarity={similarity:.2f})")

            elif key == ord("q"):
                print("👋 Exiting attendance mode.")
                break

        cap.release()
        cv2.destroyAllWindows()