import cv2
from services.flash_liveness_service import FlashLivenessService  # ✅ Import liveness module

class AttendanceService:
    def __init__(self, face_service, db_service):
        self.face_service = face_service
        self.db_service = db_service
        self.liveness_service = FlashLivenessService()  # ✅ Initialize liveness service

    def take_attendance(self):
        """Opens the camera, performs flash-based liveness detection, and marks attendance."""
        cap = cv2.VideoCapture(0)
        print("📷 Camera started... Press 's' to capture, 'q' to quit.")

        if not cap.isOpened():
            print("❌ Error: Camera could not be opened.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame.")
                break

            cv2.imshow("Attendance", frame)
            key = cv2.waitKey(1) & 0xFF

            # --- Take snapshot and process ---
            if key == ord("s"):
                print("💡 Running flash-based liveness check...")
                is_live = self.liveness_service.verify_liveness(cap)

                if not is_live:
                    print("❌ Spoof detected or failed liveness test. Try again.")
                    continue

                print("✅ Liveness confirmed! Proceeding with face recognition...")
                print("📸 Capturing frame...")

                embeddings = self.face_service.extract_embedding_from_frame(frame)
                if not embeddings:
                    print("⚠️ No face detected. Try again.")
                    continue

                recognized = False
                for emb in embeddings:
                    student_id, similarity = self.face_service.find_match(emb)

                    if student_id is not None:
                        confidence = similarity
                        student = self.db_service.get_student_by_id(student_id)

                        if student:
                            self.db_service.mark_attendance(student_id, confidence)
                            print(f"✅ {student.name} recognized (similarity={confidence:.2f})")

                            total = self.db_service.get_total_attendance_for_student(student_id)
                            print(f"📈 Total attendance for {student.name}: {total}")

                            recent = self.db_service.get_recent_attendance_for_student(student_id)
                            print(f"🕒 Last {len(recent)} attendance records for {student.name}:")
                            for a in recent:
                                ts = a.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(a, "timestamp") else "N/A"
                                print(f" - {ts}")

                            recognized = True
                        else:
                            print(f"⚠️ Student ID {student_id} not found in DB.")

                if not recognized:
                    print(f"❌ Unknown face detected. Redirecting to registration...")

            elif key == ord("q"):
                print("👋 Exiting attendance mode.")
                break

        cap.release()
        cv2.destroyAllWindows()
