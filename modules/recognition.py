import cv2

from services.face_recognition_service import FaceRecognitionService

class FaceAppController:
    def __init__(self, gui = None):
        self.gui = gui
        self.face_recognition_service = FaceRecognitionService()

    def _notify(self, message):
        """Helper: send message to GUI if available, else print."""
        if self.gui:
            self.gui.show_message(message)
        else:
            print(message)

    # GUI triggers this when user clicks “Take Attendance”
    def take_attendance(self):
        try:
            img = self.face_recognition_service.capture()
            emb = self.face_recognition_service.get_embeddings(img)
            student_id, distance = self.face_recognition_service.match_embeddings(emb)

            if student_id is None:
                self._notify("No match found.")
                return None

            self._notify(f"Attendance marked for {student_id}")
            return student_id

        except (RuntimeError, ValueError) as e:
            self._notify(str(e))
            return None

        except LookupError as e:
            self.face_recognition_service.add_embeddings(emb,1)

