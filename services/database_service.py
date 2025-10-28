from database.models import Student, Attendance, SessionLocal
import datetime

class DatabaseService:
    def __init__(self):
        self.session = SessionLocal()

    def enroll_student(self, name, roll_no, class_name, image_path):
        student = Student(name=name, roll_no=roll_no, class_name=class_name, image_path=image_path)
        self.session.add(student)
        self.session.commit()
        print(f"✅ Enrolled: {name} ({roll_no})")
        return student

    def mark_attendance(self, student_id, confidence):
        attendance = Attendance(student_id=student_id, confidence=confidence, timestamp=datetime.datetime.now())
        self.session.add(attendance)
        self.session.commit()
        print(f"🕒 Attendance marked for ID {student_id} (confidence={confidence:.3f})")

    def get_student_by_id(self, student_id):
        return self.session.query(Student).filter(Student.id == student_id).first()

    def list_students(self):
        return self.session.query(Student).all()