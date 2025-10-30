from database.models import Student, Attendance, SessionLocal
import datetime

class DatabaseService:
    def __init__(self):
        self.session = SessionLocal()

    # ------------------ Student Management ------------------

    def enroll_student(self, name, roll_no, class_name, image_path):
        """Enroll a new student in the database."""
        student = Student(name=name, roll_no=roll_no, class_name=class_name, image_path=image_path)
        self.session.add(student)
        self.session.commit()
        print(f"✅ Enrolled: {name} ({roll_no})")
        return student

    def get_student_by_id(self, student_id):
        """Fetch a student using their ID."""
        return self.session.query(Student).filter(Student.id == student_id).first()

    def get_student_by_roll(self, roll_no):
        """Fetch a student using their roll number."""
        return self.session.query(Student).filter(Student.roll_no == roll_no).first()

    def list_students(self):
        """Return all enrolled students."""
        return self.session.query(Student).all()

    # ------------------ Attendance Management ------------------

    def mark_attendance(self, student_id, confidence):
        """Mark attendance for a student with a given confidence."""
        attendance = Attendance(
            student_id=student_id,
            confidence=confidence,
            timestamp=datetime.datetime.now()
        )
        self.session.add(attendance)
        self.session.commit()
        print(f"🕒 Attendance marked for ID {student_id} (confidence={confidence:.3f})")

    def get_total_attendance_for_student(self, student_id):
        """Return total attendance count for a student."""
        return self.session.query(Attendance).filter(Attendance.student_id == student_id).count()

    def get_recent_attendance_for_student(self, student_id, limit=5):
        """Return recent attendance logs for a student."""
        return (
            self.session.query(Attendance)
            .filter(Attendance.student_id == student_id)
            .order_by(Attendance.timestamp.desc())
            .limit(limit)
            .all()
        )

    def get_all_attendance_records(self):
        """
        Fetch all attendance logs (joined with student details).
        Used in the Streamlit UI for displaying attendance history.
        """
        records = (
            self.session.query(Attendance, Student)
            .join(Student, Attendance.student_id == Student.id)
            .order_by(Attendance.timestamp.desc())
            .all()
        )

        return [
            {
                "name": student.name,
                "roll_no": student.roll_no,
                "class": student.class_name,
                "confidence": round(att.confidence, 3),
                "timestamp": att.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for att, student in records
        ]

    # ------------------ Utility ------------------

    def close(self):
        """Close the database session cleanly."""
        self.session.close()
