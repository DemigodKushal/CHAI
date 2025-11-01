from database.models import Student, Attendance, SessionLocal
import datetime
from sqlalchemy import func

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
        """Mark attendance for a student."""
        attendance = Attendance(student_id=student_id, confidence=confidence)
        self.session.add(attendance)
        self.session.commit()
        print(f"🕒 Attendance marked for ID {student_id} (confidence={confidence:.3f})")
        return attendance
    
    def is_attendance_marked_today(self, student_id):
        """Check if attendance already marked for student today"""
        today = datetime.date.today()
        
        attendance = self.session.query(Attendance).filter(
            Attendance.student_id == student_id,
            func.date(Attendance.timestamp) == today
        ).first()
        
        return attendance is not None
    
    def get_attendance_by_date_range(self, start_date=None, end_date=None):
        """Get attendance records grouped by date"""
        from datetime import timedelta
        
        if end_date is None:
            end_date = datetime.datetime.now().date()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        records = self.session.query(Attendance).join(Student).filter(
            func.date(Attendance.timestamp) >= start_date,
            func.date(Attendance.timestamp) <= end_date
        ).order_by(Attendance.timestamp.desc()).all()
        
        # Group by student and date
        attendance_dict = {}
        for record in records:
            student_key = (record.student.id, record.student.roll_no, record.student.name)
            date_key = record.timestamp.date()
            
            if student_key not in attendance_dict:
                attendance_dict[student_key] = {}
            
            attendance_dict[student_key][date_key] = {
                'time': record.timestamp.strftime('%H:%M:%S'),
                'confidence': record.confidence
            }
        
        return attendance_dict
    
    def get_all_attendance_records(self):
        """Fetch all attendance records with student names."""
        records = self.session.query(Attendance).join(Student).all()
        attendance_list = []
        for attendance in records:
            attendance_list.append({
                'roll_no': attendance.student.roll_no,
                'name': attendance.student.name,
                'timestamp': attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'confidence': f'{attendance.confidence:.1%}'
            })
        return attendance_list
    
    def close(self):
        """Close the database session cleanly."""
        self.session.close()
