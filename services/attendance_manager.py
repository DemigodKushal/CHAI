from datetime import datetime

class AttendanceManager:
    def __init__(self):
        self.records = {}

    def mark_attendance(self, student_id):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.records[student_id] = now
        print(f"[{now}] ✅ Attendance recorded for {student_id}")
