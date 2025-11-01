# app.py
from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import sys
from datetime import datetime, timedelta

from services.face_recognition_service import FaceRecognitionService
from services.database_service import DatabaseService
from services.liveness_detector import LivenessDetector
from services.frame_processor import FrameProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chai-attendance-system-2025'

# Initialize services
IMAGE_DIR = "data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

face_service = FaceRecognitionService()
db_service = DatabaseService()
liveness_detector = LivenessDetector()
frame_processor = FrameProcessor()

camera = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enroll')
def enroll_page():
    return render_template('enroll.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/dashboard')
def dashboard():
    """Date-wise attendance dashboard"""
    students = db_service.list_students()
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    attendance_dict = db_service.get_attendance_by_date_range(start_date, end_date)
    
    date_list = []
    current_date = end_date
    while current_date >= start_date:
        date_list.append(current_date)
        current_date -= timedelta(days=1)
    
    attendance_grid = []
    for student in students:
        student_key = (student.id, student.roll_no, student.name)
        row = {
            'id': student.id,
            'roll_no': student.roll_no,
            'name': student.name,
            'attendance': attendance_dict.get(student_key, {})
        }
        attendance_grid.append(row)
    
    return render_template('dashboard.html', 
                         students=attendance_grid, 
                         dates=date_list,
                         today=end_date)

def generate_frames():
    cap = get_camera()
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==================== API ENDPOINTS ====================

@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    """Enroll a new student"""
    try:
        data = request.json
        student_id = data.get('student_id')
        name = data.get('name')
        
        if not student_id or not name:
            return jsonify({'success': False, 'message': 'Missing student ID or name'}), 400
        
        if db_service.get_student_by_roll(student_id):
            return jsonify({'success': False, 'message': f'Student {student_id} already enrolled'}), 400
        
        cap = get_camera()
        ret, frame = cap.read()
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to capture image'}), 500
        
        embedding = face_service.get_embedding(frame)
        if embedding is None:
            return jsonify({'success': False, 'message': 'No face detected'}), 400
        
        image_path = os.path.join(IMAGE_DIR, f"{student_id}.jpg")
        cv2.imwrite(image_path, frame)
        
        db_service.enroll_student(name, student_id, "Default", image_path)
        face_service.add_to_index(embedding, student_id)
        
        print(f"✅ Enrolled: {name} ({student_id})")
        return jsonify({'success': True, 'message': f'{name} enrolled successfully!'})
    
    except Exception as e:
        print(f"❌ Enrollment error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/take_attendance_browser', methods=['POST'])
def api_take_attendance_browser():
    """Take attendance with multi-layer liveness detection"""
    try:
        print("\n" + "="*60)
        print("🔍 Multi-Layer Liveness Detection")
        print("="*60)
        
        data = request.json
        before_frames_b64 = data.get('before_frames', [])
        after_frames_b64 = data.get('after_frames', [])
        
        if not before_frames_b64 or not after_frames_b64:
            return jsonify({'success': False, 'message': 'Missing frames'}), 400
        
        print(f"📊 Received {len(before_frames_b64)} before, {len(after_frames_b64)} after frames")
        
        # Decode frames
        before_frames = frame_processor.decode_frames_batch(before_frames_b64)
        after_frames = frame_processor.decode_frames_batch(after_frames_b64)
        
        # Liveness analysis
        is_live, metrics, fail_reason = liveness_detector.analyze_frames(before_frames, after_frames)
        liveness_detector.print_analysis(metrics)
        
        if not is_live:
            print(f"❌ SPOOF DETECTED! {fail_reason}")
            return jsonify({'success': False, 'message': f'🚫 Spoof detected! {fail_reason}'}), 400
        
        print("✅ Liveness PASSED!")
        
        # Face recognition
        frame = after_frames[-1]
        embedding = face_service.get_embedding(frame)
        if embedding is None:
            return jsonify({'success': False, 'message': 'No face detected'}), 400
        
        match = face_service.recognize(embedding)
        if match is None:
            return jsonify({'success': False, 'message': 'Student not recognized'}), 404
        
        student_roll_no, distance = match
        confidence = 1.0 / (1.0 + float(distance))
        
        student = db_service.get_student_by_roll(student_roll_no)
        if not student:
            return jsonify({'success': False, 'message': 'Student not found in database'}), 404
        
        # Check duplicate
        if db_service.is_attendance_marked_today(student.id):
            print(f"⚠️ Already marked today for {student.name}")
            return jsonify({'success': False, 'message': f'⚠️ Already marked for {student.name} today!'}), 400
        
        # Mark attendance
        db_service.mark_attendance(student.id, confidence)
        print(f"✅ Attendance: {student.name} ({student.roll_no})")
        print("="*60 + "\n")
        
        return jsonify({
            'success': True,
            'message': f'✅ Attendance marked for {student.name}!',
            'student_id': student.roll_no,
            'name': student.name,
            'confidence': f'{confidence:.1%}'
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/students')
def api_students():
    students = db_service.list_students()
    return jsonify({'students': [
        {'id': s.id, 'roll_no': s.roll_no, 'name': s.name, 'class': s.class_name}
        for s in students
    ]})

@app.route('/api/attendance_records')
def api_attendance_records():
    return jsonify({'records': db_service.get_all_attendance_records()})

@app.route('/api/reset_all', methods=['POST'])
def reset_all():
    """Complete system reset"""
    try:
        import faiss
        from database.models import Student, Attendance
        
        face_service.index = faiss.IndexFlatL2(512)
        face_service.id_map = []
        face_service._save_index()
        
        db_service.session.query(Attendance).delete()
        db_service.session.query(Student).delete()
        db_service.session.commit()
        
        for file in os.listdir(IMAGE_DIR):
            os.unlink(os.path.join(IMAGE_DIR, file))
        
        return jsonify({'success': True, 'message': 'System reset complete!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🎓 CHAI ATTENDANCE SYSTEM - IIT BHU")
    print("  🌙 Multi-Layer Liveness Detection")
    print("="*60)
    print("  📍 http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        db_service.close()
