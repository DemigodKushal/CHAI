# app.py - Flask Web Application with Location Service

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import base64
from datetime import datetime, date
from collections import defaultdict
from datetime import datetime, date


# Import services
from services.face_recognition_service import FaceRecognitionService
from services.liveness_detector import LivenessDetector
from services.database_service import DatabaseService
from services.location_service import LocationService

app = Flask(__name__)

# Initialize services
face_service = FaceRecognitionService()
liveness_detector = LivenessDetector()
db_service = DatabaseService()
location_service = LocationService()

# OPTIONAL: Set manual classroom location for accurate geofencing
# Get coordinates from Google Maps: Right-click on location → Copy coordinates
location_service.set_server_location_manual(25.263790, 82.984952)  # IIT BHU example

# Camera instance
camera = None

print("\n" + "="*60)
print("  🎓 CHAI ATTENDANCE SYSTEM - IIT BHU")
print("  🌙 Multi-Layer Liveness Detection")
print("="*60)
print(f"  📍 http://127.0.0.1:5000")
print("="*60 + "\n")


# ==================== HELPER FUNCTIONS ====================

def decode_base64_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/enroll')
def enroll():
    """Enrollment page"""
    return render_template('enroll.html')


@app.route('/attendance')
def attendance():
    """Attendance page"""
    return render_template('attendance.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page showing attendance records"""
    try:
        students = db_service.get_all_students()
        
        # Get all unique dates with attendance
        all_dates = set()
        student_data = []
        
        for student in students:
            records = db_service.get_attendance_records(student.id)
            attendance_dict = {}
            
            for record in records:
                date_key = record.timestamp.date()
                all_dates.add(date_key)
                
                attendance_dict[date_key] = {
                    'time': record.timestamp.strftime('%H:%M:%S'),
                    'confidence': record.confidence
                }
            
            student_data.append({
                'name': student.name,
                'roll_no': student.roll_no,
                'id': student.id,
                'attendance': attendance_dict
            })
        
        # Sort dates (newest first)
        sorted_dates = sorted(all_dates, reverse=True)
        
        print(f"📊 Dashboard: {len(students)} students, {len(sorted_dates)} dates")
        
        return render_template('dashboard.html', 
                             students=student_data,
                             dates=sorted_dates,
                             today=date.today())
    
    except Exception as e:
        print(f"❌ Dashboard error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error loading dashboard: {str(e)}", 500
    """Dashboard page showing attendance records"""
    try:
        students = db_service.get_all_students()
        
        # Organize attendance by date
        attendance_by_date = {}
        
        for student in students:
            records = db_service.get_attendance_records(student.id)
            
            for record in records:
                date_key = record.timestamp.date()
                
                if date_key not in attendance_by_date:
                    attendance_by_date[date_key] = []
                
                attendance_by_date[date_key].append({
                    'student_name': student.name,
                    'student_roll': student.roll_no,
                    'time': record.timestamp.strftime('%H:%M:%S'),
                    'confidence': record.confidence  # Keep as number
                })
        
        # Sort dates (newest first)
        sorted_dates = sorted(attendance_by_date.keys(), reverse=True)
        
        return render_template('dashboard.html', 
                             students=students,
                             attendance_by_date=attendance_by_date,
                             dates=sorted_dates)
    
    except Exception as e:
        print(f"❌ Dashboard error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error loading dashboard: {str(e)}", 500
    """Dashboard page showing attendance records"""
    try:
        students = db_service.get_all_students()
        
        # Get all attendance records (returns list of dicts)
        all_records = db_service.get_all_attendance_records()
        
        # Get unique dates from timestamp strings
        dates = sorted(set(
            datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S').date() 
            for record in all_records
        ), reverse=True)
        
        # Organize attendance by student
        student_data = []
        
        for student in students:
            records = db_service.get_attendance_records(student.id)
            attendance_dict = {}
            
            for record in records:
                date_key = record.timestamp.date()
                attendance_dict[date_key] = {
                    'time': record.timestamp.strftime('%H:%M:%S'),
                    'confidence': f'{record.confidence * 100:.1f}%'
                }
            
            student_data.append({
                'name': student.name,
                'roll_no': student.roll_no,
                'attendance': attendance_dict
            })
        
        return render_template('dashboard.html', 
                             students=student_data,
                             dates=dates)
    
    except Exception as e:
        print(f"❌ Dashboard error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error loading dashboard: {str(e)}", 500
    """Dashboard page showing attendance records"""
    students = db_service.get_all_students()
    
    # Get all unique dates
    all_records = db_service.get_all_attendance_records()
    dates = sorted(set(record.date for record in all_records))
    
    # Organize attendance by student
    student_data = []
    today = date.today()
    
    for student in students:
        records = db_service.get_attendance_records(student.id)
        attendance_dict = {}
        
        for record in records:
            attendance_dict[record.date] = {
                'time': record.timestamp.strftime('%H:%M:%S'),
                'confidence': record.confidence
            }
        
        student_data.append({
            'name': student.name,
            'roll_no': student.roll_no,
            'attendance': attendance_dict
        })
    
    return render_template('dashboard.html', 
                         students=student_data,
                         dates=dates,
                         today=today)


# ==================== CAMERA ROUTES ====================

def generate_frames():
    """Generate camera frames for video streaming"""
    global camera
    
    while True:
        if camera is None or not camera.isOpened():
            break
            
        success, frame = camera.read()
        if not success:
            break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """Start camera"""
    global camera
    
    try:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            
        if camera.isOpened():
            return jsonify({'success': True, 'message': 'Camera started'})
        else:
            return jsonify({'success': False, 'message': 'Failed to open camera'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/release_camera', methods=['POST'])
def release_camera():
    """Release camera"""
    global camera
    
    try:
        if camera is not None:
            camera.release()
            camera = None
        return jsonify({'success': True, 'message': 'Camera released'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ==================== ENROLLMENT ROUTE ====================

@app.route('/api/enroll', methods=['POST'])
def enroll_student():
    """Enroll a new student"""
    global camera
    
    try:
        data = request.json
        student_id = data.get('student_id')
        name = data.get('name')
        
        if not student_id or not name:
            return jsonify({'success': False, 'message': 'Student ID and name required'}), 400
        
        # Check if student already exists
        existing = db_service.get_student_by_roll(student_id)
        if existing:
            return jsonify({'success': False, 'message': f'Student {student_id} already enrolled'}), 400
        
        # Capture frame from camera
        if camera is None or not camera.isOpened():
            return jsonify({'success': False, 'message': 'Camera not available'}), 400
        
        success, frame = camera.read()
        if not success:
            return jsonify({'success': False, 'message': 'Failed to capture image'}), 500
        
        # Extract face embedding
        embeddings = face_service.extract_embedding_from_frame(frame)
        if not embeddings:
            return jsonify({'success': False, 'message': 'No face detected. Please face the camera clearly.'}), 400
        
        embedding = embeddings[0]
        
        # Save image
        import os
        os.makedirs('data/images', exist_ok=True)
        image_path = f'data/images/{student_id}.jpg'
        cv2.imwrite(image_path, frame)
        
        # Add to database
        student = db_service.enroll_student(name, student_id, '', image_path)
        
        # Add to FAISS index
        face_service.add_to_index(embedding, student.id)
        
        return jsonify({
            'success': True,
            'message': f'Successfully enrolled {name}!',
            'student_id': student_id
        })
        
    except Exception as e:
        print(f"Enrollment error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ==================== ATTENDANCE ROUTE WITH LOCATION ====================

@app.route('/api/take_attendance_browser', methods=['POST'])
def take_attendance_browser():
    """Take attendance with liveness detection and location verification"""
    try:
        data = request.json
        before_frames = data.get('before_frames', [])
        after_frames = data.get('after_frames', [])
        
        # Get student's location
        user_lat = data.get('latitude')
        user_lon = data.get('longitude')
        
        print(f"\n📍 Location Check:")
        print(f"   Student: ({user_lat}, {user_lon})")
        
        # Verify location first
        is_location_valid, distance, location_msg = location_service.verify_location(user_lat, user_lon)
        
        print(f"   {location_msg}")
        
        if not is_location_valid:
            return jsonify({
                'success': False,
                'message': location_msg,
                'reason': 'location_out_of_range',
                'distance': f'{distance:.1f}m' if distance else 'N/A'
            }), 400
        
        # Continue with liveness and face recognition
        print(f"\n{'='*60}")
        print(f"🔍 Multi-Layer Liveness Detection")
        print(f"{'='*60}")
        print(f"📊 Received {len(before_frames)} before, {len(after_frames)} after frames")
        
        if len(before_frames) != 5 or len(after_frames) != 5:
            return jsonify({
                'success': False,
                'message': 'Expected 5 before and 5 after frames'
            }), 400
        
        # Decode frames
        before_images = [decode_base64_image(frame) for frame in before_frames]
        after_images = [decode_base64_image(frame) for frame in after_frames]
        
        if any(img is None for img in before_images) or any(img is None for img in after_images):
            return jsonify({
            'success': False,
            'message': 'Failed to decode one or more frames'
        }), 400

        
        # Liveness check
        is_live, metrics, fail_reason = liveness_detector.analyze_frames(before_images, after_images)
        
        liveness_detector.print_analysis(metrics)
        
        if not is_live:
            print(f"❌ SPOOF DETECTED! {fail_reason}")
            return jsonify({
                'success': False,
                'message': fail_reason
            }), 400
        
        print(f"✅ Liveness PASSED!")
        
        # Face recognition
        last_frame = after_images[-1]
        embeddings = face_service.extract_embedding_from_frame(last_frame)
        
        if not embeddings:
            return jsonify({
                'success': False,
                'message': 'No face detected in frames'
            }), 400
        
        embedding = embeddings[0]
        student_id, face_distance = face_service.find_match(embedding)
        
        if student_id is None:
            return jsonify({
                'success': False,
                'message': 'Face not recognized. Please enroll first.'
            }), 400
        
        # Get student info
        student = db_service.get_student_by_id(int(student_id))
        if not student:
            return jsonify({
                'success': False,
                'message': 'Student record not found'
            }), 400
        
        # Calculate confidence
        confidence = 1.0 / (1.0 + face_distance)
        
        # Check if already marked today
        today = date.today()
        existing_record = db_service.get_attendance_record(student.id, today)
        
        if existing_record:
            print(f"⚠️ Already marked today for {student.name}")
            return jsonify({
                'success': False,
                'message': f'Attendance already marked for {student.name} today at {existing_record.timestamp.strftime("%H:%M:%S")}'
            }), 400
        
        # Mark attendance
        db_service.mark_attendance(student.id, confidence)
        
        print(f"✅ Attendance marked for {student.name}")
        
        return jsonify({
            'success': True,
            'message': 'Attendance marked successfully!',
            'name': student.name,
            'student_id': student.roll_no,
            'confidence': f'{confidence*100:.1f}%',
            'location_distance': f'{distance:.1f}m'
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


# ==================== LOCATION INFO ROUTE ====================

@app.route('/api/server_location', methods=['GET'])
def get_server_location():
    """Return server's current location for frontend reference"""
    info = location_service.get_server_info()
    return jsonify({
        'success': True,
        'server': info
    })


# ==================== RUN APP ====================

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
