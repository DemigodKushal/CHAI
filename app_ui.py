import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time

from services.flash_liveness_service import FlashLivenessService
from services.database_service import DatabaseService
from services.face_recognition_service import FaceRecognitionService
from services.attendance_service import AttendanceService


# ------------------ INITIAL SETUP ------------------
st.set_page_config(page_title="CHAI — Face Attendance", layout="centered")

st.markdown(
    """
    <h1 style='text-align:center;'>📷 CHAI — Face Attendance System</h1>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Core Services
face_service = FaceRecognitionService()
db_service = DatabaseService()
attendance_service = AttendanceService(face_service, db_service)
liveness_service = FlashLivenessService()

# ------------------ SIDEBAR NAV ------------------
page = st.sidebar.radio("Navigation", ["Home", "Take Attendance", "Enroll New Student", "View Attendance Logs"])

# ------------------ HOME ------------------
if page == "Home":
    st.success("Welcome to the CHAI Face Attendance System!")
    st.markdown(
        """
        Use the sidebar to:
        - **Take Attendance**: Perform liveness check + recognition  
        - **Enroll Students**: Add a new face to the system  
        - **View Attendance Logs**: See who attended and when
        """
    )
    st.image("https://cdn-icons-png.flaticon.com/512/681/681494.png", width=250)

# ------------------ TAKE ATTENDANCE ------------------
elif page == "Take Attendance":
    st.header("📸 Take Attendance")
    st.info("Click below to start your camera. Press 'Capture' when ready.")

    camera = st.camera_input("Camera Feed", key="attendance_cam")

    if camera is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(camera.getbuffer())
            image_path = temp_img.name

        st.write("💡 Running flash-based liveness check...")

        # Liveness Verification (simulate)
        is_live = liveness_service.verify_liveness(cap)

        if not is_live:
            st.error("❌ Spoof detected or failed liveness test. Try again.")
        else:
            st.success("✅ Liveness confirmed! Proceeding with recognition...")

            emb = face_service.extract_embedding(image_path)
            if emb is None:
                st.warning("⚠️ No face detected.")
            else:
                student_id, similarity = face_service.find_match(emb)
                if student_id:
                    student = db_service.get_student_by_id(student_id)
                    db_service.mark_attendance(student_id, similarity)
                    st.success(f"✅ {student.name} recognized (similarity={similarity:.3f})")

                    total = db_service.get_total_attendance_for_student(student_id)
                    st.info(f"📈 Total attendance for {student.name}: {total}")

                    recent = db_service.get_recent_attendance_for_student(student_id)
                    if recent:
                        st.write("🕒 **Recent Attendance:**")
                        for a in recent:
                            st.write(f"- {a.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.warning(f"❌ Unknown face (similarity={similarity:.3f}). Please register.")

# ------------------ ENROLL NEW STUDENT ------------------
elif page == "Enroll New Student":
    st.header("🧑‍🎓 Enroll New Student")

    with st.form("enroll_form"):
        name = st.text_input("Full Name")
        roll_no = st.text_input("Roll Number")
        class_name = st.text_input("Class Name")
        camera = st.camera_input("Capture Face Image")

        submitted = st.form_submit_button("💾 Enroll Student")

        if submitted:
            if not name or not roll_no or not class_name:
                st.error("⚠️ Fill all fields before submitting.")
            elif camera is None:
                st.error("⚠️ Please capture a face image.")
            else:
                os.makedirs("data/images", exist_ok=True)
                image_path = f"data/images/{roll_no}.jpg"
                with open(image_path, "wb") as f:
                    f.write(camera.getbuffer())

                emb = face_service.extract_embedding(image_path)
                if emb is None:
                    st.error("❌ No face detected.")
                else:
                    student = db_service.enroll_student(name, roll_no, class_name, image_path)
                    face_service.add_to_index(emb, student.id)
                    st.success(f"✅ {name} enrolled successfully!")

# ------------------ VIEW ATTENDANCE LOGS ------------------
elif page == "View Attendance Logs":
    st.header("📋 Attendance Logs")

    all_logs = db_service.get_all_attendance_records()

    if not all_logs:
        st.info("No attendance records found yet.")
    else:
        for record in all_logs:
            st.write(
                f"🧑 {record['name']} | 🆔 {record['roll_no']} | "
                f"🏫 {record['class']} | 🔢 Conf: {record['confidence']} | "
                f"⏰ {record['timestamp']}"
            )

