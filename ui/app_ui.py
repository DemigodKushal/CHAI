from PIL import Image, ImageTk
import customtkinter as ctk
import threading
import os
import time
import cv2

from services.face_recognition_service import FaceRecognitionService
from services.database_service import DatabaseService
from services.attendance_service import AttendanceService


class AttendanceAppUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("CHAI — Face Attendance System")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Core services
        self.face_service = FaceRecognitionService()
        self.db_service = DatabaseService()
        self.attendance_service = AttendanceService(self.face_service, self.db_service)

        self._build_home_ui()

    # -------------------- HOME SCREEN --------------------
    def _build_home_ui(self):
        """Main home screen with two options."""
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self.root, width=800, height=500, corner_radius=15)
        frame.pack(padx=50, pady=50, fill="both", expand=True)

        title = ctk.CTkLabel(frame, text="CHAI — Face Attendance System", font=("Arial Bold", 28))
        title.pack(pady=40)

        attendance_btn = ctk.CTkButton(frame, text="📷 Take Attendance", width=240, height=50,
                                       command=self.open_attendance_mode)
        attendance_btn.pack(pady=20)

        enroll_btn = ctk.CTkButton(frame, text="🧑‍🎓 Enroll New Student", width=240, height=50,
                                   command=self.open_enroll_window)
        enroll_btn.pack(pady=20)

        self.status_label = ctk.CTkLabel(frame, text="", font=("Arial", 16))
        self.status_label.pack(pady=30)

    def show_message(self, text: str):
        self.status_label.configure(text=text)
        self.root.update()

    # -------------------- ATTENDANCE --------------------
    def open_attendance_mode(self):
        """Start attendance mode using AttendanceService (external camera window)."""
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = ctk.CTkFrame(self.root)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        title = ctk.CTkLabel(frame, text="📸 Attendance Mode", font=("Arial Bold", 22))
        title.pack(pady=20)

        info_label = ctk.CTkLabel(
            frame,
            text=(
                "The camera will open in a new window.\n"
                "Press 's' to capture a frame and mark attendance.\n"
                "Press 'q' to quit and return here."
            ),
            font=("Arial", 16),
            justify="center",
        )
        info_label.pack(pady=30)

        start_btn = ctk.CTkButton(frame, text="🚀 Start Attendance", width=220,
                                  command=self._start_attendance_thread)
        start_btn.pack(pady=10)

        back_btn = ctk.CTkButton(frame, text="⬅️ Back", width=150, command=self._build_home_ui)
        back_btn.pack(pady=20)

        self.status_label = ctk.CTkLabel(frame, text="", font=("Arial", 16))
        self.status_label.pack(pady=10)

    def _start_attendance_thread(self):
        """Run attendance in a separate thread so UI doesn’t freeze."""
        self.show_message("📷 Launching camera window...")
        threading.Thread(target=self.attendance_service.take_attendance, daemon=True).start()

    # -------------------- ENROLLMENT --------------------
    def open_enroll_window(self):
        """Open new window for student enrollment."""
        win = ctk.CTkToplevel(self.root)
        win.title("Enroll New Student")
        win.geometry("950x550")
        win.resizable(False, False)

        ctk.CTkLabel(win, text="🧑‍🎓 Student Enrollment", font=("Arial Bold", 22)).pack(pady=10)

        main_frame = ctk.CTkFrame(win)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left - camera
        cam_frame = ctk.CTkFrame(main_frame, width=450, height=400)
        cam_frame.pack(side="left", padx=20, pady=20)
        cam_label = ctk.CTkLabel(cam_frame, text="Starting camera...")
        cam_label.pack()

        # Right - form
        form_frame = ctk.CTkFrame(main_frame)
        form_frame.pack(side="right", padx=20, pady=20)

        name_entry = ctk.CTkEntry(form_frame, placeholder_text="Full Name", width=250)
        name_entry.pack(pady=10)

        roll_entry = ctk.CTkEntry(form_frame, placeholder_text="Roll Number", width=250)
        roll_entry.pack(pady=10)

        class_entry = ctk.CTkEntry(form_frame, placeholder_text="Class Name", width=250)
        class_entry.pack(pady=10)

        capture_btn = ctk.CTkButton(form_frame, text="📸 Capture Image", width=200)
        capture_btn.pack(pady=10)

        save_btn = ctk.CTkButton(form_frame, text="💾 Enroll Student", width=200)
        save_btn.pack(pady=10)

        status = ctk.CTkLabel(form_frame, text="", font=("Arial", 14))
        status.pack(pady=10)

        # Camera handling
        cap = cv2.VideoCapture(0)
        running = True
        captured_path = None
        current_frame = None

        def update_feed():
            nonlocal running, current_frame
            try:
                while running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    cam_label.imgtk = imgtk
                    cam_label.configure(image=imgtk)
                    current_frame = frame
                    time.sleep(0.03)
            except Exception:
                pass

        threading.Thread(target=update_feed, daemon=True).start()

        def capture_image():
            nonlocal captured_path
            roll_no = roll_entry.get().strip()
            if not roll_no:
                status.configure(text="⚠️ Enter Roll Number before capturing.")
                return
            os.makedirs("data/images", exist_ok=True)
            img_path = os.path.join("data/images", f"{roll_no}.jpg")
            if current_frame is not None:
                cv2.imwrite(img_path, current_frame)
                captured_path = img_path
                status.configure(text=f"✅ Image captured for {roll_no}.")
            else:
                status.configure(text="⚠️ No camera frame available.")

        def save_and_enroll():
            name = name_entry.get().strip()
            roll_no = roll_entry.get().strip()
            class_name = class_entry.get().strip()

            if not name or not roll_no or not class_name:
                status.configure(text="⚠️ Fill all details before saving.")
                return
            if not captured_path or not os.path.exists(captured_path):
                status.configure(text="⚠️ Capture image first.")
                return

            embedding = self.face_service.extract_embedding(captured_path)
            if embedding is None:
                status.configure(text="❌ No face detected.")
                return

            student = self.db_service.enroll_student(name, roll_no, class_name, captured_path)
            self.face_service.add_to_index(embedding, student.id)
            status.configure(text=f"✅ {name} enrolled successfully!")
            win.after(2000, on_close)

        capture_btn.configure(command=capture_image)
        save_btn.configure(command=save_and_enroll)

        def on_close():
            nonlocal running
            running = False
            if cap and cap.isOpened():
                cap.release()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    # -------------------- RUN --------------------
    def run(self):
        self.root.mainloop()
