from PIL import Image
import cv2
import os
import customtkinter as ctk
from tkinter import messagebox
from services.attendance_service import AttendanceService
from services.face_recognition_service import FaceRecognitionService
from services.database_service import DatabaseService


class AttendanceAppUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("CHAI — Face Attendance System")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize services
        self.face_service = FaceRecognitionService()
        self.db_service = DatabaseService()
        self.attendance_service = AttendanceService(self.face_service, self.db_service)

        self._build_ui()

    # ------------------ UI SETUP ------------------
    def _build_ui(self):
        frame = ctk.CTkFrame(self.root, width=800, height=500, corner_radius=15)
        frame.pack(padx=50, pady=50, fill="both", expand=True)

        title = ctk.CTkLabel(frame, text="CHAI — Face Attendance System", font=("Arial Bold", 24))
        title.pack(pady=30)

        enroll_btn = ctk.CTkButton(frame, text="Enroll New Student", command=self.enroll_student, height=45, width=220)
        enroll_btn.pack(pady=20)

        attendance_btn = ctk.CTkButton(frame, text="Take Attendance", command=self.take_attendance, height=45, width=220)
        attendance_btn.pack(pady=20)

        self.status_label = ctk.CTkLabel(frame, text="", font=("Arial", 16))
        self.status_label.pack(pady=30)

    def show_message(self, text: str):
        self.status_label.configure(text=text)
        self.root.update()

    # ------------------ ENROLLMENT ------------------
    def enroll_student(self):
        """Opens enrollment form and uses OpenCV window for camera."""
        enroll_window = ctk.CTkToplevel(self.root)
        enroll_window.title("Enroll New Student")
        enroll_window.geometry("500x400")
        enroll_window.resizable(False, False)

        # Title
        ctk.CTkLabel(enroll_window, text="Student Enrollment", font=("Arial Bold", 22)).pack(pady=15)

        # Entry fields
        entry_frame = ctk.CTkFrame(enroll_window, fg_color="transparent")
        entry_frame.pack(pady=5)

        ctk.CTkLabel(entry_frame, text="Name:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        name_entry = ctk.CTkEntry(entry_frame, width=250)
        name_entry.grid(row=0, column=1, padx=10, pady=5)

        ctk.CTkLabel(entry_frame, text="Roll No:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        roll_entry = ctk.CTkEntry(entry_frame, width=250)
        roll_entry.grid(row=1, column=1, padx=10, pady=5)

        ctk.CTkLabel(entry_frame, text="Class:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        class_entry = ctk.CTkEntry(entry_frame, width=250)
        class_entry.grid(row=2, column=1, padx=10, pady=5)

        # Buttons
        btn_frame = ctk.CTkFrame(enroll_window, fg_color="transparent")
        btn_frame.pack(pady=25)

        capture_btn = ctk.CTkButton(btn_frame, text="📸 Capture Image", width=180)
        capture_btn.grid(row=0, column=0, padx=10)

        save_btn = ctk.CTkButton(btn_frame, text="💾 Save & Enroll", width=180)
        save_btn.grid(row=0, column=1, padx=10)

        # Status label
        status_label = ctk.CTkLabel(enroll_window, text="", font=("Arial", 14))
        status_label.pack(pady=10)

        # Capture function
        def capture_image():
            roll_no = roll_entry.get().strip()
            if not roll_no:
                messagebox.showwarning("Missing Roll No", "Please enter Roll No before capturing.")
                return

            os.makedirs("data/images", exist_ok=True)
            img_path = os.path.join("data/images", f"{roll_no}.jpg")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Camera Error", "Cannot access camera.")
                return

            messagebox.showinfo("Instructions", "Press 's' to capture and 'q' to quit camera window.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                cv2.imshow("Capture Image", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    cv2.imwrite(img_path, frame)
                    messagebox.showinfo("Captured", f"Image saved for Roll No {roll_no}")
                    break
                elif key == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        # Save & Enroll function
        def save_and_enroll():
            name = name_entry.get().strip()
            roll_no = roll_entry.get().strip()
            class_name = class_entry.get().strip()

            if not name or not roll_no or not class_name:
                messagebox.showwarning("Incomplete Data", "Please fill all fields.")
                return

            img_path = os.path.join("data/images", f"{roll_no}.jpg")
            if not os.path.exists(img_path):
                messagebox.showwarning("No Image", "Capture an image first.")
                return

            embedding = self.face_service.extract_embedding(img_path)
            if embedding is None:
                messagebox.showerror("Face Error", "No face detected in image.")
                return

            student = self.db_service.enroll_student(name, roll_no, class_name, img_path)
            self.face_service.add_to_index(embedding, student.id)

            messagebox.showinfo("Success", f"{name} enrolled successfully!")
            enroll_window.destroy()

        capture_btn.configure(command=capture_image)
        save_btn.configure(command=save_and_enroll)

    # ------------------ ATTENDANCE ------------------
    def take_attendance(self):
        self.show_message("📷 Starting attendance...")
        self.attendance_service.take_attendance()
        self.show_message("✅ Attendance completed!")

    # ------------------ RUN APP ------------------
    def run(self):
        self.root.mainloop()
