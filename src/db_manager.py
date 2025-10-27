from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

os.makedirs("database", exist_ok=True)

engine = create_engine(f"sqlite:///{os.path.abspath('database/attendance.db')}")
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Students table
class Student(Base):
    __tablename__ = "students"
    student_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    roll_number = Column(String, unique=True, nullable=False)
    class_name = Column(String)
    ref_image_path = Column(String)  # one reference image path

# Attendance table
class Attendance(Base):
    __tablename__ = "attendance"
    attendance_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    snapshot_path = Column(String)

# Create tables
Base.metadata.create_all(engine)
