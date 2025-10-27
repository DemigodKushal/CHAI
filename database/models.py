from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import datetime

Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    roll_no = Column(String, unique=True)
    class_name = Column(String)
    image_path = Column(String)

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    timestamp = Column(DateTime, default=datetime.datetime.now)
    confidence = Column(Float)
    student = relationship("Student")

# Initialize DB connection and session
engine = create_engine("sqlite:///data/attendance.db", echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
