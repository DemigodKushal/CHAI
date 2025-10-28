import os
import cv2
import faiss
import numpy as np
import insightface
import pickle
from pathlib import Path


class FaceRecognitionService:
    """
    A service class for facial recognition and attendance marking using
    InsightFace embeddings and FAISS for efficient face matching.
    """

    def __init__(self, threshold=1.0, data_dir=None):
        """
        Initialize the face recognition service.

        Args:
            threshold (float): L2 distance threshold for face matching.
            data_dir (str or Path, optional): Directory to store embeddings and ID map.
                                              Defaults to ~/FaceRecognitionData/embeddings
        """
        self.threshold = threshold
        self.dimension = 512

        # Data directory setup
        if data_dir is None:
            data_dir = Path.home() / "FaceRecognitionData" / "embeddings"
        else:
            data_dir = Path(data_dir)

        data_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = str(data_dir / "faiss_index.bin")
        self.id_map_path = str(data_dir / "id_map.bin")

        # Model setup
        self.model = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
        self.model.prepare(ctx_id=0)

        # Load or initialize FAISS index and ID map
        self._load_index()

    # ---------- FACE CAPTURE ----------
    def capture_frame(self, cap=None):
        """
        Capture an image frame from a webcam feed.

        Args:
            cap (cv2.VideoCapture, optional): Existing camera object.
                                              If None, opens default webcam.

        Returns:
            np.ndarray or None: Captured image frame (BGR format) if successful,
                                otherwise None.
        """
        if cap is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Failed to open webcam.")

        print("Press S to capture face, Q to quit.")
        while True:
            ret, frame = cap.read()
            cv2.imshow("WebCam", frame)
            if not ret:
                break
            key = cv2.waitKey(1)
            if key == ord('s'):
                cap.release()
                cv2.destroyAllWindows()
                return frame
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None

    # ---------- FACE EMBEDDINGS ----------
    def extract_embedding_from_frame(self, frame):
        """
        Extract normalized face embedding(s) from a webcam frame.

        Args:
            frame (np.ndarray): Input image in BGR format.

        Returns:
            list[np.ndarray]: List of normalized embeddings for detected faces.
        """
        faces = self.model.get(frame)
        if not faces:
            return []
        return [face.normed_embedding for face in faces]

    def extract_embedding(self, img_path):
        """
        Extract a single normalized face embedding from an image file.

        Args:
            img_path (str): Path to image file.

        Returns:
            np.ndarray or None: Normalized 512-D face embedding if a face is found,
                                otherwise None.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        faces = self.model.get(img)
        if len(faces) == 0:
            print(f"No face detected in: {img_path}")
            return None
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return face.normed_embedding

    # ---------- INDEX MANAGEMENT ----------
    def add_to_index(self, embedding, student_id):
        """
        Add a student's face embedding to the FAISS index and save to disk.

        Args:
            embedding (np.ndarray): Normalized 512-D face embedding.
            student_id (str or int): Unique identifier for the student.
        """
        self.index.add(np.array([embedding], dtype=np.float32))
        self.id_map.append(str(student_id))
        self._save_index()
        print(f"Added embedding for Student ID {student_id}")

    def find_match(self, embedding):
        """
        Match a given face embedding against the stored database.

        Args:
            embedding (np.ndarray): Normalized 512-D face embedding.

        Returns:
            tuple: (matched_student_id, distance)
        """
        if self.index.ntotal == 0:
            print("No embeddings in index yet.")
            return None, None

        D, I = self.index.search(np.array([embedding], dtype='float32'), k=1)
        distance = D[0][0]
        idx = I[0][0]

        if distance > self.threshold:
            print("No match found.")
            return None, distance
        else:
            student_id = self.id_map[idx]
            return student_id, distance

    # ---------- SAVE & LOAD ----------
    def _save_index(self):
        """
        Save FAISS index and ID mapping to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)
        print("Saved FAISS index and ID map")

    def _load_index(self):
        """
        Load FAISS index and ID mapping if available, else initialize new ones.
        """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        if os.path.exists(self.id_map_path):
            with open(self.id_map_path, "rb") as f:
                self.id_map = pickle.load(f)
            print(f"Loaded existing FAISS index with {len(self.id_map)} embeddings")
        else:
            self.id_map = []
            print("No saved index found, starting fresh.")
