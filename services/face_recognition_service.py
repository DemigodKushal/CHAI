import os
import cv2
import faiss
import numpy as np
import insightface
import pickle


class FaceRecognitionService:
    """
    A service class for facial recognition and attendance marking using
    InsightFace embeddings and FAISS for efficient face matching.
    """

    def __init__(self, threshold=1.0):
        """
        Initialize the FaceRecognitionService.

        Args:
            threshold (float): Distance threshold for deciding a face match. 
                               Lower values make matching stricter.
        """
        self.threshold = threshold
        self.dimension = 512
        self.index_path = "data/embeddings/faiss_index.bin"
        self.id_map_path = "data/embeddings/id_map.bin"

        # Initialize face detection & embedding model
        self.model = insightface.app.FaceAnalysis(
            name="buffalo_l", 
            providers=["CUDAExecutionProvider"]
        )
        self.model.prepare(ctx_id=0)

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.id_map_path), exist_ok=True)

        # Load or initialize FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        # Load or initialize ID mapping list
        if os.path.exists(self.id_map_path):
            with open(self.id_map_path, "rb") as f:
                self.id_map = pickle.load(f)
        else:
            self.id_map = []

    def capture(self, cap=None):
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
                print("Cannot open camera")
                return None

        print('Press S to capture face, Q to quit.')
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

    def get_embeddings(self, img):
        """
        Extract a 512-dimensional normalized face embedding from an image.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray or None: Normalized face embedding vector if a face is found,
                                otherwise None.
        """
        faces = self.model.get(img)
        if len(faces) == 0:
            print("No faces detected")
            return None
        else:
            # Choose the largest detected face (in case multiple faces are present)
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return face.normed_embedding

    def add_embeddings(self, emb, student_id):
        """
        Add a student's face embedding to the FAISS index and save to disk.

        Args:
            emb (np.ndarray): Normalized 512-D face embedding.
            student_id (str or int): Unique identifier for the student.
        """
        self.index.add(np.array([emb], dtype=np.float32))
        self.id_map.append(str(student_id))

        # Save updated index and ID mapping
        faiss.write_index(self.index, self.index_path)
        with open(self.id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)

    def match_embeddings(self, emb):
        """
        Match a given face embedding against the stored database.

        Args:
            emb (np.ndarray): Normalized 512-D face embedding to be matched.

        Returns:
            tuple: (matched_student_id, distance)
                - matched_student_id (str or None): ID of the closest matching face, or None if no match.
                - distance (float or None): L2 distance to the matched embedding, or None if no match found.
        """
        if self.index.ntotal == 0:
            print("No faces registered in DB")
            return None, None

        # Search for the closest embedding in the index
        D, I = self.index.search(np.array([emb], dtype='float32'), k=1)
        distance = D[0][0]
        idx = I[0][0]

        if distance > self.threshold:
            print("No Match found")
            return None, None
        else:
            return self.id_map[idx], distance
