import os
import cv2
import faiss
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognitionService:
    def __init__(self, threshold=0.55):
        """
        threshold: cosine similarity threshold (higher = more similar)
        typically 0.5–0.6 works well for ArcFace embeddings.
        """
        self.threshold = threshold
        self.dimension = 512
        self.index_path = "data/embeddings/faiss_index_cosine.bin"
        self.id_map_path = "data/embeddings/id_map.npy"

        # Initialize InsightFace ArcFace model
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Initialize FAISS index (cosine = Inner Product)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map = []

        self._load_index()

    # ---------- FACE EMBEDDINGS ----------
    def _normalize(self, emb):
        """Normalize the embedding vector for cosine similarity"""
        return emb / np.linalg.norm(emb)

    def extract_embedding(self, img_path: str):
        img = cv2.imread(img_path)
        faces = self.app.get(img)
        if not faces:
            print(f"❌ No face detected in: {img_path}")
            return None

        emb = faces[0].embedding
        return self._normalize(emb).astype("float32")

    def extract_embedding_from_frame(self, frame):
        faces = self.app.get(frame)
        if not faces:
            return []
        return [self._normalize(face.embedding).astype("float32") for face in faces]

    # ---------- INDEX MANAGEMENT ----------
    def add_to_index(self, embedding, student_id):
        embedding = np.array([embedding]).astype("float32")
        self.index.add(embedding)
        self.id_map.append(student_id)
        self._save_index()
        print(f"✅ Added embedding for Student ID {student_id}")

    def find_match(self, embedding):
        if self.index.ntotal == 0:
            print("⚠️ No embeddings in index yet.")
            return None, 0.0

        emb = np.array([embedding]).astype("float32")
        similarity, idx = self.index.search(emb, 1)
        similarity = similarity[0][0]

        if similarity >= self.threshold:
            student_id = int(self.id_map[idx[0][0]])
            return student_id, similarity

        return None, similarity

    # ---------- SAVE & LOAD ----------
    def _save_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        np.save(self.id_map_path, np.array(self.id_map, dtype=np.int32))
        print("💾 Saved FAISS index and ID map")

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.id_map_path):
            self.index = faiss.read_index(self.index_path)
            self.id_map = np.load(self.id_map_path).tolist()
            print(f"📂 Loaded existing FAISS index with {len(self.id_map)} embeddings")
        else:
            print("⚙️ No saved index found, starting fresh.")
