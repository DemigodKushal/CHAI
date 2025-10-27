import pinecone
import numpy as np
from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create index if not exists
if PINECONE_INDEX not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX, dimension=128, metric="cosine")

index = pinecone.Index(PINECONE_INDEX)

def add_student_vector(student_id, embedding):
    """Store student embedding in Pinecone"""
    index.upsert([(str(student_id), embedding.tolist())])
    print(f"Student vector {student_id} stored in Pinecone")

def update_student_vector(student_id, embedding):
    """Update student embedding in Pinecone"""
    index.upsert([(str(student_id), embedding.tolist())])
    print(f"Student vector {student_id} updated in Pinecone")

def delete_student_vector(student_id):
    """Delete student vector from Pinecone"""
    index.delete(ids=[str(student_id)])
    print(f"Student vector {student_id} deleted from Pinecone")

def query_student(embedding, threshold=0.4):
    """Query Pinecone for nearest student"""
    res = index.query(vector=embedding.tolist(), top_k=1, include_values=False)
    if res["matches"] and res["matches"][0]["score"] >= (1 - threshold):
        return res["matches"][0]["id"], res["matches"][0]["score"]
    return None, None
