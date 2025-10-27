import os
import shutil

def reset_embeddings():
    paths = [
        "data/embeddings/faiss_index_cosine.bin",
        "data/embeddings/id_map.npy"
    ]
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
            print(f"🗑️ Deleted {p}")
    print("✅ Embeddings reset complete.")


def reset_images():
    img_dir = "data/images/students"
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)
        print("🗑️ All student images deleted and folder recreated.")
    else:
        print("⚠️ Image directory not found.")


def reset_database():
    db_path = "data/database/attendance.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print("🗑️ Deleted attendance database.")
    else:
        print("⚠️ Database file not found.")


def reset_all():
    reset_embeddings()
    reset_images()
    reset_database()
    print("\n🚀 System fully reset — start fresh enrollment now.")


if __name__ == "__main__":
    print("Select reset option:")
    print("1️⃣ Reset embeddings only")
    print("2️⃣ Reset images only")
    print("3️⃣ Reset database only")
    print("4️⃣ Reset everything")

    choice = input("👉 Enter option (1-4): ").strip()

    if choice == "1":
        reset_embeddings()
    elif choice == "2":
        reset_images()
    elif choice == "3":
        reset_database()
    elif choice == "4":
        reset_all()
    else:
        print("❌ Invalid choice.")
