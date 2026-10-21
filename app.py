import cv2
import pickle
import numpy as np
import requests
import os
import threading

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from insightface.app import FaceAnalysis


# ==========================================
# START LOCAL IMAGE SERVER
# ==========================================

PROJECT_FOLDER = os.path.abspath(".")


class ImageServerHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


image_server = ThreadingHTTPServer(
    ("127.0.0.1", 8000),
    ImageServerHandler
)

server_thread = threading.Thread(
    target=image_server.serve_forever,
    daemon=True
)

server_thread.start()


# ==========================================
# LOAD INSIGHTFACE MODEL
# ==========================================

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# ==========================================
# LOAD EMPLOYEE FACE DATABASE
# ==========================================

with open("models/face_database.pkl", "rb") as file:
    face_database = pickle.load(file)


# Normalize stored embeddings
for name in face_database:

    embeddings = np.asarray(
        face_database[name],
        dtype=np.float32
    )

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    face_database[name] = embeddings / norms


# ==========================================
# DISPLAY INFORMATION
# ==========================================

print()
print("===================================")
print("AI FACE RECOGNITION")
print("===================================")

print("Employees in database:")

for name in face_database:
    print("-", name)

print()
print("Local image server:")
print("http://127.0.0.1:8000")

print()


# ==========================================
# ENTER IMAGE URL
# ==========================================

image_url = input("Enter Image URL: ").strip()


# ==========================================
# DOWNLOAD IMAGE
# ==========================================

try:

    response = requests.get(
        image_url,
        timeout=15
    )

    if response.status_code != 200:

        print()
        print("Could not download image.")
        print("HTTP Status:", response.status_code)
        exit()

    image_bytes = response.content

except Exception as e:

    print()
    print("Could not access image URL.")
    print("Error:", e)
    exit()


# ==========================================
# CONVERT IMAGE
# ==========================================

image_array = np.frombuffer(
    image_bytes,
    np.uint8
)

image = cv2.imdecode(
    image_array,
    cv2.IMREAD_COLOR
)

if image is None:

    print()
    print("Invalid image.")
    print("Please enter a direct image URL.")
    exit()


# ==========================================
# DETECT FACE
# ==========================================

faces = face_app.get(image)


if len(faces) == 0:

    result = {
        "name": None,
        "recognized": False,
        "similarity": 0,
        "message": "No face detected"
    }

    print()
    print("===================================")
    print("RESULT")
    print("===================================")
    print(result)

    exit()


# Select largest face
face = max(
    faces,
    key=lambda x:
    (x.bbox[2] - x.bbox[0]) *
    (x.bbox[3] - x.bbox[1])
)


# ==========================================
# CREATE FACE EMBEDDING
# ==========================================

embedding = np.asarray(
    face.embedding,
    dtype=np.float32
)

norm = np.linalg.norm(embedding)

if norm == 0:

    result = {
        "name": None,
        "recognized": False,
        "similarity": 0,
        "message": "Invalid face embedding"
    }

    print(result)
    exit()


embedding = embedding / norm


# ==========================================
# COMPARE WITH ALL EMPLOYEES
# ==========================================

best_name = None
best_similarity = -1


for name, known_embeddings in face_database.items():

    similarities = np.dot(
        known_embeddings,
        embedding
    )

    similarity = float(
        np.max(similarities)
    )

    if similarity > best_similarity:

        best_similarity = similarity
        best_name = name


# ==========================================
# RECOGNITION THRESHOLD
# ==========================================

threshold = 0.40


if best_similarity >= threshold:

    result = {
        "name": best_name,
        "recognized": True,
        "similarity": round(best_similarity, 3)
    }

else:

    result = {
        "name": None,
        "recognized": False,
        "similarity": round(best_similarity, 3)
    }


# ==========================================
# DISPLAY JSON RESULT
# ==========================================

print()
print("===================================")
print("RESULT")
print("===================================")

print("{")

print(
    f'    "name": '
    f'"{result["name"]}",'
)

print(
    f'    "recognized": '
    f'{str(result["recognized"]).lower()},'
)

print(
    f'    "similarity": '
    f'{result["similarity"]}'
)

print("}")

print("===================================")