import cv2
import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# -----------------------------------
# Load InsightFace model
# -----------------------------------

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

# -----------------------------------
# Dataset folder
# -----------------------------------

dataset_path = "dataset"

# Dictionary to store embeddings
face_database = {}

# -----------------------------------
# Process each employee
# -----------------------------------

for person_name in os.listdir(dataset_path):

    person_path = os.path.join(
        dataset_path,
        person_name
    )

    # Skip files
    if not os.path.isdir(person_path):
        continue

    print()
    print("Processing:", person_name)

    person_embeddings = []

    # Process each image
    for filename in sorted(os.listdir(person_path)):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            person_path,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            print("Could not read:", filename)
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print("No face found:", filename)
            continue

        # Select largest detected face
        face = max(
            faces,
            key=lambda x:
            (x.bbox[2] - x.bbox[0]) *
            (x.bbox[3] - x.bbox[1])
        )

        embedding = face.embedding

        person_embeddings.append(embedding)

        print("Processed:", filename)

    # Save employee embeddings
    if len(person_embeddings) > 0:

        face_database[person_name] = np.array(
            person_embeddings
        )

        print(
            person_name,
            "embeddings:",
            len(person_embeddings)
        )

# -----------------------------------
# Save multi-employee database
# -----------------------------------

if len(face_database) == 0:

    print()
    print("No face embeddings were created.")

else:

    os.makedirs("models", exist_ok=True)

    with open(
        "models/face_database.pkl",
        "wb"
    ) as file:

        pickle.dump(
            face_database,
            file
        )

    print()
    print("===================================")
    print("Multi-employee embedding generation completed!")
    print("Employees:", list(face_database.keys()))

    for name, embeddings in face_database.items():

        print(
            name,
            "→",
            embeddings.shape
        )

    print()
    print("Saved to:")
    print("models/face_database.pkl")
    print("===================================")