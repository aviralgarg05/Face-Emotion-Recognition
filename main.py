import cv2
import numpy as np
import base64
from deepface import DeepFace
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import logging
import mediapipe as mp
import uvicorn
from collections import defaultdict
from datetime import datetime
import csv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize background removal with mediapipe's SelfieSegmentation
segmentor = None

# Initialize variables
face_cascade = None

# Global dictionary to track detected faces
next_face_id = 0

# Emotion productivity weights (can be adjusted based on your needs)
emotion_weights = {
    "happy": 1.0,
    "neutral": 0.7,
    "surprise": 0.5,
    "sad": -0.5,
    "angry": -1.0,
    "fear": -0.8,
    "disgust": -0.7,
}

# CSV file path for storing face logs
csv_file_path = "face_data.csv"


# inilialzing services like logging only once
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        filename="app_logs.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_face_cascade():
    global face_cascade
    if face_cascade is None:
        # Initialize the Haar Cascade only when needed
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
        )
    return face_cascade


def get_segmentor():
    global segmentor
    if segmentor is None:
        # Initialize SelfieSegmentation only when needed
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    return segmentor


# Initialize the CSV file by writing the headers if the file doesn't exist
def init_csv():
    try:
        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            file.seek(0)  # Move to the beginning of the file
            if file.read(1):  # Check if the file is not empty (already has header)
                return
            # Write header if CSV is empty
            writer.writerow(
                [
                    "face_id",
                    "time_in",
                    "time_out",
                    "duration",
                    "emotions",
                    "productivity",
                ]
            )
        logging.info("CSV file initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing the CSV file: {e}")


# Function to insert face data into the CSV file
def insert_face_log(face_id, time_in, time_out, duration, emotions, productivity):
    try:
        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [time_in, time_out, face_id, duration, ",".join(emotions), productivity]
            )
        logging.info(f"Inserted face data for Face ID {face_id} into CSV file.")
    except Exception as e:
        logging.error(f"Error inserting face data into CSV: {e}")


def insert_face_emotion(currTime, faceId, detectedEmotion):
    try:
        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([currTime, faceId, detectedEmotion])
        print(f"wrote csv for {faceId}{currTime}")
    except Exception as e:
        print(f"error inserting record in csv {e}")


# Analyze face for emotion using DeepFace
def analyze_face(face_roi):
    try:
        result = DeepFace.analyze(
            face_roi, actions=["emotion"], enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]
        emotion = result.get("dominant_emotion")
        return emotion
    except Exception as e:
        logging.error(f"Error analyzing face: {e}")
        return None


# Function to calculate productivity based on emotions
def calculate_productivity(emotions):
    if not emotions:
        return 0.0
    weighted_sum = sum(emotion_weights.get(emotion, 0) for emotion in emotions)
    return max(
        0, min(100, (weighted_sum / len(emotions)) * 100)
    )  # Clamp between 0 and 100


# Function to match or assign a new face ID using DeepFace embeddings
def get_or_assign_face_id(face_roi):
    global next_face_id, face_ids

    # Get face embedding using DeepFace
    try:
        face_embedding = DeepFace.represent(face_roi, enforce_detection=False)[0][
            "embedding"
        ]

        # Try to match with existing face embeddings
        for face_id, data in face_ids.items():
            if (
                "embedding" in data
                and np.linalg.norm(
                    np.array(data["embedding"]) - np.array(face_embedding)
                )
                < 0.8
            ):
                return face_id

        maxKey = None

        if len(face_ids.keys()) > 0:
            maxKey = max(face_ids.keys())
        else:
            maxKey = 0

        # If no match, assign a new face ID
        face_ids[maxKey + 1] = {"embedding": face_embedding}

        return maxKey + 1
    except Exception as e:
        logging.error(f"Error extracting face embedding: {e}")
        print("error ", e)


def is_base64(s: str) -> bool:
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


face_ids = {}  # Dictionary to store face embeddings and face IDs
face_data = defaultdict(lambda: {"time_in": None, "time_out": None, "emotions": []})


# Function to continuously capture frames from the camera
def capture_and_process_video(imageString: str) -> None:
    global face_ids, face_data
    detected_face_ids = set()

    # ret, frame = camera.read()  # Read a frame from the webcam
    isValidBaseString = is_base64(imageString)

    if not isValidBaseString:
        print("INVALID BASE64 STRING PROVIDED")
        logging.error("Failed to capture frame from webcam")
        return

    # converting base64 string in np array
    image_data = base64.b64decode(imageString)

    np_array = np.frombuffer(image_data, np.uint8)

    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Read as a color image (BGR)

    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )

    print("faces found in image", len(faces))

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Track the faces detected in this frame

    # If faces are found, process each face
    for x, y, w, h in faces:
        face_roi = image[y : y + h, x : x + w]
        face_id = get_or_assign_face_id(face_roi)

        if face_id is None:
            continue

        detected_face_ids.add(face_id)

        # if the person is already present in facedata
        if face_data[face_id]["time_out"] is not None:
            face_data[face_id]["time_in"] = current_time
            face_data[face_id]["time_out"] = None
            face_data[face_id]["emotions"] = []

        # If it's the first time we see this face, log time_in and store base64 image
        if face_data[face_id]["time_in"] is None:
            face_data[face_id]["time_in"] = current_time
            # face_data[face_id]["base64_image"] = image_to_base64(face_roi)
            # log_base64(face_id, face_data[face_id]["base64_image"])  # Log the base64 string

        # Analyze face for emotion
        emotion = analyze_face(face_roi)
        if emotion:
            face_data[face_id]["emotions"].append(emotion)
            insert_face_emotion(
                datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S"), face_id, emotion
            )
            # logging.info(f"Face ID {face_id} detected emotion: {emotion}")

    for face_id in list(face_data.keys()):
        if face_id not in detected_face_ids:

            try:

                face_data[face_id]["time_out"] = current_time
                time_in = datetime.strptime(
                    face_data[face_id]["time_in"], "%Y-%m-%d %H:%M:%S"
                )

                time_out = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                duration = (time_out - time_in).total_seconds()  # Duration in seconds

                cumulative_emotions = face_data[face_id]["emotions"]
                productivity = calculate_productivity(cumulative_emotions)

                insert_face_log(
                    face_id,
                    face_data[face_id]["time_in"],
                    current_time,
                    duration,
                    cumulative_emotions,
                    productivity,
                )

                # detected_face_ids.remove(face_id)
                del face_data[face_id]
            except Exception as e:
                print("error on server side", e)

    print("face data after", face_data)
    print("face ids we got", face_ids.keys())


hasIntialised = False


@app.post("/process_frame")
async def process_frame(params: dict = Body(...)):
    global hasIntialised

    if not hasIntialised:
        setup_logging()
        get_face_cascade()
        get_segmentor()
        hasIntialised = True

    # return {"processed": params.get("image")}

    image = params.get("image")
    prefix1 = "data:image/jpeg;base64,"
    prefix2 = "data:image/png:base64,"
    prefix3 = "data:image/jpg:base64,"

    if image.startswith(prefix1):
        image = image[len(prefix1) :]

    if image.startswith(prefix2):
        image = image[len(prefix2) :]

    if image.startswith(prefix3):
        image = image[len(prefix3) :]

    try:
        # Start the video capture and processing in a separate thread
        # thread = threading.Thread(target=capture_and_process_video, args=(image,))
        # thread.start()
        capture_and_process_video(image)

        return {
            "message": "Video processing started. Check logs for emotion detection results."
        }
    except Exception as e:
        logging.error(f"Error starting video feed: {e}")
        raise HTTPException(status_code=500, detail="Failed to log facial expressions")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}


if __name__ == "__main__":
    # start_time = time.perf_counter()

    uvicorn.run(app, host="0.0.0.0", port=8000)
