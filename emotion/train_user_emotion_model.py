import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh

def extract_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    return None

def load_data(data_path):
    X, y = [], []
    emotions = os.listdir(data_path)

    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.isdir(emotion_path):
            continue

        print(f"[INFO] Loading {emotion} samples...")
        for img_file in tqdm(os.listdir(emotion_path)):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(emotion_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            landmarks = extract_landmarks(image)
            if landmarks is not None:
                X.append(landmarks)
                y.append(emotion)
            else:
                print(f"[WARNING] No face found in: {img_path}")

    return np.array(X), np.array(y)

def train_and_save(user_name, data_dir='user_data', model_dir='models'):
    user_path = os.path.join(data_dir, user_name)
    if not os.path.exists(user_path):
        raise FileNotFoundError(f"No such user folder: {user_path}")

    X, y = load_data(user_path)
    if len(X) < 2:
        raise ValueError("Not enough samples. Please collect more images.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[INFO] Training classifier...")
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_scaled, y)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({'model': clf, 'scaler': scaler}, os.path.join(model_dir, f"{user_name}_mediapipe_emotion.pkl"))
    print(f"[SUCCESS] Model saved for user '{user_name}'.")

if __name__ == "__main__":
    username = input("Enter your username: ").strip()
    train_and_save(username)

