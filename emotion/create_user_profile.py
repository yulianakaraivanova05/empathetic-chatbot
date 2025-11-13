import cv2
import os
import time

# === Configuration ===
EMOTIONS = ['happy', 'sad', 'angry', 'surprised', 'neutral','fear','disgust','frustration']
#EMOTIONS = ['fear','disgust','frustration']
CAPTURE_DURATION = 5  # seconds per emotion
FRAME_SKIP = 3         # skip every n frames to reduce redundancy

def create_user_directory(username):
    base_path = os.path.join('user_data', username)
    os.makedirs(base_path, exist_ok=True)
    for emotion in EMOTIONS:
        os.makedirs(os.path.join(base_path, emotion), exist_ok=True)
    return base_path

def collect_emotion_data(username):
    base_path = create_user_directory(username)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print(f"\n[INFO] Starting emotion capture for user: {username}")
    print("[INFO] Press 'q' to quit early.\n")

    for emotion in EMOTIONS:
        print(f"\nPlease show the emotion: '{emotion.upper()}'")
        time.sleep(2)
        print("[INFO] Recording...")

        start_time = time.time()
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame.")
                break

            elapsed_time = time.time() - start_time
            if elapsed_time > CAPTURE_DURATION:
                break

            # Display the recording window
            cv2.imshow("Recording - Press 'q' to skip", frame)

            # Save every Nth frame to avoid duplicates
            if frame_count % FRAME_SKIP == 0:
                filename = os.path.join(base_path, emotion, f"{emotion}_{saved_count}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"[INFO] Saved {saved_count} frames for emotion '{emotion}'.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Emotion data collection complete for user: {username}")

if __name__ == "__main__":
    print("Facial Emotion Dataset Creator")
    user = input("Enter your username: ").strip().lower()
    collect_emotion_data(user)
