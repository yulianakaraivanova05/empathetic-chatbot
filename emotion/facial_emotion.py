import cv2
import threading 
import tempfile
import time
import os
from deepface import DeepFace
from collections import Counter
import numpy as np
import mediapipe as mp
import joblib


#class for the video capture object
class VideoRec:
    def __init__(self, filename=None):
        #testing to check which cameras are working
        #camera_index = self.find_working_camera()
        #self.cap = cv2.VideoCapture(camera_index)

        #creates the video capture
        self.cap = cv2.VideoCapture(1) 
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = None
        self.running = False
        self.subfolder = "video_recs"

        #some of the custom classifier iniatiations
        user_profile = "yuliana"
        self.model_path = f"emotion/models/{user_profile}_mediapipe_emotion.pkl"
        self.classifier = None
        self.scaler = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        self.load_classifier() 

        #creates a temp file for the capture to be stored in
        if filename is None:
            self.tempfile = tempfile.NamedTemporaryFile(delete=False, suffix=".avi", dir=self.subfolder)
            self.filename = self.tempfile.name
            self.tempfile.close()

        else:
            self.filename = filename

    #method to start the recording
    def start(self):
        if not self.cap.isOpened(): #error handling
            raise Exception("Webcam not accessible.")
        self.running = True #changes the running attribute to true
        #gets the width and height of the frame
        width  = int(self.cap.get(3))
        height = int(self.cap.get(4))

        #starts the recording
        self.out = cv2.VideoWriter(self.filename, self.fourcc, 20.0, (width, height))

        self.thread = threading.Thread(target=self.record)
        self.thread.start()
        print("Recording started...")

    #method to record the video
    def record(self):
        try:
            #uses a loop to keep the recording running until the stop method is called
            while self.running:
            #reads and writes to the file
                ret, frame = self.cap.read()
                if ret:
                    self.out.write(frame)
                else:
                    print("Frame capture failed.")
                    break
                time.sleep(0.05)  # 20 FPS
        except Exception as e: #error handling
            print(f"Exception in video recording thread: {e}")
            self.running = False

    #method to stop the recording
    def stop(self):
        #stops all processes and sets the running attribute to false
        self.running = False
        self.thread.join()
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print(f"Recording stopped. Saved to {self.filename}")
        return self.filename
    
    #method to find the working camera - used for testing originally
    def find_working_camera(self,max_index=5):
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.read()[0]:
                cap.release()
                return i
            cap.release()
        raise Exception("No working webcam found.")

    

#Start of video emotion detection
    def extract_emotion(self, video_path):
        cap = cv2.VideoCapture(video_path)

        deepface_emotions = []
        custom_emotions = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 5 == 0:
                # --- DeepFace ---
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    if isinstance(result, list):
                        result = result[0]
                    deepface_emotions.append(result["dominant_emotion"])
                except:
                    pass

                # --- Custom Classifier ---
                landmarks = self.get_landmarks(frame)
                if landmarks is not None and self.classifier and self.scaler:
                    features = self.scaler.transform([landmarks])
                    try:
                        pred = self.classifier.predict(features)[0]
                        custom_emotions.append(pred)
                    except Exception as e:
                        print(f"[WARN] Custom classifier failed: {e}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Most common from each method
        deepface_final = Counter(deepface_emotions).most_common(1)[0][0] if deepface_emotions else "neutral"
        custom_final = Counter(custom_emotions).most_common(1)[0][0] if custom_emotions else "neutral"

        print(f"[DEBUG] DeepFace Emotion: {deepface_final}")
        print(f"[DEBUG] Custom Model Emotion: {custom_final}")

        # return both for fusion
        return {
            "deepface": deepface_final,
            "custom": custom_final
        }
    
    def get_landmarks(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
        return None 
    
    def load_classifier(self):
        try:
            data = joblib.load(self.model_path)
            self.classifier = data['model']
            self.scaler = data['scaler']
            print(f"[INFO] Loaded classifier for user: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] Could not load custom classifier: {e}")

#OpenCV and facial emotion detection logic